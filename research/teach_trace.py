"""教学:把策略从【原始输入】一路追到【最终 Sharpe】,用真实数据、真实一根 bar。"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])  # 从仓库根运行(import strategy + data/ 路径都对)
import numpy as np
import strategy as S

cfg = S.CarryConfig()
# 直接读原始对齐矩阵,好把两腿拆开看
z = np.load("data/clean/crypto_tensor_60min_pit.npz", allow_pickle=True)
mask, adj, tdv = z["mask"], z["adj_close"].astype(float), z["tdv"].astype(float)
dates, slots = z["dates"].astype(str), z["slots"].astype(str)
fund = np.load("data/clean/funding_pit.npz", allow_pickle=True)["funding"].astype(float)
sp = np.load("data/clean/spot_pit.npz", allow_pickle=True)["spot"].astype(float)
T, N = mask.shape
a = S.load_arrays(cfg=cfg)                      # funding/f0/tdv/vp/cr/bm(与策略同口径)

# 跑策略,记录每 bar 持仓
strat = S.CarryStrategy(N, cfg)
held = np.zeros((T, N))
for t in range(T):
    held[t] = strat.step(a["funding"][t], a["bm"][t], a["tdv"][t], a["vp"][t], dt=1.0)

# 两腿(每币每bar)
f0 = a["f0"]; vp = a["vp"]
pr = np.zeros((T, N)); sr = np.zeros((T, N)); vpp = vp[:-1] & vp[1:]
pr[:-1][vpp] = adj[1:][vpp] / adj[:-1][vpp] - 1
sr[:-1][vpp] = sp[1:][vpp] / sp[:-1][vpp] - 1
fund_leg = held * (f0 / 8.0)                    # funding 收割
price_leg = held * np.clip(np.nan_to_num(sr - pr), -0.25, 0.25)  # 价格腿(对冲后)
turn = np.abs(np.diff(held, axis=0, prepend=0)).sum(1)
short_spot = np.where(held < 0, -held, 0).sum(1)
bar_ret = fund_leg.sum(1) + price_leg.sum(1) - cfg.cost_bps / 1e4 * turn - cfg.borrow_apr / cfg.ann * short_spot

# 选一根有代表性的 OOS bar(2025 年中,书已成熟)
t = int(np.where(dates == "2025-06-01 00:00:00")[0][0])
print("=" * 70)
print(f"【一根真实 bar 的全流程】 t={t}  时间={dates[t]} (UTC)")
print("=" * 70)

print("\n── 第①层:原始输入(这一小时,几个币的样子)──")
book = np.where(np.abs(held[t]) > 1e-9)[0]
order = book[np.argsort(-np.abs(held[t, book]))]
print(f"{'币':>14s} {'funding(8h)':>11s} {'≈年化':>8s} {'方向':>6s} {'权重':>7s}")
for i in order[:6]:
    side = "空永续" if held[t, i] > 0 else "空现货"
    print(f"{slots[i]:>14s} {fund[t,i]:>+11.6f} {fund[t,i]*3*365:>+7.0%} {side:>6s} {held[t,i]:>+7.2%}")
print(f"  ……共 {len(book)} 个持仓;funding>0→多现货空永续, funding<0→多永续空现货(都对冲掉价格)")

print("\n── 第②层:决策怎么来的(规则,以上面第一个币为例)──")
i = order[0]
print(f"  {slots[i]}: 当前 funding={fund[t,i]:+.6f}/8h → 方向=sign(funding)={'+ (空永续收费)' if fund[t,i]>0 else '− (空现货收费)'}")
print(f"  权重 ∝ |funding|, 每币帽 {cfg.cap:.0%}, 只在 top-{cfg.kpos}(正)/{cfg.kneg}(负) 流动币里, basis 正逆向飙的躲开")
print(f"  再经 EMA 平滑(hl={cfg.pos_hl})+ 每 {cfg.cad}bar 调仓 → 最终权重 {held[t,i]:+.2%}")

print("\n── 第③层:下一小时这笔 P&L 怎么算(同一个币 t→t+1)──")
print(f"  funding 腿 = 权重 × funding/8 = {held[t,i]:+.4f} × {f0[t,i]:+.6f}/8 = {fund_leg[t,i]:+.7f}")
print(f"  价格腿(对冲后)= 权重 ×(现货涨跌 − 永续涨跌)= {held[t,i]:+.4f} ×({sr[t,i]:+.5f} − {pr[t,i]:+.5f}) = {price_leg[t,i]:+.7f}")
print(f"  → 这个币这一 bar 贡献 = {fund_leg[t,i]+price_leg[t,i]:+.7f}  (价格被现货对冲≈抵消,留下 funding)")

print("\n── 第④层:这一小时整本书的收益(把所有币加起来 − 成本)──")
print(f"  Σfunding腿={fund_leg[t].sum():+.6f}  Σ价格腿={price_leg[t].sum():+.6f}  "
      f"成本={-cfg.cost_bps/1e4*turn[t]:+.6f}  借币={-cfg.borrow_apr/cfg.ann*short_spot[t]:+.6f}")
print(f"  ★ 这一 bar 的本书收益 = {bar_ret[t]:+.6f}  ({bar_ret[t]*100:+.4f}% 当小时)")

print("\n── 第⑤层:一串这样的 bar 收益 → 最终指标 ──")
cut = int(T * 0.4); oos = np.zeros(T, bool); oos[cut:] = True
r = bar_ret[oos]
mu, sd = r.mean(), r.std()
sharpe = mu / sd * np.sqrt(cfg.ann)
annual = mu * cfg.ann
cum = np.cumsum(r); mdd = (cum - np.maximum.accumulate(cum)).min()
print(f"  OOS 段共 {len(r)} 根 bar(=小时)。每 bar 收益的:")
print(f"    平均 μ={mu:.8f}/bar   标准差 σ={sd:.8f}/bar")
print(f"  年化 = μ × 8760 = {annual:+.1%}")
print(f"  Sharpe = μ/σ × √8760 = {mu:.2e}/{sd:.2e} × {np.sqrt(cfg.ann):.1f} = {sharpe:+.2f}")
print(f"  最大回撤 = 累计曲线最深的峰到谷 = {mdd:+.3f}")
print("\n  这就是 +15%/yr、Sharpe~4、maxDD−4% 的来历:不是单个大赢,是几万根这样的小 bar 累起来。")
print("=" * 70)
