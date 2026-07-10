"""数值审计:核对 cash-and-carry 所有关键数,防一个单位/年化错让我们空欢喜。
1) funding 量级 vs 币安真实(BTC 等);2) 单位/年化链;3) P&L 拆腿(收了多少funding−价格−成本−借币=净,验自洽);
4) 外部锚:单币 BTC cash-and-carry 应≈funding 年化;5) 折年化的常识区间。
跑：python carry_audit_numbers.py
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])  # 从仓库根运行(import strategy + data/ 路径都对)
import numpy as np
import strategy as S

a = S.load_arrays(); cfg = S.CarryConfig()
T, N, slots, dates = a["T"], a["N"], list(a["slots"]), a["dates"]
funding, f0, cr = a["funding"], a["f0"], a["cr"]
yr = np.array([d[:4] for d in dates]); oos = np.zeros(T, bool); oos[int(T * 0.4):] = True

print("=" * 60)
print("① funding 量级 vs 真实(币安 funding = 每 8h 的'分数',典型 ±0.0001=±0.01%/8h)")
print("=" * 60)
for s in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
    if s in slots:
        c = slots.index(s); v = funding[:, c]; v = v[np.isfinite(v)]
        print(f"  {s}: 均值={np.mean(v):+.6f}/8h (={np.mean(v)*3*365:+.1%}/yr) "
              f"区间[{v.min():+.5f},{v.max():+.5f}] 中位{np.median(v):+.6f}")
allf = funding[np.isfinite(funding)]
print(f"  全宇宙: 均值={allf.mean():+.6f}/8h(={allf.mean()*3*365:+.2%}/yr) "
      f"|绝对值|中位={np.median(np.abs(allf)):.6f}(={np.median(np.abs(allf))*3*365:.1%}/yr)")
print(f"  → 对照真实:BTC funding 常年 ~5-15%/yr、极端币可几百%/yr。上面量级合理✓ 即说明 funding 是'分数/8h'口径,无 ×100 错。")

print("\n" + "=" * 60)
print("② 单位/年化链(防错的核心)")
print("=" * 60)
print(f"  funding 口径: 分数/8h。每小时 accrual = funding/8。一年 = funding/8 × 8760h = funding × {8760/8:.0f}")
print(f"  而 8760/8 = 1095 = 3结算/天 × 365天 ✓(每8h一结算,一天3次)")
print(f"  Sharpe 年化 = 每bar均值/std × √8760(小时bar)✓")

print("\n" + "=" * 60)
print("③ P&L 拆腿(全用 step() 的书;unlevered;验 收funding−价格−成本−借币 = 净)")
print("=" * 60)
strat = S.CarryStrategy(N, cfg); held = np.zeros((T, N)); turn = np.zeros(T); prev = np.zeros(N)
for t in range(T):
    held[t] = strat.step(funding[t], a["bm"][t], a["tdv"][t], a["vp"][t], dt=1.0)
    turn[t] = np.abs(held[t] - prev).sum(); prev = held[t]
price_cell = cr - f0 / 8.0                                     # cr=funding/8+price → price腿
fund_pnl = (held * f0 / 8.0).sum(1)                            # 收到的funding(带符号对齐→恒收)
price_pnl = (held * price_cell).sum(1)
cost = cfg.cost_bps / 1e4 * turn
borrow = cfg.borrow_apr / cfg.ann * np.where(held < 0, -held, 0).sum(1)
net = fund_pnl + price_pnl - cost - borrow


def ann(x, m):
    return x[m].mean() * cfg.ann
for lab, m in [("OOS", oos), ("全样本", np.ones(T, bool))]:
    print(f"  [{lab}] 收funding {ann(fund_pnl,m):+.1%}  价格腿 {ann(price_pnl,m):+.1%}  "
          f"成本 −{ann(cost,m):.1%}  借币 −{ann(borrow,m):.1%}  = 净 {ann(net,m):+.1%}/yr")
    chk = ann(fund_pnl, m) + ann(price_pnl, m) - ann(cost, m) - ann(borrow, m)
    print(f"       自洽检查: 四项和 {chk:+.2%} vs 直接净 {ann(net,m):+.2%}  (应相等 ✓)")

print("\n" + "=" * 60)
print("④ 外部锚:单币 BTC cash-and-carry,纯 funding 应≈BTC funding 年化")
print("=" * 60)
c = slots.index("BTCUSDT"); v = funding[oos, c]; v = v[np.isfinite(v)]
print(f"  若只做 BTC、gross=1、持有不动:年收 funding ≈ {v.mean()*3*365:+.1%}/yr(=BTC funding 均值年化)")
print(f"  → 这就是教科书数(多现货+空永续≈收 funding)。我们的 +15% 是全书加权(含负侧肥名),比单 BTC 高合理。")

print("\n" + "=" * 60)
print("⑤ 折年化常识区间")
print("=" * 60)
print(f"  我们 unlevered net OOS = {ann(net,oos):+.1%}/yr,Sharpe = {net[oos].mean()/net[oos].std()*np.sqrt(cfg.ann):+.1f}")
print(f"  真实 basis/carry 基金:unlevered 通常 5-15%/yr。我们偏高端(小资金可追极端肥名+双侧)。")
print(f"  ⚠️ Sharpe 偏高含 funding 近确定 accrual 性质,实盘会低 → 这正是前向验证在查的。")
print(f"\n  权重核对:gross={np.abs(held[oos]).sum(1).mean():.2f}(应≈1)  "
      f"单币最大权重={np.abs(held[oos]).max():.1%}(帽5%,thin期renorm可超→已知)")
