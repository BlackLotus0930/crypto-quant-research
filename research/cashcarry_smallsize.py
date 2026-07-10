"""小资金视角:容量不再是约束(能进微盘),但微盘真实价差大。用"流动性相关成本"(按成交额估每币价差)重测。
问题从"能不能上量"变成"微盘真实价差吃完,funding 还剩不剩"。允许集中(小资金能进 LEVER 那种)。
跑：python cashcarry_smallsize.py
"""
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--tensor", default="data/clean/crypto_tensor_60min_pit.npz")
ap.add_argument("--funding", default="data/clean/funding_pit.npz")
ap.add_argument("--spot", default="data/clean/spot_pit.npz")
ap.add_argument("--ann", type=int, default=8760)
ap.add_argument("--split", type=float, default=0.4)
ap.add_argument("--cad", type=int, default=24); ap.add_argument("--rebal", type=float, default=0.3)
ap.add_argument("--band", type=float, default=0.01)
a = ap.parse_args()

z = np.load(a.tensor, allow_pickle=True)
mask = z["mask"]; adj = z["adj_close"].astype(np.float64); tdv = z["tdv"].astype(np.float64)
dates = z["dates"].astype(str); slots = z["slots"].astype(str)
T, N = mask.shape
funding = np.load(a.funding, allow_pickle=True)["funding"].astype(np.float64)
spot = np.load(a.spot, allow_pickle=True)["spot"].astype(np.float64)
yr = np.array([d[:4] for d in dates]); cut = int(T * a.split)
oos = np.zeros(T, bool); oos[cut:] = True

vp = mask & np.isfinite(spot) & (spot > 0) & (adj > 0)
f0 = np.nan_to_num(funding)
pr = np.zeros((T, N)); sr = np.zeros((T, N)); vpp = vp[:-1] & vp[1:]
pr[:-1][vpp] = adj[1:][vpp] / adj[:-1][vpp] - 1
sr[:-1][vpp] = spot[1:][vpp] / spot[:-1][vpp] - 1
basis = np.full((T, N), np.nan); basis[vp] = adj[vp] / spot[vp] - 1
basis_bad = ~np.isfinite(basis) | (np.abs(basis) > 0.5)
active = vp & np.isfinite(funding) & ~basis_bad
price_leg = np.clip(np.nan_to_num(sr - pr), -0.3, 0.3); price_leg[basis_bad] = 0.0

# 流动性相关价差(每腿单边 bps):按日成交额(tdv是7日滚动$额)分档
dvol = tdv / 7.0
spread = np.select([dvol > 50e6, dvol > 10e6, dvol > 1e6, dvol > 0.1e6],
                   [5.0, 12.0, 35.0, 100.0], default=250.0)        # 微盘价差大

W = np.zeros((T, N))
for t in range(T):
    idx = np.where(active[t])[0]
    if len(idx) == 0:
        continue
    w = np.clip(f0[t, idx], 0, None); s = w.sum()
    if s > 0:
        W[t, idx] = w / s


def run(spread_mode):
    """spread_mode: 'flat5'=平5bps; 'liq'=流动性相关; cost 双腿。"""
    cur = np.zeros(N); pnl = np.zeros(T); cost = np.zeros(T); hspr = np.zeros(T)
    for t in range(T):
        prev = cur.copy()
        if t % a.cad == 0:
            d = W[t] - cur; mv = np.abs(d) > a.band; new = cur.copy(); new[mv] = cur[mv] + a.rebal * d[mv]; cur = new
        dlt = np.abs(cur - prev)
        sp = np.full(N, 5.0) if spread_mode == "flat5" else spread[t]
        cost[t] = (dlt * sp / 1e4 * 2).sum()                       # 双腿(现货+永续)各一次单边价差
        pnl[t] = (cur * (price_leg[t] + f0[t] / 8.0)).sum() - cost[t]
        g = np.abs(cur).sum(); hspr[t] = (np.abs(cur) * spread[t]).sum() / g if g > 0 else 0
    return pnl, hspr


def sh(p, m=None):
    p = p if m is None else p[m]
    return p.mean() / p.std() * np.sqrt(a.ann) if p.std() > 0 else 0.0


def line(name, pnl):
    print(f"  {name:>24s}: OOS {sh(pnl[oos]):>+5.2f}  2024 {sh(pnl[oos&(yr=='2024')]):>+5.2f}"
          f"  2025 {sh(pnl[oos&(yr=='2025')]):>+5.2f}  2026 {sh(pnl[oos&(yr=='2026')]):>+5.2f}"
          f"  年化 {pnl[oos].mean()*a.ann:>+5.1%}")


print("小资金=容量不约束(允许集中),但微盘价差大。重测:\n")
p_flat, _ = run("flat5"); line("平5bps(原幻觉)", p_flat)
p_liq, hspr = run("liq"); line("流动性相关价差", p_liq)

# 持仓的平均价差(我们是不是全在贵微盘?)+ 微盘占比
held_spread_oos = hspr[oos][hspr[oos] > 0].mean()
print(f"\n  持仓加权平均单边价差 ≈ {held_spread_oos:.0f} bps (说明 book 主要在多贵的币上)")

# 不同档位的 P&L 来源
cur = np.zeros(N); held = np.zeros((T, N))
for t in range(T):
    if t % a.cad == 0:
        d = W[t] - cur; mv = np.abs(d) > a.band; new = cur.copy(); new[mv] = cur[mv] + a.rebal * d[mv]; cur = new
    held[t] = cur
gross_pnl = held * (price_leg + f0 / 8.0)
tiers = [("大盘>50M", dvol > 50e6), ("中10-50M", (dvol > 10e6) & (dvol <= 50e6)),
         ("小1-10M", (dvol > 1e6) & (dvol <= 10e6)), ("微<1M", dvol <= 1e6)]
print("\n  毛P&L按流动性档(看edge靠不靠微盘):")
for nm, msk in tiers:
    print(f"    {nm:>12s}: 毛P&L {(gross_pnl*msk)[oos].sum():+.3f}  仓位占比 {(np.abs(held)*msk)[oos].sum()/np.abs(held)[oos].sum()*100:.0f}%")

print("\n判读: 流动性价差下仍正=小资金可吃;若靠'微<1M'档=吃的是真微盘,价差/操作/退市风险都在那。")
