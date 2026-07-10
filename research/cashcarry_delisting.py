"""死币占 cash-and-carry 67% P&L:是真(价格中性安全收高funding的将死币)还是假(退市平仓假设太乐观)?
查:① 死币P&L=funding腿 vs 价格腿(若≈全funding=真收割)② 集中度(几个币?)③ 退市平仓惩罚 sweep(0/2/5/10%)。
跑：python cashcarry_delisting.py
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
ap.add_argument("--band", type=float, default=0.01); ap.add_argument("--bps", type=float, default=5)
a = ap.parse_args()

z = np.load(a.tensor, allow_pickle=True)
mask = z["mask"]; adj = z["adj_close"].astype(np.float64)
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
fund_leg = f0 / 8.0

W = np.zeros((T, N))
for t in range(T):
    idx = np.where(active[t])[0]
    if len(idx) == 0:
        continue
    w = np.clip(f0[t, idx], 0, None); s = w.sum()
    if s > 0:
        W[t, idx] = w / s

cur = np.zeros(N); held = np.zeros((T, N)); turn = np.zeros(T)
for t in range(T):
    prev = cur.copy()
    if t % a.cad == 0:
        d = W[t] - cur; mv = np.abs(d) > a.band; new = cur.copy(); new[mv] = cur[mv] + a.rebal * d[mv]; cur = new
    turn[t] = np.abs(cur - prev).sum(); held[t] = cur

last_active = np.array([np.where(active[:, c])[0].max() if active[:, c].any() else -1 for c in range(N)])
delisted = (last_active >= 0) & (last_active < T - 24 * 14)
fpnl = held * fund_leg; ppnl = held * price_leg


def sh(p, m=None):
    p = p if m is None else p[m]
    return p.mean() / p.std() * np.sqrt(a.ann) if p.std() > 0 else 0.0


print(f"宇宙 {int(vp.any(0).sum())} 币;退市 {int(delisted.sum())}\n")
print("=== ① 死币 P&L 拆 funding腿 vs 价格腿(全样本累计) ===")
print(f"  死币: funding {fpnl[:,delisted].sum():+.3f}  价格 {ppnl[:,delisted].sum():+.3f}  合计 {(fpnl+ppnl)[:,delisted].sum():+.3f}")
print(f"  活币: funding {fpnl[:,~delisted].sum():+.3f}  价格 {ppnl[:,~delisted].sum():+.3f}  合计 {(fpnl+ppnl)[:,~delisted].sum():+.3f}")
print("  (死币若≈全 funding=价格中性真收割,退市无方向损;价格腿大负=平仓在吃亏)")

print("\n=== ② 死币 P&L 集中度(前几个币?) ===")
dc = (fpnl + ppnl).sum(0)
order = np.argsort(dc)[::-1]
top = [o for o in order if delisted[o]][:8]
tot_d = dc[delisted].sum()
for o in top:
    print(f"  {slots[o]:18s} P&L={dc[o]:+.4f} ({dc[o]/tot_d*100:.0f}% of 死币)")

print("\n=== ③ 退市平仓惩罚 sweep(每个死币在 last_active 按 |持仓| 罚 pen%;看OOS Sharpe降多少) ===")
pnl0 = (held * (price_leg + fund_leg)).sum(1) - 2 * (a.bps / 1e4) * turn
print(f"  {'惩罚':>8s} {'OOS':>6s} {'2024':>6s} {'2025':>6s} {'2026':>6s}")
for pen in (0.0, 0.02, 0.05, 0.10):
    pnl = pnl0.copy()
    if pen > 0:
        for c in np.where(delisted)[0]:
            la = last_active[c]
            if 0 <= la < T:
                pnl[la] -= pen * abs(held[la, c])           # 退市那刻按持仓罚 pen
    print(f"  {pen*100:>6.0f}% {sh(pnl[oos]):>+6.2f} {sh(pnl[oos&(yr=='2024')]):>+6.2f}"
          f" {sh(pnl[oos&(yr=='2025')]):>+6.2f} {sh(pnl[oos&(yr=='2026')]):>+6.2f}")

print("\n判读: ①死币≈全funding+价格腿小 ②不靠1-2个币 ③5-10%重罚后OOS仍正 → 死币P&L是真收割、非平仓幻觉。")
