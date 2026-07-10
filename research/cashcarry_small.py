"""cash-and-carry 小账户版(<$10万,capacity 非约束):只在肥 funding 的中小币里、等权分散、单币帽死、regime过滤。
回答用户:多做中小/偏死的币、稳定 sharpe 行不行?不是 88% 押一个(彩票),是分散到很多肥funding小币。
构造:members=active&funding>F_ENTER;等权 1/n;gross=min(1,n/Nmin)→单币恒帽 1/Nmin,肥币太少自动退现金。
真实小币成本扫 5/10/20bps(双腿)。badbar 清洗(funding也清,不吃爆裂)。死币在内。逐年+宽度(证明非彩票)。
跑：python cashcarry_small.py
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
ap.add_argument("--Nmin", type=int, default=10, help="肥币数到此=满仓;少于则按比例退现金(单币恒帽1/Nmin)")
a = ap.parse_args()

z = np.load(a.tensor, allow_pickle=True)
mask = z["mask"]; adj = z["adj_close"].astype(np.float64); tdv = z["tdv"].astype(np.float64)
dates = z["dates"].astype(str)
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
ret_bad = (np.abs(pr) > 1.0) | (np.abs(sr) > 1.0)
basis_bad = ~np.isfinite(basis) | (np.abs(basis) > 0.5)
pr[ret_bad] = 0.0; sr[ret_bad] = 0.0
price_leg = np.nan_to_num(sr - pr); price_leg[ret_bad | basis_bad] = 0.0
f0[ret_bad | basis_bad] = 0.0                                # 坏bar清funding(不吃爆裂的funding)
active = vp & np.isfinite(funding) & ~basis_bad


def run(F, bps):
    cur = np.zeros(N); pnl = np.zeros(T); turn = np.zeros(T); nheld = np.zeros(T); maxw = np.zeros(T)
    for t in range(T):
        prev = cur.copy()
        if t % a.cad == 0:
            mem = active[t] & (funding[t] > F)
            n = int(mem.sum())
            if n > 0:
                w = mem.astype(np.float64) / n * min(1.0, n / a.Nmin)   # 等权;肥币<Nmin→退现金;单币恒帽1/Nmin
            else:
                w = np.zeros(N)
            d = w - cur; mv = np.abs(d) > a.band
            new = cur.copy(); new[mv] = cur[mv] + a.rebal * d[mv]; cur = new
        turn[t] = np.abs(cur - prev).sum()
        pnl[t] = (cur * (price_leg[t] + f0[t] / 8.0)).sum() - 2 * (bps / 1e4) * turn[t]
        held = cur > 1e-9; nheld[t] = held.sum(); maxw[t] = cur.max()
    return pnl, turn, nheld, maxw


def sh(p, m=None):
    p = p if m is None else p[m]
    return p.mean() / p.std() * np.sqrt(a.ann) if p.std() > 0 else 0.0


print(f"等权分散肥funding小币;单币帽=1/{a.Nmin}={100/a.Nmin:.0f}%。逐年+宽度(证明非单币彩票)\n")
for F, lab in [(5e-5, "F>5e-5(~5%/yr)"), (1e-4, "F>1e-4(~11%/yr)"), (2e-4, "F>2e-4(~22%/yr)"), (4e-4, "F>4e-4(~44%/yr)")]:
    print(f"--- {lab} ---")
    print(f"{'bps':>5s} {'val':>6s} {'OOS':>6s} {'2024':>6s} {'2025':>6s} {'2026':>6s} {'年化':>7s} {'均持币':>6s} {'最大单币':>7s}")
    for bps in (5, 10, 20):
        pnl, tn, nh, mw = run(F, bps)
        print(f"{bps:>5d} {sh(pnl[:cut]):>+6.2f} {sh(pnl[oos]):>+6.2f} {sh(pnl[oos&(yr=='2024')]):>+6.2f}"
              f" {sh(pnl[oos&(yr=='2025')]):>+6.2f} {sh(pnl[oos&(yr=='2026')]):>+6.2f}"
              f" {pnl[oos].mean()*a.ann:>+6.1%} {nh[oos&(nh>0)].mean():>6.1f} {mw[oos].max()*100:>6.0f}%")
    print()

print("判读: 逐年(含2026)稳定正、均持币多(分散非彩票)、扛得住10-20bps小币成本 = 真适合小账户。否则=肥funding=赔尾部钱、不稳。")
