"""回答"死币能稳定Sharpe就行,为什么是问题":死币在宇宙内是对的、且hedged=非方向灾难。
唯一真问题=死/illiquid 币的真实平仓摩擦是否远高于5bps。直接测:① 流动性缩放成本(illiquid 多付)② 退市平仓罚金。
基线=+1.72 构造(全宇宙 funding 权重、中等换手、无平滑)。跑：python cashcarry_friction.py
"""
import numpy as np
ANN = 8760
z = np.load("data/clean/crypto_tensor_60min_pit.npz", allow_pickle=True)
mask = z["mask"]; adj = z["adj_close"].astype(np.float64); tdv = z["tdv"].astype(np.float64)
dates = z["dates"].astype(str); T, N = mask.shape
funding = np.load("data/clean/funding_pit.npz", allow_pickle=True)["funding"].astype(np.float64)
spot = np.load("data/clean/spot_pit.npz", allow_pickle=True)["spot"].astype(np.float64)
yr = np.array([d[:4] for d in dates]); cut = int(T * 0.4); oos = np.zeros(T, bool); oos[cut:] = True

vp = mask & np.isfinite(spot) & (spot > 0) & (adj > 0); f0 = np.nan_to_num(funding)
pr = np.zeros((T, N)); sr = np.zeros((T, N)); vpp = vp[:-1] & vp[1:]
pr[:-1][vpp] = adj[1:][vpp] / adj[:-1][vpp] - 1; sr[:-1][vpp] = spot[1:][vpp] / spot[:-1][vpp] - 1
basis = np.full((T, N), np.nan); basis[vp] = adj[vp] / spot[vp] - 1
basis_bad = ~np.isfinite(basis) | (np.abs(basis) > 0.5)
price_leg = np.clip(np.nan_to_num(sr - pr), -0.3, 0.3); price_leg[basis_bad] = 0.0   # 因果清洗
active = vp & np.isfinite(funding) & ~basis_bad

# 退市:vp 最后一次 True 在数据末14天前
last_vp = np.array([np.where(vp[:, c])[0].max() if vp[:, c].any() else -1 for c in range(N)])
delisted = (last_vp >= 0) & (last_vp < T - 24 * 14)
# 流动性成本倍率:illiquid(低tdv)多付,base×clip((tdv_ref/tdv)^.5,1,cap)
tdv_ref = np.median(tdv[active & (tdv > 0)])
liqmult = np.ones((T, N))
good = tdv > 0
liqmult[good] = np.clip(np.sqrt(tdv_ref / tdv[good]), 1.0, 10.0)   # 最贵10×


def sim(base_bps=5.0, liq_scaled=False, delist_pen=0.0):
    cur = np.zeros(N); pnl = np.zeros(T)
    for t in range(T):
        prev = cur.copy()
        idx = np.where(active[t] & (funding[t] > 0))[0]; W = np.zeros(N)
        if len(idx):
            w = np.clip(f0[t, idx], 0, None); s = w.sum()
            if s > 0:
                W[idx] = w / s
        if t % 24 == 0:
            d = W - cur; mv = np.abs(d) > 0.01; new = cur.copy(); new[mv] = cur[mv] + 0.3 * d[mv]; cur = new
        dpos = np.abs(cur - prev)                                  # 每币换手
        bps_vec = base_bps * (liqmult[t] if liq_scaled else 1.0)
        cost = 2 * (bps_vec / 1e4 * dpos).sum()                   # 双腿、按币成本
        if delist_pen > 0:                                        # 退市当bar:被迫平掉的仓位额外罚
            dl = (last_vp == t) & delisted & (cur > 1e-9)
            if dl.any():
                cost += delist_pen * cur[dl].sum(); cur[dl] = 0.0
        pnl[t] = (cur * (price_leg[t] + f0[t] / 8.0)).sum() - cost
    return pnl


def sh(p, m): return p[m].mean() / p[m].std() * np.sqrt(ANN) if p[m].std() > 0 else 0.0


def line(name, p):
    print(f"  {name:>34s}: OOS {sh(p,oos):>+5.2f}  2024 {sh(p,oos&(yr=='2024')):>+5.2f}"
          f"  2025 {sh(p,oos&(yr=='2025')):>+5.2f}  2026 {sh(p,oos&(yr=='2026')):>+5.2f}  年化 {p[oos].mean()*ANN:>+6.1%}")


print(f"宇宙 {int(vp.any(0).sum())} 币;退市 {int(delisted.sum())};流动性成本倍率 中位={np.median(liqmult[active]):.1f} 90分位={np.percentile(liqmult[active],90):.1f}\n")
print("=== 基线 5bps 平 ===")
line("base 5bps", sim(5))
print("\n=== ① 流动性缩放成本(illiquid 多付,最贵10×) ===")
line("5bps × liq倍率", sim(5, liq_scaled=True))
print("\n=== ② 退市平仓罚金(死币离场额外滑点) ===")
for pen in (0.01, 0.03, 0.05, 0.10):
    line(f"退市罚 {pen:.0%}", sim(5, delist_pen=pen))
print("\n=== ③ 现实组合:流动性缩放 + 退市罚3% ===")
line("liq倍率 + 退市罚3%", sim(5, liq_scaled=True, delist_pen=0.03))
print("\n=== ④ 残酷组合:流动性缩放 + 退市罚10% ===")
line("liq倍率 + 退市罚10%", sim(5, liq_scaled=True, delist_pen=0.10))
print("\n判读: 现实摩擦后 OOS 仍稳、逐年仍≈正 = 死币利润真可交易、用户对。塌了 = 我那个担心成立。")
