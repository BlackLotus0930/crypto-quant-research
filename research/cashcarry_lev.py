"""③ 诚实杠杆/强平(修上版"免费止损"假象):在 ② 的低回撤两侧 book 上,诚实加杠杆。
诚实模型:杠杆 L 线性放大 P&L 与回撤(net_L = L×net);强平=纯成本:某仓基差较开仓逆向走阔 > 1/L−mm →
  收逼空滑点 + **立刻按现价复入**(不给冷却"砍损"红利,避免上版 Sharpe 随 L 升的假象)。
→ 报 L / 年化 / 真实最大回撤 / 强平次数 / 单仓最大逆向走阔。看安全杠杆上限。
跑：python cashcarry_lev.py
"""
import numpy as np

z = np.load("data/clean/crypto_tensor_60min_pit.npz", allow_pickle=True)
mask = z["mask"]; adj = z["adj_close"].astype(float); tdv = z["tdv"].astype(float)
dates = z["dates"].astype(str); slots = list(z["slots"].astype(str)); T, N = mask.shape
funding = np.load("data/clean/funding_pit.npz", allow_pickle=True)["funding"].astype(float)
spot = np.load("data/clean/spot_pit.npz", allow_pickle=True)["spot"].astype(float)
yr = np.array([d[:4] for d in dates]); cut = int(T * 0.4); ann = 8760
oos = np.zeros(T, bool); oos[cut:] = True
vp = mask & np.isfinite(spot) & (spot > 0) & (adj > 0)
f0 = np.nan_to_num(funding)
pr = np.zeros((T, N)); sr = np.zeros((T, N)); vpp = vp[:-1] & vp[1:]
pr[:-1][vpp] = adj[1:][vpp] / adj[:-1][vpp] - 1
sr[:-1][vpp] = spot[1:][vpp] / spot[:-1][vpp] - 1
cr = f0 / 8.0 + np.clip(np.nan_to_num(sr - pr), -0.25, 0.25)
basis = np.full((T, N), np.nan); basis[vp] = adj[vp] / spot[vp] - 1
bm = np.zeros((T, N)); MOMW = 48
bm[MOMW:] = np.nan_to_num(basis[MOMW:] - basis[:-MOMW])
KPOS, KNEG, CAD, REBAL, BAND, POSHL, CAP, COST, BORROW, FBAND = 100, 50, 24, 0.3, 0.01, 120, 0.05, 10.0, 0.10, 0.0
MM, LIQSLIP = 0.005, 0.005


def cap_renorm(w, C):
    w = w.copy()
    for _ in range(20):
        over = w > C + 1e-12
        if not over.any():
            break
        ex = (w[over] - C).sum(); w[over] = C; room = (w > 0) & (~over)
        if not room.any() or w[room].sum() <= 0:
            break
        w[room] += ex * w[room] / w[room].sum()
    s = w.sum(); return w / s if s > 0 else w


def make_W():
    W = np.zeros((T, N))
    for t in range(T):
        f = f0[t]; va = vp[t] & np.isfinite(funding[t])
        posc = np.where(va & (f > FBAND))[0]; negc = np.where(va & (f < -FBAND))[0]
        if len(posc) > KPOS:
            posc = posc[np.argsort(tdv[t, posc])[::-1][:KPOS]]
        if len(negc) > KNEG:
            negc = negc[np.argsort(tdv[t, negc])[::-1][:KNEG]]
        cand = np.concatenate([posc, negc])
        if len(cand) == 0:
            continue
        s_i = np.sign(f[cand]); keep = s_i * bm[t, cand] <= 0.05    # 基差动量过滤(②)
        cand = cand[keep]; s_i = s_i[keep]
        if len(cand) == 0:
            continue
        wa = np.abs(f[cand]); ss = wa.sum()
        if ss > 0:
            W[t, cand] = cap_renorm(wa / ss, CAP) * s_i
    return W


def smooth(W, h):
    al = 1 - 0.5 ** (1.0 / h); S = np.empty_like(W); S[0] = W[0]
    for t in range(1, len(W)):
        S[t] = al * W[t] + (1 - al) * S[t - 1]
    g = np.abs(S).sum(1, keepdims=True)
    return np.divide(S, g, out=np.zeros_like(S), where=g > 0)


Ws = smooth(make_W(), POSHL)
cur = np.zeros(N); held = np.zeros((T, N)); turn = np.zeros(T); prev = np.zeros(N)
for t in range(T):
    if t % CAD == 0:
        d = Ws[t] - cur; mv = np.abs(d) > BAND
        new = cur.copy(); new[mv] = cur[mv] + REBAL * d[mv]
        s = np.abs(new).sum(); cur = new / s if s > 0 else new
    turn[t] = np.abs(cur - prev).sum(); held[t] = cur; prev = cur
short_spot = np.where(held < 0, -held, 0).sum(1)
base_net = (held * cr).sum(1) - COST / 1e4 * turn - BORROW / ann * short_spot


def lever(L):
    """诚实:net_L=L×base_net − 强平滑点;强平立刻复入(无冷却红利)。"""
    thr = (1.0 / L - MM) if L > 1 else 1e9
    entry_b = np.full(N, np.nan); slip = np.zeros(T); nliq = 0; worst = 0.0
    for t in range(T):
        h = held[t]; on = np.abs(h) > 1e-9
        new_entry = on & ~np.isfinite(entry_b)                     # 新开仓→记开仓基差
        entry_b[new_entry] = basis[t, new_entry]
        entry_b[~on] = np.nan
        adv = np.sign(h) * (basis[t] - entry_b)                    # 逆向走阔(short perp:基差↑;long perp:基差↓)
        if on.any():
            worst = max(worst, np.nanmax(np.where(on & np.isfinite(adv), adv, -np.inf)))
        mc = on & np.isfinite(adv) & (adv > thr)
        if mc.any():
            slip[t] = LIQSLIP * L * np.abs(h[mc]).sum(); nliq += int(mc.sum())
            entry_b[mc] = basis[t, mc]                             # 立刻复入(重置开仓基差)
    return L * base_net - slip, nliq, worst


def sh(p, m):
    p = p[m]; return p.mean() / p.std() * np.sqrt(ann) if p.std() > 0 else 0.0


def mdd(p, m):
    c = np.cumsum(p[m]); return (c - np.maximum.accumulate(c)).min()


print(f"② 低回撤两侧 book + 诚实强平(滑点{LIQSLIP:.1%}、立刻复入、无冷却红利)\n")
print(f"{'杠杆':>4s} {'OOS Sh':>7s} {'年化':>8s} {'最大回撤':>8s} {'2026':>6s} {'强平次数':>8s} {'单仓最大逆向':>11s}")
for L in (1, 2, 3, 5):
    nL, nliq, worst = lever(L)
    print(f"{L:>3d}x {sh(nL,oos):>+7.2f} {nL[oos].mean()*ann:>+7.1%} {mdd(nL,oos):>+8.3f} {sh(nL,oos&(yr=='2026')):>+6.2f} {nliq:>8d} {worst:>10.1%}")

print("\n健全性:杠杆↑ → 年化↑、回撤↑(线性)、Sharpe≈平(不该升=假象已修)。")
print("判读:回撤可忍(如≤−10%)的最高 L = 安全杠杆 → 真实可投年化。")
