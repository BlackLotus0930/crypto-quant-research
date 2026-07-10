"""② 机械降回撤(两侧 carry 基线,负侧 top-50 可借):
- basis-momentum 入场过滤:不进"基差已在逆向飙"的币(s·Δbasis 过阈=正在被挤,躲开)。
- vol-targeting:用近期已实现波动缩放 gross(湍流期自动缩仓)。
目标:把 −7% 回撤压更低、Sharpe 升(→可加更多杠杆),且不大伤年化。
跑：python cashcarry_dd.py
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
bm = np.zeros((T, N)); MOMW = 48                                   # 基差动量(2天)
bm[MOMW:] = np.nan_to_num(basis[MOMW:] - basis[:-MOMW])
KPOS, KNEG, CAD, REBAL, BAND, POSHL, CAP, COST, BORROW, FBAND = 100, 50, 24, 0.3, 0.01, 120, 0.05, 10.0, 0.10, 0.0
VW, SCMAX = 168, 1.5                                               # 波动窗7天、最大放大1.5


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


def make_W(mom_thr):
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
        s_i = np.sign(f[cand])
        if mom_thr is not None:                                    # 躲"逆向飙"的基差(s·Δbasis 过阈)
            keep = s_i * bm[t, cand] <= mom_thr
            cand = cand[keep]; s_i = s_i[keep]
            if len(cand) == 0:
                continue
        wa = np.abs(f[cand]); ss = wa.sum()
        if ss <= 0:
            continue
        W[t, cand] = cap_renorm(wa / ss, CAP) * s_i
    return W


def smooth(W, h):
    al = 1 - 0.5 ** (1.0 / h); S = np.empty_like(W); S[0] = W[0]
    for t in range(1, len(W)):
        S[t] = al * W[t] + (1 - al) * S[t - 1]
    g = np.abs(S).sum(1, keepdims=True)
    return np.divide(S, g, out=np.zeros_like(S), where=g > 0)


def build(mom_thr):
    Ws = smooth(make_W(mom_thr), POSHL)
    cur = np.zeros(N); held = np.zeros((T, N)); turn = np.zeros(T); prev = np.zeros(N)
    for t in range(T):
        if t % CAD == 0:
            d = Ws[t] - cur; mv = np.abs(d) > BAND
            new = cur.copy(); new[mv] = cur[mv] + REBAL * d[mv]
            s = np.abs(new).sum(); cur = new / s if s > 0 else new
        turn[t] = np.abs(cur - prev).sum(); held[t] = cur; prev = cur
    return held, turn


def net_of(held, turn):
    short_spot = np.where(held < 0, -held, 0).sum(1)
    return (held * cr).sum(1) - COST / 1e4 * turn - BORROW / ann * short_spot


def vol_target(net):
    """因果:用 [t-VW:t] 的已实现波动缩放;target=全期波动→平均≈1,只重分配暴露。"""
    target = net[oos].std()
    sc = np.ones(T)
    for t in range(VW, T):
        rv = net[t - VW:t].std()
        sc[t] = min(SCMAX, target / rv) if rv > 1e-12 else 1.0
    return net * sc, sc


def sh(p, m):
    p = p[m]; return p.mean() / p.std() * np.sqrt(ann) if p.std() > 0 else 0.0


def mdd(p, m):
    c = np.cumsum(p[m]); return (c - np.maximum.accumulate(c)).min()


base_h, base_t = build(None)
base = net_of(base_h, base_t)
mom_h, mom_t = build(0.05)                                         # 基差动量过滤阈 5%
mom = net_of(mom_h, mom_t)
base_vt, _ = vol_target(base)
both_vt, _ = vol_target(mom)

print(f"两侧基线: 正top{KPOS}/负top{KNEG}, 帽{CAP:.0%}, 成本{COST}bps, 借币{BORROW:.0%}; vol-target窗{VW} 上限{SCMAX}x\n")
print(f"{'方案':>16s} {'OOS Sh':>7s} {'年化':>7s} {'最大回撤':>8s} {'年化/|回撤|':>10s} {'2024':>6s} {'2026':>6s}")
for tag, p in [("基线", base), ("+基差动量过滤", mom), ("+vol-target", base_vt), ("+两者", both_vt)]:
    print(f"{tag:>16s} {sh(p,oos):>+7.2f} {p[oos].mean()*ann:>+6.1%} {mdd(p,oos):>+8.3f}"
          f" {p[oos].mean()*ann/abs(mdd(p,oos)):>10.1f} {sh(p,oos&(yr=='2024')):>+6.2f} {sh(p,oos&(yr=='2026')):>+6.2f}")

print("\n判读:回撤降、年化/|回撤| 升(=同回撤能赚更多/能加更多杠杆)、年化不大伤 → 机械降回撤有效。")
