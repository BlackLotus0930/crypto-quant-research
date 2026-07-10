"""加 per-coin 仓位帽(真实风控:绝不让一个币占 69%)→ 看去集中后的真实分散收益。
v3 配置(K=100, hl120, exit-hl72, 双腿成本),只多一个每币权重上限 C,超了均摊给其它。
跑：python cashcarry_cap.py
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
K, CAD, REBAL, BAND, POSHL, EXHL = 100, 24, 0.3, 0.01, 120, 72
lev = slots.index("LEVERUSDT")


def cap_renorm(w, C):
    w = w.copy()
    for _ in range(20):
        over = w > C + 1e-12
        if not over.any():
            break
        excess = (w[over] - C).sum(); w[over] = C
        room = (w > 0) & (~over)
        if not room.any() or w[room].sum() <= 0:
            break
        w[room] += excess * w[room] / w[room].sum()
    s = w.sum()
    return w / s if s > 0 else w


def make_W(C):
    W = np.zeros((T, N))
    for t in range(T):
        cand = np.where(vp[t] & np.isfinite(funding[t]) & (funding[t] > 0))[0]
        if len(cand) == 0:
            continue
        if len(cand) > K:
            cand = cand[np.argsort(tdv[t, cand])[::-1][:K]]
        w = f0[t, cand]; s = w.sum()
        if s <= 0:
            continue
        w = w / s
        if C < 1:
            w = cap_renorm(w, C)
        W[t, cand] = w
    return W


def smooth(W, h):
    al = 1 - 0.5 ** (1.0 / h); S = np.empty_like(W); S[0] = W[0]
    for t in range(1, len(W)):
        S[t] = al * W[t] + (1 - al) * S[t - 1]
    g = np.abs(S).sum(1, keepdims=True)
    return np.divide(S, g, out=np.zeros_like(S), where=g > 0)


def emaT(X, hl):
    al = 1 - 0.5 ** (1.0 / hl); E = np.empty_like(X); E[0] = X[0]
    for t in range(1, len(X)):
        E[t] = al * X[t] + (1 - al) * E[t - 1]
    return E


fe = emaT(f0, EXHL)


def run(C):
    Ws = smooth(make_W(C), POSHL); cur = np.zeros(N); held = np.zeros((T, N)); turn = np.zeros(T); prev = np.zeros(N)
    for t in range(T):
        if t % CAD == 0:
            d = Ws[t] - cur; mv = np.abs(d) > BAND
            new = cur.copy(); new[mv] = cur[mv] + REBAL * d[mv]
            s = np.abs(new).sum(); cur = new / s if s > 0 else new
        cur = np.where(fe[t] < 0, 0.0, cur)
        turn[t] = np.abs(cur - prev).sum(); held[t] = cur; prev = cur
    return held, turn


def sh(p, cb, tn, m):
    net = (p - cb / 1e4 * tn)[m]
    return net.mean() / net.std() * np.sqrt(ann) if net.std() > 0 else 0.0


def ann_(p, cb, tn, m):
    return (p - cb / 1e4 * tn)[m].mean() * ann


print(f"{'每币帽':>7s} {'OOS@10':>7s} {'2024':>6s} {'2025':>6s} {'2026':>6s} {'OOS年化':>8s} {'LEVER占比':>9s} {'有效币数':>7s} {'年换手':>7s}")
for C in (0.02, 0.05, 0.10, 0.20, 1.0):
    held, tn = run(C); contrib = held * cr; pnl = contrib.sum(1)
    cpnl = contrib[oos].sum(0); tot = cpnl.sum(); poss = cpnl[cpnl > 0].sum()
    eff = poss ** 2 / (cpnl[cpnl > 0] ** 2).sum() if poss > 0 else 0
    levshare = contrib[oos][:, lev].sum() / tot * 100 if tot != 0 else 0
    tag = "无帽" if C >= 1 else f"{C*100:.0f}%"
    def s(m):
        return sh(pnl, 10, tn, m)
    print(f"{tag:>7s} {s(oos):>+7.2f} {s(oos&(yr=='2024')):>+6.2f} {s(oos&(yr=='2025')):>+6.2f} {s(oos&(yr=='2026')):>+6.2f}"
          f" {ann_(pnl,10,tn,oos):>+7.1%} {levshare:>8.0f}% {eff:>7.1f} {tn[oos].mean()*ann:>7.0f}")

print("\n判读:加帽后 LEVER 占比降、有效币数升、年化掉到真实分散水平 → 那才是可重复、可上规模的收益。")
