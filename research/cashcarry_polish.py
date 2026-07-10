"""打磨:① 风险加权(funding/vol 替代纯 funding 权重)② 动态杠杆(carry 厚时多加)。
都用现有数据、因果、理论干净。对验证基线(两侧+basis-mom过滤+负top50)做干净 A/B,OOS 报。
跑：python cashcarry_polish.py
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
# 因果每币波动(EWMA,10天半衰期)
volc = np.zeros((T, N)); var = np.zeros(N); dec = 0.5 ** (1.0 / 240)
for t in range(T):
    var = dec * var + (1 - dec) * np.nan_to_num(pr[t]) ** 2
    volc[t] = np.sqrt(var)
KPOS, KNEG, CAD, REBAL, BAND, POSHL, CAP, COST, BORROW, FBAND = 100, 50, 24, 0.3, 0.01, 120, 0.05, 10.0, 0.10, 0.0


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


def make_W(riskparity):
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
        s_i = np.sign(f[cand]); keep = s_i * bm[t, cand] <= 0.05
        cand = cand[keep]; s_i = s_i[keep]
        if len(cand) == 0:
            continue
        wa = np.abs(f[cand])
        if riskparity:
            wa = wa / (volc[t, cand] + 1e-4)                # funding/vol
        ss = wa.sum()
        if ss > 0:
            W[t, cand] = cap_renorm(wa / ss, CAP) * s_i
    return W


def smooth(W, h):
    al = 1 - 0.5 ** (1.0 / h); S = np.empty_like(W); S[0] = W[0]
    for t in range(1, len(W)):
        S[t] = al * W[t] + (1 - al) * S[t - 1]
    g = np.abs(S).sum(1, keepdims=True)
    return np.divide(S, g, out=np.zeros_like(S), where=g > 0)


def build(riskparity):
    Ws = smooth(make_W(riskparity), POSHL)
    cur = np.zeros(N); held = np.zeros((T, N)); turn = np.zeros(T); prev = np.zeros(N)
    for t in range(T):
        if t % CAD == 0:
            d = Ws[t] - cur; mv = np.abs(d) > BAND
            new = cur.copy(); new[mv] = cur[mv] + REBAL * d[mv]
            s = np.abs(new).sum(); cur = new / s if s > 0 else new
        turn[t] = np.abs(cur - prev).sum(); held[t] = cur; prev = cur
    return held, turn


def netser(held, turn):
    short_spot = np.where(held < 0, -held, 0).sum(1)
    rich = (np.abs(held) * np.abs(f0)).sum(1) / 8.0           # 每bar在收的毛funding(carry厚度)
    return (held * cr).sum(1) - COST / 1e4 * turn - BORROW / ann * short_spot, rich


def stats(p, m):
    pm = p[m]; s = pm.mean() / pm.std() * np.sqrt(ann) if pm.std() > 0 else 0.0
    c = np.cumsum(pm); dd = (c - np.maximum.accumulate(c)).min()
    return s, pm.mean() * ann, dd


def dynlev(net, rich, base_L=2.0, lo=0.5, hi=1.5):
    """carry 厚时多加杠杆(全因果:richness EWMA 滞后 + 扩张均值当基准,不偷看)。"""
    rs = np.zeros(T); r = rich[0]
    for t in range(1, T):
        r = 0.99 * r + 0.01 * rich[t - 1]                    # 平滑 richness(滞后)
        rs[t] = r
    ref = np.zeros(T); csum = 0.0                            # 扩张均值(因果)
    for t in range(T):
        ref[t] = csum / t if t > 0 else rs[0]
        csum += rs[t]
    Lt = np.clip(rs / (ref + 1e-12), lo, hi) * base_L
    return Lt * net


print("=== ① 权重:funding vs 风险加权(funding/vol);OOS,L=1 ===")
print(f"{'方案':>10s} {'Sharpe':>7s} {'年化':>7s} {'最大回撤':>8s} {'2024':>6s} {'2026':>6s}")
for tag, rp in [("funding权重", False), ("风险加权", True)]:
    h, tn = build(rp); n, rich = netser(h, tn)
    s, a, d = stats(n, oos)
    print(f"{tag:>10s} {s:>+7.2f} {a:>+6.1%} {d:>+8.3f} {stats(n,oos&(yr=='2024'))[0]:>+6.2f} {stats(n,oos&(yr=='2026'))[0]:>+6.2f}")

print("\n=== ② 动态杠杆(carry厚多加 vs 恒定;基于更优权重);OOS ===")
best_rp = False                                               # ① 显示 funding 权重更优,用它
h, tn = build(best_rp); n, rich = netser(h, tn)
print(f"{'方案':>14s} {'Sharpe':>7s} {'年化':>7s} {'最大回撤':>8s}")
s, a, d = stats(2.0 * n, oos); print(f"{'恒定 2x':>14s} {s:>+7.2f} {a:>+6.1%} {d:>+8.3f}")
ndyn = dynlev(n, rich, base_L=2.0)
s, a, d = stats(ndyn, oos); print(f"{'动态(均值2x)':>14s} {s:>+7.2f} {a:>+6.1%} {d:>+8.3f}")

print("\n判读:风险加权若 Sharpe↑ = 真改善;动态杠杆若 同回撤下年化↑/Sharpe↑ = carry择时有效。只留真涨的。")
