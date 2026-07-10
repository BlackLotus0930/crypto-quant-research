"""压力 regime 验证 + 相关性尾部检查:把 ② 的两侧 book 跑全样本(暴露此前当 val 的 2020-2022),
看它在 COVID(2020-03)、LUNA(2022-05)、FTX(2022-11)、2022 大熊里扛不扛得住;
并查回撤是不是"多名同时被挤"(系统性/相关)还是"个别币"(可分散)。
用 basis-momentum 过滤(因果),不用 vol-target(避免 target 标定的 lookahead)。
跑：python cashcarry_stress.py
"""
import numpy as np

z = np.load("data/clean/crypto_tensor_60min_pit.npz", allow_pickle=True)
mask = z["mask"]; adj = z["adj_close"].astype(float); tdv = z["tdv"].astype(float)
dates = z["dates"].astype(str); slots = list(z["slots"].astype(str)); T, N = mask.shape
funding = np.load("data/clean/funding_pit.npz", allow_pickle=True)["funding"].astype(float)
spot = np.load("data/clean/spot_pit.npz", allow_pickle=True)["spot"].astype(float)
yr = np.array([d[:4] for d in dates]); ann = 8760
day = np.array([d[:10] for d in dates])
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
        s_i = np.sign(f[cand]); keep = s_i * bm[t, cand] <= 0.05
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
contrib = held * cr
net = contrib.sum(1) - COST / 1e4 * turn - BORROW / ann * short_spot


def sh(p, m):
    p = p[m]; return p.mean() / p.std() * np.sqrt(ann) if (len(p) and p.std() > 0) else 0.0


def mdd(p, m):
    p = p[m]
    if len(p) == 0:
        return 0.0
    c = np.cumsum(p); return (c - np.maximum.accumulate(c)).min()


print("=== 全样本逐年(2020-2022 此前是 val=首次见光;2022=熊/LUNA/FTX) ===")
print(f"{'年':>6s} {'Sharpe':>7s} {'年化':>8s} {'最大回撤':>8s} {'中位持仓数':>9s} {'负侧占比':>8s}")
for y in ("2020", "2021", "2022", "2023", "2024", "2025", "2026"):
    m = yr == y
    if m.sum() < 50:
        continue
    npos = np.median((np.abs(held[m]) > 1e-9).sum(1))
    neg = np.where(held[m] < 0, -held[m], 0).sum() / (np.abs(held[m]).sum() + 1e-12)
    print(f"{y:>6s} {sh(net,m):>+7.2f} {net[m].mean()*ann:>+7.1%} {mdd(net,m):>+8.3f} {npos:>9.0f} {neg*100:>7.0f}%")

print("\n=== 危机窗口(窗内累计收益 / 窗内最大回撤) ===")
for tag, d0, d1 in [("COVID 2020-03-09~16", "2020-03-09", "2020-03-16"),
                    ("LUNA 2022-05-08~14", "2022-05-08", "2022-05-14"),
                    ("FTX  2022-11-07~12", "2022-11-07", "2022-11-12")]:
    m = (day >= d0) & (day <= d1)
    if m.sum() == 0:
        print(f"  {tag}: (无数据)"); continue
    print(f"  {tag}: 累计{net[m].sum():+.3f}  窗内回撤{mdd(net,m):+.3f}  中位持仓{np.median((np.abs(held[m])>1e-9).sum(1)):.0f}")

print("\n=== 相关性尾部:最差 1% 的 bar 上,多少持仓同时在亏(高=系统性逼空、分散失效) ===")
loss = contrib                                              # 每币每bar毛贡献
q = np.percentile(net, 1)                                   # 最差1%阈
bad = net <= q
fr = []
for t in np.where(bad)[0]:
    h = np.abs(held[t]) > 1e-9
    if h.sum() > 3:
        fr.append((loss[t][h] < 0).mean())
print(f"  最差1% bar({bad.sum()}根):平均 {np.mean(fr)*100:.0f}% 的持仓同时在亏  (>80%=高度相关/系统性)")
# 正常 bar 对照
ok = net > np.percentile(net, 50); fro = []
for t in np.where(ok)[0][::20]:
    h = np.abs(held[t]) > 1e-9
    if h.sum() > 3:
        fro.append((loss[t][h] < 0).mean())
print(f"  对照(中位以上 bar):平均 {np.mean(fro)*100:.0f}% 持仓在亏")

print("\n=== 全样本最差单次回撤(peak-to-trough) ===")
c = np.cumsum(net); peak = np.maximum.accumulate(c); ddser = c - peak
tt = ddser.argmin(); pk = c[:tt + 1].argmax()
drivers = contrib[pk:tt + 1].sum(0); od = np.argsort(drivers)
print(f"  最深回撤 {ddser[tt]:+.3f}  从 {day[pk]} 到 {day[tt]}")
print(f"  最大亏损贡献币: {[(slots[i], round(drivers[i],3)) for i in od[:5]]}")

print("\n判读:① 2022(熊/FTX)Sharpe/回撤可忍 = 含最坏 regime 也稳。② 最差bar若<80%持仓同亏 = 非系统性、分散有效。")
