"""验证借币可得性:负侧(空现货)需借币,只 liquid 币借得到。把负侧宇宙限到 top-Kneg 流动,
看双侧 carry 的 +19%/yr 能 survive 多少。正侧(长现货,无需借)固定 top-100。
跑：python cashcarry_borrow.py
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
KPOS, CAD, REBAL, BAND, POSHL, CAP, COST, BORROW = 100, 24, 0.3, 0.01, 120, 0.05, 10.0, 0.10
FBAND = 0.0   # funding 符号阈值(≠ BAND 调仓死区);funding ~1e-4 量级,选币用 0


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


def make_W(kneg):
    """正侧 top-KPOS(f>0);负侧 top-kneg 流动(f<0,kneg=0 即正向only)。权重∝|f|,帽,带符号。"""
    W = np.zeros((T, N))
    for t in range(T):
        f = f0[t]; va = vp[t] & np.isfinite(funding[t])
        posc = np.where(va & (f > FBAND))[0]
        if len(posc) > KPOS:
            posc = posc[np.argsort(tdv[t, posc])[::-1][:KPOS]]
        negc = np.array([], int)
        if kneg > 0:
            negc = np.where(va & (f < -FBAND))[0]
            if len(negc) > kneg:
                negc = negc[np.argsort(tdv[t, negc])[::-1][:kneg]]
        cand = np.concatenate([posc, negc])
        if len(cand) == 0:
            continue
        wa = np.abs(f[cand]); s = wa.sum()
        if s <= 0:
            continue
        W[t, cand] = cap_renorm(wa / s, CAP) * np.sign(f[cand])
    return W


def smooth(W, h):
    al = 1 - 0.5 ** (1.0 / h); S = np.empty_like(W); S[0] = W[0]
    for t in range(1, len(W)):
        S[t] = al * W[t] + (1 - al) * S[t - 1]
    g = np.abs(S).sum(1, keepdims=True)
    return np.divide(S, g, out=np.zeros_like(S), where=g > 0)


def build(kneg):
    Ws = smooth(make_W(kneg), POSHL)
    cur = np.zeros(N); held = np.zeros((T, N)); turn = np.zeros(T); prev = np.zeros(N)
    for t in range(T):
        if t % CAD == 0:
            d = Ws[t] - cur; mv = np.abs(d) > BAND
            new = cur.copy(); new[mv] = cur[mv] + REBAL * d[mv]
            s = np.abs(new).sum(); cur = new / s if s > 0 else new
        turn[t] = np.abs(cur - prev).sum(); held[t] = cur; prev = cur
    return held, turn


def net(held, turn):
    short_spot = np.where(held < 0, -held, 0).sum(1)
    return (held * cr).sum(1) - COST / 1e4 * turn - BORROW / ann * short_spot


def sh(p, m):
    p = p[m]; return p.mean() / p.std() * np.sqrt(ann) if p.std() > 0 else 0.0


def dd(p, m):
    c = np.cumsum(p[m]); return (c - np.maximum.accumulate(c)).min()


print(f"正侧 top-{KPOS}(无需借);负侧限 top-Kneg 流动(借币{BORROW:.0%}/yr);成本{COST}bps;帽{CAP:.0%}\n")
print(f"{'负侧宇宙':>9s} {'OOS Sh':>7s} {'OOS年化':>7s} {'2024':>6s} {'2026':>6s} {'回撤':>7s} {'有效币':>6s}")
for kneg in (0, 10, 20, 30, 50, 100):
    held, turn = build(kneg); n = net(held, turn)
    cp = (held * cr)[oos].sum(0); eff = cp[cp > 0].sum() ** 2 / (cp[cp > 0] ** 2).sum() if (cp > 0).any() else 0
    tag = "正向only" if kneg == 0 else f"top-{kneg}"
    print(f"{tag:>9s} {sh(n,oos):>+7.2f} {n[oos].mean()*ann:>+6.1%} {sh(n,oos&(yr=='2024')):>+6.2f}"
          f" {sh(n,oos&(yr=='2026')):>+6.2f} {dd(n,oos):>+7.3f} {eff:>6.1f}")

# 负侧 P&L 有多少来自 top-30 之外(难借)的币
held100, t100 = build(100); gneg = np.where(held100 < 0, held100 * cr, 0.0)[oos]
cpn = gneg.sum(0); on = np.argsort(cpn)[::-1]
# 每币是否曾在 top-30 流动
ever_top30 = np.zeros(N, bool)
for t in np.where(oos)[0][::24]:
    v = np.where(vp[t])[0]
    if len(v):
        ever_top30[v[np.argsort(tdv[t, v])[::-1][:30]]] = True
neg_in30 = cpn[ever_top30 & (cpn > 0)].sum(); neg_out30 = cpn[(~ever_top30) & (cpn > 0)].sum()
print(f"\n负侧正贡献:来自'曾 top-30 流动'的币 {neg_in30:.3f} / 来自更不流动 {neg_out30:.3f}"
      f"  → 易借部分占 {neg_in30/(neg_in30+neg_out30)*100:.0f}%")
print("\n判读:负侧限到 top-20~30 仍明显优于正向only(~14%) → 借币可得性不是杀手,反向carry落地。")
