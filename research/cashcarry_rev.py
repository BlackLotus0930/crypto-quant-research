"""① 反向 carry:把负 funding 那半边接上(做多负funding永续 + 空现货收它),看年化/Sharpe/逐年/补不补平淡年。
统一:position 方向 = sign(funding),权重 ∝ |funding|;P&L = held×cr(cr=spot_ret−perp_ret+funding/8 对两侧都成立)。
负侧=空现货,加诚实借币成本(borrow_apr,只 liquid 币可借→已 top-K 流动截顶)。CONTROL:每币|帽|5%。
对比:Book A 正向only(现状) vs Book B 双侧。borrow 敏感性。
跑：python cashcarry_rev.py
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
K, CAD, REBAL, BAND, POSHL, CAP, BANDF = 100, 24, 0.3, 0.01, 120, 0.05, 0.0
COST = 10.0


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


def make_W(both):
    W = np.zeros((T, N))
    for t in range(T):
        f = f0[t]
        sel = vp[t] & np.isfinite(funding[t]) & ((np.abs(f) > BANDF) if both else (f > BANDF))
        cand = np.where(sel)[0]
        if len(cand) == 0:
            continue
        if len(cand) > K:
            cand = cand[np.argsort(tdv[t, cand])[::-1][:K]]
        wa = np.abs(f[cand]); s = wa.sum()
        if s <= 0:
            continue
        w = cap_renorm(wa / s, CAP)                          # 帽作用在 |权重|
        W[t, cand] = w * np.sign(f[cand]) if both else w
    return W


def smooth(W, h):
    al = 1 - 0.5 ** (1.0 / h); S = np.empty_like(W); S[0] = W[0]
    for t in range(1, len(W)):
        S[t] = al * W[t] + (1 - al) * S[t - 1]
    g = np.abs(S).sum(1, keepdims=True)
    return np.divide(S, g, out=np.zeros_like(S), where=g > 0)


def build(both):
    Ws = smooth(make_W(both), POSHL)
    cur = np.zeros(N); held = np.zeros((T, N)); turn = np.zeros(T); prev = np.zeros(N)
    for t in range(T):
        if t % CAD == 0:
            d = Ws[t] - cur; mv = np.abs(d) > BAND
            new = cur.copy(); new[mv] = cur[mv] + REBAL * d[mv]
            s = np.abs(new).sum(); cur = new / s if s > 0 else new
        turn[t] = np.abs(cur - prev).sum(); held[t] = cur; prev = cur
    return held, turn


def net_of(held, turn, borrow_apr):
    g = (held * cr).sum(1)
    short_spot = np.where(held < 0, -held, 0).sum(1)         # 负侧=空现货,借币成本
    return g - COST / 1e4 * turn - borrow_apr / ann * short_spot


def sh(p, m):
    p = p[m]; return p.mean() / p.std() * np.sqrt(ann) if p.std() > 0 else 0.0


def dd(p, m):
    c = np.cumsum(p[m]); return (c - np.maximum.accumulate(c)).min()


hA, tA = build(False)
hB, tB = build(True)
nA = net_of(hA, tA, 0.0); nB = net_of(hB, tB, 0.10)        # B 负侧借币 10%/yr

print(f"成本{COST}bps; CONTROL每币帽{CAP:.0%}; K={K}; B负侧借币10%/yr\n")
print(f"{'区间':>8s} | {'A正向only Sh':>12s} {'年化':>7s} | {'B双侧 Sh':>9s} {'年化':>7s} {'回撤':>7s} {'负侧gross':>8s}")
for lab, m in [("OOS全", oos)] + [(y, oos & (yr == y)) for y in ("2023", "2024", "2025", "2026")]:
    if m.sum() < 50:
        continue
    negshare = np.where(hB[m] < 0, -hB[m], 0).sum() / np.abs(hB[m]).sum() if np.abs(hB[m]).sum() > 0 else 0
    print(f"{lab:>8s} | {sh(nA,m):>+12.2f} {nA[m].mean()*ann:>+6.1%} | {sh(nB,m):>+9.2f} {nB[m].mean()*ann:>+6.1%}"
          f" {dd(nB,m):>+7.3f} {negshare*100:>7.0f}%")

print("\n=== B 双侧:借币成本敏感性(OOS) ===")
for b in (0.0, 0.05, 0.10, 0.20, 0.40):
    nb = net_of(hB, tB, b)
    print(f"  借币{b*100:>2.0f}%/yr: OOS Sharpe={sh(nb,oos):+.2f} 年化={nb[oos].mean()*ann:+.1%} 2024={sh(nb,oos&(yr=='2024')):+.2f} 2026={sh(nb,oos&(yr=='2026')):+.2f}")

print("\n判读:B 双侧若年化/Sharpe 升、且补厚 2024/2026 平淡年、借币成本可承受 → 反向 carry 成立,值得接上。")
