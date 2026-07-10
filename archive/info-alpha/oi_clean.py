"""oi(淡大户多空)信号 干净复测:全 pit 宇宙(含死币)+ beta 对冲。
问:剥掉 beta 后,top-LSR-fade 还剩干净 alpha 吗?还是长短腿全是 regime 收益?
跑：python oi_clean.py
"""
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--tensor", default="data/clean/crypto_tensor_60min_pit.npz")
ap.add_argument("--oi", default="data/clean/oi_pit.npz")
ap.add_argument("--hl", type=int, default=24); ap.add_argument("--cad", type=int, default=24)
ap.add_argument("--rebal", type=float, default=0.3); ap.add_argument("--band", type=float, default=0.01)
ap.add_argument("--beta_hl", type=int, default=720); ap.add_argument("--ann", type=int, default=8760)
ap.add_argument("--bps", type=float, default=5)
a = ap.parse_args()

z = np.load(a.tensor, allow_pickle=True)
mask = z["mask"]; adj = z["adj_close"].astype(np.float64); dates = z["dates"].astype(str); slots = z["slots"].astype(str)
T, N = mask.shape; yr = np.array([d[:4] for d in dates])
tt = np.load(a.oi, allow_pickle=True)["tt"].astype(np.float64)
R = np.zeros((T, N)); g2 = mask[:-1] & mask[1:] & (adj[:-1] > 0) & (adj[1:] > 0); R[:-1][g2] = adj[1:][g2] / adj[:-1][g2] - 1
active = mask & (adj > 0) & np.isfinite(tt)
print(f"oi 覆盖 {np.isfinite(tt).any(0).sum()}/{N} 币(含死币); active率={active.mean():.3f}\n")

# 信号 z(-tt)
W = np.full((T, N), np.nan)
for t in range(T):
    idx = np.where(active[t])[0]
    if len(idx) < 10:
        continue
    v = -tt[t, idx]; sd = v.std()
    if sd > 1e-12:
        W[t, idx] = (v - v.mean()) / sd
Wn = np.nan_to_num(W); g = np.abs(Wn).sum(1, keepdims=True); Wn = np.divide(Wn, g, out=np.zeros_like(Wn), where=g > 0)


def smooth(W, h):
    al = 1 - 0.5 ** (1.0 / h); S = np.empty_like(W); S[0] = W[0]
    for t in range(1, len(W)):
        S[t] = al * W[t] + (1 - al) * S[t - 1]
    g = np.abs(S).sum(1, keepdims=True)
    return np.divide(S, g, out=np.zeros_like(S), where=g > 0)


Ws = smooth(Wn, a.hl); held = np.zeros((T, N)); cur = np.zeros(N); turn = np.zeros(T)
for t in range(T):
    if t % a.cad == 0:
        d = Ws[t] - cur; mv = np.abs(d) > a.band; new = cur.copy(); new[mv] = cur[mv] + a.rebal * d[mv]
        s = np.abs(new).sum(); new = new / s if s > 0 else new
        turn[t] = np.abs(new - cur).sum(); cur = new
    held[t] = cur

mkt = np.array([R[t, active[t]].mean() if active[t].sum() > 5 else 0.0 for t in range(T)])
dec = 0.5 ** (1.0 / a.beta_hl); cov = np.zeros(N); var = 1e-8; beta = np.zeros((T, N))
for t in range(T):
    cov = dec * cov + (1 - dec) * (R[t] * mkt[t]); var = dec * var + (1 - dec) * (mkt[t] ** 2); beta[t] = cov / var

book = (held * R).sum(1); net_beta = (held * beta).sum(1); hedged = book - net_beta * mkt
cost = a.bps / 1e4 * turn


def sh(p, m):
    p = p[m]
    return p.mean() / p.std() * np.sqrt(a.ann) if p.std() > 0 else 0.0


print(f"book 对 market 相关={np.corrcoef(book, mkt)[0,1]:+.3f}; 平均净beta={net_beta.mean():+.3f}\n")
print(f"{'区间':>8s} {'原book净@5':>11s} {'对冲后净@5':>11s}")
for lab, m in [("全程", np.ones(T, bool))] + [(y, yr == y) for y in ("2024", "2025", "2026")]:
    if m.sum() < 50:
        continue
    print(f"{lab:>8s} {sh(book - cost, m):>+11.2f} {sh(hedged - cost, m):>+11.2f}")

print(f"\n长短腿毛 P&L(对冲前/后):")
lp = (np.where(held > 0, held, 0) * R).sum(1); sp = (np.where(held < 0, held, 0) * R).sum(1)
lb = (np.where(held > 0, held, 0) * beta).sum(1); sb = (np.where(held < 0, held, 0) * beta).sum(1)
lph = lp - lb * mkt; sph = sp - sb * mkt
for y in ("2024", "2025", "2026"):
    m = yr == y
    if m.sum() < 50:
        continue
    print(f"  {y}: 长 {lp[m].sum():+.3f}/{lph[m].sum():+.3f}   短 {sp[m].sum():+.3f}/{sph[m].sum():+.3f}")
print("\n判读: 对冲后全程 Sharpe 还在、长短腿都翻正、逐年(含2026)正 = 有干净独立alpha → oi 值得进组合。")
print("       对冲后塌、长短腿仍互补 = 就是 regime/beta 收益 → oi 钉死、广度路到顶。")
