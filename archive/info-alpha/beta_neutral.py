"""beta 中性化测试:book 带时变市场 beta 吗? 对冲掉后,长短腿还互相抵消吗、2025-26 翻正吗?
事后对冲(不重训):market=等权全市场次bar收益;EWMA 估每币 beta;hedged_pnl = book_pnl − netβ·market。
跑：python beta_neutral.py --book walkforward_book.npz --tensor data/clean/crypto_tensor_60min_pit.npz
"""
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--book", default="walkforward_book.npz")
ap.add_argument("--tensor", default="data/clean/crypto_tensor_60min_pit.npz")
ap.add_argument("--hl", type=int, default=24); ap.add_argument("--cad", type=int, default=24)
ap.add_argument("--rebal", type=float, default=0.3); ap.add_argument("--band", type=float, default=0.01)
ap.add_argument("--beta_hl", type=int, default=720, help="beta EWMA 半衰期(bar);720=30天")
ap.add_argument("--ann", type=int, default=8760)
a = ap.parse_args()

bk = np.load(a.book, allow_pickle=True)
W, R, bdates = bk["W"].astype(np.float64), bk["R"].astype(np.float64), bk["dates"].astype(str)
T, N = W.shape
zt = np.load(a.tensor, allow_pickle=True)
mask, tdates = zt["mask"], zt["dates"].astype(str)
tidx = {d: i for i, d in enumerate(tdates)}
active = np.zeros((T, N), bool)
for t in range(T):
    ti = tidx.get(bdates[t], -1)
    if ti >= 0:
        active[t] = mask[ti]

# 市场=活跃币等权次bar收益
mkt = np.array([R[t, active[t]].mean() if active[t].sum() > 5 else 0.0 for t in range(T)])

# EWMA per-coin beta(因果)
dec = 0.5 ** (1.0 / a.beta_hl)
cov = np.zeros(N); var = 1e-8; beta = np.zeros((T, N))
for t in range(T):
    cov = dec * cov + (1 - dec) * (R[t] * mkt[t])
    var = dec * var + (1 - dec) * (mkt[t] ** 2)
    beta[t] = cov / var

# 重建持仓(慢配置)
def smooth(W, h):
    al = 1 - 0.5 ** (1.0 / h); S = np.empty_like(W); S[0] = W[0]
    for t in range(1, len(W)):
        S[t] = al * W[t] + (1 - al) * S[t - 1]
    g = np.abs(S).sum(1, keepdims=True)
    return np.divide(S, g, out=np.zeros_like(S), where=g > 0)

Ws = smooth(W, a.hl)
held = np.zeros((T, N)); cur = np.zeros(N)
for t in range(T):
    if t % a.cad == 0:
        d = Ws[t] - cur; mv = np.abs(d) > a.band
        new = cur.copy(); new[mv] = cur[mv] + a.rebal * d[mv]
        g = np.abs(new).sum(); new = new / g if g > 0 else new
        cur = new
    held[t] = cur

book_pnl = (held * R).sum(1)
net_beta = (held * beta).sum(1)                 # book 的净市场 beta 暴露
hedged_pnl = book_pnl - net_beta * mkt          # 对冲掉 beta·市场
yr = np.array([d[:4] for d in bdates])


def sh(p, m=None):
    p = p if m is None else p[m]
    return p.mean() / p.std() * np.sqrt(a.ann) if p.std() > 0 else 0.0


print(f"book {T}bar N={N}; book 平均净 beta = {net_beta.mean():+.3f} (|净beta|均值 {np.abs(net_beta).mean():.3f})")
print(f"book_pnl 对 market 的相关 = {np.corrcoef(book_pnl, mkt)[0,1]:+.3f}  (大=带市场方向=有beta污染)\n")

print(f"{'区间':>8s} {'原book Sharpe':>14s} {'对冲后 Sharpe':>14s}")
for lab, m in [("全程", None)] + [(y, yr == y) for y in ("2024", "2025", "2026")]:
    if m is not None and m.sum() < 50:
        continue
    print(f"{lab:>8s} {sh(book_pnl, m):>+14.2f} {sh(hedged_pnl, m):>+14.2f}")

# 长/短腿,对冲前后,逐年
print(f"\n长短腿毛 P&L(对冲前 / 对冲后):")
lp = (np.where(held > 0, held, 0) * R).sum(1); sp = (np.where(held < 0, held, 0) * R).sum(1)
lb = (np.where(held > 0, held, 0) * beta).sum(1); sb = (np.where(held < 0, held, 0) * beta).sum(1)
lp_h = lp - lb * mkt; sp_h = sp - sb * mkt
for y in ("2024", "2025", "2026"):
    m = yr == y
    if m.sum() < 50:
        continue
    print(f"  {y}: 长 {lp[m].sum():+.3f}/{lp_h[m].sum():+.3f}   短 {sp[m].sum():+.3f}/{sp_h[m].sum():+.3f}")
print("\n判读: 对冲后 2025-26 翻正、长短腿不再抵消、全程Sharpe升 = beta 污染是真凶(可去除)→ 下一步做干净 beta-中性构造。")
