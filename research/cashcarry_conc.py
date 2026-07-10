"""集中度:cash-and-carry(v3 最佳配置)的利润来自多少币?是一两个还是分散?
逐币 OOS 贡献 + top-K 份额 + 有效币数(1/HHI) + jackknife(去掉最赚的几个还剩多少 Sharpe) + 逐年头号币(是否轮动)。
跑：python cashcarry_conc.py
"""
import numpy as np

z = np.load("data/clean/crypto_tensor_60min_pit.npz", allow_pickle=True)
mask = z["mask"]; adj = z["adj_close"].astype(float); tdv = z["tdv"].astype(float)
dates = z["dates"].astype(str); slots = z["slots"].astype(str); T, N = mask.shape
funding = np.load("data/clean/funding_pit.npz", allow_pickle=True)["funding"].astype(float)
spot = np.load("data/clean/spot_pit.npz", allow_pickle=True)["spot"].astype(float)
yr = np.array([d[:4] for d in dates]); cut = int(T * 0.4); ann = 8760
oos = np.zeros(T, bool); oos[cut:] = True
vp = mask & np.isfinite(spot) & (spot > 0) & (adj > 0)
f0 = np.nan_to_num(funding)
pr = np.zeros((T, N)); sr = np.zeros((T, N)); vpp = vp[:-1] & vp[1:]
pr[:-1][vpp] = adj[1:][vpp] / adj[:-1][vpp] - 1
sr[:-1][vpp] = spot[1:][vpp] / spot[:-1][vpp] - 1
price = np.clip(np.nan_to_num(sr - pr), -0.25, 0.25)
cr = f0 / 8.0 + price                                    # 每币每bar毛收益(funding+价格腿)
K, CAD, REBAL, BAND, POSHL, EXHL = 100, 24, 0.3, 0.01, 120, 72


def emaT(X, hl):
    al = 1 - 0.5 ** (1.0 / hl); E = np.empty_like(X); E[0] = X[0]
    for t in range(1, len(X)):
        E[t] = al * X[t] + (1 - al) * E[t - 1]
    return E


def make_W():
    W = np.zeros((T, N))
    for t in range(T):
        cand = np.where(vp[t] & np.isfinite(funding[t]) & (funding[t] > 0))[0]
        if len(cand) == 0:
            continue
        if len(cand) > K:
            cand = cand[np.argsort(tdv[t, cand])[::-1][:K]]
        w = f0[t, cand]; s = w.sum()
        if s > 0:
            W[t, cand] = w / s
    return W


def smooth(W, h):
    al = 1 - 0.5 ** (1.0 / h); S = np.empty_like(W); S[0] = W[0]
    for t in range(1, len(W)):
        S[t] = al * W[t] + (1 - al) * S[t - 1]
    g = np.abs(S).sum(1, keepdims=True)
    return np.divide(S, g, out=np.zeros_like(S), where=g > 0)


Ws = smooth(make_W(), POSHL); fe = emaT(f0, EXHL)
cur = np.zeros(N); held = np.zeros((T, N))
for t in range(T):
    if t % CAD == 0:
        d = Ws[t] - cur; mv = np.abs(d) > BAND
        new = cur.copy(); new[mv] = cur[mv] + REBAL * d[mv]
        s = np.abs(new).sum(); cur = new / s if s > 0 else new
    cur = np.where(fe[t] < 0, 0.0, cur)
    held[t] = cur

contrib_t = held * cr                                    # 每币每bar P&L贡献
coin_pnl = contrib_t[oos].sum(0)                         # 逐币 OOS 累计毛贡献
book = contrib_t[oos].sum(1)                             # 组合每bar(OOS)


def sh(p):
    return p.mean() / p.std() * np.sqrt(ann) if p.std() > 0 else 0.0


ever = (np.abs(held[oos]) > 1e-9).any(0).sum()
medpos = np.median((np.abs(held[oos]) > 1e-9).sum(1))
tot = coin_pnl.sum()
order = np.argsort(coin_pnl)[::-1]
pos_share = coin_pnl[coin_pnl > 0].sum()
hhi = (coin_pnl[coin_pnl > 0] ** 2).sum() / pos_share ** 2 if pos_share > 0 else 1
print(f"OOS 期间:曾持有 {ever} 个币;每bar中位持仓数={medpos:.0f};总毛P&L={tot:+.3f}")
print(f"正贡献币数={int((coin_pnl>0).sum())}  负贡献币数={int((coin_pnl<0).sum())}")
print(f"有效币数(1/HHI of 正贡献)={1/hhi:.1f}  (越大=越分散;接近1=全靠一个)\n")

print("=== top-K 币占'总毛P&L'比例 ===")
cum = np.cumsum(coin_pnl[order])
for k in (1, 3, 5, 10, 20, 50):
    if k <= N:
        print(f"  top-{k:>2d} 币: 占总 {cum[k-1]/tot*100:>5.0f}%   (这{k}个累计 {cum[k-1]:+.3f})")

print("\n=== 最赚的 10 个币(OOS 毛贡献) ===")
for i in order[:10]:
    print(f"  {slots[i]:18s} {coin_pnl[i]:+.4f}  ({coin_pnl[i]/tot*100:+.1f}%总)")

print("\n=== jackknife:去掉最赚的 top-k 币后,OOS Sharpe ===")
print(f"  全部:            Sharpe={sh(book):+.2f}")
for k in (1, 3, 5, 10):
    drop = order[:k]
    book_drop = book - contrib_t[oos][:, drop].sum(1)
    print(f"  去掉top-{k:>2d}最赚: Sharpe={sh(book_drop):+.2f}  (剩余毛 {coin_pnl[order[k:]].sum():+.3f})")

print("\n=== 逐年头号贡献币(看是否同一个/轮动) ===")
for y in ("2023", "2024", "2025", "2026"):
    m = oos & (yr == y)
    if m.sum() < 50:
        continue
    cp = contrib_t[m].sum(0); o = np.argsort(cp)[::-1]
    top3 = ", ".join(f"{slots[i]}({cp[i]/cp[cp>0].sum()*100:.0f}%)" for i in o[:3])
    print(f"  {y}: {top3}")

print("\n判读:有效币数大、top-5占比低、去掉top-10仍正Sharpe、逐年头号轮动 = 真分散溢价;反之=靠少数币/运气。")
