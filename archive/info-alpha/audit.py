"""对抗审计：主动推翻 cost_aware 的 net@5 +0.83。查四件最可能灌水的事。
跑：python audit.py --book walkforward_book.npz --tensor data/clean/crypto_tensor_60min_joint.npz
"""
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--book", default="walkforward_book.npz")
ap.add_argument("--tensor", default="data/clean/crypto_tensor_60min_joint.npz")
ap.add_argument("--hl", type=int, default=24); ap.add_argument("--cad", type=int, default=24)
ap.add_argument("--rebal", type=float, default=0.3); ap.add_argument("--band", type=float, default=0.01)
ap.add_argument("--ann", type=int, default=8760)
a = ap.parse_args()
z = np.load(a.book, allow_pickle=True)
W, R, dates = z["W"].astype(np.float64), z["R"].astype(np.float64), z["dates"].astype(str)
T, N = W.shape


def smooth(W, h):
    if h <= 0:
        return W
    al = 1 - 0.5 ** (1.0 / h); S = np.empty_like(W); S[0] = W[0]
    for t in range(1, len(W)):
        S[t] = al * W[t] + (1 - al) * S[t - 1]
    g = np.abs(S).sum(1, keepdims=True)
    return np.divide(S, g, out=np.zeros_like(S), where=g > 0)


def pnl_series(Ws, R, cad, rebal, band, delay=0):
    held = np.zeros(N); pnl = np.empty(T); turn = np.empty(T)
    for t in range(T):
        src = max(0, t - delay)                       # delay=1: 用上一bar的目标(测时序稳健)
        if t % cad == 0:
            d = Ws[src] - held; mv = np.abs(d) > band
            new = held.copy(); new[mv] = held[mv] + rebal * d[mv]
            g = np.abs(new).sum(); new = new / g if g > 0 else new
            turn[t] = np.abs(new - held).sum(); held = new
        else:
            turn[t] = 0.0
        pnl[t] = (held * R[t]).sum()
    return pnl, turn


def shp(pnl, ann):
    sd = pnl.std(); return pnl.mean() / sd * np.sqrt(ann) if sd > 0 else 0.0


Ws = smooth(W, a.hl)
pnl, turn = pnl_series(Ws, R, a.cad, a.rebal, a.band)
net5 = pnl - 5 / 1e4 * turn

print("=" * 60)
print("① 年化是否虚高(per-bar √8760 vs 日块 √365)")
print(f"  per-bar 净@5 Sharpe(√8760) = {shp(net5, 8760):+.2f}")
nblk = T // a.cad
blk = net5[:nblk * a.cad].reshape(nblk, a.cad).sum(1)        # 每 cad-bar 非重叠块
print(f"  {a.cad}-bar块 净@5 Sharpe(√{8760//a.cad}) = {shp(blk, 8760 // a.cad):+.2f}  (块数={nblk})")
ac1 = np.corrcoef(net5[1:], net5[:-1])[0, 1]
print(f"  per-bar 净 pnl lag-1 自相关 = {ac1:+.3f}  (高=√8760 夸大)")

print("=" * 60)
print("② 时序稳健(再加1bar执行延迟)")
pnl_d, turn_d = pnl_series(Ws, R, a.cad, a.rebal, a.band, delay=1)
net5_d = pnl_d - 5 / 1e4 * turn_d
print(f"  原始 净@5(√8760)   = {shp(net5, 8760):+.2f}")
print(f"  +1bar延迟 净@5      = {shp(net5_d, 8760):+.2f}  (大跌=吃了同bar信息/时序脆)")

print("=" * 60)
print("③ 随机符号 sanity(信号符号打乱→该≈0)")
rng = np.random.default_rng(0); shuf = []
for _ in range(5):
    sgn = rng.choice([-1.0, 1.0], size=N)
    p, tn = pnl_series(Ws * sgn, R, a.cad, a.rebal, a.band)
    shuf.append(shp(p - 5 / 1e4 * tn, 8760))
print(f"  打乱符号 净@5(5次) = {np.mean(shuf):+.2f} ± {np.std(shuf):.2f}  (应≈0)")

print("=" * 60)
print("④ 幸存者偏差(币有没有退市)")
zt = np.load(a.tensor, allow_pickle=True)
mask, td = zt["mask"], zt["dates"].astype(str)
hold = td >= "2024-01-01"
mh = mask[hold]
active_start = mh[:24 * 7].any(0)          # 头一周活跃
active_end = mh[-24 * 7:].any(0)           # 末一周活跃
# 退市=holdout中途由活变永久不活
last_active = np.array([np.where(mh[:, j])[0].max() if mh[:, j].any() else -1 for j in range(N)])
delisted = ((last_active >= 0) & (last_active < len(mh) - 24 * 14)).sum()   # 末14天前就没了
print(f"  N={N}; 头周活跃 {active_start.sum()}, 末周活跃 {active_end.sum()}")
print(f"  holdout 中途退市(末14天前消失)的币数 = {delisted}")
print(f"  → 退市≈0 = 大概率幸存者偏差(只含活到现在的币); 有不少退市 = universe 较真实")
print("=" * 60)
print(f"判读: ①日块Sharpe≈per-bar 且自相关低 → 年化不虚; ②延迟后仍正 → 时序稳; ③打乱≈0 → 是信号; ④退市数")
