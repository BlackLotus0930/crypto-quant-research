"""双侧归因:不光看市场(谁亏)，更看我们模型的输出(它对将死的币下了什么注)。
在 pit book(含死币)上,用慢配置(hl24/cad24/rebal0.3/band0.01)重建持仓,拆 P&L 来源。
跑：python pnl_attrib.py --book walkforward_book.npz --tensor data/clean/crypto_tensor_60min_pit.npz
"""
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--book", default="walkforward_book.npz")
ap.add_argument("--tensor", default="data/clean/crypto_tensor_60min_pit.npz")
ap.add_argument("--hl", type=int, default=24); ap.add_argument("--cad", type=int, default=24)
ap.add_argument("--rebal", type=float, default=0.3); ap.add_argument("--band", type=float, default=0.01)
ap.add_argument("--bps", type=float, default=5)
a = ap.parse_args()

bk = np.load(a.book, allow_pickle=True)
W, R, bdates = bk["W"].astype(np.float64), bk["R"].astype(np.float64), bk["dates"].astype(str)
T, N = W.shape
zt = np.load(a.tensor, allow_pickle=True)
mask, adj, tdates = zt["mask"], zt["adj_close"].astype(np.float64), zt["dates"].astype(str)
slots = zt["slots"].astype(str) if "slots" in zt else np.arange(N).astype(str)

# book bar → tensor index
tidx = {d: i for i, d in enumerate(tdates)}
bk2t = np.array([tidx.get(d, -1) for d in bdates])

# 近期跌幅(120bar=5天)在决策时点
lc = np.log(np.where(adj > 0, adj, np.nan))
ret5 = np.full((T, N), np.nan)
for t in range(T):
    ti = bk2t[t]
    if ti >= 120:
        ret5[t] = lc[ti] - lc[ti - 120]

# 退市:tensor 里 mask 最后一次 True 的位置;在数据末尾前永久消失=退市
last_active = np.array([np.where(mask[:, c])[0].max() if mask[:, c].any() else -1 for c in range(N)])
delisted = (last_active >= 0) & (last_active < len(tdates) - 24 * 14)   # 末14天前就没了

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
contrib = held * R                       # 每格 P&L 贡献(毛)
yr = np.array([d[:4] for d in bdates])

print(f"book {T}bar N={N}; 退市币 {int(delisted.sum())}/{N}; 毛总 P&L={contrib.sum():+.3f}\n")

print("=== ① 退市 vs 存活 各贡献多少 P&L(毛) ===")
print(f"  退市币: {contrib[:, delisted].sum():+.3f}   存活币: {contrib[:, ~delisted].sum():+.3f}")
print(f"  退市币占总仓位(|held|)比例: {np.abs(held[:, delisted]).sum()/np.abs(held).sum():.1%}")

print("\n=== ② 长腿 vs 短腿 ===")
lc_pnl = np.where(held > 0, contrib, 0).sum(); sc_pnl = np.where(held < 0, contrib, 0).sum()
print(f"  长腿(做多): {lc_pnl:+.3f}   短腿(做空): {sc_pnl:+.3f}")
print(f"  长腿里退市币: {np.where(held > 0, contrib, 0)[:, delisted].sum():+.3f}")

print("\n=== ③ 模型是不是'越跌越买'(信号 vs 近期跌幅 横截面corr) ===")
cs = []
for t in range(T):
    v = (held[t] != 0) & np.isfinite(ret5[t])
    if v.sum() > 20 and held[t][v].std() > 1e-12 and ret5[t][v].std() > 1e-12:
        cs.append(np.corrcoef(held[t][v], ret5[t][v])[0, 1])
print(f"  corr(持仓, 近期跌幅) 均值 = {np.nanmean(cs):+.3f}  (负=越跌越做多=反转;很负=猛够飞刀)")

print("\n=== ④ 按'做多币的近期跌幅'分桶看长腿 P&L(找飞刀) ===")
lh = held > 0
rr = ret5[lh]; cc = contrib[lh]
qs = np.nanpercentile(rr, [10, 30, 50, 70, 90])
print(f"  {'近期收益桶':>16s} {'长腿P&L':>10s} {'仓位占比':>8s}")
edges = [-np.inf] + list(qs) + [np.inf]
labs = ["跌最狠10%", "10-30%", "30-50%", "50-70%", "70-90%", "涨最多10%"]
for i, lab in enumerate(labs):
    m = (rr > edges[i]) & (rr <= edges[i + 1])
    print(f"  {lab:>16s} {cc[m].sum():+10.3f} {np.abs(held[lh][m]).sum()/np.abs(held[lh]).sum():>7.1%}")

print("\n=== ⑤ 对将死的币,模型死前60天的平均持仓(>0=做多它们) ===")
for c in np.where(delisted)[0][:0]:
    pass
pre = []
for c in np.where(delisted)[0]:
    # 该币在 book 里对应退市前的 bar
    la = last_active[c]
    bm = (bk2t >= la - 24 * 60) & (bk2t <= la) & (bk2t > 0)
    if bm.sum() > 0:
        pre.append(held[bm, c].mean())
pre = np.array(pre)
print(f"  退市币死前60天平均持仓 = {np.nanmean(pre):+.4f}  (>0=模型平均在做多将死的币)")
print(f"  其中做多(>0)的比例 = {np.mean(pre > 0):.1%}")

print("\n=== ⑥ 逐年长腿/短腿/退市贡献 ===")
for y in ("2024", "2025", "2026"):
    ym = yr == y
    if ym.sum() < 50:
        continue
    l = np.where(held[ym] > 0, contrib[ym], 0).sum(); s = np.where(held[ym] < 0, contrib[ym], 0).sum()
    d = contrib[ym][:, delisted].sum()
    print(f"  {y}: 长腿{l:+.3f} 短腿{s:+.3f} 退市币{d:+.3f} 总{contrib[ym].sum():+.3f}")
