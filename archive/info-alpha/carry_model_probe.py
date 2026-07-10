"""Q1 数据推导:我们已有的反转模型,真能对冲 carry 的价格腿吗? 能避多少尾部?
关键怀疑:carry(空高funding=空"拥挤多头")和反转模型(空"近期被拉高的")可能是**同号**(都=短动量)
→ 若高度相关,模型是冗余的,救不了价格腿。用数据判。
宇宙=198 survivor(我有 model book W/R + funding_60min);survivor 只影响绝对水平,不影响"同号/能否过滤"的结论。
跑：python carry_model_probe.py
"""
import numpy as np
import pandas as pd

ANN = 8760
bk = np.load("data/clean/crypto_holdout_book.npz", allow_pickle=True)
Wm = bk["W"].astype(np.float64); R = bk["R"].astype(np.float64); dates = bk["dates"].astype(str)
T, N = Wm.shape
slots = np.load("data/clean/crypto_tensor_60min.npz", allow_pickle=True)["slots"].astype(str)
assert len(slots) == N, f"{len(slots)} != {N}"

f = pd.read_parquet("data/clean/funding_60min.parquet")
f["ds"] = pd.to_datetime(f["ts"], unit="ms", utc=True).dt.strftime("%Y-%m-%d %H:%M:%S")
fw = f.pivot_table(index="ds", columns="symbol", values="funding", aggfunc="last").reindex(index=dates, columns=slots)
F = fw.to_numpy(np.float64)
yr = np.array([d[:4] for d in dates])
print(f"book T={T} N={N}; funding 覆盖 slot={np.isfinite(F).any(0).mean():.1%} cell={np.isfinite(F).mean():.1%}\n")

active = np.isfinite(F) & np.isfinite(R) & (R != 0.0)


def gross1(W):
    g = np.abs(W).sum(1, keepdims=True)
    return np.divide(W, g, out=np.zeros_like(W), where=g > 0)


# carry 信号:每bar active 内 -zscore(funding)(空高funding),市场中性,gross1
Wc = np.zeros((T, N))
for t in range(T):
    idx = np.where(active[t])[0]
    if len(idx) < 10:
        continue
    x = F[t, idx]; sd = x.std()
    if sd > 1e-12:
        Wc[t, idx] = -(x - x.mean()) / sd
Wc = gross1(Wc)
Wmn = gross1(np.where(active, Wm, 0.0))             # 模型权重也归一、限同 active

# ---- 推导① carry 与 模型 同号还是正交?----
cs = []
for t in range(T):
    a = active[t] & (Wc[t] != 0) & (Wmn[t] != 0)
    if a.sum() > 20 and Wc[t][a].std() > 1e-12 and Wmn[t][a].std() > 1e-12:
        cs.append(np.corrcoef(Wc[t][a], Wmn[t][a])[0, 1])
print(f"① 横截面 corr(carry权重, 模型权重) 均值 = {np.nanmean(cs):+.3f}")
print(f"   (>0=同号→模型与carry押同向=冗余,救不了价格腿; ~0=正交→可能有用; <0=反向→可对冲)")


def IC(W):
    v = []
    for t in range(T):
        a = active[t] & (W[t] != 0)
        if a.sum() > 20 and W[t][a].std() > 1e-12 and R[t][a].std() > 1e-12:
            v.append(np.corrcoef(W[t][a], R[t][a])[0, 1])
    return np.nanmean(v)


print(f"\n② 各信号对次bar收益的 IC:  模型={IC(Wmn):+.4f}   carry={IC(Wc):+.4f}")
print(f"   (carry IC<0 = 价格腿确实逆我们;模型 IC>0 = 模型能排序价格)")

# ---- 推导③ 在 carry 做空的高funding币里,模型能挑出"安全空"吗 ----
short = active & (Wc < 0)                            # carry 想做空的格(高funding)
mb = Wm > 0                                          # 模型看多(危险:别空)
ms = Wm <= 0                                         # 模型不看多(相对安全空)
rs_dangerous = R[short & mb]; rs_safe = R[short & ms]
print(f"\n③ carry做空格中,按模型分:")
print(f"   模型看多(危险)的格: 次bar收益均值={rs_dangerous.mean():+.5f} n={short.__and__(mb).sum()}  ←这些空会亏(收益正)")
print(f"   模型不看多(安全)的格: 次bar收益均值={rs_safe.mean():+.5f} n={short.__and__(ms).sum()}")
print(f"   差 = {rs_dangerous.mean()-rs_safe.mean():+.5f} (>0=模型确实把'会涨的高funding'识别出来了→过滤有用)")


# ---- 推导④ 纯carry vs 模型过滤carry: 逐年 net@5 + 价格/funding + 尾部 ----
def smooth(W, h):
    if h <= 0:
        return W
    al = 1 - 0.5 ** (1.0 / h); S = np.empty_like(W); S[0] = W[0]
    for t in range(1, len(W)):
        S[t] = al * W[t] + (1 - al) * S[t - 1]
    return gross1(S)


F0 = np.nan_to_num(F)


def pnl(W, hl=0, cad=24, rebal=0.3, band=0.01):
    Ws = smooth(W, hl); cur = np.zeros(N)
    price = np.zeros(T); fp = np.zeros(T); turn = np.zeros(T)
    for t in range(T):
        if t % cad == 0:
            d = Ws[t] - cur; mv = np.abs(d) > band
            new = cur.copy(); new[mv] = cur[mv] + rebal * d[mv]
            g = np.abs(new).sum(); new = new / g if g > 0 else new
            turn[t] = np.abs(new - cur).sum(); cur = new
        price[t] = (cur * R[t]).sum(); fp[t] = -(cur * F0[t] / 8.0).sum()
    return price, fp, turn


def sh(net, m):
    p = net[m]
    return p.mean() / p.std() * np.sqrt(ANN) if p.std() > 0 else 0.0


# 模型过滤:去掉"模型看多的carry空"和"模型看空的carry多"
drop = ((Wc < 0) & (Wm > 0)) | ((Wc > 0) & (Wm < 0))
Wf = gross1(np.where(drop, 0.0, Wc))
print(f"\n④ 过滤掉 {drop.sum()/ max(1,(Wc!=0).sum()):.1%} 的carry格(模型反对的);逐年 net@5(price+funding−5bps):")
print(f"{'区间':>8s} {'纯carry':>9s} {'模型过滤':>9s} {'纯-价格腿':>9s} {'过滤-价格腿':>11s}")
for W, _ in [(Wc, 0)]:
    pass
pc, fc, tc = pnl(Wc); pf, ff, tf = pnl(Wf)
nc = pc + fc - 5e-4 * tc; nf = pf + ff - 5e-4 * tf
for lab, m in [("全程", np.ones(T, bool))] + [(y, yr == y) for y in ("2024", "2025", "2026")]:
    if m.sum() < 50:
        continue
    print(f"{lab:>8s} {sh(nc,m):>+9.2f} {sh(nf,m):>+9.2f} {pc[m].sum():>+9.3f} {pf[m].sum():>+11.3f}")

print(f"\n尾部(全程): 纯carry 最差bar={nc.min():+.4f} 1%分位={np.percentile(nc,1):+.4f}  "
      f"| 过滤 最差bar={nf.min():+.4f} 1%分位={np.percentile(nf,1):+.4f}")
print(f"价格腿累计: 纯carry={pc.sum():+.3f} → 过滤={pf.sum():+.3f} (越接近0越好=价格风险被砍)")
print(f"\n判读: ①>0.3=冗余(模型救不了,选A现货对冲); ③差>0且④过滤后2026/尾部明显改善=模型有用(B可行)。")
