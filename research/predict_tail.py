# -*- coding: utf-8 -*-
"""Q2:极端 funding/OI/波动 能不能预测下周崩盘(→提前降跨所杠杆)?
目标=下周市场尾部(驱动跨所强平的东西):下周 BTC 最坏单日 + 下周截面最大单币日动。
特征(≤本周末,因果):① 近2周已实现波动(BTC)② funding 拥挤(|funding| 90分位)③ OI 变化 ④ 多空比极端。
测:各特征→下周尾部的 Spearman IC + 3次真实崩盘前信号是否已升 + 简单降杠杆规则的命中/误报。
跑:PYTHONUTF8=1 .venv/Scripts/python.exe research/predict_tail.py
"""
import glob, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

z = np.load("data/clean/crypto_tensor_60min_pit.npz", allow_pickle=True)
adj = z["adj_close"].astype(float); slots = list(z["slots"].astype(str)); dates = z["dates"].astype(str); T, N = adj.shape
nmap = {s: i for i, s in enumerate(slots)}; mask = z["mask"]; BTC = nmap.get("BTCUSDT")
xv = np.load("data/clean/xvenue_funding.npz", allow_pickle=True)
fb = xv["f_bin"]                                       # 每8h funding(拥挤代理,用 |.| 不在意间隔)
day = np.array([d[:10] for d in dates])

# 日聚合
df = pd.DataFrame({"day": day})
v = mask[:-1] & mask[1:] & (adj[:-1] > 0)
ret = np.zeros((T, N)); ret[:-1][v] = adj[1:][v] / adj[:-1][v] - 1
absret = np.where(mask, np.abs(np.nan_to_num(ret)), np.nan)
btcr = np.zeros(T)
if BTC is not None:
    bp = adj[:, BTC]; vb = (bp[:-1] > 0) & (bp[1:] > 0); btcr[:-1][vb] = bp[1:][vb] / bp[:-1][vb] - 1
fund_abs90 = np.array([np.nanpercentile(np.abs(fb[t][np.isfinite(fb[t])]), 90) if np.isfinite(fb[t]).sum() > 5 else np.nan for t in range(T)])

g = pd.DataFrame({"day": day, "btcr": btcr, "xmed": np.nanmedian(absret, 1), "f90": fund_abs90}).groupby("day").agg(
    btc_ret=("btcr", "sum"), btc_worst=("btcr", "min"), xmed=("xmed", "mean"), f90=("f90", "mean"))
g["xmax"] = pd.DataFrame({"day": day, "x": np.nanmax(absret, 1)}).groupby("day")["x"].max()
g = g.reset_index()

# 周聚合(每7个交易日一块)
g["wk"] = np.arange(len(g)) // 7
W = g.groupby("wk").agg(btc_worst=("btc_worst", "min"), xmax=("xmax", "max"),
                        f90=("f90", "mean"), btc_ret=("btc_ret", "sum"),
                        rv=("btc_ret", "std"), day0=("day", "first")).reset_index()
W["rv2"] = W["rv"].rolling(2).mean()                  # 近2周已实现波动(≤本周末)
# 特征(本周 t,≤t)→ 目标(下周 t+1)
W["tgt_btc_worst"] = W["btc_worst"].shift(-1)         # 下周 BTC 最坏单日
W["tgt_xmax"] = W["xmax"].shift(-1)                   # 下周截面最大单币日动
D = W.dropna(subset=["rv2", "f90", "tgt_btc_worst"]).copy()


def ic(a, b):
    ra = pd.Series(a).rank(); rb = pd.Series(b).rank()
    return np.corrcoef(ra, rb)[0, 1]


print(f"周样本 {len(D)}(~{len(D)/52:.1f}年)\n")
print("=== Spearman IC:本周特征 → 下周尾部(负=预测更深崩/更大动)===")
print(f"{'特征':>16s} | {'→下周BTC最坏单日':>16s} {'→下周最大单币动':>16s}")
for name, col in [("近2周波动 rv2", "rv2"), ("funding拥挤 f90", "f90"), ("本周BTC最坏", "btc_worst"), ("本周最大单币动", "xmax")]:
    ic1 = ic(D[col], D["tgt_btc_worst"]); ic2 = ic(D[col], D["tgt_xmax"])
    print(f"{name:>16s} | {ic1:>+15.2f} {ic2:>+15.2f}")
print("  (BTC最坏是负数:特征与它正IC=特征高→下周跌得更浅;负IC=特征高→下周崩更深)")

print("\n=== 3次真实崩盘:崩盘所在周 vs 前一周,信号是否已升(分位)===")
for cd in ["2021-05", "2022-05", "2025-10"]:
    wk = W[W["day0"].str[:7] == cd]
    if len(wk) == 0: continue
    i = wk.index[0]
    for lab, j in [("前一周", i - 1), ("崩盘周", i)]:
        if j < 0 or j >= len(W): continue
        rvp = (W["rv2"] < W.loc[j, "rv2"]).mean(); fp = (W["f90"] < W.loc[j, "f90"]).mean()
        print(f"  {cd} {lab}: 近2周波动 分位{rvp:.0%} | funding拥挤 分位{fp:.0%} | 该周BTC最坏 {W.loc[j,'btc_worst']:+.1%}")

print("\n=== 降杠杆规则:近2周波动 > X分位 → 跨所降杠杆。命中崩盘 vs 误报 ===")
thr_q = 0.80
hi = D["rv2"] > D["rv2"].quantile(thr_q)
# "坏周"=下周BTC最坏 < -10%(会触发跨所强平的级别)
badnext = D["tgt_btc_worst"] < -0.10
tp = (hi & badnext).sum(); fn = (~hi & badnext).sum(); fp = (hi & ~badnext).sum()
print(f"  规则:近2周波动 > {thr_q:.0%}分位 → 降杠杆(占周数 {hi.mean():.0%})")
print(f"  下周'坏周'(BTC<-10%)共 {badnext.sum()} 个:提前降到 {tp}/{badnext.sum()} ({tp/max(badnext.sum(),1):.0%}命中)")
print(f"  漏报 {fn} | 误报(降了但下周没事){fp}/{hi.sum()} ({fp/max(hi.sum(),1):.0%})")
print("\n判读:① IC 绝对值 >0.15 = 有预测力。② 崩盘周前信号若已在高分位 = 能提前降。")
print("  ③ 命中率 vs 误报率:vol-targeting 是否值得。波动聚类通常让 rv2 有效;funding/OI 是否加分看 IC。")
