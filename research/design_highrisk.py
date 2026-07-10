# -*- coding: utf-8 -*-
"""高风险投资设计:这是可全损的10%风险仓,不为100年COVID设防。
但要用真实数据回答:(1)杀死杠杆的崩盘多久一次(不是COVID频率,是常规崩盘)
(2)给定真实崩盘频率+你能接受归零,最优杠杆是多少(Kelly on 真实崩盘分布)。
跑:PYTHONUTF8=1 .venv/Scripts/python.exe research/design_highrisk.py
"""
import glob, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import strategy as S
from venues import canon
from research.xvenue_honest import fetch_intervals, honest_legs
ANN = 8760

z = np.load("data/clean/crypto_tensor_60min_pit.npz", allow_pickle=True)
adj = z["adj_close"].astype(float); tdv = z["tdv"].astype(float)
slots = list(z["slots"].astype(str)); dates = z["dates"].astype(str); T, N = adj.shape
nmap = {s: i for i, s in enumerate(slots)}; valid = z["mask"] & (adj > 0)
xv = np.load("data/clean/xvenue_funding.npz", allow_pickle=True)
bi, yi = fetch_intervals(); biv = np.array([bi.get(s, 8) for s in slots]); yiv = np.array([yi.get(s, 8) for s in slots])
fb = xv["f_bin"] / biv[None, :]; fy = xv["f_byb"] / yiv[None, :]
grid = np.sort(pd.read_parquet("data/clean/crypto_60min_pit.parquet", columns=["ts"])["ts"].unique()).astype(np.int64)
gdf = pd.DataFrame({"ts": grid}); byb = np.full((T, N), np.nan)
for f in glob.glob("data/raw/bybit/kline/*.csv"):
    j = nmap.get(os.path.basename(f)[:-4])
    if j is None: continue
    df = pd.read_csv(f).sort_values("ts").drop_duplicates("ts", keep="last")
    if not df.empty:
        byb[:, j] = pd.merge_asof(gdf, df[["ts", "close"]], on="ts", direction="backward", tolerance=3600_001)["close"].to_numpy()
day = np.array([d[:10] for d in dates]); udays = pd.unique(day)

# (1) 崩盘频率:每天 高funding书持仓宇宙 的截面中位日内最坏动
both = (adj > 0) & np.isfinite(fb) & np.isfinite(fy) & np.isfinite(byb) & (byb > 0)
med_move = {}
for d in udays:
    idx = np.where(day == d)[0]
    if len(idx) < 6: continue
    t0 = idx[0]
    act = np.where(both[t0])[0]
    if len(act) < 10: continue
    # 选高funding 30币(书实际持仓画像)
    spr = np.abs(fb[t0, act] - fy[t0, act]); top = act[np.argsort(spr)[::-1][:30]]
    intraday = np.abs(adj[idx][:, top] / adj[t0, top] - 1)
    med_move[d] = np.nanmedian(np.nanmax(intraday, 0))
mv = np.array([v for v in med_move.values() if np.isfinite(v)])
yrs = len(mv) / 365.0
print(f"=== (1) 崩盘频率(真实,{len(mv)}天≈{yrs:.1f}年,书持仓宇宙截面中位日内最坏动)===")
for thr in [0.15, 0.20, 0.25, 0.35]:
    n = (mv > thr).sum()
    print(f"  日内动 >{thr:.0%} 的天: {n} 天 = 每 {yrs*365/max(n,1):.0f} 天一次 ≈ 每年 {n/yrs:.1f} 次")
print("  (>20% 约对应 L=3 群体强平线;>35%=2021-05-19/COVID 级)\n")

# (2) 最优杠杆:Kelly on 真实崩盘
carry = S.backtest(S.CarryConfig(leverage=1.0))["net"]
d = np.where(both, fb - fy, 0.0)
def rets(p, m): o = np.zeros((T, N)); v = m[:-1] & m[1:]; o[:-1][v] = (p[1:]/p[:-1]-1)[v]; return np.nan_to_num(o)
xnet = honest_legs(d, rets(adj, both), rets(byb, both), both, tdv, np.abs(d), "x")["net_h"]
comb = 0.5 * carry + 0.5 * xnet
base = comb.mean() * ANN
print(f"=== (2) 高风险仓最优杠杆(base 年化 {base:.0%},真实崩盘频率)===")
# 崩盘损失 lookup(L→损失,取自 stress_crash/protected 三崩盘日均)
def crashloss(L):  # 群体强平损,L=1~0,L≥3飙升
    pts = {1:0.0, 2:0.10, 3:0.55, 4:0.72, 5:0.83, 6:0.88, 8:0.93, 10:0.97}
    ks = sorted(pts);
    if L <= ks[0]: return 0.0
    for a, b in zip(ks, ks[1:]):
        if L <= b: return pts[a] + (pts[b]-pts[a])*(L-a)/(b-a)
    return 0.97
cr_per_yr = (mv > 0.20).sum() / yrs        # >20% 崩盘/年(杀L=3的频率)
print(f"  用 >20% 崩盘频率 = {cr_per_yr:.1f} 次/年\n")
print(f"  {'杠杆':>4s} {'正常年化':>8s} {'崩盘损/次':>8s} {'几何年化(扣崩盘)':>14s} {'每年归零概率*':>11s}")
best = (-9, 0)
for L in [1,2,3,4,5,6,8]:
    normal = L * base
    cl = crashloss(L)
    # 几何年化 ≈ exp( ln(1+normal) + cr_per_yr*ln(1-cl) ) - 1
    if cl >= 0.999: g = -1.0
    else: g = np.exp(np.log(1+normal) + cr_per_yr*np.log(1-cl)) - 1
    p_zero = 1-(1-min(cl,0.999))**cr_per_yr if cl>0.5 else 0.0   # 粗略:崩盘致重损概率
    tag = " ←几何最优" if g > best[0] else ""
    if g > best[0]: best = (g, L)
    print(f"  {L:>3d}x {normal:>+7.0%} {cl:>7.0%} {g:>+13.0%} {p_zero:>10.0%}{tag}")
print(f"\n  几何增长最优杠杆 ≈ {best[1]}x(扣真实崩盘频率后,复利增长最大)")
print("  *崩盘致重损(>50%)的年概率,粗估。\n")
print("判读:")
print("  - 杀杠杆的是**常规崩盘(每年 ~N 次),不是 100 年的 COVID**。")
print("  - 若你**复利这个仓**(再投):几何最优 ~上面那个 L,过了它崩盘把复利吃光。")
print("  - 若你**当一次性可全损赌注**(每年补仓、接受归零):可超过几何最优博更大单期上行,但多年期望被归零拖低。")
print("  - COVID 级(>35%)更罕见,但常规崩盘已经在每年 N 次这个量级定上限。")
