# -*- coding: utf-8 -*-
"""崩盘熔断器:能不能把跨所书从 2x 提到 3x?
根因=崩盘里单腿被强平→对冲断→孤儿腿→脉冲反转巨亏。
想法:BTC 领跌、山寨滞后。BTC 当日累计动 > 阈值 T → **整对两腿一起平**(对冲完整退出),
在山寨腿击穿强平距离前就出场。测:真实崩盘日,加熔断 vs 不加,强平%与损失。
+ 误报代价:全样本 BTC 日内动 > T 的频率 → 年化拖累(每次误报=平+重开成本+丢当日funding)。
跑:PYTHONUTF8=1 .venv/Scripts/python.exe research/crash_breaker.py
"""
import glob, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from venues import canon
from research.xvenue_honest import fetch_intervals
from strategy import _cap_renorm
MM = 0.02

z = np.load("data/clean/crypto_tensor_60min_pit.npz", allow_pickle=True)
adj = (z["adj"] if "adj" in z else z["adj_close"]).astype(float)
slots = list(z["slots"].astype(str)); dates = z["dates"].astype(str); T, N = adj.shape
nmap = {s: i for i, s in enumerate(slots)}; valid = z["mask"] & (adj > 0)
BTC = nmap.get("BTCUSDT")
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
day = np.array([d[:10] for d in dates])


def sim(d, L, T_break=None, pair_stop=None):
    """返回 (强平腿%, 总损%) at 杠杆L。T_break=BTC熔断(整书);pair_stop=每对保证金止损(亏到 f×强平距离就平该对)。"""
    idx = np.where(day == d)[0]
    if len(idx) < 6 or BTC is None: return None
    t0 = idx[0]; hrs = idx
    act = np.where(valid[t0] & np.isfinite(byb[t0]) & (byb[t0] > 0) & np.isfinite(fb[t0]) & np.isfinite(fy[t0]) & (adj[t0] > 0))[0]
    if len(act) < 5: return None
    spr = np.abs(fb[t0, act] - fy[t0, act]); w = _cap_renorm(np.power(spr, 2) / max(np.power(spr, 2).sum(), 1e-12), 0.05)
    sgn = np.sign(fb[t0, act] - fy[t0, act])
    cA = np.nan_to_num(adj[hrs][:, act] / adj[t0, act] - 1); cB = np.nan_to_num(byb[hrs][:, act] / byb[t0, act] - 1)
    btc_cum = adj[hrs, BTC] / adj[t0, BTC] - 1
    if T_break is not None:
        br = np.where(np.abs(btc_cum) > T_break)[0]; hstar = br[0] if len(br) else len(hrs)
    else:
        hstar = len(hrs)
    dliq = 1.0 / L - MM
    liq_w = 0.0; flat_loss = 0.0
    for k in range(len(act)):
        legA = -sgn[k] * cA[:, k]; legB = sgn[k] * cB[:, k]
        worst_leg = np.minimum(legA, legB)                     # 亏得最多的那条腿路径
        h_liq = next((h for h in range(len(hrs)) if worst_leg[h] < -dliq), len(hrs))
        h_stop = next((h for h in range(len(hrs)) if pair_stop is not None and worst_leg[h] < -pair_stop * dliq), len(hrs))
        h_exit = min(h_liq, h_stop, hstar)                     # 最早:被强平 / 触止损 / BTC熔断
        if h_exit == h_liq and h_liq < min(h_stop, hstar):
            liq_w += w[k] * (1 - MM * L)                        # 真被强平=孤儿腿最坏
        else:
            h = min(h_exit, len(hrs) - 1)
            flat_loss += w[k] * L * (legA[h] + legB[h])          # 止损/熔断前整对平=只吃当时基差(对冲完整)
    return liq_w / max(w.sum(), 1e-9), -liq_w * (1 - MM * L) + flat_loss


CRASH = ["2021-05-19", "2022-05-11", "2025-10-10"]
print("=== 崩盘熔断器:跨所书 @3x,BTC 熔断 vs 不加 ===\n")
print(f"{'崩盘日':>12s} | {'不加@3x':>16s} | {'熔断T=12%':>16s} {'熔断T=8%':>16s} {'熔断T=5%':>16s}")
for d in CRASH:
    row = [d]
    cells = []
    for Tb in [None, 0.12, 0.08, 0.05]:
        r = sim(d, 3, Tb)
        cells.append(f"{r[0]:>5.0%}爆/{r[1]:>+5.0%}" if r else "  --  ")
    print(f"{d:>12s} | {cells[0]:>16s} | {cells[1]:>16s} {cells[2]:>16s} {cells[3]:>16s}")

print("\n=== 对比:加熔断后 跨所能不能上 3x?(T=8%)===")
print(f"{'崩盘日':>12s} {'2x无熔断':>12s} {'3x无熔断':>12s} {'3x+熔断T8':>12s}")
for d in CRASH:
    r2 = sim(d, 2, None); r3 = sim(d, 3, None); r3b = sim(d, 3, 0.08)
    if r2 and r3 and r3b:
        print(f"{d:>12s} {r2[1]:>+11.0%} {r3[1]:>+11.0%} {r3b[1]:>+11.0%}")

# 误报频率:全样本 BTC 日内 |动| > T 的天数
print("\n=== 误报代价(BTC 日内|动|>T 的频率 → 年化拖累)===")
dd = pd.DataFrame({"day": day, "btc": adj[:, BTC]}) if BTC is not None else None
g = dd.groupby("day")["btc"].agg(["first", "min", "max"])
intramove = np.maximum(g["max"] / g["first"] - 1, 1 - g["min"] / g["first"])  # 日内最大单向幅
ndays = len(g)
for Tb in [0.12, 0.08, 0.05]:
    nfire = (intramove > Tb).sum(); peryr = nfire / ndays * 365
    drag = peryr * (0.0020 + 0.0005)   # 每次误报 ~20bps 平+重开 + 5bps 丢当日funding
    print(f"  T={Tb:.0%}: 触发 {nfire}/{ndays}天 = {peryr:.0f}次/年 → 年化拖累 ~{drag:.1%}")
print("\n=== 更好的:每对保证金止损(盯自己的腿,任何崩盘都触发;只平危险的对)===")
print(f"{'崩盘日':>12s} {'2x基线':>9s} | {'3x止损60%':>10s} {'3x止损70%':>10s} {'4x止损70%':>10s} {'4x止损60%':>10s}")
for d in CRASH:
    r2 = sim(d, 2, None, None)
    cells = [sim(d, 3, None, 0.6), sim(d, 3, None, 0.7), sim(d, 4, None, 0.7), sim(d, 4, None, 0.6)]
    out = " ".join(f"{c[0]:>4.0%}爆/{c[1]:>+4.0%}" if c else " -- " for c in cells)
    print(f"{d:>12s} {r2[1]:>+8.0%} | {out}")
print("\n判读:")
print("  ① BTC 熔断:山寨领跌的崩盘(2025-10-10)抓不到,要 T=5% 才行但拖累12.9%=不划算。")
print("  ② 每对保证金止损:盯自己腿,任何崩盘触发,只平危险对。若 3x/4x 止损后损失≈2x基线 → 真能提杠杆。")
print("  ③ 止损拖累比 BTC 熔断小得多(只平触线的对,非整书);硬限制仍是单bar瞬时gap(分钟内击穿)。前向证滑点。")
