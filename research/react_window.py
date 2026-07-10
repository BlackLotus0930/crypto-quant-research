# -*- coding: utf-8 -*-
"""Q1:每对保证金止损执行得及时吗?测真实崩盘日"触止损线→真强平"的反应窗口。
L=3 强平距离 dliq=31%;止损线=70%×dliq=22%。测每对从亏到 22% 到亏到 31% 之间有多少小时。
窗口宽(几小时)→分钟级执行能平掉=熔断可行;窗口窄(<1h/瞬时)→来不及=不可行。
跑:PYTHONUTF8=1 .venv/Scripts/python.exe research/react_window.py
"""
import glob, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from venues import canon
from research.xvenue_honest import fetch_intervals
from strategy import _cap_renorm
MM = 0.02; L = 3; dliq = 1.0 / L - MM; stop = 0.70 * dliq

z = np.load("data/clean/crypto_tensor_60min_pit.npz", allow_pickle=True)
adj = (z["adj"] if "adj" in z else z["adj_close"]).astype(float)
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
day = np.array([d[:10] for d in dates])

print(f"=== Q1 反应窗口:L={L}, 强平距离 {dliq:.0%}, 止损线 {stop:.0%}(70%)===\n")
print(f"{'崩盘日':>12s} {'触线对数':>7s} {'反应窗口(小时):中位':>16s} {'最短':>5s} {'1h内直接击穿(救不了)':>18s}")
for d in ["2021-05-19", "2022-05-11", "2025-10-10"]:
    idx = np.where(day == d)[0]
    if len(idx) < 6: continue
    t0 = idx[0]; hrs = idx
    act = np.where(valid[t0] & np.isfinite(byb[t0]) & (byb[t0] > 0) & np.isfinite(fb[t0]) & np.isfinite(fy[t0]) & (adj[t0] > 0))[0]
    sgn = np.sign(fb[t0, act] - fy[t0, act])
    cA = np.nan_to_num(adj[hrs][:, act] / adj[t0, act] - 1); cB = np.nan_to_num(byb[hrs][:, act] / byb[t0, act] - 1)
    windows = []; instant = 0; ntouch = 0
    for k in range(len(act)):
        wl = np.minimum(-sgn[k] * cA[:, k], sgn[k] * cB[:, k])    # 亏损腿路径
        h_stop = next((h for h in range(len(hrs)) if wl[h] < -stop), None)
        h_liq = next((h for h in range(len(hrs)) if wl[h] < -dliq), None)
        if h_stop is None:
            continue
        ntouch += 1
        if h_liq is not None:
            w = h_liq - h_stop; windows.append(w)
            if w <= 1:
                instant += 1          # 同一/下一根 bar 就击穿 = 来不及
    if windows:
        print(f"{d:>12s} {ntouch:>7d} {np.median(windows):>16.0f} {min(windows):>5.0f} {instant}/{len(windows)} ({instant/len(windows):.0%})")
    else:
        print(f"{d:>12s} {ntouch:>7d} {'触线后无一击穿(全可救)':>16s}")
print("\n判读:反应窗口=从'亏22%触止损'到'亏31%真强平'有几小时。")
print("  窗口≥2-3h → 分钟级执行能从容平掉=熔断可行;'1h内直接击穿'比例=瞬时gap救不了的部分(硬限制)。")
