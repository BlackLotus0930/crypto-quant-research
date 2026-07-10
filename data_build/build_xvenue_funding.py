"""把 Bybit 历史 funding 对齐到 pit 张量网格(因果前向填),与 Binance funding 拼成跨所面板。
→ data/clean/xvenue_funding.npz: f_bin / f_byb (T×N, 8h 费率, ≤t 已知), dates, slots。
跑：PYTHONUTF8=1 .venv/Scripts/python.exe build_xvenue_funding.py
"""
import glob
import os

import numpy as np
import pandas as pd

z = np.load("data/clean/crypto_tensor_60min_pit.npz", allow_pickle=True)
dates = z["dates"].astype(str); slots = list(z["slots"].astype(str))
T, N = len(dates), len(slots)
nmap = {s: i for i, s in enumerate(slots)}
grid_ts = np.sort(pd.read_parquet("data/clean/crypto_60min_pit.parquet", columns=["ts"])["ts"].unique()).astype(np.int64)
assert len(grid_ts) == T

f_bin = np.load("data/clean/funding_pit.npz", allow_pickle=True)["funding"].astype(np.float64)  # 已对齐 T×N
f_byb = np.full((T, N), np.nan, np.float64)
gdf = pd.DataFrame({"ts": grid_ts})
have = 0
for f in glob.glob("data/raw/bybit/funding/*.csv"):
    sym = os.path.basename(f)[:-4]
    j = nmap.get(sym)
    if j is None:
        continue
    df = pd.read_csv(f)
    if df.empty:
        continue
    df = df.sort_values("ts").drop_duplicates("ts", keep="last")
    m = pd.merge_asof(gdf, df[["ts", "funding"]], on="ts", direction="backward")  # ≤t 最近一次结算
    f_byb[:, j] = m["funding"].to_numpy(np.float64)
    have += 1

both = np.isfinite(f_bin) & np.isfinite(f_byb)
print(f"Binance 覆盖 {np.isfinite(f_bin).any(0).sum()}/{N};Bybit 覆盖 {have}/{N};两所都有的 bar 占比 {both.mean():.3f}")
print(f"两所都有时:|f_bin−f_byb| 年化 中位={np.nanmedian(np.abs(f_bin-f_byb)[both])*3*365:.2%} 均值={np.nanmean(np.abs(f_bin-f_byb)[both])*3*365:.2%}")
np.savez_compressed("data/clean/xvenue_funding.npz", f_bin=f_bin, f_byb=f_byb, dates=z["dates"], slots=z["slots"])
print("完成 → data/clean/xvenue_funding.npz")
