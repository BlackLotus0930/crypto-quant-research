"""把全宇宙(含死币)metrics(OI + 大户多空比,5min)对齐到 pit 张量网格 → oi_pit.npz(T×N)。
逐 symbol 流式处理(5min×732 币放不下内存):每币读它的 daily zips → merge_asof ≤t 到小时网格。
跑：PYTHONUTF8=1 .venv/Scripts/python.exe build_oi_pit.py
"""
import argparse
import concurrent.futures as cf
import glob
import io
import os
import zipfile

import numpy as np
import pandas as pd

MDIR = "data/raw/binance/data/futures/um/daily/metrics"


def read_zip(path):
    try:
        z = zipfile.ZipFile(path); raw = z.read(z.namelist()[0])
    except Exception:
        return None
    try:
        df = pd.read_csv(io.BytesIO(raw), usecols=["create_time", "sum_open_interest_value", "sum_toptrader_long_short_ratio"])
    except Exception:
        return None
    return df if len(df) else None


def main(a):
    z = np.load(a.tensor, allow_pickle=True)
    dates = z["dates"].astype(str); slots = list(z["slots"].astype(str)); nmap = {s: i for i, s in enumerate(slots)}
    T, N = len(dates), len(slots)
    grid_ts = np.sort(pd.read_parquet(a.panel, columns=["ts"])["ts"].unique()).astype(np.int64)
    assert len(grid_ts) == T
    gdf = pd.DataFrame({"ts": grid_ts})
    print(f"pit T={T} N={N}; 逐币对齐 metrics(5min→小时,≤t 因果)", flush=True)

    oi = np.full((T, N), np.nan, np.float32); tt = np.full((T, N), np.nan, np.float32)

    def proc(sym):
        j = nmap[sym]
        zips = glob.glob(f"{MDIR}/{sym}/*.zip")
        if not zips:
            return None
        dfs = [d for d in (read_zip(p) for p in zips) if d is not None]
        if not dfs:
            return None
        df = pd.concat(dfs, ignore_index=True)
        ts = pd.to_datetime(df["create_time"], utc=True, errors="coerce").values.astype("datetime64[ms]").astype("int64")
        df = pd.DataFrame({"ts": ts, "oi": pd.to_numeric(df["sum_open_interest_value"], errors="coerce"),
                           "tt": pd.to_numeric(df["sum_toptrader_long_short_ratio"], errors="coerce")}).dropna()
        if df.empty:
            return None
        df = df.sort_values("ts").drop_duplicates("ts", keep="last")
        m = pd.merge_asof(gdf, df, on="ts", direction="backward")
        return j, m["oi"].to_numpy(np.float32), m["tt"].to_numpy(np.float32)

    workers = max(1, (os.cpu_count() or 4) - 2)
    have = 0
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(proc, slots):
            if r is not None:
                j, o, t = r; oi[:, j] = o; tt[:, j] = t; have += 1
    cov = np.isfinite(tt).any(0).mean()
    print(f"覆盖 {have}/{N} 币 ({cov:.1%}); top-LSR 非空均值={np.nanmean(tt[np.isfinite(tt)]):.3f}", flush=True)
    np.savez_compressed("data/clean/oi_pit.npz", oi=oi, tt=tt, dates=dates, slots=np.array(slots, dtype=str))
    print("完成 → data/clean/oi_pit.npz", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensor", default="data/clean/crypto_tensor_60min_pit.npz")
    ap.add_argument("--panel", default="data/clean/crypto_60min_pit.parquet")
    main(ap.parse_args())
