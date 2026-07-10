"""把全宇宙(含死币)fundingRate 对齐到 pit 张量网格 → funding_pit.npz(T×N,与 crypto_tensor_60min_pit 1:1)。
funding = 永续多头拥挤度(高正=多头付费给空头);carry 信号源。前向填(merge_asof ≤t,因果)。
跑：PYTHONUTF8=1 .venv/Scripts/python.exe build_funding_pit.py
"""
import argparse
import concurrent.futures as cf
import glob
import io
import os
import zipfile

import numpy as np
import pandas as pd

FGLOB = "data/raw/binance/data/futures/um/monthly/fundingRate/*/*.zip"


def read_zip(path):
    """单个 fundingRate zip → DataFrame[symbol, calc_time(ms,int64), funding]。复用 build_funding 逻辑。"""
    sym = path.replace("\\", "/").split("/fundingRate/")[1].split("/")[0]
    try:
        z = zipfile.ZipFile(path); raw = z.read(z.namelist()[0])
    except Exception:
        return None
    df = pd.read_csv(io.BytesIO(raw))
    df = df.rename(columns={c: c.strip() for c in df.columns})
    if "calc_time" not in df.columns or "last_funding_rate" not in df.columns:
        return None
    df["calc_time"] = pd.to_numeric(df["calc_time"], errors="coerce")
    df["funding"] = pd.to_numeric(df["last_funding_rate"], errors="coerce")
    df = df.dropna(subset=["calc_time", "funding"])
    if df.empty:
        return None
    df["symbol"] = sym
    return df[["symbol", "calc_time", "funding"]]


def main(a):
    z = np.load(a.tensor, allow_pickle=True)
    dates = z["dates"].astype(str)
    slots = list(z["slots"].astype(str))
    T, N = len(dates), len(slots)
    nmap = {s: i for i, s in enumerate(slots)}
    # ms 网格直接取自面板 ts(int64 ms,与 funding calc_time 同口径;张量时间序=面板 sort(unique(ts)))
    grid_ts = np.sort(pd.read_parquet(a.panel, columns=["ts"])["ts"].unique()).astype(np.int64)
    assert len(grid_ts) == T, f"面板 ts 数 {len(grid_ts)} != 张量 T {T}"
    print(f"pit 张量 T={T} N={N}; 网格 {pd.to_datetime(grid_ts[0],unit='ms')} → {pd.to_datetime(grid_ts[-1],unit='ms')}", flush=True)

    files = [f for f in glob.glob(FGLOB)
             if f.replace("\\", "/").split("/fundingRate/")[1].split("/")[0] in nmap]
    workers = max(1, (os.cpu_count() or 4) - 2)
    print(f"{len(files)} 个 fundingRate zip(在 pit 宇宙内),线程 {workers}", flush=True)
    rows = []
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(read_zip, files):
            if r is not None:
                rows.append(r)
    fund = pd.concat(rows, ignore_index=True)
    fund["calc_time"] = fund["calc_time"].astype(np.int64)
    print(f"funding 原始 {len(fund):,} 行, {fund['symbol'].nunique()} 币", flush=True)

    funding = np.full((T, N), np.nan, np.float32)
    gdf = pd.DataFrame({"ts": grid_ts})                    # 已排序(张量按时间)
    have = 0
    for sym, g in fund.groupby("symbol", sort=False):
        j = nmap.get(sym)
        if j is None:
            continue
        g = g.sort_values("calc_time").rename(columns={"calc_time": "ts"})
        m = pd.merge_asof(gdf, g[["ts", "funding"]], on="ts", direction="backward")   # ≤t 最近一次结算(因果前向填)
        funding[:, j] = m["funding"].to_numpy(np.float32)
        have += 1

    cov = np.isfinite(funding).any(0).mean()
    nz = funding[np.isfinite(funding)]
    ann_mean = np.nanmean(nz) * 3 * 365                    # 8h rate × 3/day × 365
    print(f"覆盖 {have}/{N} 币 ({cov:.1%} slot 至少有过 funding); funding 非空均值={np.nanmean(nz):.6e}/8h ≈ {ann_mean:+.2%}/yr", flush=True)
    outp = "data/clean/funding_pit.npz"
    np.savez_compressed(outp, funding=funding, dates=dates, slots=np.array(slots, dtype=str))
    print(f"完成 → {outp}  shape={funding.shape}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensor", default="data/clean/crypto_tensor_60min_pit.npz")
    ap.add_argument("--panel", default="data/clean/crypto_60min_pit.parquet")
    main(ap.parse_args())
