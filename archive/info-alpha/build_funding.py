"""把 Binance fundingRate(8h 更新)前向填到 bar 网格 → (symbol,ts,funding) 候选 parquet，供 probe_orthogonal 探。
funding = 永续持仓拥挤度/情绪(高正=多头拥挤、付费给空头)，部分正交于价格——数据已在本地(download_binance 一起拉的)。
跑：PYTHONUTF8=1 .venv/Scripts/python.exe build_funding.py --freq 60
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
    df["last_funding_rate"] = pd.to_numeric(df["last_funding_rate"], errors="coerce")
    df = df.dropna(subset=["calc_time", "last_funding_rate"])
    if df.empty:
        return None
    df["symbol"] = sym
    return df[["symbol", "calc_time", "last_funding_rate"]]


def main(a):
    panel = pd.read_parquet(f"data/clean/crypto_{a.freq}min.parquet")
    grid_syms = set(panel["symbol"].unique())
    grid_ts = np.sort(panel["ts"].unique())
    files = [f for f in glob.glob(FGLOB)
             if f.replace("\\", "/").split("/fundingRate/")[1].split("/")[0] in grid_syms]
    workers = max(1, (os.cpu_count() or 4) - 2)
    print(f"{len(files)} 个 fundingRate zip(universe 内)，线程 {workers}", flush=True)
    rows = []
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(read_zip, files):
            if r is not None:
                rows.append(r)
    fund = pd.concat(rows, ignore_index=True)
    fund["calc_time"] = fund["calc_time"].astype(np.int64)            # merge_asof 要求 key 同 dtype(网格 ts 是 int64)
    print(f"funding 原始 {len(fund):,} 行, {fund['symbol'].nunique()} 币", flush=True)

    gdf = pd.DataFrame({"ts": grid_ts.astype(np.int64)})
    out = []
    for sym, g in fund.groupby("symbol", sort=False):
        g = g.sort_values("calc_time").rename(columns={"calc_time": "ts", "last_funding_rate": "funding"})
        m = pd.merge_asof(gdf, g[["ts", "funding"]], on="ts")     # 每个 bar 取 ≤ts 的最近一次 funding(前向填)
        m["symbol"] = sym
        out.append(m)
    res = pd.concat(out, ignore_index=True)[["symbol", "ts", "funding"]].dropna()
    outp = f"data/clean/funding_{a.freq}min.parquet"
    res.to_parquet(outp, index=False)
    print(f"完成 → {outp}: {len(res):,} 行, {res['symbol'].nunique()} 币, funding 均值={res['funding'].mean():.6f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--freq", type=int, default=60)
    main(ap.parse_args())
