"""把所有币(含死币)的 1h K线 zip → 60min 面板 parquet(point-in-time, 修幸存者偏差)。
1h kline 即 60min,直接读、不用聚合 1m。schema 对齐现有 crypto_60min.parquet。
跑：PYTHONUTF8=1 .venv/Scripts/python.exe build_panel_pit.py --workers 12
"""
import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

RAW = "data/raw/binance"
KCOLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
         "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
NUM = ["open", "high", "low", "close", "volume", "quote_volume", "count", "taker_buy_volume"]


def parse_sym(args):
    sym, files = args
    dfs = []
    for f in files:
        try:
            d = pd.read_csv(f, header=None)
        except Exception:
            continue
        if str(d.iloc[0, 0]).startswith("open_time"):     # 新文件带表头
            d = d.iloc[1:]
        d = d.iloc[:, :len(KCOLS)]
        d.columns = KCOLS[:d.shape[1]]
        dfs.append(d)
    if not dfs:
        return None
    d = pd.concat(dfs, ignore_index=True)
    for c in NUM + ["open_time"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["open_time", "close"])
    # open_time 偶有微秒(16位)→规整到毫秒
    ot = d["open_time"].astype(np.int64)
    d["ts"] = np.where(ot > 1e15, ot // 1000, ot).astype(np.int64)
    d = d.drop_duplicates(subset=["ts"]).sort_values("ts")
    d["symbol"] = sym
    return d.rename(columns={"quote_volume": "qv", "taker_buy_volume": "tbv"})[
        ["ts", "open", "high", "low", "close", "volume", "qv", "count", "tbv", "symbol"]]


def main(a):
    files = glob.glob(f"{RAW}/**/monthly/klines/**/{a.interval}/*.zip", recursive=True)
    groups = {}
    for f in files:
        parts = f.replace("\\", "/").split("/")
        sym = parts[parts.index("klines") + 1]
        groups.setdefault(sym, []).append(f)
    print(f"{len(files)} 个 zip, {len(groups)} 个 symbol(含死币)", flush=True)

    res = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, r in enumerate(ex.map(parse_sym, groups.items())):
            if r is not None and len(r):
                res.append(r)
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(groups)} symbols", flush=True)
    panel = pd.concat(res, ignore_index=True)
    # 只留 grid 上完整小时(ts 为整点 ms)
    panel = panel[panel["ts"] % 3600000 == 0]
    panel = panel.dropna(subset=["close", "volume"])
    panel = panel[panel["close"] > 0]
    out = f"data/clean/crypto_60min_pit.parquet"
    panel.to_parquet(out)
    n_sym = panel["symbol"].nunique()
    rng = pd.to_datetime([panel["ts"].min(), panel["ts"].max()], unit="ms", utc=True)
    print(f"完成 → {out}  {len(panel):,} 行, {n_sym} 币, {rng[0].date()}~{rng[1].date()}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--workers", type=int, default=12)
    main(ap.parse_args())
