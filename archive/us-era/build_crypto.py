"""Binance 1m klines → 60min 加密面板（24/7、无 session、无拆股）。
通道白送：除 OHLCV 还有 quote_volume(=USDT额)、count(笔数)、taker_buy_volume(主动买量=订单流)。
ThreadPool（文件小、解压/IO 为主，单进程稳，不像 ProcessPool 在 Windows 上 spawn 崩）。留 2 核。
跑：PYTHONUTF8=1 .venv/Scripts/python.exe build_crypto.py --freq 60
产物：data/clean/crypto_{freq}min.parquet（symbol/ts/OHLCV/quote_volume/count/taker_buy_volume）。
"""
import argparse
import concurrent.futures as cf
import glob
import io
import os
import time
import zipfile

import numpy as np
import pandas as pd

KLINE_GLOB = "data/raw/binance/data/futures/um/monthly/klines/*/1m/*.zip"
# kline 列：0 open_time,1 open,2 high,3 low,4 close,5 volume,6 close_time,7 quote_volume,8 count,9 taker_buy_vol,...
USECOLS = [0, 1, 2, 3, 4, 5, 7, 8, 9]
NAMES = ["open_time", "open", "high", "low", "close", "volume", "qv", "count", "tbv"]
_G = {}


def _init(freq_ms):
    _G["freq_ms"] = freq_ms


def _read(path):
    """一个 symbol-month 1m zip → 按 freq 重采样成 bar。"""
    sym = path.replace("\\", "/").split("/klines/")[1].split("/")[0]
    try:
        z = zipfile.ZipFile(path); raw = z.read(z.namelist()[0])
    except Exception:
        return None
    df = pd.read_csv(io.BytesIO(raw), header=None, usecols=USECOLS, names=NAMES)
    df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce")     # 表头行→NaN→丢
    df = df.dropna(subset=["open_time"])
    if df.empty:
        return None
    for c in NAMES[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["bar"] = (df["open_time"] // _G["freq_ms"]) * _G["freq_ms"]        # 向下取整到 bar 起点(ms)
    df = df.sort_values("open_time")
    agg = df.groupby("bar", sort=True).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum"), qv=("qv", "sum"),
        count=("count", "sum"), tbv=("tbv", "sum"),
    ).reset_index()
    agg["symbol"] = sym
    return agg


def main(a):
    freq_ms = a.freq * 60_000
    files = glob.glob(KLINE_GLOB)
    print(f"{len(files)} 个 kline zip → {a.freq}min bar；线程 {a.workers}", flush=True)
    _init(freq_ms)
    t0 = time.time(); rows = []; done = 0
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(_read, files):
            done += 1
            if r is not None:
                rows.append(r)
            if done % 1000 == 0:
                print(f"  {done}/{len(files)}  {time.time()-t0:.0f}s", flush=True)
    panel = pd.concat(rows, ignore_index=True).sort_values(["symbol", "bar"]).reset_index(drop=True)
    panel = panel.rename(columns={"bar": "ts"})
    out = f"data/clean/crypto_{a.freq}min.parquet"
    panel.to_parquet(out, index=False)
    print(f"完成: {len(panel):,} 行, {panel['symbol'].nunique()} 币, "
          f"{pd.to_datetime(panel['ts'].min(),unit='ms')}~{pd.to_datetime(panel['ts'].max(),unit='ms')} "
          f"→ {out}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--freq", type=int, default=60, help="bar 分钟数(60=1h)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    main(ap.parse_args())
