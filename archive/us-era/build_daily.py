"""Massive 分钟 → **干净日频** panel（全市场，喂 pipeline/build.py）。
为什么：旧 akshare 日频 mask 只有 0.298（70% 空）；Massive 派生日频 ~0.87+，密。
做法：每日文件 → 正常时段聚合成日 OHLCV（open=首、close=尾、high/low、volume/transactions 求和）
      → 按当日美元成交额地板去掉死微盘 → 全部 ticker（无幸存者偏差，build.py 再按中位额选 top-N）。
跑：PYTHONUTF8=1 .venv/Scripts/python.exe build_daily.py --min-dv 200000
产物：data/panel_long.parquet（symbol/date/OHLCV/transactions/category）。然后：
     PYTHONUTF8=1 .venv/Scripts/python.exe -m pipeline.build --panel data/panel_long.parquet --out data/clean/daily2k --v1-n 2000
"""
import argparse
import concurrent.futures as cf
import sys
import time

import numpy as np
import pandas as pd

from build_intraday import all_files, SESS_LO, SESS_HI

_G = {}


def _init(min_dv):
    _G["min_dv"] = min_dv


def _process(path):
    """读一日 → **先按美元成交额地板砍 ticker(无需时间戳,快)** → 只对幸存者转时区+时段过滤
    → 聚合成日 OHLCV。把贵的 tz_convert 从 130 万行降到几千行（提速十几倍）。"""
    df = pd.read_csv(path)                          # .gz 自动解压
    if df.empty:
        return None
    dv = (df["close"] * df["volume"]).groupby(df["ticker"]).transform("sum")   # 当日全段$vol(便宜)
    df = df[dv >= _G["min_dv"]]
    if df.empty:
        return None
    ts = pd.to_datetime(df["window_start"], unit="ns", utc=True).dt.tz_convert("America/New_York")
    mins = (ts.dt.hour * 60 + ts.dt.minute).to_numpy()
    keep = (mins >= SESS_LO) & (mins < SESS_HI)     # 正常时段
    df = df[keep].copy()
    if df.empty:
        return None
    df["date"] = ts[keep].dt.strftime("%Y-%m-%d").to_numpy()
    df = df.sort_values("window_start")
    agg = df.groupby("ticker", sort=False).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum"),
        transactions=("transactions", "sum"), date=("date", "first"),
    ).reset_index()
    return agg if not agg.empty else None


def main(a):
    files = all_files()
    print(f"{len(files)} 天文件；美元成交额地板=${a.min_dv:,.0f}/天", flush=True)
    t0 = time.time(); out = []
    with cf.ProcessPoolExecutor(max_workers=a.workers, initializer=_init, initargs=(a.min_dv,)) as ex:
        for i, r in enumerate(ex.map(_process, files, chunksize=4)):
            if r is not None:
                out.append(r)
            if (i + 1) % 300 == 0:
                print(f"  {i+1}/{len(files)}  {time.time()-t0:.0f}s", flush=True)
    panel = pd.concat(out, ignore_index=True)
    panel = panel.rename(columns={"ticker": "symbol"})
    panel["category"] = "stock"                    # build.py 的 select_v1/segment_quality 需要
    panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)
    panel.to_parquet("data/panel_long.parquet", index=False)
    print(f"完成: {len(panel):,} 行, {panel['symbol'].nunique()} symbol, "
          f"{panel['date'].min()}~{panel['date'].max()} → data/panel_long.parquet ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-dv", type=float, default=200000, dest="min_dv", help="当日美元成交额地板(去死微盘)")
    ap.add_argument("--workers", type=int, default=max(1, (__import__("os").cpu_count() or 4) - 2))
    sys.exit(main(ap.parse_args()))
