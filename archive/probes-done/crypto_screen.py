"""加密廉价信号预筛(与 fast_screen 同尺子,可直接和股票比)。
读 Binance 1m klines 月度 zip → 按 UTC 日聚合(close 末、dollar vol=quote_volume 求和)
→ 同样的反转/动量/量因子横截面 IC。24/7、无隔夜。留 2 核。
跑：PYTHONUTF8=1 .venv/Scripts/python.exe crypto_screen.py
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

from fast_screen import rank_ic

DEF_WORKERS = max(1, (os.cpu_count() or 4) - 2)
KLINE_GLOB = "data/raw/binance/data/futures/um/monthly/klines/*/1m/*.zip"


def _read_zip(path):
    """一个 symbol-month 的 1m zip → 按 UTC 日 (close 末, dollar_vol=Σquote_volume)。"""
    sym = path.replace("\\", "/").split("/klines/")[1].split("/")[0]
    try:
        z = zipfile.ZipFile(path)
        raw = z.read(z.namelist()[0])
    except Exception:
        return None
    df = pd.read_csv(io.BytesIO(raw), header=None, usecols=[0, 4, 7],
                     names=["open_time", "close", "qv"])
    df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce")   # 表头行→NaN→丢
    df = df.dropna(subset=["open_time"])
    if df.empty:
        return None
    df["date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.strftime("%Y-%m-%d")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["qv"] = pd.to_numeric(df["qv"], errors="coerce")
    g = df.groupby("date").agg(close=("close", "last"), dvol=("qv", "sum")).reset_index()
    g["symbol"] = sym
    return g


def main(a):
    files = glob.glob(KLINE_GLOB)
    print(f"{len(files)} 个 symbol-month kline zip；线程={a.workers}（ThreadPool 单进程，稳）", flush=True)
    t0 = time.time(); rows = []
    done = 0
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:   # 文件小、解压/IO 为主，线程足够且不 spawn
        for r in ex.map(_read_zip, files):
            done += 1
            if r is not None:
                rows.append(r)
            if done % 1000 == 0:
                print(f"  读 {done}/{len(files)}  {time.time()-t0:.0f}s", flush=True)
    panel = pd.concat(rows, ignore_index=True)
    close = panel.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
    dv = panel.pivot_table(index="date", columns="symbol", values="dvol", aggfunc="sum").sort_index()
    print(f"聚合 {close.shape[0]} 个UTC日 x {close.shape[1]} 币  ({time.time()-t0:.0f}s)", flush=True)

    ret = np.log(close).diff()
    medv = dv.median().sort_values(ascending=False)
    tk = [t for t in medv.index if medv[t] > 0]
    universes = {
        "all":            tk,
        "top100":         tk[:100],
        "tail(100-200)":  tk[100:200],
    }
    factors = {
        "rev1":      -ret.shift(1),
        "rev5":      -ret.rolling(5).sum().shift(1),
        "rev5_skip": -ret.rolling(5).sum().shift(2),
        "mom21":      ret.rolling(21).sum().shift(1),
        "dvolchg":    np.log(dv.clip(lower=1)).diff().shift(1),
    }
    print(f"\n{'因子':>10s} | " + " | ".join(f"{u:>16s}" for u in universes), flush=True)
    print("-" * 80, flush=True)
    for fname, f in factors.items():
        cells = []
        for uname, cols in universes.items():
            cols = [c for c in cols if c in ret.columns]
            ic, ir, n = rank_ic(f, ret, cols, min_names=10)
            cells.append(f"IC{ic:+.4f} IR{ir:+.1f}")
        print(f"{fname:>10s} | " + " | ".join(f"{c:>16s}" for c in cells), flush=True)
    print(f"\n判读：和股票同尺子。加密 |IC|/IR 明显更高 → 更低效、信息潜力更高（值得上大模型）。", flush=True)
    print(f"({time.time()-t0:.0f}s 总)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=DEF_WORKERS)
    main(ap.parse_args())
