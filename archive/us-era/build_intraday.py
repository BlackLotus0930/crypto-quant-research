"""日内数据构建：Massive minute flat files → 过滤 universe + 正常时段(9:30-16:00) → 重采样 → 日内面板。

为什么这样：
  - universe 从 **Massive 自身**近期 $vol 选（top 股票）+ 已知跨资产 ETF —— Massive-native，避开 akshare 老票/退市/格式(BRK_B vs BRK.B)问题。
  - 只留正常时段（盘前/盘后稀薄噪声大，v1 不要；以后可加）。
  - 重采样到 --freq（默认 60min，粗粒度先行）；要测频率阶梯就改 --freq 重跑。
  - 并发读（pandas C parser + gzip 都释放 GIL），过滤到 ~500 symbol 后量很小。

跑：PYTHONUTF8=1 .venv/Scripts/python.exe build_intraday.py --freq 60min --n-stocks 450
"""
import argparse
import concurrent.futures as cf
import glob
import sys
import time

import numpy as np
import pandas as pd

from download_us import CROSS

ETF_SYMS = sorted({s for lst in CROSS.values() for s in lst})   # ~79 跨资产 ETF
MIN_DIR = "data/massive/us_stocks_sip/minute_aggs_v1"
SESS_LO, SESS_HI = 570, 960     # 9:30 / 16:00，ET 当日分钟数


def all_files():
    return sorted(glob.glob(f"{MIN_DIR}/*/*/*.csv.gz"))


def read_filtered(path, symbols=None):
    """读一日文件 → 过滤 symbol + 正常时段，附 et_min/date。"""
    df = pd.read_csv(path)                         # .gz 自动解压
    if symbols is not None:
        df = df[df["ticker"].isin(symbols)]
    if df.empty:
        return df
    ts = pd.to_datetime(df["window_start"], unit="ns", utc=True).dt.tz_convert("America/New_York")
    mins = (ts.dt.hour * 60 + ts.dt.minute).to_numpy()
    keep = (mins >= SESS_LO) & (mins < SESS_HI)
    df = df[keep].copy()
    df["et_min"] = mins[keep]
    df["date"] = ts[keep].dt.strftime("%Y-%m-%d").to_numpy()
    return df


def resample_day(df, freq_min):
    """日内重采样：每 (ticker,date,slot) 聚成一根 OHLCV bar。"""
    nslot = (SESS_HI - SESS_LO - 1) // freq_min
    df = df.sort_values("window_start")
    df["slot"] = np.clip((df["et_min"] - SESS_LO) // freq_min, 0, nslot)
    return df.groupby(["ticker", "date", "slot"], sort=False).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum"),
        transactions=("transactions", "sum"), ts=("window_start", "first"),
    ).reset_index()


_G = {}


def _init(symbols, freq_min):
    _G["sym"] = symbols; _G["freq"] = freq_min


def _process(path):
    """worker：读一日 → 过滤 + 重采样 → 小 df（ProcessPool，真并行避开 GIL）。"""
    df = read_filtered(path, _G["sym"])
    if df is None or df.empty:
        return None
    return resample_day(df, _G["freq"])


def select_universe(files, n_stocks, recent=60, skip_top=0):
    """近 recent 天聚合美元成交额选股票 + 在场的跨资产 ETF。
    skip_top>0：跳过最大的 skip_top 只股票 → 取中盘带（测"越不流动越多 alpha"）。"""
    acc = {}
    for p in files[-recent:]:
        df = read_filtered(p)
        if df.empty:
            continue
        dv = (df["volume"] * df["close"]).groupby(df["ticker"]).sum()
        for tk, v in dv.items():
            acc[tk] = acc.get(tk, 0.0) + float(v)
    s = pd.Series(acc).sort_values(ascending=False)
    etfs = [t for t in ETF_SYMS if t in acc]
    stocks = [t for t in s.index if t not in set(ETF_SYMS)][skip_top: skip_top + n_stocks]
    return etfs + stocks


def main(a):
    files = all_files()
    print(f"{len(files)} 天文件 {files[0].split('/')[-1]} ~ {files[-1].split('/')[-1]}", flush=True)
    freq_min = int(a.freq.replace("min", ""))

    tag = f"_{a.tag}" if a.tag else ""
    uni_path = f"data/clean/universe_intraday{tag}.parquet"
    import os
    if os.path.exists(uni_path):
        syms = pd.read_parquet(uni_path)["symbol"].tolist()
        print(f"  复用已选 universe: {len(syms)}", flush=True)
    else:
        print(f"选 universe（近 60 天 $vol，跳过 top {a.skip_top}，取 {a.n_stocks} 股 + {len(ETF_SYMS)} ETF）...", flush=True)
        syms = select_universe(files, a.n_stocks, skip_top=a.skip_top)
        pd.DataFrame({"symbol": syms,
                      "is_etf": [s in set(ETF_SYMS) for s in syms]}).to_parquet(uni_path)
        print(f"  universe: {len(syms)}（{sum(s in set(ETF_SYMS) for s in syms)} ETF + 股票）", flush=True)
    symset = set(syms)

    t0 = time.time(); out = []
    with cf.ProcessPoolExecutor(max_workers=a.workers, initializer=_init, initargs=(symset, freq_min)) as ex:
        for i, r in enumerate(ex.map(_process, files, chunksize=4)):
            if r is not None:
                out.append(r)
            if (i + 1) % 300 == 0:
                print(f"  {i+1}/{len(files)}  {time.time()-t0:.0f}s", flush=True)
    panel = pd.concat(out, ignore_index=True).sort_values(["ticker", "ts"]).reset_index(drop=True)
    out_path = f"data/clean/intraday_{a.freq}{tag}.parquet"
    panel.to_parquet(out_path, index=False)
    print(f"完成: {len(panel):,} 行, {panel['ticker'].nunique()} ticker → {out_path}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--freq", default="60min", help="60min/30min/15min/5min/1min")
    ap.add_argument("--n-stocks", type=int, default=450, dest="n_stocks")
    ap.add_argument("--skip-top", type=int, default=0, dest="skip_top", help="跳过最大的 N 只(取中盘带)")
    ap.add_argument("--tag", default="", help="输出文件后缀(midcap 等)，隔离不同 universe 的产物")
    ap.add_argument("--workers", type=int, default=max(1, (__import__("os").cpu_count() or 4) - 2))
    sys.exit(main(ap.parse_args()))
