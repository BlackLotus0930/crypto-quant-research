"""加密面板 → 因果归一张量（喂模型，和股票张量同构）。
24/7：无拆股、无 session、无隔夜。tod = hour-of-day + day-of-week（各 sin/cos）。
年化 ann = 365×bars/天（加密 365 天/年，存进张量供 backtest/ic 用）。
跑：PYTHONUTF8=1 .venv/Scripts/python.exe build_crypto_tensor.py --freq 60
"""
import argparse
import json

import numpy as np
import pandas as pd

EPS = 1e-6
SLOW_DAYS = 60                  # 慢窗（加密 regime 比股票快，60 天）
MINP_FRAC = 0.1
TRAIN_END, VAL_END = "2023-01-01", "2024-01-01"


def normalize_causal(panel, slow_bars):
    """per-coin 因果 scale-free（慢窗=slow_bars 根 bar）。无拆股、无 session。"""
    minp = max(20, int(slow_bars * MINP_FRAC))
    df = panel.sort_values(["symbol", "ts"]).reset_index(drop=True)
    df["_logc"] = np.log(df["close"]); df["_logv"] = np.log1p(df["volume"])
    df["_logn"] = np.log1p(df["count"])                                       # 成交笔数(订单流活动度)
    df["_logsz"] = np.log((df["qv"] / df["count"].clip(lower=1)).clip(lower=EPS))  # 均单美元额(大单 proxy)
    df["_ret"] = df.groupby("symbol", sort=False)["_logc"].diff()
    g = df.groupby("symbol", sort=False)

    def rmean(c): return g[c].rolling(slow_bars, min_periods=minp).mean().reset_index(level=0, drop=True)
    def rstd(c): return g[c].rolling(slow_bars, min_periods=minp).std().reset_index(level=0, drop=True)
    ref, scale, s_ret = rmean("_logc"), rstd("_logc"), rstd("_ret")
    ref_v, scale_v = rmean("_logv"), rstd("_logv")
    ref_n, scale_n = rmean("_logn"), rstd("_logn")
    ref_sz, scale_sz = rmean("_logsz"), rstd("_logsz")
    for ch in ("open", "high", "low", "close"):
        df["n_" + ch] = (np.log(df[ch]) - ref) / (scale + EPS)
    df["n_volume"] = (df["_logv"] - ref_v) / (scale_v + EPS)
    df["n_count"] = (df["_logn"] - ref_n) / (scale_n + EPS)               # 笔数(订单流)
    df["n_avgsz"] = (df["_logsz"] - ref_sz) / (scale_sz + EPS)            # 均单额(大单/碎单)
    df["n_tbr"] = (df["tbv"] / df["volume"].clip(lower=EPS) - 0.5) * 2     # taker 买卖失衡，居中到 [-1,1]
    df["scale_ret"] = s_ret
    bad = (scale < EPS) | scale.isna() | (s_ret < EPS) | s_ret.isna()
    for ch in ("open", "high", "low", "close", "volume"):
        df.loc[bad, "n_" + ch] = np.nan
    bad_of = (scale_n < EPS) | scale_n.isna() | (scale_sz < EPS) | scale_sz.isna()
    df.loc[bad_of, ["n_count", "n_avgsz"]] = np.nan                        # 订单流自身慢尺度坏→该通道置 nan(会被 mask)
    return df


def main(a):
    bars_per_day = 1440 // a.freq                                  # 60min→24
    slow_bars = SLOW_DAYS * bars_per_day
    rich = a.channels == "rich"; oflow = a.channels == "oflow"
    print(f"freq={a.freq}min  bars/天={bars_per_day}  慢窗={slow_bars} bar  通道={a.channels}", flush=True)

    panel = pd.read_parquet(f"data/clean/crypto_{a.freq}min.parquet")
    print(f"面板 {len(panel):,} 行, {panel['symbol'].nunique()} 币", flush=True)
    panel = normalize_causal(panel, slow_bars)

    steps = pd.DataFrame({"ts": np.sort(panel["ts"].unique())})
    steps["gt"] = np.arange(len(steps))
    panel = panel.merge(steps, on="ts", how="left")
    T = len(steps)
    syms = sorted(panel["symbol"].unique())
    nmap = {s: i for i, s in enumerate(syms)}; N = len(syms)
    print(f"T={T} bar, N={N} 币", flush=True)

    base5 = ["n_open", "n_high", "n_low", "n_close", "n_volume"]
    chans = base5 + (["n_tbr", "n_count", "n_avgsz"] if oflow else ["n_tbr"] if rich else [])  # oflow=+订单流正交流
    C_data = len(chans)
    C = C_data + 4                                                 # +hour sin/cos +dow sin/cos
    n_arr = np.zeros((T, N, C), np.float32)
    mask = np.zeros((T, N), bool)
    scale_ret = np.zeros((T, N), np.float32)
    adj_close = np.zeros((T, N), np.float32)
    rows = panel["gt"].to_numpy(); cols = panel["symbol"].map(nmap).to_numpy()
    valid = panel[chans].notna().all(axis=1).to_numpy() & panel["scale_ret"].notna().to_numpy()
    mask[rows, cols] = valid
    for k, ch in enumerate(chans):
        n_arr[rows, cols, k] = np.nan_to_num(panel[ch].to_numpy(np.float32))
    scale_ret[rows, cols] = np.nan_to_num(panel["scale_ret"].to_numpy(np.float32))
    adj_close[rows, cols] = np.nan_to_num(panel["close"].to_numpy(np.float32))

    dt = pd.to_datetime(steps["ts"].to_numpy(), unit="ms", utc=True)
    hour = dt.hour.to_numpy(); dow = dt.dayofweek.to_numpy()       # 0-23 / 0-6(Mon=0)
    ah = 2 * np.pi * hour / 24; ad = 2 * np.pi * dow / 7
    n_arr[..., C_data + 0] = np.sin(ah)[:, None]
    n_arr[..., C_data + 1] = np.cos(ah)[:, None]
    n_arr[..., C_data + 2] = np.sin(ad)[:, None]
    n_arr[..., C_data + 3] = np.cos(ad)[:, None]
    n_arr[~mask] = 0.0

    dates = dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy().astype(str)
    tod = (hour % 256).astype(np.int16)
    suffix = "_oflow" if oflow else "_rich" if rich else ""
    np.savez_compressed(f"data/clean/crypto_tensor_{a.freq}min{suffix}.npz",
                        n=n_arr, mask=mask, scale_ret=scale_ret, adj_close=adj_close,
                        tod=tod, dates=dates, slots=np.array(syms, dtype=str),
                        bars_per_day=bars_per_day, ann=365 * bars_per_day)   # 加密 365 天/年
    splits = {"train": ["2018-01-01", TRAIN_END], "val": [TRAIN_END, VAL_END], "holdout": [VAL_END, None]}
    json.dump(splits, open(f"data/clean/crypto_splits_{a.freq}min.json", "w"), indent=2)
    print(f"完成 → crypto_tensor_{a.freq}min{suffix}.npz  C={C}  mask 真实率={mask.mean():.3f}  ann={365*bars_per_day}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--freq", type=int, default=60)
    ap.add_argument("--channels", default="basic", choices=["basic", "rich", "oflow"],
                    help="rich=+taker失衡; oflow=+taker失衡+笔数+均单(订单流正交流)")
    main(ap.parse_args())
