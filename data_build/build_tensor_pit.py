"""point-in-time 张量:含死币 + 每根bar只保留"当时"最流动的 top-N(修幸存者偏差,又不交易买不到的垃圾)。
通道同基线(9=OHLCV5+tod4),好和原 base WF 直接对比。死币在它流动时进入、退市/缩量后退出。
跑：PYTHONUTF8=1 .venv/Scripts/python.exe build_tensor_pit.py --top_liquid 200
"""
import argparse
import json

import numpy as np
import pandas as pd

from build_crypto_tensor import EPS, SLOW_DAYS, TRAIN_END, VAL_END, normalize_causal


def main(a):
    bars_per_day = 1440 // a.freq
    slow_bars = SLOW_DAYS * bars_per_day
    panel = pd.read_parquet(a.parquet)
    print(f"面板 {len(panel):,} 行, {panel['symbol'].nunique()} 币(含死币) → 因果归一", flush=True)
    df = normalize_causal(panel, slow_bars)
    # 因果滚动美元成交额(7天)→ 流动性排序用
    win = 7 * bars_per_day
    df["tdv"] = df.groupby("symbol", sort=False)["qv"].transform(
        lambda s: s.rolling(win, min_periods=bars_per_day).sum())

    steps = pd.DataFrame({"ts": np.sort(df["ts"].unique())})
    steps["gt"] = np.arange(len(steps))
    df = df.merge(steps, on="ts", how="left")
    T = len(steps)
    syms = sorted(df["symbol"].unique())
    nmap = {s: i for i, s in enumerate(syms)}
    N = len(syms)
    print(f"T={T} bar, N={N} 币", flush=True)

    base5 = ["n_open", "n_high", "n_low", "n_close", "n_volume"]
    C_data = len(base5)
    C = C_data + 4
    n_arr = np.zeros((T, N, C), np.float32)
    mask = np.zeros((T, N), bool)
    scale_ret = np.zeros((T, N), np.float32)
    adj_close = np.zeros((T, N), np.float32)
    tdv = np.zeros((T, N), np.float64)
    rows = df["gt"].to_numpy(); cols = df["symbol"].map(nmap).to_numpy()
    valid = df[base5].notna().all(axis=1).to_numpy() & df["scale_ret"].notna().to_numpy()
    mask[rows, cols] = valid
    for k, ch in enumerate(base5):
        n_arr[rows, cols, k] = np.nan_to_num(df[ch].to_numpy(np.float32))
    scale_ret[rows, cols] = np.nan_to_num(df["scale_ret"].to_numpy(np.float32))
    adj_close[rows, cols] = np.nan_to_num(df["close"].to_numpy(np.float32))
    tdv[rows, cols] = np.nan_to_num(df["tdv"].to_numpy(np.float64))

    # point-in-time 流动性过滤:每根 bar 只留 tdv 排名前 top_liquid 的(且 valid)
    if a.top_liquid > 0:
        liq = np.zeros((T, N), bool)
        for t in range(T):
            v = np.where(mask[t] & (tdv[t] > 0))[0]
            if len(v) == 0:
                continue
            keep = v[np.argsort(tdv[t, v])[::-1][:a.top_liquid]]
            liq[t, keep] = True
        before = mask.mean()
        mask = mask & liq
        print(f"流动性过滤 top{a.top_liquid}: mask 真实率 {before:.3f} → {mask.mean():.3f}；"
              f"每bar中位可交易={np.median(mask.sum(1)):.0f}", flush=True)

    # 时长过滤:只留在 top-N 里累计待够 min_bars 的币(对称门槛,丢短命pump-dump,不论死活)→ 控 N²显存
    if a.min_bars > 0:
        keep = mask.sum(0) >= a.min_bars
        n_arr = n_arr[:, keep]; mask = mask[:, keep]
        scale_ret = scale_ret[:, keep]; adj_close = adj_close[:, keep]
        syms = list(np.array(syms)[keep]); N = int(keep.sum())
        print(f"时长过滤(≥{a.min_bars}bar): N {len(keep)} → {N}；每bar中位可交易={np.median(mask.sum(1)):.0f}", flush=True)

    dt = pd.to_datetime(steps["ts"].to_numpy(), unit="ms", utc=True)
    hour = dt.hour.to_numpy(); dow = dt.dayofweek.to_numpy()
    ah = 2 * np.pi * hour / 24; ad = 2 * np.pi * dow / 7
    n_arr[..., C_data + 0] = np.sin(ah)[:, None]; n_arr[..., C_data + 1] = np.cos(ah)[:, None]
    n_arr[..., C_data + 2] = np.sin(ad)[:, None]; n_arr[..., C_data + 3] = np.cos(ad)[:, None]
    n_arr[~mask] = 0.0

    dates = dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy().astype(str)
    tod = (hour % 256).astype(np.int16)
    np.savez_compressed(f"data/clean/crypto_tensor_{a.freq}min_pit.npz",
                        n=n_arr, mask=mask, scale_ret=scale_ret, adj_close=adj_close,
                        tod=tod, dates=dates, slots=np.array(syms, dtype=str),
                        bars_per_day=bars_per_day, ann=365 * bars_per_day)
    splits = {"train": ["2018-01-01", TRAIN_END], "val": [TRAIN_END, VAL_END], "holdout": [VAL_END, None]}
    json.dump(splits, open(f"data/clean/crypto_splits_{a.freq}min.json", "w"), indent=2)
    print(f"完成 → crypto_tensor_{a.freq}min_pit.npz  C={C}  N={N}  mask率={mask.mean():.3f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--freq", type=int, default=60)
    ap.add_argument("--parquet", default="data/clean/crypto_60min_pit.parquet")
    ap.add_argument("--top_liquid", type=int, default=200, help="每根bar保留最流动的N个币(point-in-time)")
    ap.add_argument("--min_bars", type=int, default=2160, help="累计可交易≥此bar数才入universe(30天=720;90天=2160)控显存")
    main(ap.parse_args())
