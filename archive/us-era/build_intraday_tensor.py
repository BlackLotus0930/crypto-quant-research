"""日内面板 → 因果归一张量（喂模型）。
- 拆股：从日内聚出的**日频**序列检测（复用 pipeline.clean 验证过的逻辑）→ 因子应用到所有日内 bar。
- 归一：per-asset 因果 scale-free，**慢窗按 bar 计**（≈252 交易日 × bars/天）。
- 存 time-of-day(slot) 供模型做日内位置编码；adj_close 供回测算收益。
跑：PYTHONUTF8=1 .venv/Scripts/python.exe build_intraday_tensor.py --freq 60min
"""
import argparse
import json

import numpy as np
import pandas as pd

from pipeline.clean import drop_bad_rows, segment_identities, adjust_splits

EPS = 1e-6
SLOW_DAYS = 252                 # 慢尺度 ≈ 1 交易年（换算成 bar）
MINP_FRAC = 0.1                 # 慢窗最少样本占比
# 日内切分（2016+ 窗口）：train 2016-2022 / val 2022-2024 / holdout 2024+
TRAIN_END, VAL_END = "2022-01-01", "2024-01-01"


def split_adjust(panel):
    """日内→日频聚合检测拆股，因子贴回日内 bar（价×F、量÷F）。"""
    p = panel.sort_values(["ticker", "date", "slot"])
    daily = (p.groupby(["ticker", "date"], sort=False)
               .agg(open=("open", "first"), close=("close", "last"),
                    high=("high", "max"), low=("low", "min"), volume=("volume", "sum"))
               .reset_index().rename(columns={"ticker": "symbol"}))
    cal = sorted(daily["date"].unique()); pos = {d: i for i, d in enumerate(cal)}
    daily = drop_bad_rows(daily)
    daily = segment_identities(daily, pos)            # 加 seg_id/pos，切退市重上市
    daily = adjust_splits(daily)                       # 加 split_factor F（段内累积）
    f = daily[["symbol", "date", "split_factor"]].rename(columns={"symbol": "ticker"})
    m = panel.merge(f, on=["ticker", "date"], how="left")
    F = m["split_factor"].fillna(1.0).to_numpy(float)
    for c in ("open", "high", "low", "close"):
        m[c] = m[c].to_numpy(float) * F
    m["volume"] = m["volume"].to_numpy(float) / np.where(F > 0, F, 1.0)
    return m.drop(columns=["split_factor"])


def normalize_causal(panel, slow_bars, bars_per_day):
    """因果归一 + **日内去季节化**（E7b）。
    价格 level：per-ticker（价位无日内 U 型）。
    收益 vol（scale_ret）与成交量：**per-(ticker,slot)**——把日内 U 型(开/收大、午盘小)从 scale 里除掉，
    不让慢 scaler 把"几点钟"当信号。per-slot 窗口按"天"算(每 slot 每天 1 根)。"""
    minp = max(20, int(slow_bars * MINP_FRAC))
    slow_days = max(40, slow_bars // bars_per_day)            # per-slot 窗口（天）
    minp_d = max(15, int(slow_days * MINP_FRAC))
    df = panel.sort_values(["ticker", "gt"]).reset_index(drop=True)
    df["_logc"] = np.log(df["close"]); df["_logv"] = np.log1p(df["volume"])
    df["_ret"] = df.groupby("ticker", sort=False)["_logc"].diff()

    g = df.groupby("ticker", sort=False)                      # 价格 level + 收益 scale：整条 bar 流
    ref = g["_logc"].rolling(slow_bars, min_periods=minp).mean().reset_index(level=0, drop=True)
    scale = g["_logc"].rolling(slow_bars, min_periods=minp).std().reset_index(level=0, drop=True)
    # scale_ret 用 per-ticker 池化：per-slot 会把隔夜(slot0进/slot6出)与日内 scale 错配→边界炸；
    # 且 ic_loss 是每行横截面相关、scale 不变，池化对训练无损，对回测 sizing 也更稳。
    s_ret = g["_ret"].rolling(slow_bars, min_periods=minp).std().reset_index(level=0, drop=True)

    gs = df.groupby(["ticker", "slot"], sort=False)           # **成交量** per-slot 去季节化（U 型最猛、无隔夜进出错配）
    ref_v = gs["_logv"].rolling(slow_days, min_periods=minp_d).mean().reset_index(level=[0, 1], drop=True)
    scale_v = gs["_logv"].rolling(slow_days, min_periods=minp_d).std().reset_index(level=[0, 1], drop=True)

    for ch in ("open", "high", "low", "close"):              # Series 按 index 自动对齐
        df["n_" + ch] = (np.log(df[ch]) - ref) / (scale + EPS)
    df["n_volume"] = (df["_logv"] - ref_v) / (scale_v + EPS)
    df["scale_ret"] = s_ret
    bad = (scale < EPS) | scale.isna() | (s_ret < EPS) | s_ret.isna()
    for ch in ("open", "high", "low", "close", "volume"):
        df.loc[bad, "n_" + ch] = np.nan
    return df


def main(a):
    freq_min = int(a.freq.replace("min", ""))
    bars_per_day = (960 - 570 - 1) // freq_min + 1
    slow_bars = SLOW_DAYS * bars_per_day
    print(f"freq={a.freq}  bars/天={bars_per_day}  慢窗={slow_bars} bar (~{SLOW_DAYS}日)", flush=True)

    tag = f"_{a.tag}" if a.tag else ""
    panel = pd.read_parquet(f"data/clean/intraday_{a.freq}{tag}.parquet")
    print(f"面板 {len(panel):,} 行, {panel['ticker'].nunique()} ticker", flush=True)

    panel = split_adjust(panel)
    print("拆股还原完成", flush=True)

    # 全局时间轴：按 (date, slot) 排序 → gt 索引
    steps = panel[["date", "slot"]].drop_duplicates().sort_values(["date", "slot"]).reset_index(drop=True)
    steps["gt"] = np.arange(len(steps))
    panel = panel.merge(steps, on=["date", "slot"], how="left")
    T = len(steps);
    syms = pd.read_parquet(f"data/clean/universe_intraday{tag}.parquet")["symbol"].tolist()
    present = set(panel["ticker"].unique())                # 只建一次，别在推导式里重建
    syms = [s for s in syms if s in present]
    nmap = {s: i for i, s in enumerate(syms)}; N = len(syms)
    print(f"T={T} 个 bar, N={N} 资产", flush=True)

    panel = normalize_causal(panel, slow_bars, bars_per_day)
    panel = panel[panel["ticker"].isin(nmap)]

    chans = ["n_open", "n_high", "n_low", "n_close", "n_volume"]
    C = 7 if a.tod else 5                                    # 5 OHLCV(+2 tod sin/cos)
    n_arr = np.zeros((T, N, C), np.float32)
    mask = np.zeros((T, N), bool)
    scale_ret = np.zeros((T, N), np.float32)
    adj_close = np.zeros((T, N), np.float32)
    rows = panel["gt"].to_numpy(); cols = panel["ticker"].map(nmap).to_numpy()
    valid = panel[chans].notna().all(axis=1).to_numpy() & panel["scale_ret"].notna().to_numpy()
    mask[rows, cols] = valid
    for k, ch in enumerate(chans):
        n_arr[rows, cols, k] = np.nan_to_num(panel[ch].to_numpy(np.float32))
    scale_ret[rows, cols] = np.nan_to_num(panel["scale_ret"].to_numpy(np.float32))
    adj_close[rows, cols] = np.nan_to_num(panel["close"].to_numpy(np.float32))

    tod = steps["slot"].to_numpy(np.int16)                 # time-of-day（slot）
    if a.tod:                                              # 绝对时钟坐标当输入通道，广播到所有 N
        ang = 2 * np.pi * tod.astype(np.float32) / bars_per_day
        n_arr[..., 5] = np.sin(ang)[:, None]
        n_arr[..., 6] = np.cos(ang)[:, None]
    n_arr[~mask] = 0.0                                     # 空槽整体清零（含 tod 通道）

    dates = steps["date"].to_numpy().astype(str)
    suffix = tag + ("_tod" if a.tod else "")
    np.savez_compressed(f"data/clean/intraday_tensor_{a.freq}{suffix}.npz",
                        n=n_arr, mask=mask, scale_ret=scale_ret, adj_close=adj_close,
                        tod=tod, dates=dates, slots=np.array(syms, dtype=str),
                        bars_per_day=bars_per_day)
    splits = {"train": ["2016-01-01", TRAIN_END], "val": [TRAIN_END, VAL_END], "holdout": [VAL_END, None]}
    json.dump(splits, open(f"data/clean/intraday_splits_{a.freq}{tag}.json", "w"), indent=2)
    print(f"完成 → intraday_tensor_{a.freq}{suffix}.npz  C={C}  mask 真实率={mask.mean():.3f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--freq", default="60min")
    ap.add_argument("--tag", default="", help="读/写带此后缀的产物(midcap 等)")
    ap.add_argument("--tod", action="store_true", help="加 sin/cos time-of-day 通道(C→7)；输出带 _tod 后缀")
    main(ap.parse_args())
