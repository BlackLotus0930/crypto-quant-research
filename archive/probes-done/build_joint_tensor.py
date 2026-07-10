"""联合多流张量：OHLCV + 订单流 + 衍生品(OI/多空比/基差) → 喂同一个模型做高阶联合检验。
思路(用户)：一个一个加只能算双双关系；thesis 是高阶联合(2^K)。所以把所有正交流一次性喂进去，
让模型自己在 14 个通道里找跨流的高阶结构。缺失衍生通道填 0(中性)，绝不掩掉 OHLCV bar。
通道：base5(OHLCV) + oflow3(taker失衡/笔数/均单) + deriv6(OI z/ΔOI z/top多空比/账户多空比/taker多空比/基差)。
因果：metrics 5min 取小时内最后一笔(bar 收盘已知)；basis open_time 直接对齐 panel ts(bar 开盘 ms)。
跑：PYTHONUTF8=1 .venv/Scripts/python.exe build_joint_tensor.py --freq 60 --workers 12
"""
import argparse
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from build_crypto_tensor import EPS, MINP_FRAC, SLOW_DAYS, TRAIN_END, VAL_END, normalize_causal

RAW = "data/raw/binance"
MCOLS = ["create_time", "sum_open_interest_value", "sum_toptrader_long_short_ratio",
         "count_long_short_ratio", "sum_taker_long_short_vol_ratio"]
KCOLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
         "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]


def _group_by_symbol(pat):
    """glob 所有 zip，按上级目录名(symbol)分组 → {sym: [files]}。"""
    out = {}
    for f in glob.glob(f"{RAW}/**/{pat}/**/*.zip", recursive=True):
        sym = os.path.basename(os.path.dirname(f))
        # premiumIndexKlines 路径多一层 interval(1h)，symbol 在再上一级
        if sym in ("1h", "5m", "1m"):
            sym = os.path.basename(os.path.dirname(os.path.dirname(f)))
        out.setdefault(sym, []).append(f)
    return out


def parse_metrics(args):
    sym, files = args
    out = []
    for f in files:
        try:
            out.append(pd.read_csv(f, usecols=MCOLS))
        except Exception:
            continue
    if not out:
        return None
    d = pd.concat(out, ignore_index=True)
    d["t"] = pd.to_datetime(d["create_time"], utc=True, errors="coerce")
    d = d.dropna(subset=["t"])
    if d.empty:
        return None
    d["ts"] = d["t"].dt.floor("h").values.astype("datetime64[ms]").astype("int64")  # 强制毫秒(防 pandas2.x 非ns分辨率)
    d = d.sort_values("t").groupby("ts", as_index=False).last()        # 小时内最后一笔(因果)
    d["symbol"] = sym
    return d[["ts", "symbol", "sum_open_interest_value", "sum_toptrader_long_short_ratio",
              "count_long_short_ratio", "sum_taker_long_short_vol_ratio"]]


def parse_basis(args):
    sym, files = args
    out = []
    for f in files:
        try:
            d = pd.read_csv(f)
            if "open_time" not in d.columns:                          # 老月文件无表头
                d = pd.read_csv(f, header=None, names=KCOLS)
        except Exception:
            continue
        if "open_time" in d.columns and "close" in d.columns:
            out.append(d[["open_time", "close"]])
    if not out:
        return None
    d = pd.concat(out, ignore_index=True)
    d["ts"] = d["open_time"].astype("int64")
    d = d.groupby("ts", as_index=False)["close"].last().rename(columns={"close": "basis"})
    d["symbol"] = sym
    return d


def _parallel(fn, groups, workers, tag):
    res = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(fn, groups.items())):
            if r is not None:
                res.append(r)
            if (i + 1) % 25 == 0:
                print(f"  {tag} {i+1}/{len(groups)} symbols", flush=True)
    df = pd.concat(res, ignore_index=True) if res else pd.DataFrame()
    print(f"  {tag} 完成：{len(df):,} 行(小时级)", flush=True)
    return df


def _causal_z(df, col, slow_bars, minp):
    """per-coin 因果滚动 z(慢窗)。返回 z 序列(与 df 行对齐)。"""
    g = df.groupby("symbol", sort=False)[col]
    m = g.rolling(slow_bars, min_periods=minp).mean().reset_index(level=0, drop=True)
    s = g.rolling(slow_bars, min_periods=minp).std().reset_index(level=0, drop=True)
    return (df[col] - m) / (s + EPS)


def main(a):
    bars_per_day = 1440 // a.freq
    slow_bars = SLOW_DAYS * bars_per_day
    minp = max(20, int(slow_bars * MINP_FRAC))
    print(f"freq={a.freq}min  bars/天={bars_per_day}  慢窗={slow_bars}  workers={a.workers}", flush=True)

    mcache = f"data/clean/metrics_{a.freq}min.parquet"
    bcache = f"data/clean/basis_{a.freq}min.parquet"
    if a.use_cache and os.path.exists(mcache) and os.path.exists(bcache):
        print(f"用缓存：{mcache} / {bcache}", flush=True)
        met = pd.read_parquet(mcache); bas = pd.read_parquet(bcache)
    else:
        print("解析 metrics(OI/多空比) ...", flush=True)
        met = _parallel(parse_metrics, _group_by_symbol("metrics"), a.workers, "metrics")
        print("解析 premiumIndexKlines(基差) ...", flush=True)
        bas = _parallel(parse_basis, _group_by_symbol("premiumIndexKlines"), a.workers, "basis")
        met.to_parquet(mcache); bas.to_parquet(bcache)
        print(f"已缓存 → {mcache} / {bcache}(下次 --use_cache 跳过解析)", flush=True)

    panel = pd.read_parquet(f"data/clean/crypto_{a.freq}min.parquet")
    print(f"面板 {len(panel):,} 行, {panel['symbol'].nunique()} 币 → 因果归一 OHLCV+订单流", flush=True)
    df = normalize_causal(panel, slow_bars)                            # 有 n_open..n_volume,n_tbr,n_count,n_avgsz,scale_ret
    df["ts"] = df["ts"].astype("int64")                                # panel ts float→int64,保证与 met/bas 整数键匹配

    # 合并衍生流(symbol+ts) → 计算衍生通道
    df = df.merge(met, on=["symbol", "ts"], how="left").merge(bas, on=["symbol", "ts"], how="left")
    df = df.sort_values(["symbol", "ts"]).reset_index(drop=True)
    df["n_avail"] = df["sum_open_interest_value"].notna().astype(np.float32)   # 衍生数据是否真有(防"填0=缺失"被当信号)
    df["_loi"] = np.log1p(df["sum_open_interest_value"].clip(lower=0))
    df["n_oi"] = _causal_z(df, "_loi", slow_bars, minp)
    df["_doi"] = df.groupby("symbol", sort=False)["_loi"].diff()
    sd = df.groupby("symbol", sort=False)["_doi"].rolling(slow_bars, min_periods=minp).std().reset_index(level=0, drop=True)
    df["n_oichg"] = df["_doi"] / (sd + EPS)
    df["n_lsr_top"] = np.log(df["sum_toptrader_long_short_ratio"].clip(lower=1e-3, upper=1e3))
    df["n_lsr_acct"] = np.log(df["count_long_short_ratio"].clip(lower=1e-3, upper=1e3))
    df["n_lsr_taker"] = np.log(df["sum_taker_long_short_vol_ratio"].clip(lower=1e-3, upper=1e3))
    df["n_basis"] = (df["basis"].astype(np.float64) * 100).clip(-10, 10)

    base5 = ["n_open", "n_high", "n_low", "n_close", "n_volume"]
    oflow = ["n_tbr", "n_count", "n_avgsz"]
    deriv = ["n_oi", "n_oichg", "n_lsr_top", "n_lsr_acct", "n_lsr_taker", "n_basis"]
    flag = ["n_avail"]                                   # 显式可用性标志 → 填0安全
    chans = base5 + oflow + deriv + flag
    # 有效性只看 OHLCV(base5+scale_ret)；衍生缺失填 0(中性)，不掩 bar；n_avail 告诉模型哪是填的
    valid = df[base5].notna().all(axis=1).to_numpy() & df["scale_ret"].notna().to_numpy()
    for c in oflow + deriv:
        df[c] = df[c].fillna(0.0)
    h0 = int(pd.Timestamp(VAL_END, tz="UTC").value // 10**6)
    hm = (df["ts"].to_numpy() >= h0) & valid
    print(f"holdout(>= {VAL_END}) 衍生可用率 = {df.loc[hm, 'n_avail'].mean():.3f}  "
          f"(全样本 {df.loc[valid, 'n_avail'].mean():.3f})", flush=True)

    steps = pd.DataFrame({"ts": np.sort(df["ts"].unique())})
    steps["gt"] = np.arange(len(steps))
    df = df.merge(steps, on="ts", how="left")
    T = len(steps)
    syms = sorted(df["symbol"].unique())
    nmap = {s: i for i, s in enumerate(syms)}
    N = len(syms)
    C_data = len(chans)
    C = C_data + 4
    print(f"T={T} bar, N={N} 币, C={C}(数据{C_data}+tod4)", flush=True)

    n_arr = np.zeros((T, N, C), np.float32)
    mask = np.zeros((T, N), bool)
    scale_ret = np.zeros((T, N), np.float32)
    adj_close = np.zeros((T, N), np.float32)
    rows = df["gt"].to_numpy()
    cols = df["symbol"].map(nmap).to_numpy()
    mask[rows, cols] = valid
    for k, ch in enumerate(chans):
        n_arr[rows, cols, k] = np.nan_to_num(df[ch].to_numpy(np.float32))
    scale_ret[rows, cols] = np.nan_to_num(df["scale_ret"].to_numpy(np.float32))
    adj_close[rows, cols] = np.nan_to_num(df["close"].to_numpy(np.float32))

    dt = pd.to_datetime(steps["ts"].to_numpy(), unit="ms", utc=True)
    hour = dt.hour.to_numpy(); dow = dt.dayofweek.to_numpy()
    ah = 2 * np.pi * hour / 24; ad = 2 * np.pi * dow / 7
    n_arr[..., C_data + 0] = np.sin(ah)[:, None]
    n_arr[..., C_data + 1] = np.cos(ah)[:, None]
    n_arr[..., C_data + 2] = np.sin(ad)[:, None]
    n_arr[..., C_data + 3] = np.cos(ad)[:, None]
    n_arr[~mask] = 0.0

    dates = dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy().astype(str)
    tod = (hour % 256).astype(np.int16)
    np.savez_compressed(f"data/clean/crypto_tensor_{a.freq}min_joint.npz",
                        n=n_arr, mask=mask, scale_ret=scale_ret, adj_close=adj_close,
                        tod=tod, dates=dates, slots=np.array(syms, dtype=str),
                        bars_per_day=bars_per_day, ann=365 * bars_per_day)
    splits = {"train": ["2018-01-01", TRAIN_END], "val": [TRAIN_END, VAL_END], "holdout": [VAL_END, None]}
    json.dump(splits, open(f"data/clean/crypto_splits_{a.freq}min.json", "w"), indent=2)
    print(f"完成 → crypto_tensor_{a.freq}min_joint.npz  C={C}  mask 真实率={mask.mean():.3f}  ann={365*bars_per_day}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--freq", type=int, default=60)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--use_cache", action="store_true", help="有 metrics/basis parquet 缓存就跳过解析")
    main(ap.parse_args())
