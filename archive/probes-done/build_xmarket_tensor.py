"""跨市场联合张量：加密(198, 24/7) + US 跨资产 ETF(~12, 美股时段) 拼在资产轴。
US ETF 只作 **context**(attention 看得到、但 is_target=False 不进 loss/IC/回测)，让模型自己找"宏观状态 × 哪些币"的交互。
加密部分**直接复用 crypto_tensor_60min.npz**(与基线 bit 一致 → A/B 干净)；只新建 ETF 部分。
US ETF(ts=ns UTC, 7bar/日 @:30) 因果对齐到加密 :00 网格(bar 收盘≤当前才可见)，前向填(盘外=最后已知 US 状态)。
跑：PYTHONUTF8=1 .venv/Scripts/python.exe build_xmarket_tensor.py --freq 60
"""
import argparse
import json

import numpy as np
import pandas as pd

EPS = 1e-6
SLOW_DAYS = 60
ETFS = ["SPY", "QQQ", "DIA", "IWM", "VIXY", "VXX", "UUP", "GLD", "USO", "TLT", "IEF", "HYG"]


def etf_normalize(df, slow_bars):
    """单 ETF session 序列 → per-asset 因果 scale-free OHLCV(和加密同公式，无订单流)。"""
    minp = max(20, int(slow_bars * 0.1))
    df = df.sort_values("ms").reset_index(drop=True)
    df["_logc"] = np.log(df["close"].clip(lower=EPS)); df["_logv"] = np.log1p(df["volume"].clip(lower=0))
    df["_ret"] = df["_logc"].diff()
    r = df["_logc"].rolling(slow_bars, min_periods=minp)
    ref, scale = r.mean(), r.std(); s_ret = df["_ret"].rolling(slow_bars, min_periods=minp).std()
    rv = df["_logv"].rolling(slow_bars, min_periods=minp); ref_v, scale_v = rv.mean(), rv.std()
    for ch in ("open", "high", "low", "close"):
        df["n_" + ch] = (np.log(df[ch].clip(lower=EPS)) - ref) / (scale + EPS)
    df["n_volume"] = (df["_logv"] - ref_v) / (scale_v + EPS)
    df["scale_ret"] = s_ret
    bad = (scale < EPS) | scale.isna() | (s_ret < EPS) | s_ret.isna()
    for ch in ("open", "high", "low", "close", "volume"):
        df.loc[bad, "n_" + ch] = np.nan
    return df


def main(a):
    bpd = 1440 // a.freq
    slow_bars = SLOW_DAYS * bpd
    base = np.load(f"data/clean/crypto_tensor_{a.freq}min.npz", allow_pickle=False)   # 加密基线(C=9 basic)
    cn, cm = base["n"], base["mask"]; csr, cac = base["scale_ret"], base["adj_close"]
    T, Nc, C = cn.shape
    cslots = base["slots"].astype(str)
    grid_ts = np.sort(pd.read_parquet(f"data/clean/crypto_{a.freq}min.parquet")["ts"].unique()).astype(np.int64)
    assert len(grid_ts) == T, f"网格 {len(grid_ts)} != 张量 T {T}"
    print(f"加密基线 T={T} Nc={Nc} C={C}；对齐 {len(ETFS)} 个 US ETF", flush=True)

    intr = pd.read_parquet(f"data/clean/intraday_{a.freq}min.parquet")
    intr = intr[intr["ticker"].isin(ETFS)].copy()
    intr["ms"] = (intr["ts"] // 1_000_000).astype(np.int64)
    bar_ms = a.freq * 60_000
    gdf = pd.DataFrame({"ts": grid_ts})

    etf_n = np.zeros((T, len(ETFS), C), np.float32)
    etf_mask = np.zeros((T, len(ETFS)), bool)
    etf_sr = np.zeros((T, len(ETFS)), np.float32)            # scale_ret=0 = 哨兵：labels/training/回测都按 s>0 自动排除(只作 context)
    etf_ac = np.ones((T, len(ETFS)), np.float32)
    chans = ["n_open", "n_high", "n_low", "n_close", "n_volume"]
    for j, tk in enumerate(ETFS):
        g = intr[intr["ticker"] == tk]
        if len(g) < slow_bars:
            print(f"  ⚠️ {tk} 数据太少({len(g)})，跳过(全 mask)", flush=True); continue
        g = etf_normalize(g, slow_bars)
        g["close_ms"] = g["ms"] + bar_ms                          # bar 收盘时刻(因果：收盘≤当前网格点才可见)
        cols = chans + ["scale_ret", "close"]
        m = pd.merge_asof(gdf, g[["close_ms"] + cols].rename(columns={"close_ms": "ts"}),
                          on="ts", direction="backward")           # 前向填：每个网格点取最近一根已收盘 ETF bar
        valid = m[chans].notna().all(axis=1).to_numpy() & m["scale_ret"].notna().to_numpy()
        etf_mask[:, j] = valid
        for k, ch in enumerate(chans):
            etf_n[valid, j, k] = m[ch].to_numpy(np.float32)[valid]
        etf_n[:, j, 5:] = cn[:, 0, 5:]                            # tod 通道(时间特征,全资产相同)→ 复制加密的
        etf_ac[valid, j] = np.nan_to_num(m["close"].to_numpy(np.float32)[valid], nan=1.0)
        # etf scale_ret 保持 0(哨兵)——不进 loss/IC/回测
        print(f"  {tk}: 覆盖率={valid.mean():.2f}", flush=True)

    n = np.concatenate([cn, etf_n], axis=1)
    mask = np.concatenate([cm, etf_mask], axis=1)
    scale_ret = np.concatenate([csr, etf_sr], axis=1)
    adj_close = np.concatenate([cac, etf_ac], axis=1)
    is_target = np.concatenate([np.ones(Nc, bool), np.zeros(len(ETFS), bool)])    # 只评分加密
    slots = np.concatenate([cslots, np.array(ETFS, dtype=str)])
    out = f"data/clean/crypto_tensor_{a.freq}min_xmarket.npz"
    np.savez_compressed(out, n=n, mask=mask, scale_ret=scale_ret, adj_close=adj_close,
                        is_target=is_target, tod=base["tod"], dates=base["dates"], slots=slots,
                        bars_per_day=int(base["bars_per_day"]), ann=int(base["ann"]))
    json.dump(json.load(open(f"data/clean/crypto_splits_{a.freq}min.json")),
              open(f"data/clean/crypto_splits_{a.freq}min_xmarket.json", "w"))      # 同切分
    print(f"完成 → {out}  N={n.shape[1]}(加密{Nc}+ETF{len(ETFS)}) C={C}  加密mask={cm.mean():.3f} ETFmask={etf_mask.mean():.3f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--freq", type=int, default=60)
    main(ap.parse_args())
