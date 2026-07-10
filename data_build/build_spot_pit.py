"""把现货 1h klines 对齐到 pit 张量网格 → spot_pit.npz(spot close, T×N)。
真 cash-and-carry 用真实现货价(替掉 premiumIndex 近似):hedged P&L = spot_ret − perp_ret + funding。
跑：PYTHONUTF8=1 .venv/Scripts/python.exe build_spot_pit.py
"""
import argparse
import concurrent.futures as cf
import glob
import io
import os
import zipfile

import numpy as np
import pandas as pd

SGLOB = "data/raw/binance/data/spot/monthly/klines/*/1h/*.zip"


def read_zip(path):
    """现货 1h kline zip → DataFrame[symbol, ts(ms,int64), close]。处理可选表头 + µs时间戳。"""
    sym = path.replace("\\", "/").split("/klines/")[1].split("/")[0]
    try:
        z = zipfile.ZipFile(path); raw = z.read(z.namelist()[0])
    except Exception:
        return None
    df = pd.read_csv(io.BytesIO(raw), header=None)
    df = df[pd.to_numeric(df[0], errors="coerce").notna()]
    if df.empty:
        return None
    ts = pd.to_numeric(df[0]).astype(np.int64)
    ts = np.where(ts > 10**13, ts // 1000, ts)                        # µs→ms 归一
    close = pd.to_numeric(df[4], errors="coerce")
    out = pd.DataFrame({"symbol": sym, "ts": ts, "spot": close.to_numpy()}).dropna()
    return out if not out.empty else None


def main(a):
    z = np.load(a.tensor, allow_pickle=True)
    dates = z["dates"].astype(str)
    slots = list(z["slots"].astype(str))
    T, N = len(dates), len(slots)
    nmap = {s: i for i, s in enumerate(slots)}
    grid_ts = np.sort(pd.read_parquet(a.panel, columns=["ts"])["ts"].unique()).astype(np.int64)
    assert len(grid_ts) == T, f"面板 ts 数 {len(grid_ts)} != 张量 T {T}"
    print(f"pit 张量 T={T} N={N}; 网格 {pd.to_datetime(grid_ts[0],unit='ms')} → {pd.to_datetime(grid_ts[-1],unit='ms')}", flush=True)

    files = [f for f in glob.glob(SGLOB)
             if f.replace("\\", "/").split("/klines/")[1].split("/")[0] in nmap]
    workers = max(1, (os.cpu_count() or 4) - 2)
    print(f"{len(files)} 个现货 1h zip(在 pit 宇宙内),线程 {workers}", flush=True)
    rows = []
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(read_zip, files):
            if r is not None:
                rows.append(r)
    sp = pd.concat(rows, ignore_index=True)
    sp["ts"] = sp["ts"].astype(np.int64)
    print(f"现货 原始 {len(sp):,} 行, {sp['symbol'].nunique()} 币", flush=True)

    spot = np.full((T, N), np.nan, np.float32)
    gdf = pd.DataFrame({"ts": grid_ts})
    have = 0
    for sym, g in sp.groupby("symbol", sort=False):
        j = nmap.get(sym)
        if j is None:
            continue
        g = g.sort_values("ts").drop_duplicates("ts", keep="last")
        m = pd.merge_asof(gdf, g[["ts", "spot"]], on="ts", direction="backward")
        spot[:, j] = m["spot"].to_numpy(np.float32)
        have += 1

    cov = np.isfinite(spot).any(0).mean()
    print(f"现货覆盖 {have}/{N} 币 ({cov:.1%})（永续有、现货无的=perp-only,无法做 cash-and-carry,后续自动剔）", flush=True)
    outp = "data/clean/spot_pit.npz"
    np.savez_compressed(outp, spot=spot, dates=dates, slots=np.array(slots, dtype=str))
    print(f"完成 → {outp}  shape={spot.shape}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensor", default="data/clean/crypto_tensor_60min_pit.npz")
    ap.add_argument("--panel", default="data/clean/crypto_60min_pit.parquet")
    main(ap.parse_args())
