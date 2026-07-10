"""把全宇宙(含死币)premiumIndex(永续-现货基差)对齐到 pit 张量网格 → basis_pit.npz(T×N)。
basis = 永续溢价(premium index close)。cash-and-carry 价格腿 P&L ≈ −Δbasis(多现货空永续,价格被对冲)。
跑：PYTHONUTF8=1 .venv/Scripts/python.exe build_basis_pit.py
"""
import argparse
import concurrent.futures as cf
import glob
import io
import os
import zipfile

import numpy as np
import pandas as pd

PGLOB = "data/raw/binance/data/futures/um/monthly/premiumIndexKlines/*/1h/*.zip"


def read_zip(path):
    """premiumIndex 1h kline zip → DataFrame[symbol, ts(ms,int64), basis=close]。处理可选表头 + µs时间戳。"""
    sym = path.replace("\\", "/").split("/premiumIndexKlines/")[1].split("/")[0]
    try:
        z = zipfile.ZipFile(path); raw = z.read(z.namelist()[0])
    except Exception:
        return None
    df = pd.read_csv(io.BytesIO(raw), header=None)
    df = df[pd.to_numeric(df[0], errors="coerce").notna()]            # 丢可能的文字表头行
    if df.empty:
        return None
    ts = pd.to_numeric(df[0]).astype(np.int64)
    ts = np.where(ts > 10**13, ts // 1000, ts)                        # µs→ms 归一(部分新文件)
    basis = pd.to_numeric(df[4], errors="coerce")                     # premium index 的 close
    out = pd.DataFrame({"symbol": sym, "ts": ts, "basis": basis.to_numpy()}).dropna()
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

    files = [f for f in glob.glob(PGLOB)
             if f.replace("\\", "/").split("/premiumIndexKlines/")[1].split("/")[0] in nmap]
    workers = max(1, (os.cpu_count() or 4) - 2)
    print(f"{len(files)} 个 premiumIndex zip(在 pit 宇宙内),线程 {workers}", flush=True)
    rows = []
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(read_zip, files):
            if r is not None:
                rows.append(r)
    bas = pd.concat(rows, ignore_index=True)
    bas["ts"] = bas["ts"].astype(np.int64)
    print(f"basis 原始 {len(bas):,} 行, {bas['symbol'].nunique()} 币", flush=True)

    basis = np.full((T, N), np.nan, np.float32)
    gdf = pd.DataFrame({"ts": grid_ts})
    have = 0
    for sym, g in bas.groupby("symbol", sort=False):
        j = nmap.get(sym)
        if j is None:
            continue
        g = g.sort_values("ts").drop_duplicates("ts", keep="last")
        m = pd.merge_asof(gdf, g[["ts", "basis"]], on="ts", direction="backward")    # ≤t 最近一根 premium 收盘(因果)
        basis[:, j] = m["basis"].to_numpy(np.float32)
        have += 1

    cov = np.isfinite(basis).any(0).mean()
    nz = basis[np.isfinite(basis)]
    print(f"覆盖 {have}/{N} 币 ({cov:.1%}); basis 非空均值={np.nanmean(nz):+.6e} (绝对均值 {np.nanmean(np.abs(nz)):.6e})", flush=True)
    outp = "data/clean/basis_pit.npz"
    np.savez_compressed(outp, basis=basis, dates=dates, slots=np.array(slots, dtype=str))
    print(f"完成 → {outp}  shape={basis.shape}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensor", default="data/clean/crypto_tensor_60min_pit.npz")
    ap.add_argument("--panel", default="data/clean/crypto_60min_pit.parquet")
    main(ap.parse_args())
