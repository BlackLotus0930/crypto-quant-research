"""盘中极值面板:perp_high(永续bar最高,来自panel) + spot_low(现货bar最低,来自现货zip),对齐张量。
诚实强平要盘中插针:最坏基差爆裂 = perp_high/spot_low − 1(永续插上去、现货砸下来)。只用收盘会系统性低估尾部。
跑：PYTHONUTF8=1 .venv/Scripts/python.exe build_intrabar_pit.py
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


def read_low(path):
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
    ts = np.where(ts > 10**13, ts // 1000, ts)
    low = pd.to_numeric(df[3], errors="coerce")                       # kline col3 = low
    out = pd.DataFrame({"symbol": sym, "ts": ts, "low": low.to_numpy()}).dropna()
    return out if not out.empty else None


def main(a):
    z = np.load(a.tensor, allow_pickle=True)
    slots = list(z["slots"].astype(str)); dates = z["dates"].astype(str)
    T, N = len(dates), len(slots); nmap = {s: i for i, s in enumerate(slots)}

    # perp_high:从面板(有 high)对齐
    panel = pd.read_parquet(a.panel, columns=["ts", "high", "symbol"])
    grid_ts = np.sort(panel["ts"].unique()).astype(np.int64)
    assert len(grid_ts) == T, f"面板 ts {len(grid_ts)} != T {T}"
    perp_high = np.full((T, N), np.nan, np.float32)
    gt = np.searchsorted(grid_ts, panel["ts"].to_numpy())
    col = panel["symbol"].map(nmap).to_numpy()
    ok = ~np.isnan(col)
    perp_high[gt[ok], col[ok].astype(int)] = panel["high"].to_numpy(np.float32)[ok]
    print(f"perp_high 对齐完成 覆盖 {np.isfinite(perp_high).any(0).sum()}/{N}", flush=True)

    # spot_low:从现货 zip
    files = [f for f in glob.glob(SGLOB)
             if f.replace("\\", "/").split("/klines/")[1].split("/")[0] in nmap]
    workers = max(1, (os.cpu_count() or 4) - 2)
    print(f"{len(files)} 现货 zip 取 low,线程 {workers}", flush=True)
    rows = []
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(read_low, files):
            if r is not None:
                rows.append(r)
    sp = pd.concat(rows, ignore_index=True); sp["ts"] = sp["ts"].astype(np.int64)
    spot_low = np.full((T, N), np.nan, np.float32)
    gdf = pd.DataFrame({"ts": grid_ts})
    for sym, g in sp.groupby("symbol", sort=False):
        j = nmap.get(sym)
        if j is None:
            continue
        g = g.sort_values("ts").drop_duplicates("ts", keep="last")
        m = pd.merge_asof(gdf, g[["ts", "low"]], on="ts", direction="backward")
        spot_low[:, j] = m["low"].to_numpy(np.float32)
    print(f"spot_low 对齐完成 覆盖 {np.isfinite(spot_low).any(0).sum()}/{N}", flush=True)

    np.savez_compressed("data/clean/intrabar_pit.npz", perp_high=perp_high, spot_low=spot_low,
                        dates=dates, slots=np.array(slots, dtype=str))
    print("完成 → data/clean/intrabar_pit.npz", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensor", default="data/clean/crypto_tensor_60min_pit.npz")
    ap.add_argument("--panel", default="data/clean/crypto_60min_pit.parquet")
    main(ap.parse_args())
