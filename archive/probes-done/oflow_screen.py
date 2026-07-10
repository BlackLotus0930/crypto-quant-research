"""订单流有效时间横扫（无 GPU，几分钟）：bar 级 taker 失衡 → 未来收益 的横截面 IC，扫 horizon。
回答三件事：① 订单流在哪个时间尺度有信号 ② 在我们可交易 band(15-60min) 还剩多少 ③ 和价格(当根收益/反转)多冗余。
直接读 1m klines（限近窗口省内存），采样时间戳估横截面 Spearman IC。这是 E8 市场快筛的同款"先廉价探针、再决定要不要砸 GPU"。
跑：PYTHONUTF8=1 .venv/Scripts/python.exe oflow_screen.py --since 2026-01
"""
import argparse
import concurrent.futures as cf
import glob
import io
import os
import re
import zipfile

import numpy as np
import pandas as pd

KGLOB = "data/raw/binance/data/futures/um/monthly/klines/*/1m/*.zip"
USECOLS = [0, 4, 5, 9]                                     # open_time, close, volume, taker_buy_vol
NAMES = ["t", "c", "v", "tbv"]


def read_zip(path):
    sym = path.replace("\\", "/").split("/klines/")[1].split("/")[0]
    try:
        z = zipfile.ZipFile(path); raw = z.read(z.namelist()[0])
    except Exception:
        return None
    df = pd.read_csv(io.BytesIO(raw), header=None, usecols=USECOLS, names=NAMES)
    df["t"] = pd.to_numeric(df["t"], errors="coerce"); df = df.dropna(subset=["t"])
    if df.empty:
        return None
    for cc in ("c", "v", "tbv"):
        df[cc] = pd.to_numeric(df[cc], errors="coerce")
    df["sym"] = sym
    return df


def fwd_at(logc, h):                                       # fwd_h[t] = logc[t+h]-logc[t]
    out = np.full_like(logc, np.nan); out[:-h] = logc[h:] - logc[:-h]; return out


def xsic(fac, fwd, nsamp=40000):
    """采样时间戳，每个时间戳算跨币 Spearman IC，返回 (均值IC, t值, n)。"""
    T = fac.shape[0]
    idx = np.unique(np.linspace(0, T - 61, min(nsamp, T - 61)).astype(int))
    ics = []
    for t in idx:
        f, r = fac[t], fwd[t]
        m = np.isfinite(f) & np.isfinite(r)
        if m.sum() < 10:
            continue
        x, y = f[m], r[m]
        if x.std() < 1e-12 or y.std() < 1e-12:
            continue
        xr = x.argsort().argsort().astype(float); yr = y.argsort().argsort().astype(float)
        ics.append(np.corrcoef(xr, yr)[0, 1])
    a = np.array(ics)
    return (a.mean(), a.mean() / a.std() * np.sqrt(len(a)) if a.std() > 0 else 0.0, len(a)) if len(a) else (0.0, 0.0, 0)


def main(a):
    files = [f for f in glob.glob(KGLOB)
             if (m := re.search(r"(\d{4}-\d{2})\.zip$", f)) and m.group(1) >= a.since]
    workers = max(1, (os.cpu_count() or 4) - 2)
    print(f"{len(files)} 个 1m zip（>= {a.since}），线程 {workers} 读取中…", flush=True)
    rows = []
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(read_zip, files):
            if r is not None:
                rows.append(r)
    panel = pd.concat(rows, ignore_index=True)
    panel = panel[panel["v"] > 0]
    print(f"面板 {len(panel):,} 行, {panel['sym'].nunique()} 币", flush=True)

    ts = np.sort(panel["t"].unique()); tmap = {x: i for i, x in enumerate(ts)}
    syms = sorted(panel["sym"].unique()); smap = {s: i for i, s in enumerate(syms)}
    T, N = len(ts), len(syms)
    close = np.full((T, N), np.nan, np.float32); imb = np.full((T, N), np.nan, np.float32)
    ri = panel["t"].map(tmap).to_numpy(); ci = panel["sym"].map(smap).to_numpy()
    close[ri, ci] = panel["c"].to_numpy(np.float32)
    imb[ri, ci] = ((2 * panel["tbv"] - panel["v"]) / panel["v"].clip(lower=1e-9)).to_numpy(np.float32)
    with np.errstate(all="ignore"):
        logc = np.where(close > 0, np.log(close), np.nan).astype(np.float32)
    print(f"网格 T={T:,}(1min) × N={N}  时段 {pd.to_datetime(ts[0],unit='ms')}~{pd.to_datetime(ts[-1],unit='ms')}\n", flush=True)

    ret1 = np.full_like(logc, np.nan); ret1[1:] = logc[1:] - logc[:-1]      # 当根收益
    rev = -ret1                                                            # 1min 反转(价格因子对照)
    print(f"{'horizon':>8s} {'失衡IC':>9s} {'失衡t':>7s} | {'反转IC':>9s} {'反转t':>7s} | {'corr(失衡,当根收益)':>16s}")
    for h in a.horizons:
        fwd = fwd_at(logc, h)
        ic_i, t_i, n = xsic(imb, fwd)
        ic_r, t_r, _ = xsic(rev, fwd)
        # 失衡与当根收益的冗余：采样时间戳上 corr(imb, ret1) 均值
        idx = np.unique(np.linspace(0, T - 61, min(40000, T - 61)).astype(int))
        cc = []
        for t in idx:
            mm = np.isfinite(imb[t]) & np.isfinite(ret1[t])
            if mm.sum() >= 10 and imb[t][mm].std() > 1e-12 and ret1[t][mm].std() > 1e-12:
                cc.append(np.corrcoef(imb[t][mm], ret1[t][mm])[0, 1])
        print(f"{h:>6d}m {ic_i:>+9.4f} {t_i:>+7.1f} | {ic_r:>+9.4f} {t_r:>+7.1f} | {np.mean(cc):>+16.2f}")
    print("\n判读：失衡IC 在哪个 horizon 还显著(t≥3)、到 15-60min 衰减多少；corr 高=失衡≈当根涨跌(冗余)。", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-01", help="只读 >= 该 YYYY-MM 的月度 zip（省内存/时间）")
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 15, 30, 60])
    main(ap.parse_args())
