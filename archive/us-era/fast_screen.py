"""廉价信号预筛(无 GPU、无训练)：手搓简单因子的横截面 IC，比大盘 vs 中盘。
逻辑：若反转/动量等笨因子在中盘 IC 明显高于大盘 → 宇宙更低效、更值得上大模型；若≈ → 别白跑。
只读近 --days 个交易日(连续，反转/动量要连续日)。复用 build_daily 的优化聚合(地板前置，快)。
跑：PYTHONUTF8=1 .venv/Scripts/python.exe fast_screen.py --days 504
"""
import argparse
import concurrent.futures as cf
import os
import time

DEF_WORKERS = max(1, (os.cpu_count() or 4) - 2)        # 留 2 核给系统，别卡死机器

import numpy as np
import pandas as pd

from build_intraday import all_files
from build_daily import _process, _init


def rank_ic(factor: pd.DataFrame, target: pd.DataFrame, cols, min_names=20):
    """逐日横截面 Spearman IC（factor 在 t 已知，预测 t 当日 target）。返回 (meanIC, 年化IR, 天数)。"""
    ics = []
    f = factor[cols]; y = target[cols]
    for d in f.index:
        fv = f.loc[d]; yv = y.loc[d]
        m = fv.notna() & yv.notna()
        if m.sum() < min_names:
            continue
        a, c = fv[m], yv[m]
        if a.std() > 0 and c.std() > 0:
            ics.append(np.corrcoef(a.rank(), c.rank())[0, 1])
    ics = np.array(ics)
    if len(ics) == 0:
        return (0.0, 0.0, 0)
    return (ics.mean(), ics.mean() / ics.std() * np.sqrt(252) if ics.std() > 0 else 0.0, len(ics))


def main(a):
    files = all_files()[-a.days:]
    print(f"读近 {len(files)} 天 {files[0].split(chr(92))[-1]} ~ {files[-1].split(chr(92))[-1]}；地板 ${a.min_dv:,.0f}", flush=True)
    t0 = time.time(); rows = []
    with cf.ProcessPoolExecutor(max_workers=a.workers, initializer=_init, initargs=(a.min_dv,)) as ex:
        for r in ex.map(_process, files, chunksize=4):
            if r is not None:
                rows.append(r)
    panel = pd.concat(rows, ignore_index=True)
    panel["dv"] = panel["close"] * panel["volume"]
    close = panel.pivot(index="date", columns="ticker", values="close").sort_index()
    dv = panel.pivot(index="date", columns="ticker", values="dv").sort_index()
    print(f"聚合完成 {close.shape[0]} 天 x {close.shape[1]} 票  ({time.time()-t0:.0f}s)", flush=True)

    ret = np.log(close).diff()                              # 当日 log 收益
    medv = dv.median().sort_values(ascending=False)          # 按窗口内中位美元成交额排名
    tk = [t for t in medv.index if medv[t] > 0]
    universes = {
        "large(top300)":   tk[:300],
        "mid(300-800)":    tk[300:800],
        "small(800-1500)": tk[800:1500],
    }
    # 因子全部 shift(1) → 只用过去，预测当日 ret（无前视）
    factors = {
        "rev1":      -ret.shift(1),                              # 1日反转（受买卖价弹跳污染）
        "rev5":      -ret.rolling(5).sum().shift(1),             # 5日反转
        "rev5_skip": -ret.rolling(5).sum().shift(2),             # 跳过最近1天的5日反转（去弹跳，关键对照）
        "mom21":      ret.rolling(21).sum().shift(1),            # 月动量
        "dvolchg":    np.log(dv.clip(lower=1)).diff().shift(1),  # 量变化
    }
    print(f"\n{'因子':>10s} | " + " | ".join(f"{u:>16s}" for u in universes), flush=True)
    print("-" * 80, flush=True)
    for fname, f in factors.items():
        cells = []
        for uname, cols in universes.items():
            cols = [c for c in cols if c in ret.columns]
            ic, ir, n = rank_ic(f, ret, cols)
            cells.append(f"IC{ic:+.4f} IR{ir:+.1f}")
        print(f"{fname:>10s} | " + " | ".join(f"{c:>16s}" for c in cells), flush=True)
    print(f"\n判读：同一因子，中/小盘 |IC| 明显>大盘 → 越不流动越多 alpha（值得上大模型）；≈ → 低效假设不成立。", flush=True)
    print(f"({time.time()-t0:.0f}s 总)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=504, help="近 N 个交易日(连续)")
    ap.add_argument("--min-dv", type=float, default=200000, dest="min_dv")
    ap.add_argument("--workers", type=int, default=DEF_WORKERS)
    main(ap.parse_args())
