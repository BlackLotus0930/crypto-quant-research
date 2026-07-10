"""E5：在**已清洗+归一**的 clean_panel 上换更大 universe → 新张量。跳过清洗/拆股/归一（重活已做过）。
跑： PYTHONUTF8=1 .venv/Scripts/python.exe rebuild_universe.py --v1-n 1024 --out data/clean/n1024
只重做 select_v1 + 张量化，复用 pipeline.build 的函数，保证口径一致。
"""
import argparse
import time

import numpy as np
import pandas as pd

from pipeline import config as C
from pipeline.build import select_v1, build_v1_tensor
from pipeline.splits import make_splits, save_splits


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def run(n, out_prefix):
    t0 = time.time()
    uni = pd.read_parquet(C.UNIVERSE)                 # 含 med_dollar_vol/category/drop（小）
    log(f"universe: {len(uni)} segment")
    cols = ["seg_id", "pos", "date", "close", "scale_ret"] + ["n_" + c for c in C.CHANNELS]
    log("读 clean_panel（只取必要列）...")
    panel = pd.read_parquet(C.CLEAN_PANEL, columns=cols)
    log(f"  {len(panel):,} 行")

    # 从 panel 的 pos/date 还原日历（cal[pos]=date）
    cal_df = panel[["pos", "date"]].drop_duplicates("pos").sort_values("pos")
    cal = cal_df["date"].to_numpy()
    assert (cal_df["pos"].to_numpy() == np.arange(len(cal))).all(), "pos 不连续，日历对不上"
    log(f"日历 {len(cal)} 交易日 {cal[0]}~{cal[-1]}")

    sel = select_v1(uni, n)
    n_cross = int((sel["category"] != "stock").sum())
    log(f"v1 universe: {len(sel)} slot（跨资产 {n_cross} + 股票 {len(sel)-n_cross}）")

    tensor = build_v1_tensor(panel, sel, cal)
    out_tensor = f"{out_prefix}_v1_tensor.npz"
    out_uni = f"{out_prefix}_universe.parquet"
    np.savez_compressed(out_tensor, **tensor)
    uni.merge(sel[["seg_id", "slot"]], on="seg_id", how="left").to_parquet(out_uni, index=False)
    save_splits(make_splits())
    log(f"完成 ({time.time()-t0:.0f}s)  tensor={out_tensor}  "
        f"shape={tensor['n'].shape}  mask 真实率={tensor['mask'].mean():.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1-n", type=int, default=1024)
    ap.add_argument("--out", default="data/clean/n1024")
    a = ap.parse_args()
    run(a.v1_n, a.out)
