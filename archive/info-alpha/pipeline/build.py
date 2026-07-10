"""编排器：raw panel → clean_panel + universe + v1_tensor + splits。

跑：  PYTHONUTF8=1 .venv/Scripts/python.exe -m pipeline.build
dev： ... -m pipeline.build --panel data/clean/dev_panel.parquet --v1-n 128 --out data/clean/dev
"""
import argparse
import sys
import time

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from . import config as C
from .calendar import build_calendar
from .clean import dedup_dates, drop_bad_rows, segment_identities, adjust_splits, segment_quality
from .normalize import normalize
from .splits import make_splits, save_splits


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def select_v1(uni: pd.DataFrame, n: int) -> pd.DataFrame:
    """v1 universe = 全部跨资产层(非 stock，过质量) + 成交额最高的股票补足到 n。
    保证'全球联合状态'那层一定在里面。返回带 slot 列的子集。"""
    keep = uni[~uni["drop"]].copy()
    cross = keep[keep["category"] != "stock"].sort_values("med_dollar_vol", ascending=False)
    stock = keep[keep["category"] == "stock"].sort_values("med_dollar_vol", ascending=False)
    n_stock = max(0, n - len(cross))
    sel = pd.concat([cross, stock.head(n_stock)], ignore_index=True)
    sel = sel.sort_values("med_dollar_vol", ascending=False).head(n).reset_index(drop=True)
    sel["slot"] = np.arange(len(sel))
    return sel


def build_v1_tensor(panel: pd.DataFrame, sel: pd.DataFrame, cal: np.ndarray):
    """把 v1 选中的 segment 散点到稠密 [num_days, N] 张量。"""
    num_days, N = len(cal), len(sel)
    slot_of = dict(zip(sel["seg_id"], sel["slot"]))
    sub = panel[panel["seg_id"].isin(slot_of)].copy()
    sub["slot"] = sub["seg_id"].map(slot_of)
    rows = sub["pos"].to_numpy()
    cols = sub["slot"].to_numpy()

    chans = ["n_" + c for c in C.CHANNELS]              # 5 输入通道
    n_arr = np.zeros((num_days, N, len(chans)), np.float32)
    mask = np.zeros((num_days, N), bool)
    scale_ret = np.zeros((num_days, N), np.float32)
    adj_close = np.zeros((num_days, N), np.float32)

    valid = sub[chans].notna().all(axis=1).to_numpy() & sub["scale_ret"].notna().to_numpy()
    mask[rows, cols] = valid
    for k, ch in enumerate(chans):
        vals = np.nan_to_num(sub[ch].to_numpy(np.float32))
        n_arr[rows, cols, k] = vals
    scale_ret[rows, cols] = np.nan_to_num(sub["scale_ret"].to_numpy(np.float32))
    adj_close[rows, cols] = np.nan_to_num(sub["close"].to_numpy(np.float32))
    # 归一无效的格子，输入清 0
    n_arr[~mask] = 0.0
    return dict(dates=cal.astype(str), slots=sel["seg_id"].to_numpy().astype(str),
                n=n_arr, mask=mask, scale_ret=scale_ret, adj_close=adj_close)


def run(panel_path: str, out_prefix: str, v1_n: int):
    t0 = time.time()
    log(f"读 {panel_path}")
    panel = pd.read_parquet(panel_path)
    log(f"  {len(panel):,} 行, {panel['symbol'].nunique()} symbol")

    cal, pos = build_calendar(panel)
    log(f"日历: {len(cal)} 交易日 {cal[0]}~{cal[-1]}")

    panel = dedup_dates(panel);            log(f"去重后 {len(panel):,}")
    panel = drop_bad_rows(panel);          log(f"删错后 {len(panel):,}")
    panel = segment_identities(panel, pos)
    log(f"身份切分后 {len(panel):,} 行, {panel['seg_id'].nunique()} segment")
    panel = adjust_splits(panel);          log("拆股还原完成")
    panel = normalize(panel);              log("归一化完成")

    uni = segment_quality(panel)
    log(f"质量: {len(uni)} segment, drop(vol≈0)={int(uni['drop'].sum())}")
    sel = select_v1(uni, v1_n)
    log(f"v1 universe: {len(sel)} slot（跨资产 {int((sel['category']!='stock').sum())} + 股票 {int((sel['category']=='stock').sum())}）")

    # 落盘
    out_panel = f"{out_prefix}_clean_panel.parquet" if out_prefix != C.CLEAN_DIR else C.CLEAN_PANEL
    out_uni = f"{out_prefix}_universe.parquet" if out_prefix != C.CLEAN_DIR else C.UNIVERSE
    out_tensor = f"{out_prefix}_v1_tensor.npz" if out_prefix != C.CLEAN_DIR else C.V1_TENSOR
    if out_prefix == C.CLEAN_DIR:
        out_panel, out_uni, out_tensor = C.CLEAN_PANEL, C.UNIVERSE, C.V1_TENSOR

    keep_cols = (["seg_id", "symbol", "category", "date", "pos",
                  "open", "high", "low", "close", "volume", "split_factor",
                  "scale_ret", "scale_lvl", "ref_lvl"]
                 + ["n_" + c for c in C.CHANNELS])
    panel[keep_cols].to_parquet(out_panel, index=False)
    uni.merge(sel[["seg_id", "slot"]], on="seg_id", how="left").to_parquet(out_uni, index=False)
    tensor = build_v1_tensor(panel, sel, cal)
    np.savez_compressed(out_tensor, **tensor)
    save_splits(make_splits())

    log(f"完成 ({time.time()-t0:.0f}s)。clean_panel={out_panel}  tensor={out_tensor}  "
        f"mask 真实格子率={tensor['mask'].mean():.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=C.RAW_PANEL)
    ap.add_argument("--out", default=C.CLEAN_DIR, help="输出前缀；默认写正式路径")
    ap.add_argument("--v1-n", type=int, default=C.V1_N)
    a = ap.parse_args()
    run(a.panel, a.out, a.v1_n)
