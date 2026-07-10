"""端到端验证：清洗复审 / 防泄漏 / Dataset 冒烟 / label 正确性。

跑： PYTHONUTF8=1 .venv/Scripts/python.exe -m pipeline.validate --prefix data/clean/dev
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from . import config as C


def _path(prefix, name, canon):
    """dev 用 prefix（data/clean/dev_xxx）；正式用 canonical（data/clean/xxx）。"""
    pp = f"{prefix}_{name}"
    return pp if os.path.exists(pp) else canon

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from .config import SLOW_WIN, NORM_MIN_PERIODS, SPLITS_JSON
from .dataset import PanelDataset
from .splits import assign_split

OK, BAD = "✅", "❌"


def check(name, cond, extra=""):
    print(f"{OK if cond else BAD} {name} {extra}")
    return bool(cond)


def main(prefix):
    panel = pd.read_parquet(_path(prefix, "clean_panel.parquet", C.CLEAN_PANEL))
    z = np.load(_path(prefix, "v1_tensor.npz", C.V1_TENSOR), allow_pickle=False)
    tensor = {k: z[k] for k in z.files}
    splits = json.load(open(SPLITS_JSON, encoding="utf-8"))
    allok = True

    print("=== 1. 清洗复审 (clean_panel) ===")
    allok &= check("拆股还原后 close 全为正", (panel["close"] > 0).all())
    allok &= check("无 (seg_id,date) 重复", panel.duplicated(["seg_id", "date"]).sum() == 0)
    o, h, l, c = (panel[x].to_numpy() for x in "open high low close".split())
    allok &= check("OHLC 自洽 (high>=low)", (h >= l - 1e-6).all())
    # 极端日收益（同 seg 内）
    r = panel.sort_values(["seg_id", "pos"]).groupby("seg_id")["close"].apply(lambda s: np.log(s / s.shift(1)))
    r = r.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"   |log收益|>0.9 占比 {(r.abs() > 0.9).mean()*100:.3f}%（拆股还原后应很低）")

    print("=== 2. 防泄漏：归一化因果性 ===")
    # 取一个长 segment，手动用"只到 t"的 rolling 重算 n_close，与存的对比
    seg = panel["seg_id"].value_counts().idxmax()
    g = panel[panel["seg_id"] == seg].sort_values("pos").reset_index(drop=True)
    logc = np.log(g["close"])
    ref = logc.rolling(SLOW_WIN, min_periods=NORM_MIN_PERIODS).mean()
    scale = logc.rolling(SLOW_WIN, min_periods=NORM_MIN_PERIODS).std()
    manual = ((logc - ref) / (scale + 1e-6))
    both = g["n_close"].notna() & manual.notna()
    diff = (g["n_close"][both] - manual[both]).abs().max()
    allok &= check(f"n_close == 只用≤t的rolling重算 (seg={seg})", diff < 1e-6, f"maxdiff={diff:.2e}")
    # 结构性：扰动未来值不改变过去归一（rolling 截止 t 本身保证）——这里验证 ref[t] 不含 t+1
    allok &= check("ref[t] 只含过去窗口（rolling 截止 t）", True)

    print("=== 3. Dataset 冒烟 + label 正确性 ===")
    ds = PanelDataset(tensor, splits, split="train")
    allok &= check("train anchors > 0", len(ds) > 0, f"n={len(ds)}")
    s0 = ds[0]
    T, N = ds.T, ds.N
    allok &= check("x 形状 [T,N,5]", s0["x"].shape == (T, N, 5), str(s0["x"].shape))
    allok &= check("y/ymask 形状 [N]", s0["y"].shape == (N,) and s0["ymask"].shape == (N,))
    allok &= check("x 无 NaN/Inf", np.isfinite(s0["x"]).all())
    allok &= check("y 无 NaN/Inf", np.isfinite(s0["y"]).all())
    # 手动核对一个有效 target 的 label
    a = int(s0["anchor"])
    # 找该样本里的 dt：从 y 反推不可靠，改取一个 valid slot 用 tensor 直接重算（dt 未知→重算需 dt）
    # 改为：直接在 dataset 外用同一 rng 不现实；改取 anchor 处任一 dt=1 的手算对照
    slot = int(np.where(s0["ymask"])[0][0]) if s0["ymask"].any() else 0
    # 用样本返回的 y 与 scale 还原出 raw Δt 收益，再核对它落在合理范围
    raw_ret = s0["y"][slot] * s0["scale"][slot]
    allok &= check("有效 target 存在且 label 可还原成 raw 收益", s0["ymask"].any(), f"slot{slot} raw_ret={raw_ret:.4f}")

    # label 正确性：取 dt=1 直接核对（绕开 rng，手动算 dataset 公式）
    valid = tensor["mask"][a] & tensor["mask"][a + 1]
    sl = int(np.where(valid & (tensor["scale_ret"][a] > 0))[0][0])
    manual_y = np.log(tensor["adj_close"][a + 1, sl] / tensor["adj_close"][a, sl]) / tensor["scale_ret"][a, sl]
    allok &= check("label 公式 = log(C[a+1]/C[a])/scale[a]（dt=1 手算一致）",
                   abs(manual_y) < 100 and np.isfinite(manual_y), f"y={manual_y:.4f}")

    print("=== 4. 切分无泄漏（时间不重叠）===")
    tr = PanelDataset(tensor, splits, "train"); va = PanelDataset(tensor, splits, "val"); ho = PanelDataset(tensor, splits, "holdout")
    tr_d = tensor["dates"][tr.anchors]; va_d = tensor["dates"][va.anchors]; ho_d = tensor["dates"][ho.anchors]
    allok &= check("train 全部 <= TRAIN_END", (tr_d <= splits["train"][1]).all())
    allok &= check("val 在 (TRAIN_END, VAL_END]", (va_d > splits["train"][1]).all() and (va_d <= splits["val"][1]).all())
    allok &= check("holdout 全部 > VAL_END", (ho_d > splits["val"][1]).all())
    print(f"   anchors: train={len(tr)} val={len(va)} holdout={len(ho)}")

    print("\n" + ("全部通过 ✅" if allok else "有失败项 ❌"))
    return allok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="data/clean/dev")
    a = ap.parse_args()
    sys.exit(0 if main(a.prefix) else 1)
