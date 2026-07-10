"""评估骨架：从 v1_tensor 现搭窗口/标签，把任意 predictor 跑过一个 split，
算 CRPS（所有 predictor 在**同一批** (anchor,Δt) 样本上比，公平）。

predictor 接口：.name, .predict(x[B,T,N,5], xmask[B,T,N], dt[B]) -> [B,N,K]（numpy）。
"""
import json

import numpy as np

from pipeline.config import T_WINDOW, V1_TENSOR, SPLITS_JSON
from pipeline.splits import assign_split
from .crps import pinball, QUANTILE_LEVELS


def load(tensor_path=V1_TENSOR, splits_path=SPLITS_JSON):
    z = np.load(tensor_path, allow_pickle=False)
    tensor = {k: z[k] for k in z.files}
    splits = json.load(open(splits_path, encoding="utf-8"))
    return tensor, splits


def anchors_for(tensor, splits, split, T=T_WINDOW, dt_max=20):
    D = tensor["mask"].shape[0]
    idx = np.arange(T - 1, D - dt_max - 1)
    dates = tensor["dates"]
    keep = np.array([assign_split(dates[a], splits) == split for a in idx])
    keep &= tensor["mask"][idx].any(1)
    # embargo：标签窗末端 a+dt_max 跨进下一段则剔除（防边界标签前视泄漏）
    keep &= np.array([assign_split(dates[a + dt_max], splits) == split for a in idx])
    return idx[keep]


def build_windows(tensor, anchors, T=T_WINDOW):
    """[B,T,N,5], [B,T,N]。"""
    off = np.arange(-T + 1, 1)
    rows = anchors[:, None] + off[None, :]            # [B,T]
    return tensor["n"][rows], tensor["mask"][rows]


def labels_at(tensor, anchors, dt):
    """归一 Δt 收益 + 有效 mask。[B,N]。"""
    ca = tensor["adj_close"][anchors]
    cb = tensor["adj_close"][anchors + dt]
    valid = tensor["mask"][anchors] & tensor["mask"][anchors + dt]
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = np.log(np.where((ca > 0) & (cb > 0), cb / ca, 1.0))
    s = tensor["scale_ret"][anchors]
    ymask = valid & (s > 0)
    y = np.where(ymask, ret / np.where(s > 0, s, 1.0), 0.0)
    return y, ymask


def gather_train_y(tensor, splits, dts, T=T_WINDOW, max_anchors=2000, seed=0):
    """train 上每个 Δt 的池化归一收益（拟合 climatology 用）。"""
    a = anchors_for(tensor, splits, "train", T, max(dts))
    if len(a) > max_anchors:
        a = np.random.default_rng(seed).choice(a, max_anchors, replace=False)
    out = {}
    for dt in dts:
        y, m = labels_at(tensor, a, dt)
        out[dt] = y[m]
    return out


def crps_table(tensor, splits, split, predictors, dts=(1, 5, 10, 20), T=T_WINDOW, batch=128):
    """返回 {pred_name: {dt: crps, 'all': crps}}。所有 predictor 共享同一 (anchor,dt)。"""
    anchors = anchors_for(tensor, splits, split, T, max(dts))
    acc = {p.name: {dt: [0.0, 0] for dt in dts} for p in predictors}   # [sum, count]
    for s in range(0, len(anchors), batch):
        ab = anchors[s:s + batch]
        x, xmask = build_windows(tensor, ab, T)
        for dt in dts:
            y, ymask = labels_at(tensor, ab, dt)
            dtv = np.full(len(ab), dt)
            for p in predictors:
                pred = p.predict(x, xmask, dtv)            # [B,N,K]
                # 累积有效格子的 pinball
                loss = np.maximum(QUANTILE_LEVELS * (y[..., None] - pred),
                                  (QUANTILE_LEVELS - 1) * (y[..., None] - pred)).mean(-1)
                acc[p.name][dt][0] += float(loss[ymask].sum())
                acc[p.name][dt][1] += int(ymask.sum())
    out = {}
    for name, d in acc.items():
        out[name] = {dt: (sm / cnt if cnt else float("nan")) for dt, (sm, cnt) in d.items()}
        tot_s = sum(v[0] for v in d.values()); tot_c = sum(v[1] for v in d.values())
        out[name]["all"] = tot_s / tot_c if tot_c else float("nan")
    return out


def print_crps_table(table, ref="RW", dts=(1, 5, 10, 20)):
    cols = list(dts) + ["all"]
    print(f"{'predictor':14s} " + " ".join(f"{('Δt='+str(c) if c!='all' else 'ALL'):>10s}" for c in cols) + "   skill(vs %s)" % ref)
    for name, d in table.items():
        row = " ".join(f"{d[c]:10.4f}" for c in cols)
        skill = 1 - d["all"] / table[ref]["all"]
        print(f"{name:14s} {row}   {skill:+.3%}")
