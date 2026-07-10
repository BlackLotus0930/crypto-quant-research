"""IC（information coefficient）—— 方向信号的主检测器。

CRPS 对"可交易的方向 edge"近乎失明（对均值的敏感度是二阶小量）；IC 直接量。
IC = 每日**横截面** corr(方向信号, 次日归一收益)，再对天平均。
比较三种从分布里取的方向信号：mean / median / P(up)。
IC≈0.02–0.05、年化 IR≥0.5 就算可交易。
"""
import numpy as np

from pipeline.config import T_WINDOW
from .harness import anchors_for, build_windows, labels_at

MIN_NAMES = 10   # 当日有效标的 < 此数不算横截面 IC


def signals_from_pred(pred):
    """pred [B,N,K] → 三种方向信号 [B,N]。"""
    K = pred.shape[-1]
    return {
        "mean": pred.mean(-1),                 # E[r]：期望收益（理论正确）
        "median": pred[..., K // 2],           # 中位数（抗噪对照）
        "p_up": (pred > 0).mean(-1) - 0.5,     # P(r>0)：用整条分布的方向概率
    }


def ic_table(tensor, splits, split, predictor, dt=1, T=T_WINDOW, batch=128):
    ann = int(tensor["ann"]) if "ann" in tensor else 252 * (int(tensor["bars_per_day"]) if "bars_per_day" in tensor else 1)  # 加密 365×bpd；股票 252×bpd
    anchors = anchors_for(tensor, splits, split, T, dt_max=dt)
    daily = {k: [] for k in ("mean", "median", "p_up")}
    for s in range(0, len(anchors), batch):
        ab = anchors[s:s + batch]
        x, xmask = build_windows(tensor, ab, T)
        y, ymask = labels_at(tensor, ab, dt)
        pred = predictor.predict(x, xmask, np.full(len(ab), dt))
        sigs = signals_from_pred(pred)
        for b in range(len(ab)):
            m = ymask[b]
            if m.sum() < MIN_NAMES:
                continue
            yr = y[b, m]
            if yr.std() < 1e-12:
                continue
            for k, sig in sigs.items():
                sv = sig[b, m]
                if sv.std() > 1e-12:
                    daily[k].append(np.corrcoef(sv, yr)[0, 1])
    out = {}
    for k, v in daily.items():
        v = np.array(v)
        out[k] = dict(IC=float(v.mean()) if len(v) else 0.0,
                      IR=float(v.mean() / v.std() * np.sqrt(ann)) if len(v) and v.std() > 0 else 0.0,
                      n=len(v))
    return out


def print_ic(tbl, dt=1):
    print(f"--- IC（每日横截面 corr(信号, Δt={dt} 归一收益)，年化 IR）---")
    print(f"{'signal':8s} {'IC':>9s} {'年化IR':>9s} {'天数':>6s}")
    best = max(tbl, key=lambda k: tbl[k]["IR"])
    for k, d in tbl.items():
        star = " ←最强" if k == best else ""
        print(f"{k:8s} {d['IC']:9.4f} {d['IR']:9.2f} {d['n']:6d}{star}")
    return best, tbl[best]
