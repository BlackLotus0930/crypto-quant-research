"""阶段 6：因果 scale-free 归一化（设计 §4.1）。

per-资产、因果、逐标的用自身统计 → 保等变、无人类因子。
- 价：log 后减**慢速**因果均值、除慢速因果标准差 → 去任意价位/拆股残留，
  慢尺度滞后 → 近期波动 regime 不被抹平（spike 会显成更大的归一值）。
- 量：log1p 后同样慢尺度归一。
- 另存每日**收益尺度** s_ret（日 log 收益的慢速因果 std）→ label 在尺度归一空间，
  serving 乘回还原（设计 §3.2/§4.3）。
所有 rolling 窗口到 t 截止（含 t，t 是"现在"非未来），严格因果。
"""
import numpy as np
import pandas as pd

from .config import SLOW_WIN, NORM_MIN_PERIODS

EPS = 1e-6


def _roll_mean(g, col):
    return g[col].rolling(SLOW_WIN, min_periods=NORM_MIN_PERIODS).mean().reset_index(level=0, drop=True)


def _roll_std(g, col):
    return g[col].rolling(SLOW_WIN, min_periods=NORM_MIN_PERIODS).std().reset_index(level=0, drop=True)


def normalize(panel: pd.DataFrame) -> pd.DataFrame:
    """panel 需已拆股还原、含 seg_id/pos。加列 n_open/n_high/n_low/n_close/n_volume,
    scale_ret(收益尺度), scale_lvl/ref_lvl(价位归一用)。归一不出(历史不足/flat)→NaN，build 里 mask。"""
    df = panel.sort_values(["seg_id", "pos"]).reset_index(drop=True)
    df["_logc"] = np.log(df["close"])
    df["_logv"] = np.log1p(df["volume"])
    df["_ret"] = df.groupby("seg_id", sort=False)["_logc"].diff()

    g = df.groupby("seg_id", sort=False)
    ref = _roll_mean(g, "_logc")          # 价位因果均值（慢）
    scale = _roll_std(g, "_logc")         # 价位因果 std（慢）
    s_ret = _roll_std(g, "_ret")          # 收益尺度（慢）
    ref_v = _roll_mean(g, "_logv")
    scale_v = _roll_std(g, "_logv")

    for ch in ("open", "high", "low", "close"):
        df["n_" + ch] = (np.log(df[ch]) - ref) / (scale + EPS)
    df["n_volume"] = (df["_logv"] - ref_v) / (scale_v + EPS)
    df["scale_ret"] = s_ret
    df["scale_lvl"] = scale
    df["ref_lvl"] = ref

    # flat / 尺度退化 → 标 NaN（build 里据此 mask）
    bad = (scale < EPS) | scale.isna() | (s_ret < EPS) | s_ret.isna()
    for ch in ("open", "high", "low", "close", "volume"):
        df.loc[bad, "n_" + ch] = np.nan

    return df.drop(columns=["_logc", "_logv", "_ret"])
