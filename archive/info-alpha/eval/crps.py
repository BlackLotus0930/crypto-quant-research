"""CRPS / pinball（分位）损失 —— 同时是模型训练损失和评估指标，避免两套实现漂移。

K 个固定分位 level；CRPS 用"K 个 level 的 pinball 损失平均"估计（离散 CRPS，
跨预测器比较时常数因子抵消）。纯 numpy，无 scipy 依赖（norm ppf 用 Acklam 近似）。
"""
import numpy as np

K = 21
QUANTILE_LEVELS = np.linspace(0.5 / K, 1 - 0.5 / K, K)   # 中点分位，对称覆盖 (0,1)


def norm_ppf(p):
    """标准正态分位（Acklam 有理近似），免 scipy。"""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    p = np.asarray(p, float)
    x = np.zeros_like(p)
    lo, hi = 0.02425, 1 - 0.02425
    m = p < lo
    q = np.sqrt(-2 * np.log(np.clip(p[m], 1e-300, None)))
    x[m] = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    m = (p >= lo) & (p <= hi)
    q = p[m] - 0.5; r = q*q
    x[m] = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    m = p > hi
    q = np.sqrt(-2 * np.log(np.clip(1 - p[m], 1e-300, None)))
    x[m] = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    return x


Z = norm_ppf(QUANTILE_LEVELS)   # 标准正态在各 level 的分位（baseline 用）


def pinball(pred, y, levels=QUANTILE_LEVELS, mask=None):
    """pred [...,K], y [...]，返回有效格子上的平均 pinball（≈CRPS）。"""
    diff = y[..., None] - pred                      # [...,K]
    loss = np.maximum(levels * diff, (levels - 1) * diff).mean(-1)   # [...]
    if mask is not None:
        loss = loss[mask]
    return float(loss.mean()) if loss.size else float("nan")
