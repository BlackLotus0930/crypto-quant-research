"""3 个分布 baseline（归一空间）。模型必须样本外打过它们才算有信号。

目标 y 是日尺度归一收益，所以归一空间里**每个资产**的 Δt 收益都 ~ 标准化、std≈√Δt，
与是哪个资产无关 → baseline 对所有 slot 给同一分布，预测形状 [B,K] 广播到 [B,N,K]。

统一接口 predict(x, xmask, dt) -> [B,N,K]（baseline 忽略 x）。
"""
import numpy as np

from .crps import QUANTILE_LEVELS, Z


def _broadcast(base, N):
    """[B,K] -> [B,N,K]。"""
    return np.repeat(base[:, None, :], N, axis=1)


class RW:
    """随机游走：N(0, √Δt)。最经典的鞅 baseline。"""
    name = "RW"

    def predict(self, x, xmask, dt):
        dt = np.asarray(dt, float)
        base = Z[None, :] * np.sqrt(dt)[:, None]        # [B,K]
        return _broadcast(base, x.shape[2])


class ClimoScaled:
    """气候态(经验)：1 日归一收益的经验分位 × √Δt。带真实肥尾，比高斯强。"""
    name = "ClimoScaled"

    def fit(self, y1):
        self.q1 = np.quantile(np.asarray(y1, float), QUANTILE_LEVELS)
        return self

    def predict(self, x, xmask, dt):
        dt = np.asarray(dt, float)
        base = self.q1[None, :] * np.sqrt(dt)[:, None]
        return _broadcast(base, x.shape[2])


class ClimoPerDt:
    """每个 Δt 直接拟合经验分位（不假设 √Δt 标度）。最强的无条件 baseline。"""
    name = "ClimoPerDt"

    def fit(self, y_by_dt: dict):
        self.q = {int(dt): np.quantile(np.asarray(y, float), QUANTILE_LEVELS)
                  for dt, y in y_by_dt.items()}
        self._dts = np.array(sorted(self.q))
        return self

    def predict(self, x, xmask, dt):
        dt = np.asarray(dt, float)
        B, _, N, _ = (x.shape[0], 0, x.shape[2], 0)
        out = np.empty((B, len(QUANTILE_LEVELS)), float)
        for i, d in enumerate(dt):
            di = int(self._dts[np.argmin(np.abs(self._dts - d))])  # 最近已拟合的 Δt
            out[i] = self.q[di]
        return _broadcast(out, N)
