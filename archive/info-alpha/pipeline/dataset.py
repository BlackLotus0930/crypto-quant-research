"""阶段 7+9：窗口 + 多 target 标签 + Δt。

从 v1_tensor.npz 现搭训练样本（数据已进内存，不预存所有窗口）。
- 一个样本 = 一个 anchor 日 a + 采样一个 Δt → 预测当天所有有效 slot（设计 §3.1）。
- 输入窗口只用 ≤a 的归一值（因果）；label 用 a→a+Δt 的 log 收益、在尺度归一空间。
- gap/未上市/退市 全靠 mask；同一 slot=同一 segment，故"同身份"天然满足。

返回 numpy（torch DataLoader 默认 collate 可直接转 tensor；建模时再包 torch）。
"""
import numpy as np

from .config import T_WINDOW, DT_MIN, DT_MAX, V1_TENSOR, SPLITS_JSON
from .splits import assign_split


def load_tensor(path: str = V1_TENSOR):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


_K_MOM = 120                                 # 动量回看(bar)，与 eval/report.py 量 novelty 用的因子一致


def _factor_resid(y, valid, fac):
    """y[N] 对 fac[N,K] 横截面 OLS 残差(valid 内)：剔掉已知因子能解释的部分，只留 novelty。"""
    m = valid & np.isfinite(y) & np.all(np.isfinite(fac), axis=1)
    out = y.copy()
    if m.sum() < 20:
        return out
    X = np.column_stack([np.ones(m.sum()), fac[m]])
    beta, *_ = np.linalg.lstsq(X, y[m], rcond=None)
    out[m] = y[m] - X @ beta
    return out


class PanelDataset:
    def __init__(self, tensor, splits, split="train",
                 T=T_WINDOW, dt_min=DT_MIN, dt_max=DT_MAX, seed=0, residualize=False):
        self.n = tensor["n"]                 # [D,N,5] 归一输入
        self.mask = tensor["mask"]           # [D,N] bool
        self.scale = tensor["scale_ret"]     # [D,N] 收益尺度
        self.adj = tensor["adj_close"]       # [D,N] 拆股还原收盘（算 label）
        self.residualize = residualize       # novelty-聚焦：目标对已知因子残差化
        self.dates = tensor["dates"]
        self.D, self.N = self.mask.shape
        self.T, self.dt_min, self.dt_max = T, dt_min, dt_max
        self.rng = np.random.default_rng(seed)
        # 合法 anchor：有足够历史、有 label 空间、属于该 split、当天有观测
        lo, hi = T - 1, self.D - dt_max - 1
        idx = np.arange(lo, hi + 1)
        in_split = np.array([assign_split(self.dates[a], splits) == split for a in idx])
        has_obs = self.mask[idx].any(1)
        # embargo：标签窗末端 a+dt_max 跨进下一段则剔除（防边界标签前视泄漏）
        no_spill = np.array([assign_split(self.dates[a + dt_max], splits) == split for a in idx])
        self.anchors = idx[in_split & has_obs & no_spill]

    def __len__(self):
        return len(self.anchors)

    def _novelty_target(self, a, y, ymask):
        """标签 y 对 6 个已知因子(rev1/rev5/mom120/vol/nvol/dvol，全 OHLCV 衍生)横截面残差化。
        训练目标 = report.py 量的 novelty → 逼模型全容量挖"反转之外"，别再退回复刻反转。"""
        with np.errstate(divide="ignore", invalid="ignore"):
            lc = np.log(np.where(self.adj[a] > 0, self.adj[a], np.nan))
            l1 = np.log(np.where(self.adj[a - 1] > 0, self.adj[a - 1], np.nan))
            l5 = np.log(np.where(self.adj[a - 5] > 0, self.adj[a - 5], np.nan))
            lk = (np.log(np.where(self.adj[a - _K_MOM] > 0, self.adj[a - _K_MOM], np.nan))
                  if a - _K_MOM >= 0 else np.full(self.N, np.nan))
        rev1 = -(lc - l1); rev5 = -(lc - l5); mom = lc - lk
        vol = self.scale[a]
        nvol = self.n[a, :, 4]; dvol = nvol - self.n[a - 1, :, 4]
        fac = np.column_stack([rev1, rev5, mom, vol, nvol, dvol])
        return _factor_resid(y, ymask, fac).astype(np.float32)

    def __getitem__(self, i):
        a = int(self.anchors[i])
        dt = int(self.rng.integers(self.dt_min, self.dt_max + 1))
        x = self.n[a - self.T + 1: a + 1]            # [T,N,5] 因果窗口
        xmask = self.mask[a - self.T + 1: a + 1]     # [T,N]
        # label：a→a+dt，仅两端都观测的 slot 有效（同 slot 同身份，天然不跨身份）
        valid = self.mask[a] & self.mask[a + dt]
        ca, cb = self.adj[a], self.adj[a + dt]
        with np.errstate(divide="ignore", invalid="ignore"):
            ret = np.log(np.where((ca > 0) & (cb > 0), cb / ca, 1.0))
        s = self.scale[a]
        y = np.where(valid & (s > 0), ret / np.where(s > 0, s, 1.0), 0.0).astype(np.float32)
        ymask = (valid & (s > 0)).astype(bool)
        if self.residualize:                         # novelty-聚焦：目标剔掉已知因子，逼模型只挖高阶
            y = self._novelty_target(a, y, ymask)
        return {
            "x": x.astype(np.float32),               # [T,N,5]
            "xmask": xmask.astype(bool),             # [T,N]
            "dt": np.float32(dt),                    # 标量
            "y": y,                                  # [N] 尺度归一 Δt 收益
            "ymask": ymask,                          # [N] 哪些 slot 是有效 target
            "scale": s.astype(np.float32),           # [N] 收益尺度（eval 还原用）
            "anchor": np.int64(a),
        }
