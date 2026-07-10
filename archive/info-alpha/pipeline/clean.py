"""阶段 1-5：删错 / 身份切分 / 拆股还原 / 质量标 / （对齐+mask 在 build 里做）。

只做机械数据卫生——不算因子、不注入人类观点。真实极端波动一律保留。
"""
from math import gcd

import numpy as np
import pandas as pd

from .config import (
    OHLC_TOL, GAP_THRESH, MIN_SEG_LEN,
    SPLIT_TOL_CLOSE, SPLIT_TOL_OPEN, SPLIT_MIN_MOVE, SPLIT_MAX_K, MAX_SPLITS_PER_SEG,
    QUALITY_MIN_MED_VOL,
)


# ---------- 阶段 1：删错 ----------
def dedup_dates(panel: pd.DataFrame) -> pd.DataFrame:
    """同 (symbol,date) 去重，保留最后一条。"""
    return (panel.sort_values(["symbol", "date"])
                 .drop_duplicates(["symbol", "date"], keep="last")
                 .reset_index(drop=True))


def drop_bad_rows(panel: pd.DataFrame) -> pd.DataFrame:
    """删可证明的记录错误：非正价、high<low、OHLC 不自洽。真实极端值保留。"""
    o, h, l, c, v = (panel[x].to_numpy(float) for x in ("open", "high", "low", "close", "volume"))
    good = (o > 0) & (h > 0) & (l > 0) & (c > 0) & (v >= 0)
    good &= h >= l * (1 - OHLC_TOL)
    good &= h >= np.maximum(o, c) * (1 - OHLC_TOL)
    good &= l <= np.minimum(o, c) * (1 + OHLC_TOL)
    return panel[good].reset_index(drop=True)


# ---------- 阶段 2：身份切分 ----------
def segment_identities(panel: pd.DataFrame, pos: dict) -> pd.DataFrame:
    """大空洞(>GAP_THRESH 交易日)处把 ticker 切成独立 segment，防退市重上市/ticker复用拼接。
    加列 pos(日历索引)、seg、seg_id。丢弃 < MIN_SEG_LEN 的段。"""
    df = panel.copy()
    df["pos"] = df["date"].map(pos)
    df = df.dropna(subset=["pos"])
    df["pos"] = df["pos"].astype(int)
    df = df.sort_values(["symbol", "pos"]).reset_index(drop=True)

    prev = df.groupby("symbol", sort=False)["pos"].shift(1)
    gap = df["pos"] - prev                      # 连续=1；中间缺 gap-1 天
    new_seg = gap.isna() | (gap > GAP_THRESH + 1)
    df["seg"] = new_seg.groupby(df["symbol"], sort=False).cumsum().astype(int) - 1
    df["seg_id"] = df["symbol"] + "#" + df["seg"].astype(str)

    sizes = df.groupby("seg_id", sort=False)["pos"].transform("size")
    return df[sizes >= MIN_SEG_LEN].reset_index(drop=True)


# ---------- 阶段 3：拆股还原 ----------
def split_candidates(max_k: int = SPLIT_MAX_K) -> np.ndarray:
    """**保守**的常见拆股比例集合。只收 ≥3:2 的明确比例，**故意不收 0.75~0.9 这段
    密集分数**——那一段会和真实 ±20~30% 财报跳混淆，宁可漏掉罕见小拆股(残留成离群
    值、和真实大动无异)，也不要误判抹掉真实收益。正向 k:1 + 3:2，及其反向。"""
    fwd = [1 / 2, 1 / 3, 1 / 4, 1 / 5, 1 / 6, 1 / 7, 1 / 8, 1 / 10, 2 / 3]
    fwd = [r for r in fwd if 1 / r <= max_k or r >= 1 / max_k]
    s = set(fwd) | {1 / r for r in fwd}        # 加反向拆
    return np.array(sorted(s))


_CANDS = split_candidates()


def _adjust_one(g: pd.DataFrame) -> pd.DataFrame:
    """单 segment：检测拆股指纹 → 反推连续正价(后复权式)，volume 同步。"""
    c = g["close"].to_numpy(float)
    o = g["open"].to_numpy(float)
    n = len(c)
    F = np.ones(n)
    n_split = 0
    if n >= 2:
        with np.errstate(divide="ignore", invalid="ignore"):
            rc = np.ones(n); rc[1:] = c[1:] / c[:-1]     # close 比
            ro = np.ones(n); ro[1:] = o[1:] / c[:-1]     # open 比（相对昨收）
        big = np.abs(np.log(np.where(ro > 0, ro, 1.0))) >= SPLIT_MIN_MOVE
        err_o = np.abs(ro[:, None] - _CANDS[None, :]) / _CANDS[None, :]  # [n,K]
        j = err_o.argmin(1)                              # 用 open 比选最近的干净分数
        oerr = err_o[np.arange(n), j]
        cerr = np.abs(rc - _CANDS[j]) / _CANDS[j]
        is_split = big & (oerr < SPLIT_TOL_OPEN) & (cerr < SPLIT_TOL_CLOSE)
        n_split = int(is_split.sum())
        if n_split <= MAX_SPLITS_PER_SEG:                # 误判主导则整段不调整
            mult = np.where(is_split, _CANDS[j], 1.0)    # 拆股日=比例，否则=1
            rev = np.cumprod(mult[::-1])[::-1]           # rev[t]=prod mult[t..end]
            F[:-1] = rev[1:]                             # F[t]=prod_{s>t} mult
    out = g.copy()
    for col in ("open", "high", "low", "close"):
        out[col] = out[col].to_numpy(float) * F          # 价 ×F（旧价对齐到近期）
    out["volume"] = g["volume"].to_numpy(float) / np.where(F > 0, F, 1.0)  # 量 ÷F（成交额连续）
    out["split_factor"] = F
    out["n_splits"] = n_split                            # 检出拆股数（>MAX 的段会被质量标 drop）
    return out


def adjust_splits(panel: pd.DataFrame) -> pd.DataFrame:
    """对每个 segment 还原拆股。panel 需已切好 seg_id 且按 pos 有序。
    显式迭代分组（pandas 3.0 的 groupby.apply 会剥掉分组列，迭代不会）。"""
    parts = [_adjust_one(g) for _, g in panel.groupby("seg_id", sort=False)]
    return pd.concat(parts, ignore_index=True)


# ---------- 阶段 4：质量 ----------
def segment_quality(panel: pd.DataFrame) -> pd.DataFrame:
    """每 segment 元数据 + 质量标。只把 vol≈0(中位量<QUALITY_MIN_MED_VOL) 标 drop。"""
    dv = (panel["close"] * panel["volume"])
    q = (panel.assign(_dv=dv)
              .groupby("seg_id", sort=False)
              .agg(symbol=("symbol", "first"), category=("category", "first"),
                   n_obs=("pos", "size"), first=("date", "min"), last=("date", "max"),
                   med_vol=("volume", "median"), med_dollar_vol=("_dv", "median"),
                   n_splits=("n_splits", "first"))
              .reset_index())
    # drop：vol≈0 不可信报价；或拆股误判主导(仙股/权证，序列已不可信)
    q["drop"] = (q["med_vol"] < QUALITY_MIN_MED_VOL) | (q["n_splits"] > MAX_SPLITS_PER_SEG)
    return q
