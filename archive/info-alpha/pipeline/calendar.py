"""阶段 0：稳健美国交易日历。

单一时间网格，所有资产对齐到它。用"当日 >= CAL_MIN_SYMBOLS 只交易的日期"
过滤掉个别标的数据里的乌龙节假日（否则会让所有人都看着像有空洞）。
"""
import numpy as np
import pandas as pd

from .config import CAL_MIN_SYMBOLS


def build_calendar(panel: pd.DataFrame, min_symbols: int = CAL_MIN_SYMBOLS):
    """panel 含列 date(str)。返回 (cal: np.ndarray[str] 升序, pos: dict date->idx)。"""
    counts = panel.groupby("date")["symbol"].size()
    cal = np.sort(counts.index[counts >= min_symbols].to_numpy())
    pos = {d: i for i, d in enumerate(cal)}
    return cal, pos
