"""阶段 8：防泄漏时间切分（设计 §5）。

按**时间**切 train/val/holdout——绝不 shuffle 跨时间。最终 holdout 只碰 1-2 次。
归一化已是因果(只用过去)，所以切分本身只需按 anchor 日期分桶即可，无统计泄漏。
"""
import json

from .config import TRAIN_START, TRAIN_END, VAL_END, SPLITS_JSON


def make_splits(train_start: str = TRAIN_START, train_end: str = TRAIN_END, val_end: str = VAL_END) -> dict:
    """返回 {train:[start,end], val:(end,val_end], holdout:(val_end,None]}。
    train 有下界 → 早年薄横截面被排除（标为 'pre'，不进任何桶）。"""
    return {
        "train": [train_start, train_end],
        "val": [train_end, val_end],
        "holdout": [val_end, None],
    }


def assign_split(date: str, splits: dict) -> str:
    """某 anchor 日期属于哪个桶；早于 train 起点 → 'pre'（被排除）。
    holdout 有上界时(walk-forward 每块 [t_r,t_r+chunk])，超出 → 'post'（被排除），
    防每块评估到数据末尾、块间重叠（普通运行 holdout 上界=None → 开区间，行为不变）。"""
    start = splits["train"][0]
    if start is not None and date < start:
        return "pre"
    if date <= splits["train"][1]:
        return "train"
    if date <= splits["val"][1]:
        return "val"
    hi = splits["holdout"][1] if len(splits.get("holdout", [])) > 1 else None
    if hi is not None and date > hi:
        return "post"
    return "holdout"


def save_splits(splits: dict, path: str = SPLITS_JSON):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)


def walk_forward(cal, n_folds: int = 5, min_train: int = 252 * 8):
    """walk-forward 折生成器（P3/P4 用）：每折 train=起点→t，test=t→t+step。
    cal: 升序日历(np.ndarray[str])。yield (train_end_idx, test_start_idx, test_end_idx)。"""
    n = len(cal)
    if n <= min_train:
        return
    step = (n - min_train) // n_folds
    for k in range(n_folds):
        tr_end = min_train + k * step
        te_end = min(tr_end + step, n)
        if tr_end >= n:
            break
        yield tr_end, tr_end, te_end
