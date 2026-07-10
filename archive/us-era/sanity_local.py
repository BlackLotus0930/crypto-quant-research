"""本地 CPU 小 sanity：确认 tod 张量(C=7) 端到端前向+反向不崩、机器在学。不训练、不上卡。"""
import numpy as np
from eval import harness
from model.train import sanity_overfit

tensor, splits = harness.load("data/clean/intraday_tensor_60min_tod.npz",
                              "data/clean/intraday_splits_60min.json")
C = tensor["n"].shape[-1]
print(f"张量 C={C}  N={tensor['n'].shape[1]}  T_total={tensor['n'].shape[0]}  mask={tensor['mask'].mean():.3f}")
assert C == 7, "tod 张量应是 7 通道"

# 极小、极快：小网络、小子集、CPU
f, l = sanity_overfit(tensor, splits, "cpu", T=16, n=32, steps=25, d=64, L=2, bs=4, C=C)
print(f"sanity overfit  {f:.4f} -> {l:.4f}  ({'✓ 在学(C=7 前向OK)' if l < f * 0.7 else '✗ 没在学'})")
