"""加载 ckpt 在 holdout 上 dump 原始预测（[days,N,K] 分位 + R + tradeable），供离线测各种建仓法。
不重训，只一次前向。跑：python dump_signal.py [--tensor path] [--ckpt path] [--out path]
"""
import argparse
import numpy as np
import torch

from eval import harness, backtest
from model.train import Net, TorchPredictor

ap = argparse.ArgumentParser()
ap.add_argument("--tensor", default=None)
ap.add_argument("--ckpt", default="ckpt_e3.pt")
ap.add_argument("--out", default="holdout_pred.npz")
a = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
tensor, splits = harness.load(a.tensor) if a.tensor else harness.load()
net = Net(d=256, L=4, T=64).to(dev)
net.load_state_dict(torch.load(a.ckpt, map_location=dev))
model = TorchPredictor(net, dev, T=64)

vkey = "scale_ret" if "scale_ret" in tensor else ("scale" if "scale" in tensor else None)
print("tensor keys:", list(tensor.keys()), "| vol key:", vkey)
days = harness.anchors_for(tensor, splits, "holdout", 64, dt_max=1)
P, R, dates, M, V = [], [], [], [], []
for s in range(0, len(days), 128):
    ab = days[s:s + 128]
    x, xmask = harness.build_windows(tensor, ab, 64)
    P.append(model.predict(x, xmask, np.ones(len(ab))))
    ca, cb = tensor["adj_close"][ab], tensor["adj_close"][ab + 1]
    valid = tensor["mask"][ab] & tensor["mask"][ab + 1]
    R.append(np.where(valid & (ca > 0), cb / np.where(ca > 0, ca, 1) - 1, 0.0))
    M.append(tensor["mask"][ab].astype(float))
    V.append(tensor[vkey][ab] if vkey else np.ones((len(ab), x.shape[1])))
    dates.append(tensor["dates"][ab])
np.savez_compressed(a.out,
                    pred=np.concatenate(P), R=np.concatenate(R), vol=np.concatenate(V),
                    mask=np.concatenate(M), dates=np.concatenate(dates))
print(f"saved {a.out}", np.concatenate(P).shape)
