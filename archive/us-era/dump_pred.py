"""导出模型在 holdout 的**原始输出**(不只标量 IC):每个 anchor×资产的 signal=分布均值 + spread=(q84-q16)。
配合本地张量(算反转因子+真实收益)解剖：模型到底是不是纯反转、残差有没有 IC、分布校不校准。
跑(pod)：python dump_pred.py --tensor data/clean/crypto_tensor_60min.npz --splits data/clean/crypto_splits_60min.json --ckpt ckpt_e4.pt --dt 5 --out crypto_pred.npz
"""
import argparse
import numpy as np
import torch

from eval import harness
from eval.crps import QUANTILE_LEVELS
from model.train import Net, TorchPredictor

_LO = int(np.argmin(np.abs(QUANTILE_LEVELS - 0.16)))
_HI = int(np.argmin(np.abs(QUANTILE_LEVELS - 0.84)))


def main(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tensor, splits = harness.load(a.tensor, a.splits)
    C = tensor["n"].shape[-1]
    net = Net(d=a.d, L=a.L, T=a.T, C=C).to(dev)
    net.load_state_dict(torch.load(a.ckpt, map_location=dev))
    model = TorchPredictor(net, dev, T=a.T)

    anchors = harness.anchors_for(tensor, splits, a.split, a.T, dt_max=a.dt)
    sig = np.zeros((len(anchors), tensor["n"].shape[1]), np.float32)
    spr = np.zeros_like(sig)
    B = 256
    for s in range(0, len(anchors), B):
        ab = anchors[s:s + B]
        x, xmask = harness.build_windows(tensor, ab, a.T)
        pred = model.predict(x, xmask, np.full(len(ab), a.dt))     # [b,N,K]
        sig[s:s + len(ab)] = pred.mean(-1)
        spr[s:s + len(ab)] = (pred[..., _HI] - pred[..., _LO]) / 2
    np.savez_compressed(a.out, anchors=anchors, signal=sig, spread=spr,
                        dates=tensor["dates"][anchors], dt=a.dt)
    print(f"导出 {a.out}: {sig.shape} anchors, signal/spread；dt={a.dt}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensor", required=True); ap.add_argument("--splits", required=True)
    ap.add_argument("--ckpt", required=True); ap.add_argument("--out", default="pred_dump.npz")
    ap.add_argument("--split", default="holdout"); ap.add_argument("--dt", type=int, default=5)
    ap.add_argument("--T", type=int, default=64); ap.add_argument("--d", type=int, default=256); ap.add_argument("--L", type=int, default=4)
    main(ap.parse_args())
