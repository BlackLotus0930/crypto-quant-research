"""受控实验：同一个模型(ckpt_e3.pt)，只换"仓位定大小方式"，看 holdout 回测。
诊断：方向信号(IC)稳健，但 P&L 之前靠 conviction=signal/spread，而 spread 来自没校准的分布头。
对照 conviction vs signal(纯方向 z-score) vs rank(纯方向多空) → 看是不是 sizing 把 alpha 毁了。
跑（pod）： python rebook.py
"""
import sys
import numpy as np
import torch

from eval import harness, backtest
from model.train import Net, TorchPredictor


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tensor, splits = harness.load()
    net = Net(d=256, L=4, T=64).to(dev)
    net.load_state_dict(torch.load("ckpt_e3.pt", map_location=dev))
    model = TorchPredictor(net, dev, T=64)

    for book in ("conviction", "signal", "rank", "riskeq"):
        print(f"\n##### 仓位方式 = {book} #####", flush=True)
        bt = backtest.run(tensor, splits, model, split="holdout", T=64, book=book)
        backtest.report(bt, name=book)
        backtest.sweep_execution(bt)

    # 存 holdout 原始预测 → 以后调 book/执行/成本完全离线，免再占 GPU
    from eval.harness import anchors_for, build_windows
    days = anchors_for(tensor, splits, "holdout", 64, dt_max=1)
    preds = []
    for s in range(0, len(days), 128):
        ab = days[s:s + 128]
        x, xmask = build_windows(tensor, ab, 64)
        preds.append(model.predict(x, xmask, np.ones(len(ab))))
    np.savez_compressed("holdout_pred.npz",
                        pred=np.concatenate(preds),
                        vol=tensor["scale_ret"][days],
                        adj0=tensor["adj_close"][days], adj1=tensor["adj_close"][days + 1],
                        mask0=tensor["mask"][days], mask1=tensor["mask"][days + 1],
                        dates=tensor["dates"][days])
    print("\n已存 holdout_pred.npz（pred/vol/价/mask）→ 调 sizing 免重训", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
