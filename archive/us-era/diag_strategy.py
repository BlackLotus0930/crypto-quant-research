"""诊断：被成本绑(H1) vs 策略漏信号(H2)。
关键数 = 我们真实 book 在 rebal=1.0/band=0/0bps 的**毛 Sharpe**：
  ≈ IC年化IR(1.46) → sizing 没问题、纯成本(H1)；≪ 1.46 → sizing 接不住信号(H2，策略问题)。
跑(pod)：python diag_strategy.py
"""
import argparse
import numpy as np
import torch

from eval import harness, backtest
from model.train import Net, TorchPredictor


def main(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tensor, splits = harness.load(a.tensor, a.splits)
    net = Net(d=a.d, L=a.L, T=a.T).to(dev)
    net.load_state_dict(torch.load(a.ckpt, map_location=dev))
    model = TorchPredictor(net, dev, T=a.T)
    print(f"device={dev} ckpt={a.ckpt}  对照基准：IC年化IR≈1.46（理想满仓追信号的毛Sharpe上界）", flush=True)

    # ---- 1) 三种 sizing 的毛 Sharpe（直接验 H2：换 sizing 毛是否变高） ----
    print("\n[A] sizing 对比（pred_dt=5，holdout）：毛=rebal1.0/0bps，净=rebal0.3/2bps", flush=True)
    print(f"  {'book':>9s} {'毛@rebal1':>10s} {'毛换手':>8s} {'净@2(rb.3)':>11s} {'净@2(rb.1)':>11s}", flush=True)
    for book in ("riskeq", "signal", "rank"):
        bt = backtest.run(tensor, splits, model, split="holdout", T=a.T, pred_dt=5, book=book)
        g = backtest.metrics(bt, 0, 1.0, 0.0)
        n3 = backtest.metrics(bt, 2, 0.3, 0.0)
        n1 = backtest.metrics(bt, 2, 0.1, 0.0)
        print(f"  {book:>9s} {g['sharpe']:+10.2f} {g['turnover']:8.3f} {n3['sharpe']:+11.2f} {n1['sharpe']:+11.2f}", flush=True)

    # ---- 2) riskeq 的完整 毛→净 衰减面（rebal × band × 成本） ----
    for pdt in (1, 5):
        bt = backtest.run(tensor, splits, model, split="holdout", T=a.T, pred_dt=pdt, book="riskeq")
        g = backtest.metrics(bt, 0, 1.0, 0.0)
        print(f"\n[B] riskeq pred_dt={pdt} | 毛 Sharpe(rebal1.0/band0/0bps) = {g['sharpe']:+.2f}  "
              f"换手={g['turnover']:.3f}  年化={g['ann_return']:+.2%}", flush=True)
        print(f"  {'rebal':>6s}{'band':>7s}{'毛@0':>8s}{'净@2':>8s}{'净@5':>8s}{'换手':>8s}", flush=True)
        for rb in (1.0, 0.5, 0.3, 0.2, 0.1):
            for bd in (0.0, 0.001, 0.003):
                m0 = backtest.metrics(bt, 0, rb, bd)
                m2 = backtest.metrics(bt, 2, rb, bd)
                m5 = backtest.metrics(bt, 5, rb, bd)
                print(f"  {rb:6.2f}{bd:7.3f}{m0['sharpe']:+8.2f}{m2['sharpe']:+8.2f}"
                      f"{m5['sharpe']:+8.2f}{m0['turnover']:8.3f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensor", default="data/clean/intraday_tensor_60min.npz")
    ap.add_argument("--splits", default="data/clean/intraday_splits_60min.json")
    ap.add_argument("--ckpt", default="ckpt_e4.pt")
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--L", type=int, default=4)
    main(ap.parse_args())
