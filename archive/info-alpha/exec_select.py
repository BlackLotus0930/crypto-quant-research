"""干净执行选参：cadence×rebal×band 在 **val** 上选(按 val 净@cost),固定后应用到 **holdout** 报。
不偷看 holdout(diag_freq 是 holdout 偷看的诊断;这个才是干净 GO 数)。
回答:E14 慢信号配执行,扣 taker 成本、不偷看,净 Sharpe 到底过不过 0.5。
跑(pod)：python exec_select.py --ckpt ckpt_e4_resid.pt --tensor data/clean/crypto_tensor_60min.npz --splits data/clean/crypto_splits_60min.json --dt 5 --cost 5
"""
import argparse
import numpy as np
import torch

from eval import harness, backtest
from model.train import Net, TorchPredictor

CADENCES = [1, 2, 3, 4, 6, 12, 24, 72]
REBALS = [1.0, 0.5, 0.3, 0.2, 0.1]
BANDS = [0.0, 0.003, 0.01]


def sim(W, R, cadence, rebal, band, bps, ann):
    held = np.zeros(W.shape[1]); pnl = np.empty(len(W)); turn = np.empty(len(W))
    for t in range(len(W)):
        if t % cadence == 0:
            d = W[t] - held; mv = np.abs(d) > band
            new = held.copy(); new[mv] = held[mv] + rebal * d[mv]
            g = np.abs(new).sum(); new = new / g if g > 0 else new
            turn[t] = np.abs(new - held).sum(); held = new
        else:
            turn[t] = 0.0
        pnl[t] = (held * R[t]).sum()
    net = pnl - bps / 1e4 * turn; sd = net.std()
    return (net.mean() / sd * np.sqrt(ann) if sd > 0 else 0.0, float(turn.mean() * ann))


def main(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tensor, splits = harness.load(a.tensor, a.splits)
    C = tensor["n"].shape[-1]
    sd = torch.load(a.ckpt, map_location=dev); sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    net = Net(d=a.d, L=a.L, T=a.T, C=C).to(dev); net.load_state_dict(sd)
    model = TorchPredictor(net, dev, T=a.T)
    ann = int(tensor["ann"]) if "ann" in tensor else 8760
    vb = backtest.run(tensor, splits, model, split="val", T=a.T, pred_dt=a.dt)
    hb = backtest.run(tensor, splits, model, split="holdout", T=a.T, pred_dt=a.dt)
    Wv, Rv = vb["W"].astype(np.float64), vb["R"].astype(np.float64)
    Wh, Rh = hb["W"].astype(np.float64), hb["R"].astype(np.float64)
    print(f"val {len(Wv)} bar / holdout {len(Wh)} bar, N={Wv.shape[1]}, ann={ann}, dt={a.dt}, 选参成本={a.cost}bps\n", flush=True)

    best = None
    for cad in CADENCES:
        for rb in REBALS:
            for bd in BANDS:
                sv, _ = sim(Wv, Rv, cad, rb, bd, a.cost, ann)         # 只用 val 选
                if best is None or sv > best[0]:
                    best = (sv, cad, rb, bd)
    sv, cad, rb, bd = best
    print(f"val 选定: 每{cad}bar 调仓, rebal={rb}, band={bd}  (val 净@{a.cost}bps={sv:+.2f})")
    print(f"\n→ 应用到 holdout(不偷看):")
    print(f"{'bps':>5s} {'净Sharpe':>9s} {'年换手':>8s}")
    for bps in (2, 5, 10):
        sh, to = sim(Wh, Rh, cad, rb, bd, bps, ann)
        print(f"{bps:>5d} {sh:>+9.2f} {to:>8.0f}")
    sh5, _ = sim(Wh, Rh, cad, rb, bd, 5, ann)
    print(f"\n判读: holdout 净@5bps={sh5:+.2f} —— ≥0.5 = 干净过 taker 门(§0.2 gate);<0.5 但正 = 接近;负 = 执行救不回。", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--tensor", required=True); ap.add_argument("--splits", required=True)
    ap.add_argument("--dt", type=int, default=5); ap.add_argument("--cost", type=float, default=5)
    ap.add_argument("--T", type=int, default=64); ap.add_argument("--d", type=int, default=256); ap.add_argument("--L", type=int, default=4)
    main(ap.parse_args())
