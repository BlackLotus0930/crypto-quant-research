"""在线/滚动适应(walk-forward)：直接打 #1 问题=非平稳(val→holdout 塌)。
每隔 chunk 天,用"截至当下的全部数据"重训一个模型,预测接下来 chunk 天,往前滚。
模型永远只见 ≤ 预测点的数据(因果)、且始终最新 regime → 测"始终最新"能不能消掉那个崩。
判据:① 每块 holdout IC 是否保持在 val 水平(不塌) ② 拼起来的 book 净 Sharpe。
对照:静态训练 holdout IC ~0.015(崩自 val ~0.03)。若滚动 IC ~0.025-0.03=适应有效。
跑(pod)：python walk_forward.py --tensor data/clean/crypto_tensor_60min.npz --splits data/clean/crypto_splits_60min.json --chunk 180 --epochs 4 --dt 3
"""
import argparse
from datetime import datetime, timedelta

import numpy as np
import torch

from eval import harness, backtest
from eval.ic import ic_table, print_ic
from model.train import fit, TorchPredictor

GLOBAL_START = "2020-01-01"


def addd(dstr, days):
    return (datetime.strptime(str(dstr)[:10], "%Y-%m-%d") + timedelta(days=days)).strftime("%Y-%m-%d")


def main(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tensor, splits = harness.load(a.tensor, a.splits)
    C = tensor["n"].shape[-1]
    h_start = splits["holdout"][0][:10]
    last = str(tensor["dates"][-1])[:10]
    ann = int(tensor["ann"]) if "ann" in tensor else 8760
    # 滚动重训点：holdout 起点开始,每 chunk 天一个
    pts = []
    t = h_start
    while t < last:
        pts.append(t); t = addd(t, a.chunk)
    print(f"滚动适应：holdout {h_start}~{last}，{len(pts)} 个重训点(每{a.chunk}天)，每点训≤当下、epochs={a.epochs}、dt={a.dt}", flush=True)

    chunk_rows = []; Ws = []; Rs = []; Ds = []
    for i, t_r in enumerate(pts):
        sr = {"train": [GLOBAL_START, addd(t_r, -a.val_days)],
              "val": [addd(t_r, -a.val_days), t_r],
              "holdout": [t_r, addd(t_r, a.chunk)]}
        print(f"\n[{i+1}/{len(pts)}] 重训点 {t_r}：train≤{sr['train'][1]} val[{sr['val'][0]}~{t_r}] 预测[{t_r}~{sr['holdout'][1]}]", flush=True)
        try:
            net, vic = fit(tensor, sr, dev, T=a.T, d=a.d, L=a.L, bs=a.bs, lr=a.lr, epochs=a.epochs,
                           dt_min=a.dt_min, dt_max=a.dt_max, accum=a.accum, lam=a.lam, compile_model=a.compile, C=C, workers=a.workers)
        except Exception as e:
            print(f"  ⚠️ 该点训练失败,跳过: {e}", flush=True); continue
        model = TorchPredictor(net, dev, T=a.T)
        ictbl = ic_table(tensor, sr, "holdout", model, dt=a.dt, T=a.T)
        _, bic = print_ic(ictbl, dt=a.dt)
        bt = backtest.run(tensor, sr, model, split="holdout", T=a.T, pred_dt=a.dt)
        chunk_rows.append((t_r, bic["IC"], bic["IR"], vic, len(bt["W"])))
        Ws.append(bt["W"]); Rs.append(bt["R"]); Ds.append(bt["dates"])
        print(f"  → 本块 holdout IC={bic['IC']:+.4f} (该点 val_IC={vic:+.4f})", flush=True)

    print("\n" + "=" * 64 + "\n[滚动适应汇总]\n" + "=" * 64)
    print(f"{'重训点':>12s} {'val_IC':>8s} {'块holdoutIC':>11s} {'IR':>7s}")
    tot_ic = []; w = []
    for t_r, ic, ir, vic, n in chunk_rows:
        print(f"{t_r:>12s} {vic:>+8.4f} {ic:>+11.4f} {ir:>+7.2f}")
        tot_ic.append(ic); w.append(n)
    if tot_ic:
        wic = float(np.average(tot_ic, weights=w))
        wvic = float(np.average([r[3] for r in chunk_rows], weights=w))
        print(f"\n加权 holdout IC = {wic:+.4f}  (各点 val_IC 加权 = {wvic:+.4f})")
        print(f"→ 对照静态 holdout IC ~0.015(崩自 val ~0.03)。滚动 IC 若 ≥0.025/接近 val = 适应消掉了崩。")
        # 拼接 book → 净 Sharpe(每4bar 调仓,粗看)
        W = np.concatenate(Ws); R = np.concatenate(Rs)
        for cad in (1, 4, 24):
            held = np.zeros(W.shape[1]); pnl = np.empty(len(W)); to = np.empty(len(W))
            for tt in range(len(W)):
                if tt % cad == 0:
                    new = held + 0.3 * (W[tt] - held); g = np.abs(new).sum(); new = new / g if g > 0 else new
                    to[tt] = np.abs(new - held).sum(); held = new
                else:
                    to[tt] = 0.0
                pnl[tt] = (held * R[tt]).sum()
            net5 = pnl - 5 / 1e4 * to; sd = net5.std()
            sh = net5.mean() / sd * np.sqrt(ann) if sd > 0 else 0.0
            print(f"   拼接 book 每{cad}bar 净@5bps Sharpe={sh:+.2f} 年换手={to.mean()*ann:.0f}")
        np.savez_compressed("walkforward_book.npz", W=W, R=R, dates=np.concatenate(Ds))
        print("  已存 walkforward_book.npz(可 exec_select 干净选参)")
    print("=" * 64, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensor", required=True); ap.add_argument("--splits", required=True)
    ap.add_argument("--chunk", type=int, default=180, help="每块天数(重训频率)")
    ap.add_argument("--val_days", type=int, default=60, help="重训时留作早停的 val 尾巴天数")
    ap.add_argument("--epochs", type=int, default=4); ap.add_argument("--dt", type=int, default=3)
    ap.add_argument("--dt_min", type=int, default=1); ap.add_argument("--dt_max", type=int, default=5)
    ap.add_argument("--T", type=int, default=64); ap.add_argument("--d", type=int, default=256); ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--bs", type=int, default=32); ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--lam", type=float, default=0.2)
    ap.add_argument("--accum", type=int, default=1); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--compile", action="store_true")
    main(ap.parse_args())
