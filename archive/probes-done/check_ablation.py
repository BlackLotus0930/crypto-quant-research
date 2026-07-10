"""验零输入 ablation 到底是泄漏还是 IR 灌水 artifact。
run_gonogo 的 ablation 只打了年化 IR;IR=mean/std×√ann 对"近常数信号(std极小)"会爆。
这里直接看横截面 **IC 的量级**:|mean_IC|<0.001=artifact(可忽略);≥0.005=真泄漏要查。
跑(pod)：python check_ablation.py --tensor data/clean/crypto_tensor_60min.npz --splits data/clean/crypto_splits_60min.json --ckpt ckpt_e4_resid.pt --dt 5
"""
import argparse
import numpy as np
import torch

from eval import harness
from model.train import Net, TorchPredictor


def main(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tensor, splits = harness.load(a.tensor, a.splits)
    C = tensor["n"].shape[-1]
    sd = torch.load(a.ckpt, map_location=dev)
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    net = Net(d=a.d, L=a.L, T=a.T, C=C).to(dev); net.load_state_dict(sd)
    model = TorchPredictor(net, dev, T=a.T)
    anchors = harness.anchors_for(tensor, splits, "holdout", a.T, dt_max=a.dt)
    real_ics, zero_ics, zero_stds = [], [], []
    for s in range(0, len(anchors), 128):
        ab = anchors[s:s + 128]
        x, xm = harness.build_windows(tensor, ab, a.T)
        y, ym = harness.labels_at(tensor, ab, a.dt)
        dtv = np.full(len(ab), a.dt)
        for tag, xin, acc in (("real", x, real_ics), ("zero", np.zeros_like(x), zero_ics)):
            sig = model.predict(xin, xm, dtv).mean(-1)
            for b in range(len(ab)):
                m = ym[b]
                if m.sum() < 10:
                    continue
                sv, yv = sig[b, m], y[b, m]
                if tag == "zero":
                    zero_stds.append(float(sv.std()))
                if sv.std() > 1e-12 and yv.std() > 1e-12:
                    acc.append(float(np.corrcoef(sv, yv)[0, 1]))
    for tag, ics in (("真实输入", real_ics), ("零输入", zero_ics)):
        a_ = np.array(ics)
        if len(a_):
            print(f"{tag}: 有效bar={len(a_)} mean_IC={a_.mean():+.6f} std_IC={a_.std():.6f} "
                  f"t={a_.mean()/a_.std()*np.sqrt(len(a_)):+.2f}")
    zs = np.array(zero_stds)
    print(f"零输入信号的横截面 std(均值)={zs.mean():.2e}  (越小=输出越近常数→IR 越易灌水)")
    z = np.array(zero_ics)
    verdict = "✅ artifact(零输入 IC 量级可忽略,IR 是小方差灌水)" if len(z) and abs(z.mean()) < 0.001 \
        else "❌ 真泄漏(零输入仍有显著 IC)" if len(z) and abs(z.mean()) >= 0.005 else "🟡 介于之间,看量级"
    print("判读:", verdict)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensor", required=True); ap.add_argument("--splits", required=True)
    ap.add_argument("--ckpt", required=True); ap.add_argument("--dt", type=int, default=5)
    ap.add_argument("--T", type=int, default=64); ap.add_argument("--d", type=int, default=256); ap.add_argument("--L", type=int, default=4)
    main(ap.parse_args())
