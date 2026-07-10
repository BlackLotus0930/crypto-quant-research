"""不变量 / 泄漏自动检查：抓"不报错但真有害"的隐性 bug（如 accum-LR 退火 bug）。
方法论：任何"不该变却变了 / 该变却没变"都是 bug 的指纹。
  1. 张量卫生：mask=True 处无 NaN/inf；tod∈[-1,1]；adj_close/scale_ret 有限正。
  2. 因果归一审计（最重要）：扰动某币的**未来** bar → 重算归一 → 该币**过去**的归一值必须逐位不变；
     变了 = 归一用到了未来 = 泄漏（虚高一切结果）。
  3. 资产置换等变（需 --ckpt）：打乱币顺序 → 模型预测必须同样置换；不等变 = 有 ticker/位置泄漏。
跑(本地,管线检查)：PYTHONUTF8=1 .venv/Scripts/python.exe check_invariants.py --tensor data/clean/crypto_tensor_60min.npz --parquet data/clean/crypto_60min.parquet
跑(pod,加架构检查)：python check_invariants.py --tensor ... --parquet ... --ckpt ckpt_e4.pt
"""
import argparse
import numpy as np

FAIL = []


def ok(name, cond, detail=""):
    tag = "✅" if cond else "❌"
    print(f"  {tag} {name}{('  '+detail) if detail else ''}", flush=True)
    if not cond:
        FAIL.append(name)


def tensor_sanity(tpath):
    print(f"[1] 张量卫生 {tpath}", flush=True)
    Z = np.load(tpath, allow_pickle=True)
    n, mask = Z["n"], Z["mask"]
    C = n.shape[-1]
    nm = n[mask]                                                  # 只看可交易格
    ok("mask=True 处无 NaN/inf", bool(np.isfinite(nm).all()),
       f"非有限 {int((~np.isfinite(nm)).sum())} 个")
    tod = n[..., C - 4:]                                          # 末 4 通道=tod sin/cos
    ok("tod∈[-1,1]", float(np.abs(tod).max()) <= 1.001, f"max|tod|={np.abs(tod).max():.4f}")
    ac = Z["adj_close"]
    ok("adj_close[mask]>0", bool((ac[mask] > 0).all()), f"非正 {int((ac[mask] <= 0).sum())} 个")
    sr = Z["scale_ret"]
    ok("scale_ret[mask] 有限正", bool(np.isfinite(sr[mask]).all() and (sr[mask] > 0).all()))
    ok("mask 真实率合理(0.05~0.95)", 0.05 < float(mask.mean()) < 0.95, f"={mask.mean():.3f}")


def causal_norm_audit(parquet, slow_bars=1440):
    print(f"[2] 因果归一审计 {parquet}（扰动未来→过去归一须不变）", flush=True)
    import pandas as pd
    from build_crypto_tensor import normalize_causal
    df = pd.read_parquet(parquet)
    cnt = df.groupby("symbol").size()
    sym = cnt[cnt > 3 * slow_bars].index[0]                       # 取历史够长的币
    sub = df[df["symbol"] == sym].sort_values("ts").reset_index(drop=True)
    t0 = len(sub) * 2 // 3
    n1 = normalize_causal(sub.copy(), slow_bars).reset_index(drop=True)
    pert = sub.copy()
    fut = pert.index > t0                                         # 篡改未来 bar
    pert.loc[fut, ["open", "high", "low", "close"]] *= 5.0
    pert.loc[fut, ["volume", "qv", "count", "tbv"]] *= 7.0
    n2 = normalize_causal(pert, slow_bars).reset_index(drop=True)
    cols = [c for c in ("n_close", "n_volume", "n_count", "n_avgsz", "n_tbr") if c in n1.columns]
    a = n1.loc[:t0, cols].to_numpy(); b = n2.loc[:t0, cols].to_numpy()
    md = float(np.nanmax(np.abs(a - b))) if a.size else 0.0
    ok(f"过去({sym} ≤{t0}) 归一不受未来影响", md < 1e-9, f"maxdiff={md:.2e}")


def equivariance(tpath, ckpt, T=64, d=256, L=4, dt=3):
    print(f"[3] 资产置换等变 ckpt={ckpt}（打乱币顺序→预测须同样置换）", flush=True)
    import torch
    from eval import harness
    from model.train import Net, TorchPredictor
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tensor, splits = harness.load(tpath, tpath.replace("tensor", "splits").replace(".npz", ".json"))
    C = tensor["n"].shape[-1]
    sd = torch.load(ckpt, map_location=dev)
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}                # 兼容 compile 存的前缀
    net = Net(d=d, L=L, T=T, C=C).to(dev); net.load_state_dict(sd)
    model = TorchPredictor(net, dev, T=T)
    anchors = harness.anchors_for(tensor, splits, "holdout", T, dt_max=dt)[:8]
    x, xmask = harness.build_windows(tensor, anchors, T)
    p0 = model.predict(x, xmask, np.full(len(anchors), dt))      # [b,N,K]
    rng = np.random.default_rng(0); perm = rng.permutation(x.shape[2])
    p1 = model.predict(x[:, :, perm], xmask[:, :, perm], np.full(len(anchors), dt))
    md = float(np.nanmax(np.abs(p1 - p0[:, perm])))
    ok("打乱资产→预测同样置换", md < 1e-3, f"maxdiff={md:.2e}（仅看 mask 内更严，>1e-3 疑位置泄漏）")


def main(a):
    print("=" * 60 + "\n不变量/泄漏检查\n" + "=" * 60, flush=True)
    tensor_sanity(a.tensor)
    if a.parquet:
        causal_norm_audit(a.parquet)
    if a.ckpt:
        equivariance(a.tensor, a.ckpt)
    print("=" * 60)
    print(("❌ 失败: " + ", ".join(FAIL)) if FAIL else "✅ 全部通过", flush=True)
    return 1 if FAIL else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensor", required=True)
    ap.add_argument("--parquet", default=None, help="原始面板（做因果归一审计）")
    ap.add_argument("--ckpt", default=None, help="已训模型（做置换等变检查，需 GPU/CPU torch）")
    import sys
    sys.exit(main(ap.parse_args()))
