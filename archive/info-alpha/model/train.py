"""最小模型训练（torch, pod 4090, bf16 AMP）。
pinball=分布损失（复用 eval.crps levels）+ λ·均值MSE（让梯度也推方向/钱那一维）。
warmup+cosine、梯度累积、bf16 自动混合精度（省显存+提速）、按 **val IC** 选模型。
"""
import contextlib
import math
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from pipeline.dataset import PanelDataset
from eval.crps import QUANTILE_LEVELS
from .net import Encoder
from .head import QuantileHead

K = len(QUANTILE_LEVELS)
LEVELS = torch.tensor(QUANTILE_LEVELS, dtype=torch.float32)


def amp_ctx(dev):
    return torch.autocast("cuda", dtype=torch.bfloat16) if str(dev).startswith("cuda") else contextlib.nullcontext()


class Net(nn.Module):
    def __init__(self, d=256, L=4, heads=4, T=64, C=5):
        super().__init__()
        self.enc = Encoder(C=C, d=d, L=L, heads=heads, T=T)
        self.head = QuantileHead(d, K=K)

    def forward(self, x, xmask, dt):
        return self.head(self.enc(x, xmask), dt)


def pinball(pred, y, ymask, levels):
    diff = y.unsqueeze(-1) - pred
    loss = torch.maximum(levels * diff, (levels - 1) * diff).mean(-1)
    sel = loss[ymask]
    return sel.mean() if sel.numel() > 0 else pred.sum() * 0.0


def ic_loss(signal, y, ymask):
    """主损失：每行(=一天)横截面 Pearson corr，最大化 → loss = -mean(corr)。
    目标直接对齐我们要的"横截面方向排序"，而非 CRPS 那条只拉向无条件分布的梯度。
    全向量化（无 Python 循环/无 .cpu() 同步）：等价于逐行 masked Pearson。"""
    m = ymask.to(signal.dtype)                         # [B,N]
    n = m.sum(1, keepdim=True)                          # 每行有效数
    s_mean = (signal * m).sum(1, keepdim=True) / n.clamp_min(1)
    r_mean = (y * m).sum(1, keepdim=True) / n.clamp_min(1)
    sc = (signal - s_mean) * m                          # 居中并把 pad 清零（不进协方差/方差）
    rc = (y - r_mean) * m
    cov = (sc * rc).sum(1)
    ss = (sc * sc).sum(1).clamp_min(1e-12).sqrt()
    rr = (rc * rc).sum(1).clamp_min(1e-12).sqrt()
    corr = cov / (ss * rr).clamp_min(1e-8)              # [B]
    valid = ymask.sum(1) >= 5
    return -corr[valid].mean() if valid.any() else signal.sum() * 0.0


def _batch_to(b, dev):
    nb = str(dev).startswith("cuda")                   # pin_memory + non_blocking → 异步 H2D，与计算重叠
    return tuple(b[k].to(dev, non_blocking=nb) for k in ("x", "xmask", "dt", "y", "ymask"))


def _xs_ic(pred_mean, y, ymask):
    """每行(=一天)横截面 IC：Pearson + Spearman(rank-IC，对极端值稳、更贴近排序多空)。"""
    pm = pred_mean.float().detach().cpu().numpy(); yy = y.cpu().numpy(); mm = ymask.cpu().numpy()
    ic, ric = [], []
    for b in range(pm.shape[0]):
        m = mm[b]
        if m.sum() < 10:
            continue
        a, c = pm[b, m], yy[b, m]
        if a.std() > 1e-9 and c.std() > 1e-9:
            ic.append(float(np.corrcoef(a, c)[0, 1]))
            ar = a.argsort().argsort().astype(float); cr = c.argsort().argsort().astype(float)
            ric.append(float(np.corrcoef(ar, cr)[0, 1]))   # rank 上的 Pearson = Spearman
    return ic, ric


def evaluate(net, loader, dev, lv):
    """返回 (val_crps, mean_IC, IR=meanIC/stdIC, mean_rankIC)。IR 是 IC 的跨时间稳定性。"""
    net.eval(); s = c = 0.0; ics = []; rics = []
    with torch.no_grad():
        for b in loader:
            x, xmask, dt, y, ymask = _batch_to(b, dev)
            with amp_ctx(dev):
                pred = net(x, xmask, dt).float()
            diff = y.unsqueeze(-1) - pred
            loss = torch.maximum(lv * diff, (lv - 1) * diff).mean(-1)[ymask]
            s += loss.sum().item(); c += ymask.sum().item()
            i2, r2 = _xs_ic(pred.mean(-1), y, ymask)
            ics += i2; rics += r2
    mic = float(np.mean(ics)) if ics else 0.0
    ir = mic / (float(np.std(ics)) + 1e-9) if ics else 0.0
    return (s / c if c else float("nan"), mic, ir, float(np.mean(rics)) if rics else 0.0)


def sanity_overfit(tensor, splits, dev, T=64, n=64, steps=80, d=256, L=4, bs=8, C=5, residualize=False):
    full = PanelDataset(tensor, splits, "train", T=T, dt_min=1, dt_max=1, residualize=residualize)
    idx = np.random.default_rng(0).choice(len(full), min(n, len(full)), replace=False)
    loader = DataLoader(Subset(full, idx.tolist()), batch_size=bs, shuffle=True, drop_last=True)
    net = Net(d=d, L=L, T=T, C=C).to(dev); lv = LEVELS.to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    net.train(); first = last = None
    for _ in range(steps):
        for b in loader:
            x, xmask, dt, y, ymask = _batch_to(b, dev)
            opt.zero_grad()
            with amp_ctx(dev):
                loss = pinball(net(x, xmask, dt).float(), y, ymask, lv)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            if torch.isfinite(gn):
                opt.step()
            last = loss.item()
            if first is None:
                first = last
    return first, last


def fit(tensor, splits, dev, T=64, d=256, L=4, bs=32, lr=3e-4, epochs=25, patience=6,
        dt_min=1, dt_max=1, accum=1, lam=0.5, warmup_frac=0.05, compile_model=False, C=5, swa_k=5, workers=4,
        residualize=False):
    # 数据加载：训练 pod(Linux) 上开 num_workers，让数据搬运与 GPU 计算重叠（瓶颈在喂数据，不在算）。
    # worker 只在 pod 上跑、不占用户本地机器；workers=0 退回单进程（本地 Windows 调试用，避僵尸占显存）。
    pin = str(dev).startswith("cuda")
    tr_kw = dict(num_workers=workers, persistent_workers=True, prefetch_factor=4, pin_memory=pin) if workers > 0 else {}
    va_kw = dict(num_workers=min(workers, 2), persistent_workers=True, pin_memory=pin) if workers > 0 else {}
    tr = DataLoader(PanelDataset(tensor, splits, "train", T=T, dt_min=dt_min, dt_max=dt_max, residualize=residualize),
                    batch_size=bs, shuffle=True, drop_last=True, **tr_kw)
    va = DataLoader(PanelDataset(tensor, splits, "val", T=T, dt_min=dt_min, dt_max=dt_max, residualize=residualize),
                    batch_size=bs, shuffle=False, **va_kw)
    _trd = PanelDataset(tensor, splits, "train", T=T, dt_min=dt_min, dt_max=dt_max, residualize=residualize)   # train_IC 诊断
    _ridx = np.random.default_rng(0).choice(len(_trd), min(2000, len(_trd)), replace=False)  # 看 train-val gap=过拟合诊断
    tre = DataLoader(Subset(_trd, _ridx.tolist()), batch_size=bs, shuffle=False)
    net = Net(d=d, L=L, T=T, C=C).to(dev); lv = LEVELS.to(dev)
    if compile_model and str(dev).startswith("cuda"):   # 默认关：compile 子进程崩了会变僵尸占显存
        try:
            net = torch.compile(net)
        except Exception as e:
            print(f"  torch.compile 跳过: {e}", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    # total = **优化器步数**(每 accum 批走一步)。曾用 epochs*len(tr)(=批数)是 bug：accum>1 时 sch.step
    # 只被调用 1/accum 次 → cosine 只走 1/accum、LR 永不退火到低位(E9 等旧run就是这么欠退火训出来的)。
    total = epochs * max(1, len(tr) // accum); warm = max(1, int(total * warmup_frac))

    def lr_lambda(step):
        if step < warm:
            return step / warm
        p = (step - warm) / max(1, total - warm)
        return 0.5 * (1 + math.cos(math.pi * p))
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    snaps = []          # 每 epoch 存 (val_IC, cpu_state)，末了对 top-k 做权重平均(SWA)
    best_ic = -1e9; bad = 0
    for ep in range(epochs):
        t_ep = time.time()
        net.train(); opt.zero_grad()
        for i, b in enumerate(tr):
            x, xmask, dt, y, ymask = _batch_to(b, dev)
            with amp_ctx(dev):
                pred = net(x, xmask, dt).float()
                li = ic_loss(pred.mean(-1), y, ymask)      # 主：横截面 IC
                lc = pinball(pred, y, ymask, lv)           # 辅：保持分布校准（sizing 用）
                loss = (li + lam * lc) / accum
            if torch.isfinite(loss):
                loss.backward()
            if (i + 1) % accum == 0:
                gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                if torch.isfinite(gn):
                    opt.step()
                opt.zero_grad(); sch.step()
        vc, vi, vir, vric = evaluate(net, va, dev, lv)
        _, ti, _, _ = evaluate(net, tre, dev, lv)                  # train_IC：train−val gap 大 = 过拟合(容量该不该加的直接诊断)
        print(f"  epoch {ep+1}/{epochs} ({time.time()-t_ep:.0f}s)  train_IC={ti:+.4f} val_IC={vi:+.4f} gap={ti-vi:+.4f}  "
              f"crps={vc:.4f} IR={vir:+.2f} rankIC={vric:+.4f}", flush=True)
        snaps.append((vi, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}))
        if vi > best_ic + 1e-5:
            best_ic = vi; bad = 0
        else:
            bad += 1
            if bad >= patience:
                print("  早停（val_IC 不再升）"); break
    # SWA：取 val-IC 最高的 top-k 个 checkpoint 做**权重平均**（cosine 末段低 LR + LayerNorm 无 running
    # stats → 平均合法）。比挑单个抖动峰更稳、更可信，且去掉"挑 val 峰"的乐观偏差。
    snaps.sort(key=lambda s: s[0], reverse=True)
    topk = snaps[:max(1, min(swa_k, len(snaps)))]
    base = topk[0][1]
    avg = {}
    for k, v in base.items():
        if torch.is_floating_point(v):
            avg[k] = sum(s[1][k].float() for s in topk) / len(topk)
        else:
            avg[k] = v.clone()                       # 非浮点(如索引/常量buffer)取最佳那个
    net.load_state_dict(avg)
    vc, vi, vir, vric = evaluate(net, va, dev, lv)   # 重测平均模型的 val 指标（用它当返回）
    print(f"  SWA 平均 top-{len(topk)}（val_IC≥{topk[-1][0]:+.4f}）→ val_IC={vi:+.4f} IR={vir:+.2f}", flush=True)
    return net, vi


class TorchPredictor:
    name = "Model"

    def __init__(self, net, dev, T=64, batch=16):
        self.net = net.eval(); self.dev = dev; self.T = T; self.batch = batch

    def predict(self, x, xmask, dt):
        out = []
        with torch.no_grad():
            for s in range(0, len(x), self.batch):
                xb = torch.from_numpy(x[s:s+self.batch]).float().to(self.dev)
                mb = torch.from_numpy(xmask[s:s+self.batch]).bool().to(self.dev)
                db = torch.from_numpy(np.asarray(dt[s:s+self.batch])).float().to(self.dev)
                with amp_ctx(self.dev):
                    p = self.net(xb, mb, db)
                out.append(p.float().cpu().numpy())
        return np.concatenate(out)
