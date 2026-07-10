"""伴生研究报告：每次 run 自动产出"模型学到了什么"，而不只 GO/NO-GO 标量。
教训(E9)：差点从标量 IC 判"模型=线性因子、thesis 死"，一看分解才发现高阶残差。
五块：① 因子分解(IC+正交残差=novelty) ② 分布校准(spread→真实|收益|) ③ 信号半衰期
      ④ 频率/成本前沿(调仓周期×成本) ⑤ val→holdout 稳定性。
"""
import numpy as np

from .harness import anchors_for, build_windows
from .crps import QUANTILE_LEVELS
from . import backtest

_LO = int(np.argmin(np.abs(QUANTILE_LEVELS - 0.16)))
_HI = int(np.argmin(np.abs(QUANTILE_LEVELS - 0.84)))
K_MOM = 120


def _xs(a, b, v):
    m = v & np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return np.nan
    x, y = a[m], b[m]
    if x.std() < 1e-12 or y.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _resid(sig, factors, v):
    m = v & np.isfinite(sig) & np.all([np.isfinite(f) for f in factors], axis=0)
    out = np.full_like(sig, np.nan)
    if m.sum() < 20:
        return out
    X = np.column_stack([np.ones(m.sum())] + [f[m] for f in factors])
    beta, *_ = np.linalg.lstsq(X, sig[m], rcond=None)
    out[m] = sig[m] - X @ beta
    return out


def _stat(x, ann):
    x = np.array([v for v in x if np.isfinite(v)])
    if len(x) == 0:
        return 0.0, 0.0
    return float(x.mean()), float(x.mean() / x.std() * np.sqrt(ann)) if x.std() > 0 else 0.0


def _tval(x):
    """IC 序列的 t 统计量 = mean/std×√n（跨市场可比；年化 IR 受年化期数影响、不可跨市场比）。"""
    x = np.array([v for v in x if np.isfinite(v)])
    return float(x.mean() / x.std() * np.sqrt(len(x))) if len(x) > 1 and x.std() > 0 else 0.0


def _dump_signal(tensor, splits, model, split, dt, T):
    anchors = anchors_for(tensor, splits, split, T, dt_max=dt)
    sig = np.zeros((len(anchors), tensor["n"].shape[1]), np.float32); spr = np.zeros_like(sig)
    for s in range(0, len(anchors), 256):
        ab = anchors[s:s + 256]
        x, xmask = build_windows(tensor, ab, T)
        pred = model.predict(x, xmask, np.full(len(ab), dt))
        sig[s:s + len(ab)] = pred.mean(-1); spr[s:s + len(ab)] = (pred[..., _HI] - pred[..., _LO]) / 2
    return anchors, sig, spr


def report(tensor, splits, model, T, dt, split="holdout", bt=None):
    ann = int(tensor["ann"]) if "ann" in tensor else 252 * (int(tensor["bars_per_day"]) if "bars_per_day" in tensor else 1)
    adj = tensor["adj_close"].astype(np.float64); mask = tensor["mask"]; scale = tensor["scale_ret"].astype(np.float64)
    nvolch = tensor["n"][:, :, 4].astype(np.float64); Tn, N = adj.shape
    with np.errstate(all="ignore"):
        logc = np.where(adj > 0, np.log(adj), np.nan)
    A, SIG, SPR = _dump_signal(tensor, splits, model, split, dt, T)

    print("\n" + "=" * 64 + "\n[研究报告] 模型学到了什么(不只 GO/NO-GO)\n" + "=" * 64, flush=True)

    # ① 因子分解 + ② 校准
    ic_m, ic_r5, ic_res2, ic_resF, align, sprvol = [], [], [], [], [], []
    for i, a in enumerate(A):
        if a - K_MOM < 0 or a + dt >= Tn:
            continue
        v = mask[a] & mask[a - 5] & mask[a - 6] & mask[a - K_MOM] & mask[a + dt] & (adj[a] > 0) & (scale[a] > 0)  # scale>0 只评分加密(context 资产 scale=0 排除)
        rev1 = -(logc[a] - logc[a - 1]); rev5 = -(logc[a] - logc[a - 5])
        mom = logc[a] - logc[a - K_MOM]; vol = scale[a]
        nvol = nvolch[a]; dvol = nvolch[a] - nvolch[a - 1]
        fwd = (logc[a + dt] - logc[a]) / np.where(scale[a] > 0, scale[a], np.nan)
        sg = SIG[i]
        ic_m.append(_xs(sg, fwd, v)); ic_r5.append(_xs(rev5, fwd, v))
        ic_res2.append(_xs(_resid(sg, [rev1, rev5], v), fwd, v))
        ic_resF.append(_xs(_resid(sg, [rev1, rev5, mom, vol, nvol, dvol], v), fwd, v))
        align.append(_xs(sg, rev5, v)); sprvol.append(_xs(SPR[i], np.abs(fwd), v))
    print(f"① 因子分解(IC / 年化IR):")
    tot_ic, tot_ir = _stat(ic_m, ann); nov_ic, nov_ir = _stat(ic_resF, ann)   # 真实(原始) / 新颖(⟂全因子残差)
    ic_t = _tval(ic_m); nov_t = _tval(ic_resF)                                # t 值=毕业标准(§0.2，跨市场可比)
    for nm, ar in [("  模型 signal→真实", ic_m), ("  反转 rev5→真实", ic_r5),
                   ("  模型⟂反转 残差→真实", ic_res2), ("  模型⟂全因子 残差→真实(=novelty)", ic_resF)]:
        mu, ir = _stat(ar, ann); print(f"{nm:<30s} {mu:>+.4f} / {ir:>+.2f}")
    am, _ = _stat(align, ann); nov_pct = (nov_ic / tot_ic) if tot_ic else 0.0
    print(f"  → 模型与反转相关={am:+.2f}；novelty 占比≈{nov_pct:.0%}；t值 模型={ic_t:.1f}/novelty={nov_t:.1f}(毕业线 t≥3)")
    sv, svir = _stat(sprvol, ann)
    print(f"② 分布校准: spread→真实|收益| IC={sv:+.4f}(IR{svir:+.0f})  (>0=不确定性可信、可用于 sizing)")

    # ③ 信号半衰期
    acs = [np.corrcoef(SIG[1:, j], SIG[:-1, j])[0, 1] for j in range(N) if SIG[:, j].std() > 1e-9]
    ac = float(np.nanmean(acs)); hl = (np.log(0.5) / np.log(ac)) if 0 < ac < 1 else np.inf
    print(f"③ 信号性格: bar间自相关={ac:.3f} → 半衰期≈{hl:.0f} bar  (短=信号快=换手高)")

    # ④ 频率/成本前沿
    if bt is None:                                                 # run_gonogo 传入已算账本→省 1 次 holdout 前向
        bt = backtest.run(tensor, splits, model, split=split, T=T, pred_dt=dt)
    W, R = bt["W"], bt["R"]
    print(f"④ 频率/成本前沿(调仓周期 × 成本,净 Sharpe):")
    print(f"   {'周期':>6s} {'净@2':>7s} {'净@5':>7s} {'净@10':>7s} {'年换手':>7s}")
    front2 = front5 = -1e9                                          # 全前沿最优(任意周期×rebal)→可交易 gate
    for cad in (1, 4, 24, 72):
        best = None
        for rb in (1.0, 0.5, 0.3, 0.1):
            held = np.zeros(W.shape[1]); pnl = np.empty(len(W)); to = np.empty(len(W))
            for t in range(len(W)):
                if t % cad == 0:
                    new = held + rb * (W[t] - held); g = np.abs(new).sum(); new = new / g if g > 0 else new
                    to[t] = np.abs(new - held).sum(); held = new
                else:
                    to[t] = 0.0
                pnl[t] = (held * R[t]).sum()
            def sh(bps):
                p = pnl - bps / 1e4 * to; sd = p.std()
                return p.mean() / sd * np.sqrt(ann) if sd > 0 else 0.0
            s2, s5, s10 = sh(2), sh(5), sh(10)
            front2 = max(front2, s2); front5 = max(front5, s5)
            if best is None or s5 > best[1]:
                best = (s2, s5, s10, float(to.mean() * ann))
        lab = {1: "每bar", 4: "每4", 24: "每24", 72: "每72"}[cad]
        print(f"   {lab:>6s} {best[0]:>+7.2f} {best[1]:>+7.2f} {best[2]:>+7.2f} {best[3]:>7.0f}")
    print("=" * 64, flush=True)
    return {"ic": tot_ic, "ic_ir": tot_ir, "ic_t": ic_t, "nov_ic": nov_ic, "nov_ir": nov_ir, "nov_t": nov_t,
            "nov_pct": nov_pct, "align": am, "spr_ic": sv, "half_life": float(hl),
            "front_net2": float(front2), "front_net5": float(front5)}
