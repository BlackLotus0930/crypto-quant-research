"""分布 → 固定不学习策略 → P&L（设计 §5）。这就是"基于模型输出分布交易"的最小种子：
人不给市场观点，只给机械翻译器。Δt=1 次日、日频 rebalance、扣 2/5/10 bps。
"""
import numpy as np

from pipeline.config import T_WINDOW
from .crps import QUANTILE_LEVELS
from .harness import anchors_for, build_windows

_LO = int(np.argmin(np.abs(QUANTILE_LEVELS - 0.16)))
_HI = int(np.argmin(np.abs(QUANTILE_LEVELS - 0.84)))
CAP = 3.0
BPS_LIST = (2, 5, 10)


def _book_weights(pred, tradeable, book="riskeq", vol=None):
    """pred [B,N,K] -> 仓位权重 [B,N]（横截面 gross=1）。signal=分布均值=方向信号。
    book 决定**如何把信号变成仓位大小**：
      conviction: signal/spread（旧默认；spread 来自分布头，没校准时这步是噪音/有害）。
      signal:     横截面 z-score(signal) 再 clip（纯方向定大小，等"美元"权重）。
      rank:       每天按 signal 排序，多头前 1/3、空头后 1/3 等权（纯方向）。
      riskeq:     signal/vol（vol=已知因果波动率 scale_ret）→ **等风险**。
                  信号在归一空间预测，按 1/vol 定仓使原始收益 P&L = 归一空间 IC 的变现，最对路。
    """
    signal = pred.mean(-1)                                   # [B,N] 归一期望收益≈方向信号
    tr = tradeable > 0
    if book == "conviction":
        spread = (pred[..., _HI] - pred[..., _LO]) / 2 + 1e-6
        raw = np.clip(signal / spread, -CAP, CAP) * tradeable
    elif book in ("signal", "riskeq"):
        raw = np.zeros_like(signal)
        for b in range(signal.shape[0]):
            m = tr[b]
            if m.sum() < 2:
                continue
            s = signal[b, m] - signal[b, m].mean()           # 去均值 → dollar/市场中性
            if book == "riskeq":
                v = vol[b, m]
                s = s / np.maximum(v, np.quantile(v, 0.05))  # 反波动率，低vol地板防爆
            raw[b, m] = np.clip(s / (s.std() + 1e-9), -CAP, CAP)
    elif book == "rank":
        raw = np.zeros_like(signal)
        for b in range(signal.shape[0]):
            idx = np.where(tr[b])[0]
            n = len(idx)
            if n < 6:
                continue
            order = idx[np.argsort(signal[b, idx])]
            k = max(1, n // 3)
            raw[b, order[-k:]] = 1.0; raw[b, order[:k]] = -1.0
    else:
        raise ValueError(book)
    gross = np.abs(raw).sum(1, keepdims=True)
    return np.where(gross > 0, raw / gross, 0.0)


def _ret_matrix(tensor):
    """全历史次日简单收益矩阵 [T_days,N]（因果 β 用）。缺失/停牌→0。"""
    ac = tensor["adj_close"].astype(np.float64); m = tensor["mask"]
    prev = np.roll(ac, 1, axis=0); prev[0] = 0
    valid = (m & np.roll(m, 1, axis=0)) & (prev > 0); valid[0] = False
    return np.where(valid, ac / np.where(prev > 0, prev, 1) - 1, 0.0)


def neutralize_book(W, days, retmat, fac_idx, win=120, minobs=40):
    """因果因子中性化：每个 anchor 日，用**过去** win 日的收益估各股票对因子(fac_idx)的 β（OLS），
    把 book 权重里的因子敞口投影掉 w'=w−B(BᵀB)⁻¹Bᵀw → book 对因子净 β≈0，只留残差选股。
    依据：离线验证——因子敞口是快变噪声，配换手控制会把账户拖垮；剔掉后残差信号持续、净 Sharpe 抬升。"""
    Wn = W.copy()
    for i, a in enumerate(days):
        lo = max(0, a - win)
        if a - lo < minobs:
            continue
        idx = np.where(np.abs(W[i]) > 0)[0]
        idx = idx[~np.isin(idx, fac_idx)]              # 只中性化股票腿（因子槽本身不动）
        if len(idx) < 5:
            continue
        Fw = retmat[lo:a][:, fac_idx]; Sw = retmat[lo:a][:, idx]
        FtF = Fw.T @ Fw + 1e-6 * np.eye(len(fac_idx))
        B = np.linalg.solve(FtF, Fw.T @ Sw).T          # [Ns,K] 每股票因子载荷
        wi = W[i][idx]
        BtB = B.T @ B + 1e-6 * np.eye(B.shape[1])
        wn = wi - B @ np.linalg.solve(BtB, B.T @ wi)
        out = W[i].copy(); out[idx] = wn
        g = np.abs(out).sum(); Wn[i] = out / g if g > 0 else out
    return Wn


def run(tensor, splits, predictor, split="holdout", T=T_WINDOW, batch=128, book="riskeq", pred_dt=1,
        neutralize=False, factor_slots=None, neut_win=120):
    """跑模型 → 每日**目标**仓位 W[days,N] + 次日真实收益 R[days,N]。
    book = 仓位定大小方式（见 _book_weights）。执行策略（换手控制）是 W 之上的纯后处理。
    pred_dt = 喂给模型的 Δt：>1 时把模型当**更慢的信号**，book 日间更稳→换手更低，仍按次日结算。
    neutralize = 因果因子中性化（需 factor_slots=因子槽索引）。"""
    days = anchors_for(tensor, splits, split, T, dt_max=1)   # 升序
    W, R, dates = [], [], []
    for s in range(0, len(days), batch):
        ab = days[s:s + batch]
        x, xmask = build_windows(tensor, ab, T)
        pred = predictor.predict(x, xmask, np.full(len(ab), pred_dt))
        tradeable = (tensor["mask"][ab] & (tensor["scale_ret"][ab] > 0)).astype(float)   # scale>0 排除 context 资产(US ETF scale=0);加密无变化
        W.append(_book_weights(pred, tradeable, book, vol=tensor["scale_ret"][ab]))
        ca, cb = tensor["adj_close"][ab], tensor["adj_close"][ab + 1]
        valid = tensor["mask"][ab] & tensor["mask"][ab + 1]
        R.append(np.where(valid & (ca > 0), cb / np.where(ca > 0, ca, 1) - 1, 0.0))
        dates.append(tensor["dates"][ab])
    W = np.concatenate(W); R = np.concatenate(R); dates = np.concatenate(dates)
    if neutralize and factor_slots is not None and len(factor_slots) > 0:
        W = neutralize_book(W, np.asarray(days), _ret_matrix(tensor), np.asarray(factor_slots), win=neut_win)
    bpd = int(tensor["bars_per_day"]) if "bars_per_day" in tensor else 1
    ann = int(tensor["ann"]) if "ann" in tensor else 252 * bpd  # 加密存 365×bpd；股票回落 252×bpd
    return {"dates": dates, "W": W, "R": R, "ann": ann}         # ann=每年结算期数


def live_factor_slots(tensor, universe_path, presence=0.9, cats=("broad_etf", "sector_etf"), recent=1000):
    """从 universe parquet 取因子槽（市场+行业 ETF），只留**近期**在场率高的活槽。
    在场率必须在近 recent 日上测——按全 1980+ 历史会把后来才上市的行业 ETF(如 XLRE 2015)误杀。"""
    import pandas as pd
    u = pd.read_parquet(universe_path)
    col = "slot_y" if "slot_y" in u.columns else "slot"
    u = u.dropna(subset=[col]); u[col] = u[col].astype(int)
    cand = [s for s in u[u["category"].isin(cats)][col].tolist() if s < tensor["mask"].shape[1]]
    mrec = tensor["mask"][-recent:]
    return [s for s in cand if mrec[:, s].mean() > presence]


def apply_execution(W, rebal=1.0, band=0.0):
    """换手控制（纯机械、无市场观点）：每天朝目标 W_t 走 rebal 比例，且单名变动 < band 不动；
    再归一回 gross=1 部署。rebal=1,band=0 即"每天全量对齐目标"=原始基线。返回实际持仓 held[days,N]。"""
    held = np.zeros_like(W)
    cur = np.zeros(W.shape[1])
    for t in range(len(W)):
        delta = W[t] - cur
        move = np.abs(delta) > band
        new = cur.copy()
        new[move] = cur[move] + rebal * delta[move]
        g = np.abs(new).sum()
        new = new / g if g > 0 else new
        held[t] = new; cur = new
    return held


def _pnl_turnover(bt, rebal=1.0, band=0.0):
    held = apply_execution(bt["W"], rebal, band)
    pnl_gross = (held * bt["R"]).sum(1)
    turnover = np.abs(np.diff(held, axis=0, prepend=0)).sum(1)
    return pnl_gross, turnover


def metrics(bt, bps, rebal=1.0, band=0.0):
    ann = bt.get("ann", 252)                                   # 日内=252×bars/天；日频=252
    pg, to = _pnl_turnover(bt, rebal, band)
    pnl = pg - bps / 1e4 * to
    eq = np.cumprod(1 + pnl)
    dd = 1 - eq / np.maximum.accumulate(eq)
    sd = pnl.std()
    return {
        "ann_return": float(pnl.mean() * ann),
        "sharpe": float(pnl.mean() / sd * np.sqrt(ann)) if sd > 0 else 0.0,
        "max_dd": float(dd.max()),
        "hit": float((pnl > 0).mean()),
        "turnover": float(to.mean()),
    }


def report(bt, name="model", rebal=1.0, band=0.0):
    print(f"--- 回测 [{name}]  ({len(bt['dates'])} 天 {bt['dates'][0]}~{bt['dates'][-1]})"
          f"{'' if (rebal==1 and band==0) else f'  rebal={rebal} band={band}'} ---")
    print(f"{'bps':>5s} {'年化':>8s} {'Sharpe':>8s} {'最大回撤':>8s} {'命中':>6s} {'换手':>7s}")
    res = {}
    for bps in BPS_LIST:
        m = metrics(bt, bps, rebal, band); res[bps] = m
        print(f"{bps:5d} {m['ann_return']:8.2%} {m['sharpe']:8.2f} {m['max_dd']:8.2%} {m['hit']:6.1%} {m['turnover']:7.3f}")
    pg, to = _pnl_turnover(bt, rebal, band)
    ann = bt.get("ann", 252)
    yrs = np.array([d[:4] for d in bt["dates"]])
    pnl5 = pg - 5 / 1e4 * to
    by = [f"{y}:{(pnl5[yrs==y].mean()/pnl5[yrs==y].std()*np.sqrt(ann) if pnl5[yrs==y].std()>0 else 0):.1f}"
          for y in sorted(set(yrs))]
    print("  逐年 Sharpe(5bps):", " ".join(by))
    return res


def best_rebal_at(bt, bps, rebals=(1.0, 0.7, 0.5, 0.3, 0.2, 0.1)):
    """在给定成本 bps 下选净 Sharpe 最高的 rebal。**应在 val/train 上调用**（别在 holdout 上扫=过拟合）。
    最优 rebal 强依赖成本：成本低→高换手(rebal大)，成本高→低换手。由真实成本定，不由 holdout 定。"""
    return max(rebals, key=lambda r: metrics(bt, bps, r, 0.0)["sharpe"])


def best_exec_at(bt, bps, rebals=(1.0, 0.7, 0.5, 0.3, 0.2, 0.1), bands=(0.0, 0.001, 0.003)):
    """同时选 rebal+band（no-trade 带）使净 Sharpe 最高。**在 val 上调用**。
    band 让仓位扛过小抖动→压换手而不丢毛信号（诊断里 band=0.003 把净显著抬高）。返回 (rebal, band)。"""
    best = None
    for rb in rebals:
        for bd in bands:
            sh = metrics(bt, bps, rb, bd)["sharpe"]
            if best is None or sh > best[0]:
                best = (sh, rb, bd)
    return best[1], best[2]


def sweep_execution(bt, rebals=(1.0, 0.5, 0.3, 0.2), bands=(0.0, 0.001, 0.002)):
    """扫换手控制参数 → 净Sharpe-vs-换手前沿（5/10bps 各一列）。挑在真实成本下最赚的执行点。"""
    print(f"\n  换手控制扫频（净 Sharpe @5bps / @10bps · 换手）：")
    print(f"  {'rebal':>6s} {'band':>6s} {'Shrp@5':>8s} {'Shrp@10':>8s} {'换手':>7s} {'年化@5':>8s}")
    best = None
    for rb in rebals:
        for bd in bands:
            m5 = metrics(bt, 5, rb, bd); m10 = metrics(bt, 10, rb, bd)
            print(f"  {rb:6.2f} {bd:6.3f} {m5['sharpe']:8.2f} {m10['sharpe']:8.2f} {m5['turnover']:7.3f} {m5['ann_return']:8.2%}")
            score = m5["sharpe"] + m10["sharpe"]        # 偏向在高成本下也稳的执行点
            if best is None or score > best[0]:
                best = (score, rb, bd, m5["sharpe"], m10["sharpe"])
    print(f"  → 最优执行点 rebal={best[1]} band={best[2]}：Sharpe@5={best[3]:.2f} @10={best[4]:.2f}")
    return best
