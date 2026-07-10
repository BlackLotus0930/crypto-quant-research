"""成本感知组合(便宜杠杆#1)：在拼接 book 上，不再朴素 rebalance，而是
① 对信号做 EMA 时间平滑(砍掉噪声驱动的换手，半衰期~6bar 的真信号留住)
② 带 no-trade band 的成本感知调仓
③ 干净不偷看：参数在前 40% 选(按 val 净@cost)，应用到后 60% 报。
回答：聪明交易能不能把 taker 净@5 从 -0.13 推过 0？跑：python cost_aware.py --book walkforward_book.npz --cost 5
"""
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--book", default="walkforward_book.npz")
ap.add_argument("--cost", type=float, default=5, help="选参用的成本 bps")
ap.add_argument("--ann", type=int, default=8760)
ap.add_argument("--split", type=float, default=0.4, help="前多少比例做选参(val)")
a = ap.parse_args()
z = np.load(a.book, allow_pickle=True)
W, R = z["W"].astype(np.float64), z["R"].astype(np.float64)
T, N = W.shape
cut = int(T * a.split)
HALF = [0, 3, 6, 12, 24]            # 0=不平滑
CADS = [1, 2, 4, 12, 24]
REBALS = [1.0, 0.3, 0.1]
BANDS = [0.0, 0.003, 0.01]


def smooth(W, h):
    if h <= 0:
        return W
    alpha = 1 - 0.5 ** (1.0 / h)
    S = np.empty_like(W); S[0] = W[0]
    for t in range(1, len(W)):
        S[t] = alpha * W[t] + (1 - alpha) * S[t - 1]
    g = np.abs(S).sum(1, keepdims=True)                 # 重归一到 gross 1
    return np.divide(S, g, out=np.zeros_like(S), where=g > 0)


def sim(Ws, R, cad, rebal, band, bps, lo, hi):
    held = np.zeros(N); pnl = []; turn = []
    for t in range(hi):
        if t % cad == 0:
            d = Ws[t] - held; mv = np.abs(d) > band
            new = held.copy(); new[mv] = held[mv] + rebal * d[mv]
            g = np.abs(new).sum(); new = new / g if g > 0 else new
            tn = np.abs(new - held).sum(); held = new
        else:
            tn = 0.0
        if t >= lo:
            pnl.append((held * R[t]).sum()); turn.append(tn)
    pnl = np.array(pnl); turn = np.array(turn)
    net = pnl - bps / 1e4 * turn; sd = net.std()
    return (net.mean() / sd * np.sqrt(a.ann) if sd > 0 else 0.0, float(turn.mean() * a.ann))


print(f"book {T} bar, N={N}; 选参[0:{cut}] 测试[{cut}:{T}] 选参成本={a.cost}bps\n")
best = None
for h in HALF:
    Ws = smooth(W, h)
    for cad in CADS:
        for rb in REBALS:
            for bd in BANDS:
                s, _ = sim(Ws, R, cad, rb, bd, a.cost, 0, cut)        # 只在前段选
                if best is None or s > best[0]:
                    best = (s, h, cad, rb, bd)
s, h, cad, rb, bd = best
print(f"val 选定: 平滑半衰期={h}bar 调仓每{cad}bar rebal={rb} band={bd}  (val 净@{a.cost}bps={s:+.2f})")
Ws = smooth(W, h)
print(f"\n→ 应用到后 60%(不偷看):")
print(f"{'bps':>5s} {'净Sharpe':>9s} {'年换手':>8s}")
for bps in (2, 5, 10):
    sh, to = sim(Ws, R, cad, rb, bd, bps, cut, T)
    print(f"{bps:>5d} {sh:>+9.2f} {to:>8.0f}")
sh5, _ = sim(Ws, R, cad, rb, bd, 5, cut, T)
print(f"\n判读: 干净 holdout 净@5bps={sh5:+.2f} —— 对照 crude diag_freq 偷看最优 -0.13；>0=聪明交易救回了 taker。")
