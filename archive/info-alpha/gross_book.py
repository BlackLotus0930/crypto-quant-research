"""毛 Sharpe 天花板：在拼接 book 上算 0bps(执行完全免费)的净 Sharpe，
扫 cadence×rebal 取最优。回答"若做 maker 把成本压到 ~0，最高能到多少"——决定 maker 这条路值不值得。
跑：python gross_book.py --book walkforward_book.npz
"""
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--book", default="walkforward_book.npz")
ap.add_argument("--ann", type=int, default=8760)
a = ap.parse_args()
z = np.load(a.book, allow_pickle=True)
W, R = z["W"].astype(np.float64), z["R"].astype(np.float64)
T, N = W.shape


def sim(cad, rebal, bps):
    held = np.zeros(N); pnl = np.empty(T); turn = np.empty(T)
    for t in range(T):
        if t % cad == 0:
            new = held + rebal * (W[t] - held)
            g = np.abs(new).sum(); new = new / g if g > 0 else new
            turn[t] = np.abs(new - held).sum(); held = new
        else:
            turn[t] = 0.0
        pnl[t] = (held * R[t]).sum()
    net = pnl - bps / 1e4 * turn; sd = net.std()
    return (net.mean() / sd * np.sqrt(a.ann) if sd > 0 else 0.0,
            float(net.mean() * a.ann), float(turn.mean() * a.ann))


print(f"book {T} bar, N={N}, ann={a.ann}")
print(f"{'cad':>5s} {'rebal':>6s} | {'毛@0bps':>8s} {'年化':>8s} | {'净@1bps':>8s} {'净@2bps':>8s} | {'年换手':>8s}")
for cad in (1, 4, 24, 72):
    for rb in (1.0, 0.3, 0.1):
        s0, ann0, to = sim(cad, rb, 0)
        s1, _, _ = sim(cad, rb, 1)
        s2, _, _ = sim(cad, rb, 2)
        print(f"{cad:>5d} {rb:>6.1f} | {s0:>+8.2f} {ann0:>+7.1%} | {s1:>+8.2f} {s2:>+8.2f} | {to:>8.0f}")
