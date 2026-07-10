"""频率/成本诊断（零 GPU、零重训）：在**已训模型存下的 book** 上，扫"每隔几根 bar 才调仓"。
回答："把同一个信号交易得更慢，净 Sharpe 能不能从负翻正？" 翻正→真凶是频率/成本，值得训日频；
不翻→信号本身过不了成本，省下重训。**这是 holdout 上的诊断扫描(偷看)，不是干净 GO 数。**
跑：PYTHONUTF8=1 .venv/Scripts/python.exe diag_freq.py
"""
import argparse
import numpy as np

_ap = argparse.ArgumentParser()
_ap.add_argument("--book", default="data/clean/crypto_holdout_book.npz", help="已存的 holdout 账本 npz(W/R)")
_a = _ap.parse_args()

ANN = 8760                      # 加密 60min：365×24
z = np.load(_a.book, allow_pickle=False)
W, R = z["W"].astype(np.float64), z["R"].astype(np.float64)   # [T,N] 目标仓位 / 次bar收益
T, N = W.shape


def sim(cadence, rebal, band, bps):
    """每 cadence 根 bar 才朝目标 W 调 rebal 比例(带 no-trade band)，其余 bar 持仓不动。"""
    held = np.zeros(N); pnl = np.empty(T); turn = np.empty(T)
    for t in range(T):
        if t % cadence == 0:
            delta = W[t] - held
            mv = np.abs(delta) > band
            new = held.copy(); new[mv] = held[mv] + rebal * delta[mv]
            g = np.abs(new).sum(); new = new / g if g > 0 else new
            turn[t] = np.abs(new - held).sum(); held = new
        else:
            turn[t] = 0.0
        pnl[t] = (held * R[t]).sum()
    net = pnl - bps / 1e4 * turn
    sd = net.std()
    return (net.mean() / sd * np.sqrt(ANN) if sd > 0 else 0.0,
            float(net.mean() * ANN), float(turn.mean() * ANN))   # sharpe, 年化收益, 年化换手


print(f"book: {T} bar(holdout 2024-26), N={N}, ann={ANN}\n")
print(f"{'调仓周期':>10s} {'rebal':>6s} {'band':>6s} | {'净@2':>7s} {'净@5':>7s} {'净@10':>7s} | {'年换手':>8s} {'年化@5':>9s}")
print("-" * 78)
for cad, label in [(1, "每小时"), (4, "每4h"), (24, "每天"), (72, "每3天")]:
    best = None
    for rebal in (1.0, 0.5, 0.3, 0.1):
        for band in (0.0, 0.003, 0.01):
            s2, _, _ = sim(cad, rebal, band, 2)
            s5, ann5, to = sim(cad, rebal, band, 5)
            s10, _, _ = sim(cad, rebal, band, 10)
            if best is None or s5 > best[0]:
                best = (s5, rebal, band, s2, s10, to, ann5)
    s5, rb, bd, s2, s10, to, ann5 = best
    print(f"{label:>10s} {rb:>6.1f} {bd:>6.3f} | {s2:>+7.2f} {s5:>+7.2f} {s10:>+7.2f} | {to:>8.0f} {ann5:>+8.1%}")
print("\n(每行=该调仓周期下、扫 rebal×band 取净@5bps 最优的配置；这是 holdout 偷看的诊断，不是 GO)")
