"""把 cost_aware 选出的配置,在拼接 book 上逐年拆 net Sharpe —— 抓"聚合正但2026负"(LSR陷阱)。
配置: 平滑半衰期=24, 每24bar调仓, rebal=0.3, band=0.01。
跑：python peryear_book.py --book walkforward_book.npz --hl 24 --cad 24 --rebal 0.3 --band 0.01
"""
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--book", default="walkforward_book.npz")
ap.add_argument("--hl", type=int, default=24); ap.add_argument("--cad", type=int, default=24)
ap.add_argument("--rebal", type=float, default=0.3); ap.add_argument("--band", type=float, default=0.01)
ap.add_argument("--ann", type=int, default=8760)
a = ap.parse_args()
z = np.load(a.book, allow_pickle=True)
W, R, dates = z["W"].astype(np.float64), z["R"].astype(np.float64), z["dates"].astype(str)
T, N = W.shape
yr = np.array([d[:4] for d in dates])


def smooth(W, h):
    if h <= 0:
        return W
    al = 1 - 0.5 ** (1.0 / h); S = np.empty_like(W); S[0] = W[0]
    for t in range(1, len(W)):
        S[t] = al * W[t] + (1 - al) * S[t - 1]
    g = np.abs(S).sum(1, keepdims=True)
    return np.divide(S, g, out=np.zeros_like(S), where=g > 0)


Ws = smooth(W, a.hl)
held = np.zeros(N); pnl = np.empty(T); turn = np.empty(T)
for t in range(T):
    if t % a.cad == 0:
        d = Ws[t] - held; mv = np.abs(d) > a.band
        new = held.copy(); new[mv] = held[mv] + a.rebal * d[mv]
        g = np.abs(new).sum(); new = new / g if g > 0 else new
        turn[t] = np.abs(new - held).sum(); held = new
    else:
        turn[t] = 0.0
    pnl[t] = (held * R[t]).sum()


def sh(p, t, bps):
    net = p - bps / 1e4 * t; sd = net.std()
    return (net.mean() / sd * np.sqrt(a.ann) if sd > 0 else 0.0, float(net.mean() * a.ann), float(t.mean() * a.ann))


print(f"配置: 平滑hl={a.hl} 每{a.cad}bar rebal={a.rebal} band={a.band}\n")
print(f"{'区间':>10s} {'净@2':>7s} {'净@5':>7s} {'净@10':>7s} {'年化@5':>8s} {'年换手':>7s}")
for lab, m in [("全程", np.ones(T, bool))] + [(y, yr == y) for y in ("2024", "2025", "2026")]:
    if m.sum() < 50:
        continue
    s2, _, _ = sh(pnl[m], turn[m], 2); s5, a5, to = sh(pnl[m], turn[m], 5); s10, _, _ = sh(pnl[m], turn[m], 10)
    print(f"{lab:>10s} {s2:>+7.2f} {s5:>+7.2f} {s10:>+7.2f} {a5:>+7.1%} {to:>7.0f}")
print("\n判读: 2026 也 ≥0(尤其净@5) = 真稳健 → 🟢/🟡; 2026 转负 = LSR 同款陷阱 → 🔴。")
