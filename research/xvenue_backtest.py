"""跨所 funding 套利回测(Binance↔Bybit):短高funding所永续 + 长低funding所永续(同币)→ 收价差 |f_bin−f_byb|。
两腿都是永续、同币 → 价格按构造对冲、**免借现货**。同纪律:top-K流动、5%帽、平滑、调仓、扣双腿成本、val/OOS、逐年。
关键检查:与 Binance 单所 carry 的相关性(低=近独立流=√N 提 Sharpe)。
⚠️ 本版 income-only:假设两所永续价格完美track(价格腿=0)。真实有跨所基差噪声(需 Bybit 价格才能建)→ 当前数字是毛/偏乐观上界。
跑：python xvenue_backtest.py
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])  # 从仓库根运行(import strategy + data/ 路径都对)
import numpy as np

z = np.load("data/clean/crypto_tensor_60min_pit.npz", allow_pickle=True)
mask = z["mask"]; adj = z["adj_close"].astype(float); tdv = z["tdv"].astype(float)
dates = z["dates"].astype(str); slots = z["slots"].astype(str); T, N = mask.shape
xv = np.load("data/clean/xvenue_funding.npz", allow_pickle=True)
f_bin = xv["f_bin"].astype(float); f_byb = xv["f_byb"].astype(float)
yr = np.array([d[:4] for d in dates]); cut = int(T * 0.4); ann = 8760
oos = np.zeros(T, bool); oos[cut:] = True
both = mask & (adj > 0) & np.isfinite(f_bin) & np.isfinite(f_byb)   # 两所都可交易(免现货!)
d = np.where(both, f_bin - f_byb, 0.0)                              # 跨所价差(短高/长低 → 收 |d|)
spread = np.abs(d)
K, CAD, REBAL, BAND, POSHL, CAP, COST = 100, 24, 0.3, 0.01, 120, 0.05, 10.0


def cap_renorm(w, C):
    w = w.copy()
    for _ in range(20):
        over = w > C + 1e-12
        if not over.any():
            break
        ex = (w[over] - C).sum(); w[over] = C; room = (w > 0) & (~over)
        if not room.any() or w[room].sum() <= 0:
            break
        w[room] += ex * w[room] / w[room].sum()
    s = w.sum(); return w / s if s > 0 else w


def smooth(W, h):
    al = 1 - 0.5 ** (1.0 / h); S = np.empty_like(W); S[0] = W[0]
    for t in range(1, len(W)):
        S[t] = al * W[t] + (1 - al) * S[t - 1]
    g = np.abs(S).sum(1, keepdims=True)
    return np.divide(S, g, out=np.zeros_like(S), where=g > 0)


W = np.zeros((T, N))
for t in range(T):
    cand = np.where(both[t] & (spread[t] > 0))[0]
    if len(cand) == 0:
        continue
    if len(cand) > K:
        cand = cand[np.argsort(tdv[t, cand])[::-1][:K]]            # top-K 流动(Binance tdv 代理)
    w = spread[t, cand]; s = w.sum()
    if s > 0:
        W[t, cand] = cap_renorm(w / s, CAP)
Ws = smooth(W, POSHL)
cur = np.zeros(N); held = np.zeros((T, N)); turn = np.zeros(T); prev = np.zeros(N)
income = np.zeros(T)
for t in range(T):
    if t % CAD == 0:
        dd = Ws[t] - cur; mv = np.abs(dd) > BAND
        new = cur.copy(); new[mv] = cur[mv] + REBAL * dd[mv]
        g = np.abs(new).sum(); cur = new / g if g > 0 else new
    income[t] = (cur * spread[t] / 8.0).sum()                      # 收价差(价格腿假设=0)
    turn[t] = np.abs(cur - prev).sum(); prev = cur
net = income - COST / 1e4 * turn


def sh(p, m):
    p = p[m]; return p.mean() / p.std() * np.sqrt(ann) if (len(p) and p.std() > 0) else 0.0


print(f"⚠️ income-only(假设两所永续价格完美track,价格腿=0)→ 毛/偏乐观上界。真实需 Bybit 价建跨所基差噪声。\n")
print(f"{'区间':>8s} {'净income Sh':>11s} {'年化':>8s} {'年换手':>7s} {'中位持仓':>8s}")
for lab, m in [("OOS全", oos)] + [(y, oos & (yr == y)) for y in ("2023", "2024", "2025", "2026")] \
        + [("全样本", np.ones(T, bool))] + [(y, yr == y) for y in ("2021", "2022")]:
    if m.sum() < 50:
        continue
    npos = np.median((np.abs(held[m]) > 1e-9).sum(1)) if held.any() else 0
    npos = np.median((np.abs(([cur])[0]) > 1e-9).sum()) if False else np.median((np.abs(W[m]) > 1e-9).sum(1))
    print(f"{lab:>8s} {sh(net,m):>+11.2f} {net[m].mean()*ann:>+7.1%} {turn[m].mean()*ann:>7.0f} {npos:>8.0f}")

# 独立性:与 Binance 单所 carry 的相关
try:
    import strategy as S
    res = S.backtest(S.CarryConfig(leverage=1.0))
    bnet = res["net"]
    c_all = np.corrcoef(net, bnet)[0, 1]; c_oos = np.corrcoef(net[oos], bnet[oos])[0, 1]
    print(f"\n=== 独立性(√N 检验)===\n  与 Binance 单所 carry P&L 相关: 全样本 {c_all:+.2f}, OOS {c_oos:+.2f}  (低=近独立→组合提 Sharpe)")
except Exception as e:
    print(f"\n相关性检查跳过: {e}")

print("\n判读:① 年化 income 有多厚(median spread 薄、但肥尾episodic)。② 与 Binance carry 相关低=真新流。")
print("      下一步去乐观化:下 Bybit 价格,建跨所基差噪声(价格腿),看扣掉后净剩多少(类比 spot-carry 的 premiumIndex→真实价)。")
