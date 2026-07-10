"""跨所套利真实净额(扣跨所基差噪声)+ 组合书提升检验。
位置:短高funding所永续 + 长低funding所永续(同币)。
  funding income/bar = w·|f_bin−f_byb|/8
  price leg/bar = w·(−sign(d)·dret),dret = Binance_perp_ret − Bybit_perp_ret(跨所收益分歧,真实 haircut)
  净 = income + price − 双腿成本。
关键:把 net 和 Binance 单所 carry 等波动组合,看组合 Sharpe vs 单 Binance(到底有没有提升)。
跑(需先下完 Bybit klines):python xvenue_net.py
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])  # 从仓库根运行(import strategy + data/ 路径都对)
import glob
import os

import numpy as np
import pandas as pd

z = np.load("data/clean/crypto_tensor_60min_pit.npz", allow_pickle=True)
mask = z["mask"]; adj = z["adj_close"].astype(float); tdv = z["tdv"].astype(float)
dates = z["dates"].astype(str); slots = list(z["slots"].astype(str)); T, N = mask.shape
nmap = {s: i for i, s in enumerate(slots)}
xv = np.load("data/clean/xvenue_funding.npz", allow_pickle=True)
f_bin = xv["f_bin"].astype(float); f_byb = xv["f_byb"].astype(float)
grid_ts = np.sort(pd.read_parquet("data/clean/crypto_60min_pit.parquet", columns=["ts"])["ts"].unique()).astype(np.int64)
yr = np.array([d[:4] for d in dates]); cut = int(T * 0.4); ann = 8760
oos = np.zeros(T, bool); oos[cut:] = True

# Bybit 1h close 对齐到网格
byb_close = np.full((T, N), np.nan)
gdf = pd.DataFrame({"ts": grid_ts}); nf = 0
for f in glob.glob("data/raw/bybit/kline/*.csv"):
    sym = os.path.basename(f)[:-4]; j = nmap.get(sym)
    if j is None:
        continue
    df = pd.read_csv(f)
    if df.empty:
        continue
    df = df.sort_values("ts").drop_duplicates("ts", keep="last")
    m = pd.merge_asof(gdf, df[["ts", "close"]], on="ts", direction="backward", tolerance=3600_000)
    byb_close[:, j] = m["close"].to_numpy(); nf += 1
print(f"Bybit 价格覆盖 {nf}/{N} 币")

both = mask & (adj > 0) & np.isfinite(f_bin) & np.isfinite(f_byb) & np.isfinite(byb_close) & (byb_close > 0)
d = np.where(both, f_bin - f_byb, 0.0); spread = np.abs(d); sgn = np.sign(d)
binret = np.zeros((T, N)); bybret = np.zeros((T, N))
v2 = both[:-1] & both[1:]
binret[:-1][v2] = (adj[1:] / adj[:-1] - 1)[v2]
bybret[:-1][v2] = (byb_close[1:] / byb_close[:-1] - 1)[v2]
dret = np.clip(np.nan_to_num(binret - bybret), -0.10, 0.10)        # 跨所收益分歧(haircut),winsor 防坏价
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
        cand = cand[np.argsort(tdv[t, cand])[::-1][:K]]
    w = spread[t, cand]; s = w.sum()
    if s > 0:
        W[t, cand] = cap_renorm(w / s, CAP)
Ws = smooth(W, POSHL)
cur = np.zeros(N); turn = np.zeros(T); prev = np.zeros(N)
inc = np.zeros(T); pxl = np.zeros(T)
for t in range(T):
    if t % CAD == 0:
        dd = Ws[t] - cur; mv = np.abs(dd) > BAND
        new = cur.copy(); new[mv] = cur[mv] + REBAL * dd[mv]
        g = np.abs(new).sum(); cur = new / g if g > 0 else new
    inc[t] = (cur * spread[t] / 8.0).sum()
    pxl[t] = (cur * (-sgn[t] * dret[t])).sum()                    # 价格腿(跨所基差噪声)
    turn[t] = np.abs(cur - prev).sum(); prev = cur
xnet = inc + pxl - COST / 1e4 * turn


def sh(p, m):
    p = p[m]; return p.mean() / p.std() * np.sqrt(ann) if (len(p) and p.std() > 0) else 0.0


def mdd(p, m):
    p = p[m]; c = np.cumsum(p); return (c - np.maximum.accumulate(c)).min() if len(p) else 0.0


print("\n=== 跨所套利 真实净额(扣跨所基差 price leg + 成本) ===")
print(f"{'区间':>8s} {'Sharpe':>7s} {'年化':>8s} {'回撤':>8s} | {'funding腿':>9s} {'价格腿(haircut)':>13s}")
for lab, m in [("OOS全", oos)] + [(y, oos & (yr == y)) for y in ("2023", "2024", "2025", "2026")] + [("全样本", np.ones(T, bool))]:
    if m.sum() < 50:
        continue
    print(f"{lab:>8s} {sh(xnet,m):>+7.2f} {xnet[m].mean()*ann:>+7.1%} {mdd(xnet,m):>+8.3f} | "
          f"{inc[m].mean()*ann:>+8.1%} {pxl[m].mean()*ann:>+12.1%}")

# === 组合书提升检验 ===
import strategy as S
bnet = S.backtest(S.CarryConfig(leverage=1.0))["net"]
print("\n=== 组合书 vs 单 Binance(等波动组合;到底有没有提升)===")
print(f"  相关性(OOS) corr(xvenue, binance) = {np.corrcoef(xnet[oos], bnet[oos])[0,1]:+.2f}")
sb, sx = bnet[oos].std(), xnet[oos].std()
comb = bnet / sb + xnet / sx if sb > 0 and sx > 0 else bnet      # 等波动 1:1


def report(name, p):
    print(f"  {name:>16s}: OOS Sharpe {sh(p,oos):>+5.2f}  年化(归一前看Sharpe)")
print(f"  {'单 Binance':>16s}: OOS Sharpe {sh(bnet,oos):+.2f}")
print(f"  {'单 跨所':>16s}: OOS Sharpe {sh(xnet,oos):+.2f}")
print(f"  {'组合(等波动)':>16s}: OOS Sharpe {sh(comb,oos):+.2f}  ← 若 > 单 Binance = 真提升(√N)")

# === 组合书(50/50 资本,=实盘 paper_live 配法)逐年:真实年化+回撤 ===
comb50 = 0.5 * bnet + 0.5 * xnet


def dsh(p, m):
    p = p[m]; n = len(p) // 24
    b = p[:n * 24].reshape(n, 24).sum(1)
    return b.mean() / b.std() * np.sqrt(365) if (len(b) and b.std() > 0) else 0.0


def mdd2(p, m):
    p = p[m]; c = np.cumsum(p)
    return (c - np.maximum.accumulate(c)).min() if len(p) else 0.0


print("\n=== 组合书(50/50 资本)逐年 — 真实年化(跨所有数据的 2023-2026)===")
print(f"{'区间':>8s} {'组合年化':>9s} {'日Sharpe':>9s} {'最大回撤':>9s} | {'carry年化':>9s} {'跨所年化':>9s}")
for lab, m in [("OOS全", oos)] + [(y, yr == y) for y in ("2023", "2024", "2025", "2026")]:
    if m.sum() < 50:
        continue
    print(f"{lab:>8s} {comb50[m].mean()*ann:>+9.1%} {dsh(comb50,m):>+9.2f} {mdd2(comb50,m):>+9.3f} | "
          f"{bnet[m].mean()*ann:>+9.1%} {xnet[m].mean()*ann:>+9.1%}")
print("  注:50/50=各 50% 资本(L1 无杠杆);2x 杠杆翻倍。日Sharpe 比逐bar诚实。")

# === Sharpe 诚实性:逐bar近确定accrual → √8760 年化会虚高;按 日/周 聚合(更独立)重算 ===
print("\n=== Sharpe 诚实性:按聚合粒度重算(逐bar虚高?)+ 一阶自相关 ===")
print(f"{'流':>12s} {'逐bar(√8760)':>13s} {'日聚合(√365)':>12s} {'周聚合(√52)':>11s} {'lag1自相关':>10s}")
for nm, p in [("Binance", bnet[oos]), ("跨所", xnet[oos]), ("组合", comb[oos])]:
    row = []
    for w, a in [(1, 8760), (24, 365), (168, 52)]:
        n = len(p) // w; b = p[:n * w].reshape(n, w).sum(1)
        row.append(b.mean() / b.std() * np.sqrt(a) if b.std() > 0 else 0.0)
    ac = np.corrcoef(p[:-1], p[1:])[0, 1]
    print(f"{nm:>12s} {row[0]:>+13.2f} {row[1]:>+12.2f} {row[2]:>+11.2f} {ac:>+10.2f}")
print("  (逐bar≫周聚合 = √8760 把近确定 accrual 的 Sharpe 吹大了;周聚合更接近真实可信值。)")
print(f"\n判读:看周聚合 Sharpe(去自相关虚高后)组合是否仍明显 > 单 Binance;真实执行摩擦另算。")
