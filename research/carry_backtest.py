"""横截面 funding carry 回测(结构性溢价,干净):空高funding/多低funding,市场中性,扣成本,逐年。
两腿:价格腿(held·R) + funding收割(-held·funding/8 每bar)。死币在内、退市尾部计入。
信号=当前已知funding(≤t,因果)的横截面 zscore(负号:高funding=多头拥挤→做空)。仓位平滑+慢调仓控成本。
干净不偷看:hl 在前40%选(按net@5),报后60%。
跑：python carry_backtest.py
"""
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--tensor", default="data/clean/crypto_tensor_60min_pit.npz")
ap.add_argument("--funding", default="data/clean/funding_pit.npz")
ap.add_argument("--ann", type=int, default=8760)
ap.add_argument("--split", type=float, default=0.4, help="前多少比例选参(val)")
ap.add_argument("--cad", type=int, default=24); ap.add_argument("--rebal", type=float, default=0.3)
ap.add_argument("--band", type=float, default=0.01)
a = ap.parse_args()

z = np.load(a.tensor, allow_pickle=True)
mask = z["mask"]; adj = z["adj_close"].astype(np.float64); tdv = z["tdv"].astype(np.float64)
dates = z["dates"].astype(str); slots = z["slots"].astype(str)
T, N = mask.shape
zf = np.load(a.funding, allow_pickle=True)
funding = zf["funding"].astype(np.float64)
assert list(zf["dates"].astype(str)) == list(dates) and list(zf["slots"].astype(str)) == list(slots), "funding 与张量未对齐"
yr = np.array([d[:4] for d in dates]); cut = int(T * a.split)

# 次bar简单收益(双边 mask 有效才计;否则0=退市按最后标记平仓)
R = np.zeros((T, N))
good = mask[:-1] & mask[1:] & (adj[:-1] > 0) & (adj[1:] > 0)
R[:-1][good] = (adj[1:][good] / adj[:-1][good]) - 1.0
active = mask & np.isfinite(funding)                       # 要有 funding 才能交易


def make_W(fund, act):
    """每bar:active 内 funding 横截面 zscore,负号(高funding→空),gross 归一=1。市场中性(z 去均→ΣW≈0)。"""
    W = np.zeros((T, N))
    for t in range(T):
        idx = np.where(act[t])[0]
        if len(idx) < 10:
            continue
        x = fund[t, idx]; sd = x.std()
        if sd < 1e-12:
            continue
        W[t, idx] = -(x - x.mean()) / sd
    g = np.abs(W).sum(1, keepdims=True)
    return np.divide(W, g, out=np.zeros_like(W), where=g > 0)


def smooth(W, h):
    if h <= 0:
        return W
    al = 1 - 0.5 ** (1.0 / h); S = np.empty_like(W); S[0] = W[0]
    for t in range(1, len(W)):
        S[t] = al * W[t] + (1 - al) * S[t - 1]
    g = np.abs(S).sum(1, keepdims=True)
    return np.divide(S, g, out=np.zeros_like(S), where=g > 0)


fund0 = np.nan_to_num(funding)                             # 计 funding P&L 用(inactive=0)


def run(W, hl, keep_held=False):
    """→ price(T,), fpnl(T,), turn(T,)[, held_all(T×N)]。fpnl=funding收割,8h rate 摊到每hour。"""
    Ws = smooth(W, hl)
    cur = np.zeros(N); price = np.zeros(T); fpnl = np.zeros(T); turn = np.zeros(T)
    held_all = np.zeros((T, N)) if keep_held else None
    for t in range(T):
        if t % a.cad == 0:
            d = Ws[t] - cur; mv = np.abs(d) > a.band
            new = cur.copy(); new[mv] = cur[mv] + a.rebal * d[mv]
            g = np.abs(new).sum(); new = new / g if g > 0 else new
            turn[t] = np.abs(new - cur).sum(); cur = new
        price[t] = (cur * R[t]).sum()
        fpnl[t] = -(cur * fund0[t] / 8.0).sum()            # long 付正 funding → -held·funding
        if keep_held:
            held_all[t] = cur
    return (price, fpnl, turn, held_all) if keep_held else (price, fpnl, turn)


def sh(net, m=None):
    p = net if m is None else net[m]
    return p.mean() / p.std() * np.sqrt(a.ann) if p.std() > 0 else 0.0


def net_of(price, fpnl, turn, bps):
    return price + fpnl - bps / 1e4 * turn


# ---- 选参:hl 在前40%按 net@5 ----
W_full = make_W(funding, active)
val = slice(0, cut)
best = None
for hl in (0, 6, 24, 72):
    pr, fp, tn = run(W_full, hl)
    s = sh(net_of(pr, fp, tn, 5)[val])
    print(f"  [val] hl={hl:>2d}  net@5(前40%)={s:+.2f}", flush=True)
    if best is None or s > best[0]:
        best = (s, hl)
HL = best[1]
print(f"\nval 选定 hl={HL}; cad={a.cad} rebal={a.rebal} band={a.band}\n")

# ---- 后60% 报告 ----
pr, fp, tn, held = run(W_full, HL, keep_held=True)
oos = np.zeros(T, bool); oos[cut:] = True
print("=== 后60%(OOS)逐年 净Sharpe ===")
print(f"{'区间':>10s} {'净@2':>7s} {'净@5':>7s} {'净@10':>7s} {'年化@5':>8s} {'年换手':>7s}")
for lab, m in [("OOS全", oos)] + [(y, oos & (yr == y)) for y in ("2023", "2024", "2025", "2026")]:
    if m.sum() < 50:
        continue
    n5 = net_of(pr, fp, tn, 5)
    print(f"{lab:>10s} {sh(net_of(pr,fp,tn,2),m):>+7.2f} {sh(n5,m):>+7.2f} {sh(net_of(pr,fp,tn,10),m):>+7.2f}"
          f" {n5[m].mean()*a.ann:>+7.1%} {tn[m].mean()*a.ann:>7.0f}")

print("\n=== 两腿分解(OOS;溢价来自 carry 收割还是价格漂移?) ===")
print(f"  价格腿 Sharpe={sh(pr[oos]):+.2f} (年化 {pr[oos].mean()*a.ann:+.1%})   "
      f"funding腿 Sharpe={sh(fp[oos]):+.2f} (年化 {fp[oos].mean()*a.ann:+.1%})")
print(f"  价格腿累计={pr[oos].sum():+.3f}  funding腿累计={fp[oos].sum():+.3f}  合计毛={pr[oos].sum()+fp[oos].sum():+.3f}")

print("\n=== 长/短腿毛 P&L(价格+funding,OOS逐年) ===")
gross = held * R - held * fund0 / 8.0
lp = np.where(held > 0, gross, 0).sum(1); spnl = np.where(held < 0, gross, 0).sum(1)
for y in ("2024", "2025", "2026"):
    m = oos & (yr == y)
    if m.sum() < 50:
        continue
    print(f"  {y}: 长 {lp[m].sum():+.3f}   短 {spnl[m].sum():+.3f}")

print("\n=== 容量探针(每bar限 top-K 流动 tdv;net@5 OOS) ===")
for K in (50, 100, N):
    if K >= N:
        actK = active
    else:
        actK = np.zeros((T, N), bool)
        for t in range(T):
            v = np.where(active[t])[0]
            if len(v):
                actK[t, v[np.argsort(tdv[t, v])[::-1][:K]]] = True
    WK = make_W(funding, actK)
    prK, fpK, tnK = run(WK, HL)
    nK = net_of(prK, fpK, tnK, 5)
    lab = f"top-{K}" if K < N else f"全({N})"
    print(f"  {lab:>8s}: net@5 OOS={sh(nK[oos]):+.2f}  2026={sh(nK[oos & (yr=='2026')]):+.2f}  年化={nK[oos].mean()*a.ann:+.1%}")

print("\n=== 退市/逼空尾部(做空后被逼空/退市吃掉多少短腿) ===")
last_active = np.array([np.where(mask[:, c])[0].max() if mask[:, c].any() else -1 for c in range(N)])
delisted = (last_active >= 0) & (last_active < T - 24 * 14)
sg = np.where(held < 0, gross, 0)                          # 短腿每格毛
print(f"  退市币 {int(delisted.sum())}/{N}; 短腿在退市币上累计 P&L={sg[:, delisted].sum():+.3f} "
      f"(OOS {sg[oos][:, delisted].sum():+.3f})  ←负=逼空尾部吃短腿")
print(f"  短腿总累计={sg.sum():+.3f}; 退市占短腿比例={sg[:, delisted].sum()/ (sg.sum() if abs(sg.sum())>1e-9 else 1):+.1%}")

print(f"\n判读: OOS net@5≥~0.8 且逐年(尤其2026)正 且容量截断仍正 且尾部不致命 → 🟢 进 Phase3(模型当风控)。")
