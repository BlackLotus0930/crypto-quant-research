"""keep 的高年化是只靠 LEVER 一个币,还是每年都有'LEVER-2'这种 fat 名字撑?
- keep(无帽)逐年:全量 / 去掉LEVER / 去掉全期top5。
- 逐年头号贡献币 + 占比(看 fat 名字是否年年有,还是只在 LEVER 那两年)。
- 全史扫'LEVER-like'(sustained 高funding+流动)候选,看活跃年份分布。
跑：python cashcarry_keepdiag.py
"""
import numpy as np

z = np.load("data/clean/crypto_tensor_60min_pit.npz", allow_pickle=True)
mask = z["mask"]; adj = z["adj_close"].astype(float); tdv = z["tdv"].astype(float)
dates = z["dates"].astype(str); slots = list(z["slots"].astype(str)); T, N = mask.shape
funding = np.load("data/clean/funding_pit.npz", allow_pickle=True)["funding"].astype(float)
spot = np.load("data/clean/spot_pit.npz", allow_pickle=True)["spot"].astype(float)
yr = np.array([d[:4] for d in dates]); cut = int(T * 0.4); ann = 8760
oos = np.zeros(T, bool); oos[cut:] = True
vp = mask & np.isfinite(spot) & (spot > 0) & (adj > 0)
f0 = np.nan_to_num(funding)
pr = np.zeros((T, N)); sr = np.zeros((T, N)); vpp = vp[:-1] & vp[1:]
pr[:-1][vpp] = adj[1:][vpp] / adj[:-1][vpp] - 1
sr[:-1][vpp] = spot[1:][vpp] / spot[:-1][vpp] - 1
cr = f0 / 8.0 + np.clip(np.nan_to_num(sr - pr), -0.25, 0.25)
K, CAD, REBAL, BAND, POSHL, EXHL = 100, 24, 0.3, 0.01, 120, 72


def make_W(excl):
    W = np.zeros((T, N))
    for t in range(T):
        cand = np.where(vp[t] & np.isfinite(funding[t]) & (funding[t] > 0))[0]
        cand = np.array([c for c in cand if c not in excl])
        if len(cand) == 0:
            continue
        if len(cand) > K:
            cand = cand[np.argsort(tdv[t, cand])[::-1][:K]]
        w = f0[t, cand]; s = w.sum()
        if s > 0:
            W[t, cand] = w / s
    return W


def smooth(W, h):
    al = 1 - 0.5 ** (1.0 / h); S = np.empty_like(W); S[0] = W[0]
    for t in range(1, len(W)):
        S[t] = al * W[t] + (1 - al) * S[t - 1]
    g = np.abs(S).sum(1, keepdims=True)
    return np.divide(S, g, out=np.zeros_like(S), where=g > 0)


def emaT(X, hl):
    al = 1 - 0.5 ** (1.0 / hl); E = np.empty_like(X); E[0] = X[0]
    for t in range(1, len(X)):
        E[t] = al * X[t] + (1 - al) * E[t - 1]
    return E


fe = emaT(f0, EXHL)


def run(excl=set()):
    Ws = smooth(make_W(excl), POSHL); cur = np.zeros(N); contrib = np.zeros((T, N))
    for t in range(T):
        if t % CAD == 0:
            d = Ws[t] - cur; mv = np.abs(d) > BAND
            new = cur.copy(); new[mv] = cur[mv] + REBAL * d[mv]
            s = np.abs(new).sum(); cur = new / s if s > 0 else new
        cur = np.where(fe[t] < 0, 0.0, cur)
        contrib[t] = cur * cr[t]
    return contrib


lev = slots.index("LEVERUSDT")
full = run()
pnl_full = full.sum(1)
# 全期 top5 贡献币
top5 = np.argsort(full[oos].sum(0))[::-1][:5]
ex_lev = run({lev}).sum(1)
ex_top5 = run(set(top5.tolist())).sum(1)


def sh(p, m):
    return p[m].mean() / p[m].std() * np.sqrt(ann) if p[m].std() > 0 else 0.0


print("=== keep 逐年年化(双腿成本前的毛,看结构):全量 / 去LEVER / 去全期top5 ===")
print(f"{'年':>6s} {'全量':>9s} {'去LEVER':>9s} {'去top5':>9s}   {'全量Sh':>7s} {'去LEVER Sh':>10s}")
for y in ("2023", "2024", "2025", "2026"):
    m = oos & (yr == y)
    if m.sum() < 50:
        continue
    print(f"{y:>6s} {pnl_full[m].mean()*ann:>+8.1%} {ex_lev[m].mean()*ann:>+8.1%} {ex_top5[m].mean()*ann:>+8.1%}"
          f"   {sh(pnl_full,m):>+7.2f} {sh(ex_lev,m):>+10.2f}")
m = oos
print(f"{'OOS全':>6s} {pnl_full[m].mean()*ann:>+8.1%} {ex_lev[m].mean()*ann:>+8.1%} {ex_top5[m].mean()*ann:>+8.1%}"
      f"   {sh(pnl_full,m):>+7.2f} {sh(ex_lev,m):>+10.2f}")

print("\n=== 逐年头号贡献币 + 占该年正P&L比例(fat名字是否年年有?) ===")
for y in ("2023", "2024", "2025", "2026"):
    m = oos & (yr == y)
    if m.sum() < 50:
        continue
    cp = full[m].sum(0); o = np.argsort(cp)[::-1]
    poss = cp[cp > 0].sum()
    s = " ".join(f"{slots[i]}({cp[i]/poss*100:.0f}%)" for i in o[:3])
    print(f"  {y}: {s}")

print("\n=== 'LEVER-like'候选:OOS内 sustained 高funding(funding>0.0005/8h 即年化~55%+ 占其活跃 >30%)且流动 ===")
cnt = 0
for c in range(N):
    v = vp[:, c] & oos
    if v.sum() < 24 * 60:                                   # 至少活跃60天
        continue
    hi = (funding[v, c] > 0.0005)
    if np.nanmean(hi) > 0.30 and np.nanmedian(tdv[v, c]) > 2e7:
        yrs = sorted(set(yr[v]))
        print(f"  {slots[c]:16s} 高funding占比{np.nanmean(hi)*100:.0f}% 中位$vol{np.nanmedian(tdv[v,c]):.1e} 活跃年{yrs}")
        cnt += 1
print(f"  共 {cnt} 个 LEVER-like 候选")

print("\n判读:去LEVER后2025/26若塌=高年化只靠LEVER;逐年头号若只有LEVER那两年是大占比、其余年~2%=fat名字非年年有=不可重复。")
