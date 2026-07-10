"""② 诚实 per-name 强平/尾部:keep(集中,无帽) vs control(5%帽),看谁扛得住自己的尾部 → 定 keep/control。
强平诚实建模(非免费止损):杠杆L下,某币基差较开仓走阔 > (1/L−mm) → 强制平仓:
  - 已累计的 L×(−Δbasis) 亏损是真的(在cr里随价格走);
  - 逼空滑点 LIQSLIP×L×权重;
  - 冷却 COOLDOWN bar 不可复入 → 锁在最差点、错过反弹+错过funding(强平真正的代价)。
尾部测量:价格腿宽 winsor(0.5,只挡纯glitch、让真逼空入账)。健全性:杠杆↑ 该 Sharpe↓/回撤↑。
跑：python cashcarry_tail.py
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
price = np.clip(np.nan_to_num(sr - pr), -0.5, 0.5)             # 尾部测量:宽winsor,真逼空入账
cr = f0 / 8.0 + price
basis = np.full((T, N), np.nan); basis[vp] = adj[vp] / spot[vp] - 1
K, CAD, REBAL, BAND, POSHL, EXHL = 100, 24, 0.3, 0.01, 120, 72
MM, LIQSLIP, COOLDOWN, COST = 0.005, 0.005, 168, 10            # 维持保证金、强平滑点50bps、冷却7天、双腿成本10bps
lev_i = slots.index("LEVERUSDT")


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


def make_W(C):
    W = np.zeros((T, N))
    for t in range(T):
        cand = np.where(vp[t] & np.isfinite(funding[t]) & (funding[t] > 0))[0]
        if len(cand) == 0:
            continue
        if len(cand) > K:
            cand = cand[np.argsort(tdv[t, cand])[::-1][:K]]
        w = f0[t, cand]; s = w.sum()
        if s <= 0:
            continue
        w = w / s
        W[t, cand] = cap_renorm(w, C) if C < 1 else w
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


def sim(Ws, L):
    thr = (1.0 / L - MM) if L > 1 else 1e9
    cur = np.zeros(N); b0 = np.zeros(N); cd = np.zeros(N, int)
    pnl = np.zeros(T); turn = np.zeros(T); prev = np.zeros(N); nliq = 0; levloss = 0.0
    for t in range(T):
        if L > 1 and cur.any():
            dd = basis[t] - b0
            liq = (cur > 1e-9) & np.isfinite(basis[t]) & (dd > thr)
            if liq.any():
                loss = LIQSLIP * L * cur[liq].sum(); pnl[t] -= loss
                turn[t] += cur[liq].sum(); nliq += int(liq.sum())
                if liq[lev_i]:
                    levloss += loss
                cur[liq] = 0.0; cd[liq] = COOLDOWN
        cd[cd > 0] -= 1
        if t % CAD == 0:
            d = Ws[t] - cur; mv = np.abs(d) > BAND
            new = cur.copy(); new[mv] = cur[mv] + REBAL * d[mv]
            s = np.abs(new).sum(); cur = new / s if s > 0 else new
            newly = (prev <= 1e-9) & (cur > 1e-9)
            b0[newly] = np.where(np.isfinite(basis[t, newly]), basis[t, newly], 0.0)
        cur = np.where(fe[t] < 0, 0.0, cur)
        cur = np.where(cd > 0, 0.0, cur)
        turn[t] += np.abs(cur - prev).sum()
        pnl[t] += L * (cur * cr[t]).sum() - (COST / 1e4) * L * turn[t]
        prev = cur
    return pnl, nliq, levloss


def sh(p, m):
    p = p[m]; return p.mean() / p.std() * np.sqrt(ann) if p.std() > 0 else 0.0


def maxdd(p, m):
    c = np.cumsum(p[m]); return (c - np.maximum.accumulate(c)).min()


print(f"诚实强平:滑点{LIQSLIP:.1%}, 冷却{COOLDOWN}bar(7天), mm{MM}, 成本{COST}bps双腿; 价格腿winsor0.5\n")
for tag, C in [("KEEP(无帽,集中)", 1.0), ("CONTROL(5%帽)", 0.05)]:
    Ws = smooth(make_W(C), POSHL)
    print(f"=== {tag} ===")
    print(f"{'杠杆':>4s} {'OOS Sharpe':>11s} {'OOS年化':>8s} {'2026':>6s} {'最大回撤':>9s} {'强平次数':>8s} {'其中LEVER损':>11s}")
    for L in (1, 2, 3, 5):
        pnl, nliq, ll = sim(Ws, L)
        print(f"{L:>3d}x {sh(pnl,oos):>+11.2f} {pnl[oos].mean()*ann:>+8.1%} {sh(pnl,oos&(yr=='2026')):>+6.2f}"
              f" {maxdd(pnl,oos):>+9.3f} {nliq:>8d} {ll:>+11.3f}")
    print()

print("健全性:杠杆↑ 年化该↑、回撤该↑、强平次数该↑。")
print("判读:keep 高L若回撤爆/被LEVER单名拖垮、control 明显更稳 → control;keep 高L回撤仍可忍 → 对冲让尾部小 → 可keep。")
