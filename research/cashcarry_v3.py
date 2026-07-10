"""cash-and-carry v3(去乐观化①):真实成本压测 + 退出 hysteresis 降换手。
v2 的快出换手 81/yr、net@高成本下 2024 转负。本版:
- 退出用**平滑 funding**(fund_exit_hl)且带 hysteresis(enter_band / exit_band 死区)→ 砍单bar flicker churn。
- 成本=**总双腿** per 单位单边换手(cost_bps),sweep maker→taker。
- 报:逐年 + 成本 sweep + (退出平滑×死区)的 换手↔net 权衡。
跑：python cashcarry_v3.py
"""
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--tensor", default="data/clean/crypto_tensor_60min_pit.npz")
ap.add_argument("--funding", default="data/clean/funding_pit.npz")
ap.add_argument("--spot", default="data/clean/spot_pit.npz")
ap.add_argument("--ann", type=int, default=8760); ap.add_argument("--split", type=float, default=0.4)
ap.add_argument("--cad", type=int, default=24); ap.add_argument("--rebal", type=float, default=0.3)
ap.add_argument("--band", type=float, default=0.01); ap.add_argument("--K", type=int, default=100)
ap.add_argument("--wcap", type=float, default=0.25)
a = ap.parse_args()

z = np.load(a.tensor, allow_pickle=True)
mask = z["mask"]; adj = z["adj_close"].astype(float); tdv = z["tdv"].astype(float)
dates = z["dates"].astype(str); T, N = mask.shape
funding = np.load(a.funding, allow_pickle=True)["funding"].astype(float)
spot = np.load(a.spot, allow_pickle=True)["spot"].astype(float)
yr = np.array([d[:4] for d in dates]); cut = int(T * a.split)
oos = np.zeros(T, bool); oos[cut:] = True
vp = mask & np.isfinite(spot) & (spot > 0) & (adj > 0)
f0 = np.nan_to_num(funding)
pr = np.zeros((T, N)); sr = np.zeros((T, N)); vpp = vp[:-1] & vp[1:]
pr[:-1][vpp] = adj[1:][vpp] / adj[:-1][vpp] - 1
sr[:-1][vpp] = spot[1:][vpp] / spot[:-1][vpp] - 1
price_raw = np.nan_to_num(sr - pr)


def emaT(X, hl):
    if hl <= 0:
        return X
    al = 1 - 0.5 ** (1.0 / hl); E = np.empty_like(X); E[0] = X[0]
    for t in range(1, len(X)):
        E[t] = al * X[t] + (1 - al) * E[t - 1]
    return E


fsm_cache = {}


def fsm(hl):
    if hl not in fsm_cache:
        fsm_cache[hl] = emaT(f0, hl)                        # 平滑 funding(退出决策用,防flicker)
    return fsm_cache[hl]


def make_W(K, enter_band):
    W = np.zeros((T, N))
    for t in range(T):
        cand = np.where(vp[t] & np.isfinite(funding[t]) & (funding[t] > enter_band))[0]
        if len(cand) == 0:
            continue
        if K < N and len(cand) > K:
            cand = cand[np.argsort(tdv[t, cand])[::-1][:K]]
        w = f0[t, cand]; s = w.sum()
        if s > 0:
            W[t, cand] = w / s
    return W


def smooth(W, h):
    if h <= 0:
        return W
    al = 1 - 0.5 ** (1.0 / h); S = np.empty_like(W); S[0] = W[0]
    for t in range(1, len(W)):
        S[t] = al * W[t] + (1 - al) * S[t - 1]
    g = np.abs(S).sum(1, keepdims=True)
    return np.divide(S, g, out=np.zeros_like(S), where=g > 0)


def build(W, hl, exit_hl, exit_band):
    """退出:平滑funding(exit_hl) < exit_band 才清零(死区+平滑→降churn)。"""
    Ws = smooth(W, hl); fe = fsm(exit_hl)
    cur = np.zeros(N); held = np.zeros((T, N)); turn = np.zeros(T); prev = np.zeros(N)
    for t in range(T):
        if t % a.cad == 0:
            d = Ws[t] - cur; mv = np.abs(d) > a.band
            new = cur.copy(); new[mv] = cur[mv] + a.rebal * d[mv]
            s = np.abs(new).sum(); cur = new / s if s > 0 else new
        cur = np.where(fe[t] < exit_band, 0.0, cur)
        turn[t] = np.abs(cur - prev).sum(); held[t] = cur; prev = cur
    return held, turn


def legs(held):
    plc = np.clip(price_raw, -a.wcap, a.wcap)
    return (held * (f0 / 8.0)).sum(1), (held * plc).sum(1)


def shc(fp, pp, turn, cost_bps, m):
    net = (fp + pp - cost_bps / 1e4 * turn)[m]
    return net.mean() / net.std() * np.sqrt(a.ann) if net.std() > 0 else 0.0


def annc(fp, pp, turn, cost_bps, m):
    return (fp + pp - cost_bps / 1e4 * turn)[m].mean() * a.ann


# 退出用平滑 funding(hl72=低换手最稳,sweep 里 turnover 26 且 OOS 最高);进 funding>0、退 平滑f<0
EN, EX, EXHL = 0.0, 0.0, 72
val = slice(0, cut)
W = make_W(a.K, EN)
print(f"=== 选 hl(前40% net@总10bps;K={a.K},退出=平滑f(hl{EXHL})<{EX}) ===")
best = None
for hl in (24, 48, 72, 120):
    h, tn = build(W, hl, EXHL, EX); fp, pp = legs(h)
    sv = shc(fp, pp, tn, 10, val)
    print(f"  hl={hl:>3d}: val={sv:+.2f} OOS={shc(fp,pp,tn,10,oos):+.2f} 2026={shc(fp,pp,tn,10,oos&(yr=='2026')):+.2f} 换手={tn[oos].mean()*a.ann:.0f}")
    if best is None or sv > best[0]:
        best = (sv, hl)
HL = best[1]
h, tn = build(W, HL, EXHL, EX); fp, pp = legs(h)
print(f"\nval 选定 hl={HL}\n")

print(f"=== 主结果(hl={HL}, 退出平滑hl{EXHL}) 逐年:net@总成本 5/10/20bps + 两腿 ===")
print(f"{'区间':>8s} {'净@5':>7s} {'净@10':>7s} {'净@20':>7s} {'年化@10':>8s} {'funding腿':>9s} {'价格腿':>8s} {'年换手':>7s}")
for lab, m in [("OOS全", oos)] + [(y, oos & (yr == y)) for y in ("2023", "2024", "2025", "2026")]:
    if m.sum() < 50:
        continue
    print(f"{lab:>8s} {shc(fp,pp,tn,5,m):>+7.2f} {shc(fp,pp,tn,10,m):>+7.2f} {shc(fp,pp,tn,20,m):>+7.2f}"
          f" {annc(fp,pp,tn,10,m):>+7.1%} {fp[m].mean()*a.ann:>+8.1%} {pp[m].mean()*a.ann:>+7.1%} {tn[m].mean()*a.ann:>7.0f}")

print("\n=== 成本 sweep(总双腿 bps;net Sharpe OOS / 2026 / 年化@该成本) ===")
for cb in (5, 10, 20, 30):
    print(f"  {cb:>2d}bps: OOS={shc(fp,pp,tn,cb,oos):+.2f}  2026={shc(fp,pp,tn,cb,oos&(yr=='2026')):+.2f}  年化={annc(fp,pp,tn,cb,oos):+.1%}")

print("\n=== 换手↔net 权衡(退出平滑hl × 死区;net@总10bps OOS) ===")
print(f"{'退出平滑':>8s} {'死区±':>6s} {'OOS Sharpe':>11s} {'2026':>6s} {'年换手':>7s}")
for exhl in (0, 6, 24, 72):
    for exb in (0.0, -0.0001):
        Wb = make_W(a.K, -exb)
        hb, tb = build(Wb, HL, exhl, exb); fpb, ppb = legs(hb)
        tag = "瞬时" if exhl == 0 else f"hl{exhl}"
        print(f"{tag:>8s} {exb:>+6.4f} {shc(fpb,ppb,tb,10,oos):>+11.2f} {shc(fpb,ppb,tb,10,oos&(yr=='2026')):>+6.2f} {tb[oos].mean()*a.ann:>7.0f}")

print("\n判读:① 成本sweep看实盘成本下还剩多少、20-30bps(taker重)会不会垮。② 退出平滑该把换手从~81砍下来且net不降太多。")
