"""真 cash-and-carry(真实现货价,替掉 premiumIndex 近似)+ 杠杆/强平尾部建模。
hedged 单位 P&L = spot_ret − perp_ret + funding/8(多现货 +s, 空永续 −p, 收 funding)。
价格腿现在是真实 spot−perp(含真实基差爆裂),不是 −Δbasis 近似。
强平:孤立空永续杠杆 L,基差(perp/spot−1)较开仓走阔 > ~1/L → 强平(吃损+被迫退出错过回归)。
跑：python cashcarry_real.py
"""
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--tensor", default="data/clean/crypto_tensor_60min_pit.npz")
ap.add_argument("--funding", default="data/clean/funding_pit.npz")
ap.add_argument("--spot", default="data/clean/spot_pit.npz")
ap.add_argument("--ann", type=int, default=8760)
ap.add_argument("--split", type=float, default=0.4)
ap.add_argument("--hl", type=int, default=72); ap.add_argument("--cad", type=int, default=24)
ap.add_argument("--rebal", type=float, default=0.3); ap.add_argument("--band", type=float, default=0.01)
ap.add_argument("--bps", type=float, default=5)
ap.add_argument("--mm", type=float, default=0.005, help="维持保证金率(强平阈值=1/L−mm)")
ap.add_argument("--liqslip", type=float, default=0.003, help="强平额外滑点(逼空里成交差)")
a = ap.parse_args()

z = np.load(a.tensor, allow_pickle=True)
mask = z["mask"]; adj = z["adj_close"].astype(np.float64); tdv = z["tdv"].astype(np.float64)
dates = z["dates"].astype(str); slots = z["slots"].astype(str)
T, N = mask.shape
funding = np.load(a.funding, allow_pickle=True)["funding"].astype(np.float64)
spot = np.load(a.spot, allow_pickle=True)["spot"].astype(np.float64)
yr = np.array([d[:4] for d in dates]); cut = int(T * a.split)
oos = np.zeros(T, bool); oos[cut:] = True

vp = mask & np.isfinite(spot) & (spot > 0) & (adj > 0)               # 永续+现货都在
active = vp & np.isfinite(funding)
f0 = np.nan_to_num(funding)
pr = np.zeros((T, N)); sr = np.zeros((T, N))
vpp = vp[:-1] & vp[1:]
pr[:-1][vpp] = adj[1:][vpp] / adj[:-1][vpp] - 1
sr[:-1][vpp] = spot[1:][vpp] / spot[:-1][vpp] - 1
basis_real = np.full((T, N), np.nan); basis_real[vp] = adj[vp] / spot[vp] - 1
# 数据质量清洗:illiquid alt 现货有坏价(perp/spot 偏离离谱、单bar收益爆表)→ 剔除这些bar(否则污染真实P&L与尾部)
ret_bad = (np.abs(pr) > 1.0) | (np.abs(sr) > 1.0)                    # 单bar >100% = glitch
basis_bad = ~np.isfinite(basis_real) | (np.abs(basis_real) > 0.5)   # perp/spot 偏离 >50% = 坏价/不可交易
pr[ret_bad] = 0.0; sr[ret_bad] = 0.0
cr = np.nan_to_num(sr - pr) + f0 / 8.0                               # 真实 hedged 单位收益
badbar = ret_bad | basis_bad
cr[badbar] = 0.0                                                     # 坏bar不计P&L(持仓穿过也不吃假价)
basis_real[basis_bad] = np.nan
active = vp & np.isfinite(funding) & ~basis_bad                      # 坏bar不建仓

print(f"现货覆盖宇宙: {vp.any(0).sum()}/{N} 币有现货(可做cash-and-carry); active mask率={active.mean():.3f}")
print(f"清洗:剔除 {badbar.mean()*100:.2f}% 的bar(坏现货价:基差>50% 或 单bar收益>100%)\n")


def make_W(act):
    wpos = np.where(act, np.clip(f0, 0, None), 0.0)
    g = wpos.sum(1, keepdims=True)
    return np.divide(wpos, g, out=np.zeros_like(wpos), where=g > 0)


def smooth(W, h):
    if h <= 0:
        return W
    al = 1 - 0.5 ** (1.0 / h); S = np.empty_like(W); S[0] = W[0]
    for t in range(1, len(W)):
        S[t] = al * W[t] + (1 - al) * S[t - 1]
    g = np.abs(S).sum(1, keepdims=True)
    return np.divide(S, g, out=np.zeros_like(S), where=g > 0)


def sh(p, m=None):
    p = p if m is None else p[m]
    return p.mean() / p.std() * np.sqrt(a.ann) if p.std() > 0 else 0.0


def sim(Ws, L):
    """杠杆 L、含强平的真实模拟。pnl 已乘 L(杠杆放大收益与波动),capital=1。"""
    thr = 1.0 / L - a.mm
    cur = np.zeros(N); b0 = np.zeros(N); pnl = np.zeros(T); turn = np.zeros(T); nliq = 0
    for t in range(T):
        if L > 1 and cur.any():                                     # 强平检查:基差较开仓走阔过阈
            dd = basis_real[t] - b0
            liq = (cur > 1e-9) & np.isfinite(basis_real[t]) & (dd > thr)
            if liq.any():
                pnl[t] -= a.liqslip * L * cur[liq].sum()            # 强平额外滑点(损本身已在cr里随价格走)
                turn[t] += cur[liq].sum(); nliq += int(liq.sum()); cur[liq] = 0.0
        prev = cur.copy()
        if t % a.cad == 0:
            d = Ws[t] - cur; mv = np.abs(d) > a.band
            new = cur.copy(); new[mv] = cur[mv] + a.rebal * d[mv]
            s = np.abs(new).sum(); new = new / s if s > 0 else new
            turn[t] += np.abs(new - cur).sum(); cur = new
            newly = (prev <= 1e-9) & (cur > 1e-9)
            b0[newly] = np.where(np.isfinite(basis_real[t, newly]), basis_real[t, newly], 0.0)
        pnl[t] += L * (cur * cr[t]).sum() - 2 * (a.bps / 1e4) * L * turn[t]
    return pnl, turn, nliq


W = smooth(make_W(active), a.hl)

print("=== ① 真实价(L=1,无杠杆无强平)vs premiumIndex 近似 ===")
pnl, tn, _ = sim(W, 1)
print(f"{'区间':>10s} {'净@5':>7s} {'年化':>8s}")
for lab, m in [("OOS全", oos)] + [(y, oos & (yr == y)) for y in ("2024", "2025", "2026")]:
    if m.sum() < 50:
        continue
    print(f"{lab:>10s} {sh((pnl-0)[m]):>+7.2f} {pnl[m].mean()*a.ann:>+8.1%}")
print("  (对照 E20 premiumIndex 近似 OOS +4.15;接近=近似可靠)")

print("\n=== ② 真实基差爆裂尾部(假设无关,直接看价格) ===")
# 每个持仓 episode 的最大基差走阔(perp 相对 spot 跳多高)
held1 = np.zeros((T, N)); cur = np.zeros(N)
for t in range(T):
    if t % a.cad == 0:
        d = W[t] - cur; mv = np.abs(d) > a.band; new = cur.copy(); new[mv] = cur[mv] + a.rebal * d[mv]
        s = np.abs(new).sum(); cur = new / s if s > 0 else new
    held1[t] = cur
hpos = held1 > 1e-9
# 持仓中基差相对各币持仓期起点的走阔
maxwiden = []
for c in range(N):
    h = hpos[:, c]
    if not h.any():
        continue
    seg = []
    inpos = False; entry = 0.0
    for t in range(T):
        if h[t] and not inpos:
            inpos = True; entry = basis_real[t, c] if np.isfinite(basis_real[t, c]) else 0.0
        if inpos and np.isfinite(basis_real[t, c]):
            seg.append(basis_real[t, c] - entry)
        if not h[t] and inpos:
            inpos = False
            if seg:
                maxwiden.append(max(seg)); seg = []
    if inpos and seg:
        maxwiden.append(max(seg))
mw = np.array(maxwiden)
print(f"  持仓 episode 数={len(mw)}; 基差走阔>5%:{(mw>0.05).mean():.1%}  >10%:{(mw>0.10).mean():.1%}  "
      f">20%:{(mw>0.20).mean():.1%}  >50%:{(mw>0.50).mean():.1%}  最大={mw.max():.1%}")

print("\n=== ②b 两腿分解(真实价,L=1,逐年;判 2026 负是结构性还是选币) ===")
fl = np.where(badbar, 0.0, f0 / 8.0); plc = np.where(badbar, 0.0, np.nan_to_num(sr - pr))
fpnl = (held1 * fl).sum(1); ppnl = (held1 * plc).sum(1)
for lab, m in [("OOS全", oos)] + [(y, oos & (yr == y)) for y in ("2024", "2025", "2026")]:
    if m.sum() < 50:
        continue
    print(f"  {lab:>6s}: funding {fpnl[m].mean()*a.ann:>+6.1%}(Sh{sh(fpnl[m]):>+5.1f})  "
          f"price(spot−perp) {ppnl[m].mean()*a.ann:>+7.1%}(Sh{sh(ppnl[m]):>+5.1f})")

print("\n=== ③ 杠杆/强平 sweep(net@5 OOS;年化随L放大,Sharpe看强平何时吃掉) ===")
print(f"{'L':>4s} {'OOS Sharpe':>11s} {'OOS年化':>8s} {'2026':>7s} {'强平次数':>8s}")
for L in (1, 2, 3, 5, 10):
    pnl, tn, nliq = sim(W, L)
    print(f"{L:>4d} {sh(pnl[oos]):>+11.2f} {pnl[oos].mean()*a.ann:>+8.1%} {sh(pnl[oos & (yr=='2026')]):>+7.2f} {nliq:>8d}")

print("\n判读: ① 真实价≈近似=机制稳。② 爆裂尾部小=强平罕见→低杠杆安全。③ Sharpe 在某 L 前平、之后被强平吃掉=安全杠杆上限。")
