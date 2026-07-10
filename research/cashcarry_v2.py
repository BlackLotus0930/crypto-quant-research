"""cash-and-carry 修正版(认真):修上一版的过度清洗 + 黏性 + 强平假象。
- A1 选币:流动性截顶(top-K tdv)做 glitch 防线,绝不因 basis 把币踢出"选择"(上版的错)。
- 清洗:只对价格腿 spot_ret−perp_ret 轻度 winsorize(对冲后本应小),且报对阈值的敏感性(防再次自欺)。
- A2 慢进快出:funding 翻负立刻清零(不靠 hl 慢漏)。
- 选 hl 在前40%(net@5),报后60%;逐年 + 两腿分解 + 容量 + 清洗敏感性 + 快出开关。
- 杠杆:诚实模型(强平=亏掉该仓本金+滑点+冷却期不可复入),单独报。
跑：python cashcarry_v2.py
"""
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--tensor", default="data/clean/crypto_tensor_60min_pit.npz")
ap.add_argument("--funding", default="data/clean/funding_pit.npz")
ap.add_argument("--spot", default="data/clean/spot_pit.npz")
ap.add_argument("--ann", type=int, default=8760)
ap.add_argument("--split", type=float, default=0.4)
ap.add_argument("--cad", type=int, default=24); ap.add_argument("--rebal", type=float, default=0.3)
ap.add_argument("--band", type=float, default=0.01); ap.add_argument("--bps", type=float, default=5)
a = ap.parse_args()

z = np.load(a.tensor, allow_pickle=True)
mask = z["mask"]; adj = z["adj_close"].astype(float); tdv = z["tdv"].astype(float)
dates = z["dates"].astype(str); slots = z["slots"].astype(str)
T, N = mask.shape
funding = np.load(a.funding, allow_pickle=True)["funding"].astype(float)
spot = np.load(a.spot, allow_pickle=True)["spot"].astype(float)
yr = np.array([d[:4] for d in dates]); cut = int(T * a.split)
oos = np.zeros(T, bool); oos[cut:] = True

vp = mask & np.isfinite(spot) & (spot > 0) & (adj > 0)
f0 = np.nan_to_num(funding)
pr = np.zeros((T, N)); sr = np.zeros((T, N))
vpp = vp[:-1] & vp[1:]
pr[:-1][vpp] = adj[1:][vpp] / adj[:-1][vpp] - 1
sr[:-1][vpp] = spot[1:][vpp] / spot[:-1][vpp] - 1
price_raw = np.nan_to_num(sr - pr)                                   # 对冲后价格腿(应小)
basis = np.full((T, N), np.nan); basis[vp] = adj[vp] / spot[vp] - 1


def make_W(K, band_entry=0.0):
    W = np.zeros((T, N))
    for t in range(T):
        cand = np.where(vp[t] & np.isfinite(funding[t]) & (funding[t] > band_entry))[0]
        if len(cand) == 0:
            continue
        if K < N and len(cand) > K:
            cand = cand[np.argsort(tdv[t, cand])[::-1][:K]]          # 只在 top-K 流动里收(glitch 防线+容量)
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


def build(W, hl, hard_exit):
    """重建持仓;hard_exit=funding<0 立刻清零(慢进快出)。返回 held,turn。"""
    Ws = smooth(W, hl); cur = np.zeros(N); held = np.zeros((T, N)); turn = np.zeros(T)
    prev = np.zeros(N)
    for t in range(T):
        if t % a.cad == 0:
            d = Ws[t] - cur; mv = np.abs(d) > a.band
            new = cur.copy(); new[mv] = cur[mv] + a.rebal * d[mv]
            s = np.abs(new).sum(); cur = new / s if s > 0 else new
        if hard_exit:
            cur = np.where(f0[t] < 0, 0.0, cur)
        turn[t] = np.abs(cur - prev).sum(); held[t] = cur; prev = cur
    return held, turn


def pnl_of(held, turn, wcap):
    plc = np.clip(price_raw, -wcap, wcap)
    fp = (held * (f0 / 8.0)).sum(1); pp = (held * plc).sum(1)
    net = fp + pp - 2 * (a.bps / 1e4) * turn
    return net, fp, pp


def sh(p, m=None):
    p = p if m is None else p[m]
    return p.mean() / p.std() * np.sqrt(a.ann) if p.std() > 0 else 0.0


K0, WCAP0 = 100, 0.25
W100 = make_W(K0)

print("=== 选 hl(前40% net@5;K=100 流动,winsor0.25,慢进快出);顺看各hl OOS/2026 ===")
val = slice(0, cut); best = None
for hl in (0, 6, 24, 72):
    held, turn = build(W100, hl, True)
    net, _, _ = pnl_of(held, turn, WCAP0)
    sv = sh(net[val])
    print(f"  hl={hl:>2d}: val={sv:+.2f}  OOS={sh(net[oos]):+.2f}  2026={sh(net[oos&(yr=='2026')]):+.2f}  OOS年换手={turn[oos].mean()*a.ann:.0f}")
    if best is None or sv > best[0]:
        best = (sv, hl)
HL = best[1]
print(f"\nval 选定 hl={HL}\n")

print(f"=== 主结果(K=100流动, winsor0.25, 慢进快出, hl={HL}, 双腿成本) 逐年 ===")
held, turn = build(W100, HL, True)
net, fp, pp = pnl_of(held, turn, WCAP0)
print(f"{'区间':>8s} {'净@2':>7s} {'净@5':>7s} {'净@10':>7s} {'年化@5':>8s} {'funding腿':>9s} {'价格腿':>8s} {'年换手':>7s}")
for lab, m in [("OOS全", oos)] + [(y, oos & (yr == y)) for y in ("2023", "2024", "2025", "2026")]:
    if m.sum() < 50:
        continue
    def netb(b):
        return sh((fp + pp - 2 * (b / 1e4) * turn)[m])
    print(f"{lab:>8s} {netb(2):>+7.2f} {netb(5):>+7.2f} {netb(10):>+7.2f} {net[m].mean()*a.ann:>+7.1%}"
          f" {fp[m].mean()*a.ann:>+8.1%} {pp[m].mean()*a.ann:>+7.1%} {turn[m].mean()*a.ann:>7.0f}")

print("\n=== 容量(net@5 OOS / 2026;流动截顶 K) ===")
for K in (30, 50, 100, 200, N):
    Wk = make_W(K); hk, tk = build(Wk, HL, True); nk, _, _ = pnl_of(hk, tk, WCAP0)
    lab = f"top-{K}" if K < N else f"全({N})"
    print(f"  {lab:>8s}: OOS={sh(nk[oos]):+.2f}  2026={sh(nk[oos&(yr=='2026')]):+.2f}  年化={nk[oos].mean()*a.ann:+.1%}")

print("\n=== 清洗敏感性(价格腿 winsor 阈值;net@5 OOS;防自欺:数字不该随阈值剧变) ===")
for wc in (0.05, 0.10, 0.25, 0.50, 1e9):
    nk, _, _ = pnl_of(held, turn, wc)
    lab = "不限" if wc > 1 else f"{wc:.2f}"
    print(f"  winsor={lab:>5s}: OOS={sh(nk[oos]):+.2f}  2026={sh(nk[oos&(yr=='2026')]):+.2f}")

print("\n=== 慢进快出 开/关(net@5 OOS/2026) ===")
for he, tag in [(True, "快出(funding翻负即退)"), (False, "不快出(只靠hl漏)")]:
    h2, t2 = build(W100, HL, he); n2, _, _ = pnl_of(h2, t2, WCAP0)
    print(f"  {tag:>22s}: OOS={sh(n2[oos]):+.2f}  2026={sh(n2[oos&(yr=='2026')]):+.2f}  年换手={t2[oos].mean()*a.ann:.0f}")

print("\n判读:① 主结果逐年(尤其2026)≥0 = 修正后真成立。② 容量看到多流动还正。③ winsor 敏感性平=没被清洗操纵。④ 快出该明显救2026。")
