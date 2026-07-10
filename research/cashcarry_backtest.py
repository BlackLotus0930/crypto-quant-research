"""cash-and-carry(多现货+空永续,同币,价格中性)funding 收割 —— 干净验证 A 方向。
每单位(coin i)P&L/bar = funding/8(空永续收正funding) − Δbasis(价格腿被现货对冲,只剩基差收敛)。
只在 funding>0 的币上做(空永续收,长现货无需借币);按 funding 权重;双腿成本(2×bps)。死币在内、扣成本、逐年。
干净不偷看:hl 前40%选(net@5),报后60%。诊断:funding翻负、基差爆裂尾部、容量。
跑：python cashcarry_backtest.py
"""
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--tensor", default="data/clean/crypto_tensor_60min_pit.npz")
ap.add_argument("--funding", default="data/clean/funding_pit.npz")
ap.add_argument("--basis", default="data/clean/basis_pit.npz")
ap.add_argument("--ann", type=int, default=8760)
ap.add_argument("--split", type=float, default=0.4)
ap.add_argument("--hl", type=int, default=-1, help="平滑半衰期;-1=按val自选,否则强制(慢carry建议72)")
ap.add_argument("--cad", type=int, default=24); ap.add_argument("--rebal", type=float, default=0.3)
ap.add_argument("--band", type=float, default=0.01)
a = ap.parse_args()

z = np.load(a.tensor, allow_pickle=True)
mask = z["mask"]; tdv = z["tdv"].astype(np.float64)
dates = z["dates"].astype(str); slots = z["slots"].astype(str)
T, N = mask.shape
funding = np.load(a.funding, allow_pickle=True)["funding"].astype(np.float64)
zb = np.load(a.basis, allow_pickle=True)
basis = zb["basis"].astype(np.float64)
assert list(zb["dates"].astype(str)) == list(dates), "basis 与张量未对齐"
yr = np.array([d[:4] for d in dates]); cut = int(T * a.split)

active = mask & np.isfinite(funding) & np.isfinite(basis)
f0 = np.nan_to_num(funding)
db = np.zeros_like(basis); db[1:] = np.diff(basis, axis=0); db = np.nan_to_num(db)   # Δbasis(币入场处diff为nan→0)
cr = f0 / 8.0 - db                                              # 每单位 cash-and-carry 每bar收益
fund_lo = f0 / 8.0                                              # funding 收割腿
basis_lo = -db                                                 # 基差腿(-Δbasis)


def make_W(act):
    """funding>0 的币按 funding 权重(收得多权重大),归一 Σw=1。价格中性按构造(每单位现货对冲永续)。"""
    W = np.zeros((T, N))
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


def run(W, hl, keep=False):
    Ws = smooth(W, hl)
    cur = np.zeros(N); pnl = np.zeros(T); fpnl = np.zeros(T); bpnl = np.zeros(T); turn = np.zeros(T)
    held_all = np.zeros((T, N)) if keep else None
    for t in range(T):
        if t % a.cad == 0:
            d = Ws[t] - cur; mv = np.abs(d) > a.band
            new = cur.copy(); new[mv] = cur[mv] + a.rebal * d[mv]
            s = np.abs(new).sum(); new = new / s if s > 0 else new   # gross 归一
            turn[t] = np.abs(new - cur).sum(); cur = new
        pnl[t] = (cur * cr[t]).sum(); fpnl[t] = (cur * fund_lo[t]).sum(); bpnl[t] = (cur * basis_lo[t]).sum()
        if keep:
            held_all[t] = cur
    return (pnl, fpnl, bpnl, turn, held_all) if keep else (pnl, fpnl, bpnl, turn)


def sh(p, m=None):
    p = p if m is None else p[m]
    return p.mean() / p.std() * np.sqrt(a.ann) if p.std() > 0 else 0.0


def net(pnl, turn, bps):
    return pnl - 2 * bps / 1e4 * turn                          # 双腿(现货+永续)各付一次


# ---- 选 hl(前40%,net@5)+ 顺带看各 hl 的 OOS/2026 与换手(诊断 2026 是不是换手/成本问题) ----
W = make_W(active)
val = slice(0, cut)
oos = np.zeros(T, bool); oos[cut:] = True
y26 = oos & (yr == "2026")
best = None
for hl in (0, 6, 24, 72):
    pn, fp, bp, tn = run(W, hl)
    n5 = net(pn, tn, 5); sv = sh(n5[val])
    print(f"  hl={hl:>2d} net@5: val(前40%)={sv:+.2f}  OOS={sh(n5[oos]):+.2f}  2026={sh(n5[y26]):+.2f}  OOS年换手={tn[oos].mean()*a.ann:.0f}", flush=True)
    if best is None or sv > best[0]:
        best = (sv, hl)
HL = a.hl if a.hl >= 0 else best[1]
print(f"\nval 选定 hl={HL}; cad={a.cad} rebal={a.rebal} band={a.band}; 双腿成本\n")

pn, fp, bp, tn, held = run(W, HL, keep=True)
oos = np.zeros(T, bool); oos[cut:] = True
print("=== 后60%(OOS)逐年 净Sharpe(双腿成本) ===")
print(f"{'区间':>10s} {'净@2':>7s} {'净@5':>7s} {'净@10':>7s} {'年化@5':>8s} {'年换手':>7s} {'仓位%':>6s}")
for lab, m in [("OOS全", oos)] + [(y, oos & (yr == y)) for y in ("2023", "2024", "2025", "2026")]:
    if m.sum() < 50:
        continue
    n5 = net(pn, tn, 5); gross = np.abs(held).sum(1)
    print(f"{lab:>10s} {sh(net(pn,tn,2),m):>+7.2f} {sh(n5,m):>+7.2f} {sh(net(pn,tn,10),m):>+7.2f}"
          f" {n5[m].mean()*a.ann:>+7.1%} {tn[m].mean()*a.ann:>7.0f} {gross[m].mean():>5.0%}")

print("\n=== 两腿分解(OOS) ===")
print(f"  funding收割: Sharpe={sh(fp[oos]):+.2f} 年化={fp[oos].mean()*a.ann:+.1%} 累计={fp[oos].sum():+.3f}")
print(f"  基差腿(-Δb): Sharpe={sh(bp[oos]):+.2f} 年化={bp[oos].mean()*a.ann:+.1%} 累计={bp[oos].sum():+.3f}  (有界均值回归=噪声)")

print("\n=== 风险诊断(OOS) ===")
hpos = held > 1e-9
wfund = np.where(hpos, held * f0, 0).sum(1) / np.where(held.sum(1) > 1e-9, held.sum(1), 1)
print(f"  持仓加权 funding 均值={np.nanmean(wfund[oos]):+.6e}/8h (>0=在收;翻负=在付)")
print(f"  funding翻负的持仓bar占比={np.mean((wfund[oos] < 0))*100:.1f}%")
nz = net(pn, tn, 5)[oos]; cum = np.cumsum(nz); dd = (cum - np.maximum.accumulate(cum)).min()
print(f"  net@5 最大回撤={dd:+.3f}; 基差腿最差单bar={bp[oos].min():+.4f}; funding腿最差单bar={fp[oos].min():+.4f}")

print("\n=== 容量探针(每bar限 top-K 流动 tdv;net@5 OOS) ===")
for K in (20, 50, 100, N):
    if K >= N:
        actK = active
    else:
        actK = np.zeros((T, N), bool)
        for t in range(T):
            v = np.where(active[t])[0]
            if len(v):
                actK[t, v[np.argsort(tdv[t, v])[::-1][:K]]] = True
    pnK, fpK, bpK, tnK = run(make_W(actK), HL)
    nK = net(pnK, tnK, 5)
    lab = f"top-{K}" if K < N else f"全({N})"
    print(f"  {lab:>8s}: net@5 OOS={sh(nK[oos]):+.2f}  2026={sh(nK[oos & (yr=='2026')]):+.2f}  年化={nK[oos].mean()*a.ann:+.1%}")

print(f"\n判读: OOS net@5≥~1.0 且逐年(尤其2026)正 且 top-50/100 仍正 且翻负占比低/回撤可控 → 🟢 真 cash-and-carry 成立。")
