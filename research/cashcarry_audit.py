"""审计 +1.71 真实 cash-and-carry 有没有猫腻。构造=全宇宙 funding 权重、中等换手(那个稳健基线)。
查:① 未来信息清洗(ret_bad用t+1) vs 因果清洗 ② +1bar 滞后(防泄漏金标准) ③ 成本2/5/10/20
④ 幸存者(死币在不在/占多少) ⑤ 符号placebo(反号该亏、打乱该归零)。
跑：python cashcarry_audit.py
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
ap.add_argument("--band", type=float, default=0.01)
a = ap.parse_args()
rng = np.random.default_rng(0)

z = np.load(a.tensor, allow_pickle=True)
mask = z["mask"]; adj = z["adj_close"].astype(np.float64)
dates = z["dates"].astype(str); slots = z["slots"].astype(str)
T, N = mask.shape
funding = np.load(a.funding, allow_pickle=True)["funding"].astype(np.float64)
spot = np.load(a.spot, allow_pickle=True)["spot"].astype(np.float64)
yr = np.array([d[:4] for d in dates]); cut = int(T * a.split)
oos = np.zeros(T, bool); oos[cut:] = True

vp = mask & np.isfinite(spot) & (spot > 0) & (adj > 0)
f0 = np.nan_to_num(funding)
pr = np.zeros((T, N)); sr = np.zeros((T, N)); vpp = vp[:-1] & vp[1:]
pr[:-1][vpp] = adj[1:][vpp] / adj[:-1][vpp] - 1
sr[:-1][vpp] = spot[1:][vpp] / spot[:-1][vpp] - 1
basis = np.full((T, N), np.nan); basis[vp] = adj[vp] / spot[vp] - 1
basis_bad = ~np.isfinite(basis) | (np.abs(basis) > 0.5)              # 当下基差,≤t,因果
active = vp & np.isfinite(funding) & ~basis_bad

raw_pl = np.nan_to_num(sr - pr)
# 两种清洗:future_zero=按 t+1 收益判glitch置0(原版,含未来);causal_winsor=只夹幅度(无未来)
ret_bad = (np.abs(pr) > 1.0) | (np.abs(sr) > 1.0)                    # 用 t+1 → 未来
pl_future = raw_pl.copy(); pl_future[ret_bad | basis_bad] = 0.0
pl_causal = np.clip(raw_pl, -0.3, 0.3); pl_causal[basis_bad] = 0.0   # 因果:夹±30%、只按当下basis剔


def target(fthresh=0.0, shuffle=False, neg=False):
    W = np.zeros((T, N))
    for t in range(T):
        idx = np.where(active[t] & (funding[t] > fthresh))[0]
        if len(idx) == 0:
            continue
        fv = -f0[t, idx] if neg else f0[t, idx]
        if shuffle:
            fv = rng.permutation(fv)                                 # 打乱 funding↔币 关系
        w = np.clip(fv, 0, None); s = w.sum()
        if s > 0:
            W[t, idx] = w / s
    return W


def run(W, pl, fsign=1.0, lag=0):
    cur = np.zeros(N); pnl = np.zeros(T); turn = np.zeros(T); bps = 5
    for t in range(T):
        prev = cur.copy()
        ti = t - lag
        if t % a.cad == 0 and ti >= 0:
            d = W[ti] - cur; mv = np.abs(d) > a.band
            new = cur.copy(); new[mv] = cur[mv] + a.rebal * d[mv]; cur = new
        turn[t] = np.abs(cur - prev).sum()
        pnl[t] = (cur * (pl[t] + fsign * f0[t] / 8.0)).sum() - 2 * (bps / 1e4) * turn[t]
    return pnl, turn


def sh(p, m=None):
    p = p if m is None else p[m]
    return p.mean() / p.std() * np.sqrt(a.ann) if p.std() > 0 else 0.0


def line(name, pnl, tn=None):
    extra = f" 年换手={tn[oos].mean()*a.ann:.0f}" if tn is not None else ""
    print(f"  {name:>30s}: OOS {sh(pnl[oos]):>+5.2f}  2024 {sh(pnl[oos&(yr=='2024')]):>+5.2f}"
          f"  2025 {sh(pnl[oos&(yr=='2025')]):>+5.2f}  2026 {sh(pnl[oos&(yr=='2026')]):>+5.2f}{extra}")


W = target()
print("=== ① 清洗方式:未来信息(原) vs 因果 ===")
p1, t1 = run(W, pl_future); line("future_zero(原,含未来)", p1, t1)
p2, t2 = run(W, pl_causal); line("causal_winsor(因果,无未来)", p2, t2)

print("\n=== ② +1bar 滞后(防泄漏金标准;扛住=干净) ===")
line("无滞后", run(W, pl_causal)[0]); line("+1bar 滞后", run(W, pl_causal, lag=1)[0])
line("+2bar 滞后", run(W, pl_causal, lag=2)[0])

print("\n=== ③ 成本敏感(因果清洗) ===")
for bps in (2, 5, 10, 20):
    cur = np.zeros(N); pnl = np.zeros(T); turn = np.zeros(T)
    for t in range(T):
        prev = cur.copy()
        if t % a.cad == 0:
            d = W[t] - cur; mv = np.abs(d) > a.band; new = cur.copy(); new[mv] = cur[mv] + a.rebal * d[mv]; cur = new
        turn[t] = np.abs(cur - prev).sum()
        pnl[t] = (cur * (pl_causal[t] + f0[t] / 8.0)).sum() - 2 * (bps / 1e4) * turn[t]
    line(f"bps={bps}(双腿)", pnl)

print("\n=== ④ 幸存者(死币在不在) ===")
last_active = np.array([np.where(active[:, c])[0].max() if active[:, c].any() else -1 for c in range(N)])
delisted = (last_active >= 0) & (last_active < T - 24 * 14)
# 死币 P&L 贡献:重建held,拆死/活
cur = np.zeros(N); held = np.zeros((T, N))
for t in range(T):
    if t % a.cad == 0:
        d = W[t] - cur; mv = np.abs(d) > a.band; new = cur.copy(); new[mv] = cur[mv] + a.rebal * d[mv]; cur = new
    held[t] = cur
contrib = held * (pl_causal + f0 / 8.0)
print(f"  cash-and-carry 宇宙 {int(vp.any(0).sum())} 币;其中后来退市 {int(delisted.sum())} 个")
print(f"  死币累计P&L={contrib[:, delisted].sum():+.3f}  活币累计P&L={contrib[:, ~delisted].sum():+.3f}"
      f"  (死币占比 {contrib[:,delisted].sum()/contrib.sum()*100:+.0f}%)")

print("\n=== ⑤ 符号 placebo(反号该亏、打乱该归零) ===")
line("正常", run(W, pl_causal)[0])
line("funding反号(反向carry)", run(target(neg=True), pl_causal, fsign=-1.0)[0])
line("funding横截面打乱", run(target(shuffle=True), pl_causal, fsign=1.0)[0])

print("\n判读: ①因果≈原 ②滞后不崩 ③成本不敏感 ④死币非主导 ⑤反号亏/打乱归零 → 干净无猫腻。")
