"""信号盘点:我们手里到底能造几个"稳定 + 互相独立"的薄信号?
全部横截面 rank、因果(≤t)、h=24 前瞻 IC 逐年(稳定=逐年同号且|IC|≥~0.008);两两横截面相关(独立=|ρ|<0.3)。
只有"稳定且互独"≥4-5 个,组合/变现才值得做。跑：python signal_inventory.py
"""
import numpy as np
import pandas as pd

H = 24
z = np.load("data/clean/crypto_tensor_60min_pit.npz", allow_pickle=True)
mask = z["mask"]; adj = z["adj_close"].astype(np.float64); dates = z["dates"].astype(str); slots = list(z["slots"].astype(str))
T, N = mask.shape; nmap = {s: i for i, s in enumerate(slots)}; yr = np.array([d[:4] for d in dates])
grid_ts = np.sort(pd.read_parquet("data/clean/crypto_60min_pit.parquet", columns=["ts"])["ts"].unique()).astype(np.int64)
tpos = {int(t): i for i, t in enumerate(grid_ts)}
dow = pd.to_datetime(grid_ts, unit="ms").dayofweek.to_numpy()

funding = np.load("data/clean/funding_pit.npz", allow_pickle=True)["funding"].astype(np.float64)
basis = np.load("data/clean/basis_pit.npz", allow_pickle=True)["basis"].astype(np.float64)
m = pd.read_parquet("data/clean/metrics_60min.parquet"); m = m[m["symbol"].isin(nmap)].copy()
m["ti"] = m["ts"].astype(np.int64).map(tpos); m = m.dropna(subset=["ti"]); m["ti"] = m["ti"].astype(int); m["j"] = m["symbol"].map(nmap)
topLS = np.full((T, N), np.nan); topLS[m["ti"].to_numpy(), m["j"].to_numpy()] = m["sum_toptrader_long_short_ratio"].to_numpy(np.float64)
oi = np.full((T, N), np.nan); oi[m["ti"].to_numpy(), m["j"].to_numpy()] = m["sum_open_interest_value"].to_numpy(np.float64)

lc = np.log(np.where(adj > 0, adj, np.nan))
fwd = np.full((T, N), np.nan); fwd[:-H] = lc[H:] - lc[:-H]
ret1 = np.full((T, N), np.nan); ret1[1:] = lc[1:] - lc[:-1]


def lag(x, k):
    o = np.full_like(x, np.nan); o[k:] = x[k:] - x[:-k]; return o


def trail_std(r, w):
    s = pd.DataFrame(r).rolling(w, min_periods=w // 2).std().to_numpy(); return s


# 季节性(因果 day-of-week 期望):每币按 dow 的历史均值(只用过去)
def seasonal():
    sig = np.full((T, N), np.nan)
    # 用日级:取每天 hour==0 的 bar 当日代表,日收益按 dow 累计均值(shift 因果)
    day_idx = np.where(pd.to_datetime(grid_ts, unit="ms").hour.to_numpy() == 0)[0]
    for j in range(N):
        dr = lc[day_idx, j]; dr = dr[1:] - dr[:-1]; dd = dow[day_idx][1:]
        run = {}; exp = np.full(len(dr), np.nan)
        for i in range(len(dr)):
            if dd[i] in run and len(run[dd[i]]) >= 3:
                exp[i] = np.mean(run[dd[i]])
            if np.isfinite(dr[i]):
                run.setdefault(dd[i], []).append(dr[i])
        # broadcast 日 exp 到当天所有 hour bar
        for i, di in enumerate(day_idx[1:]):
            nxt = day_idx[i + 2] if i + 2 < len(day_idx) else T
            sig[di:nxt, j] = exp[i]
    return sig


act = mask & (adj > 0)
volw = trail_std(ret1, 168)

cands = {
    "反转1d": -lag(lc, 24),
    "反转5d": -lag(lc, 120),
    "动量30d": lag(lc, 720),
    "低波动": -volw,
    "funding(空高)": -funding,
    "basis(空高)": -basis,
    "反顶级多空比": -topLS,
    "ΔOI(24h)": lag(np.log(np.where(oi > 0, oi, np.nan)), 24),
    "季节性dow": seasonal(),
}


def xs(x):
    Z = np.full((T, N), np.nan)
    for t in range(T):
        idx = np.where(act[t] & np.isfinite(x[t]))[0]
        if len(idx) < 10:
            continue
        v = x[t, idx]
        if np.nanstd(v) < 1e-12:
            continue
        r = v.argsort().argsort().astype(np.float64)
        Z[t, idx] = (r - r.mean()) / (r.std() + 1e-12)
    return Z


def ic_year(s, m):
    out = []
    for t in np.where(m)[0]:
        idx = np.where(act[t] & np.isfinite(s[t]) & np.isfinite(fwd[t]))[0]
        if len(idx) > 10 and s[t, idx].std() > 1e-12 and fwd[t, idx].std() > 1e-12:
            out.append(np.corrcoef(s[t, idx], fwd[t, idx])[0, 1])
    return np.nanmean(out) if out else np.nan


S = {k: xs(v) for k, v in cands.items()}
names = list(S)
print(f"{'信号':>14s} {'IC全':>8s} {'2024':>8s} {'2025':>8s} {'2026':>8s} {'稳定?':>6s}")
stable = []
for k in names:
    s = S[k]; ica = ic_year(s, np.ones(T, bool))
    i24, i25, i26 = ic_year(s, yr == "2024"), ic_year(s, yr == "2025"), ic_year(s, yr == "2026")
    yrs = [i for i in (i24, i25, i26) if np.isfinite(i)]
    ok = len(yrs) >= 2 and abs(ica) >= 0.008 and all(np.sign(i) == np.sign(ica) for i in yrs)
    if ok:
        stable.append(k)
    print(f"{k:>14s} {ica:>+8.4f} {i24:>+8.4f} {i25:>+8.4f} {i26:>+8.4f} {'✓' if ok else '×':>6s}")

print(f"\n稳定信号({len(stable)}个): {stable}")
print("\n两两横截面相关(独立=|ρ|<0.3;每24bar采样):")
print("        " + " ".join(f"{k[:6]:>7s}" for k in names))
corr = np.zeros((len(names), len(names)))
for i in range(len(names)):
    for k in range(len(names)):
        cs = []
        for t in range(0, T, 24):
            idx = np.where(act[t] & np.isfinite(S[names[i]][t]) & np.isfinite(S[names[k]][t]))[0]
            if len(idx) > 10 and S[names[i]][t, idx].std() > 1e-12 and S[names[k]][t, idx].std() > 1e-12:
                cs.append(np.corrcoef(S[names[i]][t, idx], S[names[k]][t, idx])[0, 1])
        corr[i, k] = np.nanmean(cs) if cs else np.nan
    print(f"{names[i]:>8s} " + " ".join(f"{corr[i,k]:>+7.2f}" for k in range(len(names))))

# 稳定信号里挑互独子集(贪心:|ρ|<0.3)
si = [names.index(k) for k in stable]
indep = []
for i in si:
    if all(abs(corr[i, j]) < 0.3 for j in indep):
        indep.append(i)
print(f"\n稳定 且 互独(|ρ|<0.3)的信号: {[names[i] for i in indep]}  → 共 {len(indep)} 个")
print("判读: ≥4-5 个 = 变现(组合)值得做;<4 = 独立宽度不够,组合也救不了,该认。")
