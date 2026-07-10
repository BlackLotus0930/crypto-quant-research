"""信号#3 探针:OI/持仓(metrics)里有没有独立 IC?且和 funding 信号正交吗(variance machine 要独立)?
metrics: sum_open_interest_value(OI) + 3 个多空比(top交易者/账户数/taker)。对齐到 pit 张量,横截面 IC,逐年。
候选信号(≤t 因果,横截面 z):反 top-LSR(淡化大户拥挤)/反 retail-LSR/聪明-散户分歧/ΔOI。
跑：python oi_probe.py
"""
import argparse
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--tensor", default="data/clean/crypto_tensor_60min_pit.npz")
ap.add_argument("--panel", default="data/clean/crypto_60min_pit.parquet")
ap.add_argument("--metrics", default="data/clean/metrics_60min.parquet")
ap.add_argument("--funding", default="data/clean/funding_pit.npz")
ap.add_argument("--h", type=int, default=24, help="前瞻 horizon(bar);24=1天")
a = ap.parse_args()

z = np.load(a.tensor, allow_pickle=True)
mask = z["mask"]; adj = z["adj_close"].astype(np.float64)
dates = z["dates"].astype(str); slots = list(z["slots"].astype(str))
T, N = mask.shape
nmap = {s: i for i, s in enumerate(slots)}
grid_ts = np.sort(pd.read_parquet(a.panel, columns=["ts"])["ts"].unique()).astype(np.int64)
yr = np.array([d[:4] for d in dates])

# 对齐 metrics 到 pit 张量(merge_asof ≤t 因果)
m = pd.read_parquet(a.metrics)
cols = ["sum_open_interest_value", "sum_toptrader_long_short_ratio", "count_long_short_ratio", "sum_taker_long_short_vol_ratio"]
OI = np.full((T, N), np.nan); TT = np.full((T, N), np.nan); CN = np.full((T, N), np.nan); TK = np.full((T, N), np.nan)
arrs = {"sum_open_interest_value": OI, "sum_toptrader_long_short_ratio": TT, "count_long_short_ratio": CN, "sum_taker_long_short_vol_ratio": TK}
gdf = pd.DataFrame({"ts": grid_ts})
have = 0
for sym, g in m.groupby("symbol", sort=False):
    j = nmap.get(sym)
    if j is None:
        continue
    g = g.sort_values("ts").drop_duplicates("ts", keep="last")
    mm = pd.merge_asof(gdf, g[["ts"] + cols], on="ts", direction="backward")
    for c in cols:
        arrs[c][:, j] = mm[c].to_numpy(np.float64)
    have += 1
print(f"metrics 对齐 {have}/{N} 币到 pit; OI 覆盖率={np.isfinite(OI).any(0).mean():.1%}\n")

# 前瞻收益
fwd = np.full((T, N), np.nan)
good = mask.copy()
lc = np.log(np.where(adj > 0, adj, np.nan))
fwd[:-a.h] = lc[a.h:] - lc[:-a.h]
active = mask & np.isfinite(OI) & np.isfinite(TT) & np.isfinite(CN)

dOI = np.full((T, N), np.nan); dOI[a.h:] = np.log(np.where(OI[a.h:] > 0, OI[a.h:], np.nan)) - np.log(np.where(OI[:-a.h] > 0, OI[:-a.h], np.nan))

# 候选信号(原始值;IC 的符号告诉方向)
cands = {
    "top-LSR(大户多空比)": TT,
    "retail-LSR(账户多空比)": CN,
    "taker-LSR": TK,
    "聪明-散户分歧(top−retail)": TT - CN,
    "ΔOI": dOI,
    "log-OI": np.log(np.where(OI > 0, OI, np.nan)),
}


def xs_z(x, act):
    """每bar横截面 zscore(active 内)。"""
    Z = np.full_like(x, np.nan)
    for t in range(T):
        idx = np.where(act[t] & np.isfinite(x[t]))[0]
        if len(idx) < 10:
            continue
        v = x[t, idx]; sd = v.std()
        if sd > 1e-12:
            Z[t, idx] = (v - v.mean()) / sd
    return Z


def ic_series(sig, act):
    """每bar横截面 pearson(sig, fwd);返回 T 向量。"""
    out = np.full(T, np.nan)
    for t in range(T):
        idx = np.where(act[t] & np.isfinite(sig[t]) & np.isfinite(fwd[t]))[0]
        if len(idx) < 10:
            continue
        s = sig[t, idx]; f = fwd[t, idx]
        if s.std() > 1e-12 and f.std() > 1e-12:
            out[t] = np.corrcoef(s, f)[0, 1]
    return out


# funding 信号(carry):−z(funding),做独立性对照
fund = np.load(a.funding, allow_pickle=True)["funding"].astype(np.float64)
fsig = xs_z(-fund, mask & np.isfinite(fund))

print(f"=== 各候选信号 横截面 IC(h={a.h}bar前瞻;全程 + 逐年) ===")
print(f"{'信号':>22s} {'全程IC':>8s} {'2024':>7s} {'2025':>7s} {'2026':>7s}")
best = None
for name, raw in cands.items():
    sig = xs_z(raw, active)
    ics = ic_series(sig, active)
    full = np.nanmean(ics)
    row = [full] + [np.nanmean(ics[yr == y]) for y in ("2024", "2025", "2026")]
    print(f"{name:>22s} {row[0]:>+8.4f} {row[1]:>+7.4f} {row[2]:>+7.4f} {row[3]:>+7.4f}")
    if best is None or abs(full) > abs(best[1]):
        best = (name, full, sig)

# 最强 OI 信号 vs funding 信号:横截面相关(独立性)
print(f"\n=== 独立性:最强 OI 信号「{best[0]}」 vs funding 信号 横截面相关 ===")
corrs = []
for t in range(T):
    idx = np.where(active[t] & np.isfinite(best[2][t]) & np.isfinite(fsig[t]))[0]
    if len(idx) > 10 and best[2][t, idx].std() > 1e-12 and fsig[t, idx].std() > 1e-12:
        corrs.append(np.corrcoef(best[2][t, idx], fsig[t, idx])[0, 1])
print(f"  平均横截面相关 = {np.nanmean(corrs):+.3f}  (≈0=独立=variance machine 的好料;±高=冗余)")
print(f"\n判读: |全程IC|≳0.01 且逐年(含2026)同号 且与funding相关≈0 → 是个独立弱信号,值得进组合。")
