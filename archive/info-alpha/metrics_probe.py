"""币安期货 metrics 探针:4个信号(OI/顶级多空比/散户多空比/吃单买卖比)有没有独立 IC?是 alpha 还是 regime/beta?
同 netflow_probe 框架:横截面 rank → 前瞻 IC(h=24)逐年 + 慢配置 net@5 + beta/长短腿 scrutiny。
跑：python metrics_probe.py
"""
import argparse
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--tensor", default="data/clean/crypto_tensor_60min_pit.npz")
ap.add_argument("--panel", default="data/clean/crypto_60min_pit.parquet")
ap.add_argument("--metrics", default="data/clean/metrics_60min.parquet")
ap.add_argument("--funding", default="data/clean/funding_pit.npz")
ap.add_argument("--h", type=int, default=24)
ap.add_argument("--hl", type=int, default=24); ap.add_argument("--cad", type=int, default=24)
ap.add_argument("--rebal", type=float, default=0.3); ap.add_argument("--band", type=float, default=0.01)
ap.add_argument("--ann", type=int, default=8760); ap.add_argument("--bps", type=float, default=5)
a = ap.parse_args()

z = np.load(a.tensor, allow_pickle=True)
mask = z["mask"]; adj = z["adj_close"].astype(np.float64); dates = z["dates"].astype(str); slots = list(z["slots"].astype(str))
T, N = mask.shape; nmap = {s: i for i, s in enumerate(slots)}; yr = np.array([d[:4] for d in dates])
grid_ts = np.sort(pd.read_parquet(a.panel, columns=["ts"])["ts"].unique()).astype(np.int64)
tpos = {int(t): i for i, t in enumerate(grid_ts)}

COLS = ["sum_open_interest_value", "sum_toptrader_long_short_ratio", "count_long_short_ratio", "sum_taker_long_short_vol_ratio"]
m = pd.read_parquet(a.metrics)
m = m[m["symbol"].isin(nmap)].copy()
m["ti"] = m["ts"].astype(np.int64).map(tpos); m = m.dropna(subset=["ti"]); m["ti"] = m["ti"].astype(int)
m["j"] = m["symbol"].map(nmap)
M = {c: np.full((T, N), np.nan) for c in COLS}
for c in COLS:
    M[c][m["ti"].to_numpy(), m["j"].to_numpy()] = m[c].to_numpy(np.float64)
cov = np.isfinite(M[COLS[0]]).any(0).sum()
print(f"metrics 覆盖 {cov}/{N} 币; active bar率={np.isfinite(M[COLS[0]]).mean():.3f}\n")

lc = np.log(np.where(adj > 0, adj, np.nan))
fwd = np.full((T, N), np.nan); fwd[:-a.h] = lc[a.h:] - lc[:-a.h]
R = np.zeros((T, N)); g2 = mask[:-1] & mask[1:] & (adj[:-1] > 0) & (adj[1:] > 0); R[:-1][g2] = adj[1:][g2] / adj[:-1][g2] - 1
oi = M["sum_open_interest_value"]
doi = np.full((T, N), np.nan); doi[a.h:] = np.log(np.where(oi[a.h:] > 0, oi[a.h:], np.nan)) - np.log(np.where(oi[:-a.h] > 0, oi[:-a.h], np.nan))


def xs(x, act, sign=1.0):
    Z = np.full_like(x, np.nan)
    for t in range(T):
        idx = np.where(act[t] & np.isfinite(x[t]))[0]
        if len(idx) < 10:
            continue
        r = pd.Series(sign * x[t, idx]).rank().to_numpy()
        Z[t, idx] = (r - r.mean()) / (r.std() + 1e-12)
    return Z


def ic(s, act):
    out = np.full(T, np.nan)
    for t in range(T):
        idx = np.where(act[t] & np.isfinite(s[t]) & np.isfinite(fwd[t]))[0]
        if len(idx) > 10 and s[t, idx].std() > 1e-12 and fwd[t, idx].std() > 1e-12:
            out[t] = np.corrcoef(s[t, idx], fwd[t, idx])[0, 1]
    return out


def smooth(W, h):
    al = 1 - 0.5 ** (1.0 / h); S = np.empty_like(W); S[0] = np.nan_to_num(W[0])
    for t in range(1, len(W)):
        S[t] = al * np.nan_to_num(W[t]) + (1 - al) * S[t - 1]
    g = np.abs(S).sum(1, keepdims=True)
    return np.divide(S, g, out=np.zeros_like(S), where=g > 0)


def book(s):
    W = np.nan_to_num(s); g = np.abs(W).sum(1, keepdims=True); W = np.divide(W, g, out=np.zeros_like(W), where=g > 0)
    Ws = smooth(W, a.hl); cur = np.zeros(N); pnl = np.zeros(T); turn = np.zeros(T); held = np.zeros((T, N))
    for t in range(T):
        if t % a.cad == 0:
            d = Ws[t] - cur; mv = np.abs(d) > a.band; new = cur.copy(); new[mv] = cur[mv] + a.rebal * d[mv]
            s2 = np.abs(new).sum(); new = new / s2 if s2 > 0 else new
            turn[t] = np.abs(new - cur).sum(); cur = new
        held[t] = cur
    return pnl + (held * R).sum(1) - a.bps / 1e4 * turn, held


def sh(p, mm):
    p = p[mm]; return p.mean() / p.std() * np.sqrt(a.ann) if p.std() > 0 else 0.0


mkt = np.array([R[t, mask[t]].mean() if mask[t].sum() > 5 else 0.0 for t in range(T)])
# 候选信号:每个 metric 的 level 和方向(凭直觉给个号,IC 会告诉真相)
act = mask & (adj > 0)
sigs = {
    "ΔOI(24h)↑": xs(doi, act & np.isfinite(doi), +1.0),
    "顶级多空比↑(跟庄)": xs(M["sum_toptrader_long_short_ratio"], act & np.isfinite(M["sum_toptrader_long_short_ratio"]), +1.0),
    "散户多空比↓(反散户)": xs(M["count_long_short_ratio"], act & np.isfinite(M["count_long_short_ratio"]), -1.0),
    "吃单买卖比↑(跟买盘)": xs(M["sum_taker_long_short_vol_ratio"], act & np.isfinite(M["sum_taker_long_short_vol_ratio"]), +1.0),
}
print(f"{'信号':>20s} {'IC全':>8s} {'IC2025':>8s} {'IC2026':>8s} {'净@5全':>7s} {'2026':>7s} {'beta相关':>8s}")
for name, s in sigs.items():
    asig = act & np.isfinite(s); ics = ic(s, asig); pnl, held = book(s)
    bc = np.corrcoef(pnl, mkt)[0, 1]
    print(f"{name:>20s} {np.nanmean(ics):>+8.4f} {np.nanmean(ics[yr=='2025']):>+8.4f} {np.nanmean(ics[yr=='2026']):>+8.4f}"
          f" {sh(pnl, np.ones(T,bool)):>+7.2f} {sh(pnl, yr=='2026'):>+7.2f} {bc:>+8.2f}")

# 独立性矩阵
print("\n独立性(信号两两横截面相关均值;≈0=独立):")
names = list(sigs); arr = list(sigs.values())
for i in range(len(names)):
    row = []
    for k in range(len(names)):
        cs = []
        for t in range(0, T, 24):
            idx = np.where(act[t] & np.isfinite(arr[i][t]) & np.isfinite(arr[k][t]))[0]
            if len(idx) > 10 and arr[i][t, idx].std() > 1e-12 and arr[k][t, idx].std() > 1e-12:
                cs.append(np.corrcoef(arr[i][t, idx], arr[k][t, idx])[0, 1])
        row.append(np.nanmean(cs))
    print(f"  {names[i]:>20s} " + " ".join(f"{v:>+5.2f}" for v in row))

print("\n判读: |IC|≥0.01且逐年同号且 beta≈0 = 有独立信号;IC≈0或翻号或全靠beta = 没用(同 netflow/oi)。")
