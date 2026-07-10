"""信号#4 探针:链上交易所净流(Dune ERC20→CEX)有没有独立 IC?和 funding 正交吗?
netflow_usd>0 = 净流入交易所 = 抛压 → 做空;<0 = 流出/囤 → 做多。信号=横截面 rank(−netflow)。
因果:日 D 净流在 D+1 00:00 才完整 → 用"已完成日"对齐(+1天)。h=24bar 前瞻 IC,逐年,独立性。
跑：python netflow_probe.py
"""
import argparse
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--tensor", default="data/clean/crypto_tensor_60min_pit.npz")
ap.add_argument("--panel", default="data/clean/crypto_60min_pit.parquet")
ap.add_argument("--csv", default="data/raw/dune_xflow_2025_2026.csv")
ap.add_argument("--funding", default="data/clean/funding_pit.npz")
ap.add_argument("--h", type=int, default=24)
ap.add_argument("--roll", type=int, default=7, help="净流累计天数(1=单日;7=周累计,更密更稳)")
ap.add_argument("--hl", type=int, default=24); ap.add_argument("--cad", type=int, default=24)
ap.add_argument("--rebal", type=float, default=0.3); ap.add_argument("--band", type=float, default=0.01)
ap.add_argument("--ann", type=int, default=8760); ap.add_argument("--bps", type=float, default=5)
a = ap.parse_args()

z = np.load(a.tensor, allow_pickle=True)
mask = z["mask"]; adj = z["adj_close"].astype(np.float64); dates = z["dates"].astype(str); slots = list(z["slots"].astype(str))
T, N = mask.shape; nmap = {s: i for i, s in enumerate(slots)}; yr = np.array([d[:4] for d in dates])
grid_ts = np.sort(pd.read_parquet(a.panel, columns=["ts"])["ts"].unique()).astype(np.int64)


def base(s):
    s = s[:-4] if s.endswith("USDT") else s
    for p in ("1000000", "1000"):
        if s.startswith(p):
            s = s[len(p):]
    return s.upper()


perp_base = {base(s): s for s in slots}
df = pd.read_csv(a.csv); df["b"] = df["symbol"].astype(str).map(base)
df = df[df["b"].isin(perp_base)].copy(); df["j"] = df["b"].map(perp_base).map(nmap)
df["day"] = pd.to_datetime(df["day"], utc=True)
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")

netflow = np.full((T, N), np.nan)
gdf = pd.DataFrame({"ts": grid_ts})
for j, g in df.groupby("j"):
    s = g.groupby("day")["netflow_usd"].sum().sort_index()
    s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D", tz="UTC"), fill_value=0.0)
    s = s.rolling(a.roll, min_periods=1).sum()                       # 滚动累计净流(更密更稳)
    eff = ((s.index - EPOCH) // pd.Timedelta("1ms")).values.astype(np.int64) + 86_400_000   # +1天=因果(日完整后才可见)
    dd = pd.DataFrame({"ts": eff, "nf": s.values}).sort_values("ts")
    netflow[:, int(j)] = pd.merge_asof(gdf, dd, on="ts", direction="backward")["nf"].to_numpy(np.float64)

cov = np.isfinite(netflow).any(0).sum()
print(f"netflow 覆盖 {cov}/{N} 币(ERC20∩宇宙); active bar率={np.isfinite(netflow).mean():.3f}\n")

lc = np.log(np.where(adj > 0, adj, np.nan))
fwd = np.full((T, N), np.nan); fwd[:-a.h] = lc[a.h:] - lc[:-a.h]
R = np.zeros((T, N)); g2 = mask[:-1] & mask[1:] & (adj[:-1] > 0) & (adj[1:] > 0); R[:-1][g2] = adj[1:][g2] / adj[:-1][g2] - 1
active = mask & (adj > 0) & np.isfinite(netflow)


def xs_rank(x, act):
    """每bar横截面 rank(−netflow)∈[-1,1](robust to 量级/离群)。"""
    Z = np.full_like(x, np.nan)
    for t in range(T):
        idx = np.where(act[t])[0]
        if len(idx) < 10:
            continue
        r = pd.Series(-x[t, idx]).rank().to_numpy()
        Z[t, idx] = (r - r.mean()) / (r.std() + 1e-12)
    return Z


sig = xs_rank(netflow, active)
fund = np.load(a.funding, allow_pickle=True)["funding"].astype(np.float64)
fsig = xs_rank(-fund, mask & np.isfinite(fund))   # funding 信号(同样 rank)对照独立性


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


def booknet(s):
    W = np.nan_to_num(s); g = np.abs(W).sum(1, keepdims=True); W = np.divide(W, g, out=np.zeros_like(W), where=g > 0)
    Ws = smooth(W, a.hl); cur = np.zeros(N); pnl = np.zeros(T); turn = np.zeros(T)
    for t in range(T):
        if t % a.cad == 0:
            d = Ws[t] - cur; mv = np.abs(d) > a.band; new = cur.copy(); new[mv] = cur[mv] + a.rebal * d[mv]
            s2 = np.abs(new).sum(); new = new / s2 if s2 > 0 else new
            turn[t] = np.abs(new - cur).sum(); cur = new
        pnl[t] = (cur * R[t]).sum()
    return pnl - a.bps / 1e4 * turn


def sh(p, m):
    p = p[m]
    return p.mean() / p.std() * np.sqrt(a.ann) if p.std() > 0 else 0.0


ics = ic(sig, active); net = booknet(sig)
print(f"=== 交易所净流信号 IC(h={a.h}) + 慢配置 net@5 逐年 ===")
print(f"{'区间':>8s} {'IC':>8s} {'净@5':>7s} {'年换手':>7s}")
W = np.nan_to_num(sig); g = np.abs(W).sum(1, keepdims=True)
for lab, m in [("全程", np.ones(T, bool))] + [(y, yr == y) for y in ("2025", "2026")]:
    if m.sum() < 50:
        continue
    print(f"{lab:>8s} {np.nanmean(ics[m]):>+8.4f} {sh(net, m):>+7.2f}")

# 独立性 vs funding
cs = []
for t in range(0, T, 6):
    idx = np.where(active[t] & np.isfinite(sig[t]) & np.isfinite(fsig[t]))[0]
    if len(idx) > 10 and sig[t, idx].std() > 1e-12 and fsig[t, idx].std() > 1e-12:
        cs.append(np.corrcoef(sig[t, idx], fsig[t, idx])[0, 1])
print(f"\n独立性: netflow 信号 vs funding 信号 横截面相关 = {np.nanmean(cs):+.3f} (≈0=独立)")

# scrutiny: net@5 是干净 alpha 还是 regime/beta?(同 oi 的扒法)
W = np.nan_to_num(sig); g = np.abs(W).sum(1, keepdims=True); W = np.divide(W, g, out=np.zeros_like(W), where=g > 0)
Ws = smooth(W, a.hl); held = np.zeros((T, N)); cur = np.zeros(N)
for t in range(T):
    if t % a.cad == 0:
        d = Ws[t] - cur; mv = np.abs(d) > a.band; new = cur.copy(); new[mv] = cur[mv] + a.rebal * d[mv]
        s2 = np.abs(new).sum(); cur = new / s2 if s2 > 0 else new
    held[t] = cur
mkt = np.array([R[t, active[t]].mean() if active[t].sum() > 5 else 0.0 for t in range(T)])
bpnl = (held * R).sum(1)
print(f"\n=== scrutiny: 干净 alpha 还是 regime/beta? ===")
print(f"  净敞口|Σw|均值={np.abs(held.sum(1)).mean():.3f}; book对市场相关={np.corrcoef(bpnl, mkt)[0,1]:+.3f}")
lp = np.where(held > 0, held * R, 0).sum(1); sp = np.where(held < 0, held * R, 0).sum(1)
for y in ("2025", "2026"):
    m = yr == y
    print(f"  {y}: 长腿 {lp[m].sum():+.3f}  短腿 {sp[m].sum():+.3f}  市场 {mkt[m].mean()*a.ann:+.0%}/yr")
print("\n判读: 长短腿逐年都正、beta≈0 = 干净独立 sleeve;长短腿 regime 互补、beta大 = 又是 regime/beta(同 oi)。")
