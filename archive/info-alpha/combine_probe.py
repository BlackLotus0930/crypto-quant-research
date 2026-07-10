"""variance machine POC:把几个独立弱信号(都从 pit 原始数据直接算)堆起来,组合 IR 是否 > 任何单个?
信号(横截面 z,≤t 因果):rev=反转(−过去24bar收益)/ oi=淡化大户多空比 / fund=funding拥挤 / doi=−ΔOI。
组合=等权 z 平均(不偷看)。每个 + 组合:IC + 慢配置 net@5 逐年。还出信号相关矩阵(看独立性)。
注:真模型反转更强(IC~0.025);这里用简单反转做 POC,只问"堆叠到底有没有用"。
跑：python combine_probe.py
"""
import argparse
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--tensor", default="data/clean/crypto_tensor_60min_pit.npz")
ap.add_argument("--panel", default="data/clean/crypto_60min_pit.parquet")
ap.add_argument("--metrics", default="data/clean/metrics_60min.parquet")
ap.add_argument("--funding", default="data/clean/funding_pit.npz")
ap.add_argument("--rw", type=int, default=24, help="反转回看窗(bar)")
ap.add_argument("--hl", type=int, default=24); ap.add_argument("--cad", type=int, default=24)
ap.add_argument("--rebal", type=float, default=0.3); ap.add_argument("--band", type=float, default=0.01)
ap.add_argument("--ann", type=int, default=8760); ap.add_argument("--bps", type=float, default=5)
a = ap.parse_args()

z = np.load(a.tensor, allow_pickle=True)
mask = z["mask"]; adj = z["adj_close"].astype(np.float64)
dates = z["dates"].astype(str); slots = list(z["slots"].astype(str)); nmap = {s: i for i, s in enumerate(slots)}
T, N = mask.shape
yr = np.array([d[:4] for d in dates])
grid_ts = np.sort(pd.read_parquet(a.panel, columns=["ts"])["ts"].unique()).astype(np.int64)

lc = np.log(np.where(adj > 0, adj, np.nan))
R = np.zeros((T, N)); g2 = mask[:-1] & mask[1:] & (adj[:-1] > 0) & (adj[1:] > 0)
R[:-1][g2] = adj[1:][g2] / adj[:-1][g2] - 1                                  # 次bar简单收益(交易+IC用)

# funding
fund = np.load(a.funding, allow_pickle=True)["funding"].astype(np.float64)
# metrics OI/top-LSR 对齐
m = pd.read_parquet(a.metrics); TT = np.full((T, N), np.nan); OI = np.full((T, N), np.nan)
gdf = pd.DataFrame({"ts": grid_ts})
for sym, gg in m.groupby("symbol", sort=False):
    j = nmap.get(sym)
    if j is None:
        continue
    gg = gg.sort_values("ts").drop_duplicates("ts", keep="last")
    mm = pd.merge_asof(gdf, gg[["ts", "sum_toptrader_long_short_ratio", "sum_open_interest_value"]], on="ts", direction="backward")
    TT[:, j] = mm["sum_toptrader_long_short_ratio"].to_numpy(np.float64)
    OI[:, j] = mm["sum_open_interest_value"].to_numpy(np.float64)
dOI = np.full((T, N), np.nan); loi = np.log(np.where(OI > 0, OI, np.nan)); dOI[a.rw:] = loi[a.rw:] - loi[:-a.rw]

rev_raw = np.full((T, N), np.nan); rev_raw[a.rw:] = -(lc[a.rw:] - lc[:-a.rw])  # 反转=负过去收益


def xs_z(x, act):
    Z = np.full_like(x, np.nan)
    for t in range(T):
        idx = np.where(act[t] & np.isfinite(x[t]))[0]
        if len(idx) < 10:
            continue
        v = x[t, idx]; sd = v.std()
        if sd > 1e-12:
            Z[t, idx] = (v - v.mean()) / sd
    return Z


base = mask & (adj > 0)
sigs = {
    "rev(反转)": xs_z(rev_raw, base),
    "oi(淡大户多空)": xs_z(-TT, base & np.isfinite(TT)),
    "fund(carry拥挤)": xs_z(-fund, base & np.isfinite(fund)),
    "doi(−ΔOI)": xs_z(-dOI, base & np.isfinite(dOI)),
}
# 等权组合(不偷看):每bar对可用信号取均值
stack = np.stack(list(sigs.values()))                                       # [S,T,N]
combo = np.nanmean(stack, axis=0)
combo[~np.isfinite(combo)] = np.nan
sigs["★组合(等权)"] = combo


def ic(sig, act):
    out = np.full(T, np.nan)
    for t in range(T):
        idx = np.where(act[t] & np.isfinite(sig[t]) & np.isfinite(R[t]))[0]
        if len(idx) > 10 and sig[t, idx].std() > 1e-12 and R[t, idx].std() > 1e-12:
            out[t] = np.corrcoef(sig[t, idx], R[t, idx])[0, 1]
    return out


def smooth(W, h):
    al = 1 - 0.5 ** (1.0 / h); S = np.empty_like(W); S[0] = np.nan_to_num(W[0])
    for t in range(1, len(W)):
        S[t] = al * np.nan_to_num(W[t]) + (1 - al) * S[t - 1]
    g = np.abs(S).sum(1, keepdims=True)
    return np.divide(S, g, out=np.zeros_like(S), where=g > 0)


def booknet(sig):
    """z信号→gross1多空→慢配置→net@bps 的 pnl 序列。"""
    W = np.nan_to_num(sig); g = np.abs(W).sum(1, keepdims=True)
    W = np.divide(W, g, out=np.zeros_like(W), where=g > 0)
    Ws = smooth(W, a.hl); cur = np.zeros(N); pnl = np.zeros(T); turn = np.zeros(T)
    for t in range(T):
        if t % a.cad == 0:
            d = Ws[t] - cur; mv = np.abs(d) > a.band; new = cur.copy(); new[mv] = cur[mv] + a.rebal * d[mv]
            s = np.abs(new).sum(); new = new / s if s > 0 else new
            turn[t] = np.abs(new - cur).sum(); cur = new
        pnl[t] = (cur * R[t]).sum()
    return pnl - a.bps / 1e4 * turn


def sh(p, msk):
    p = p[msk]
    return p.mean() / p.std() * np.sqrt(a.ann) if p.std() > 0 else 0.0


print(f"=== 各信号 IC(全程) + 慢配置 net@5 逐年 Sharpe ===")
print(f"{'信号':>16s} {'IC':>8s} {'净@5全':>7s} {'2024':>7s} {'2025':>7s} {'2026':>7s}")
ys = ("2024", "2025", "2026")
for name, sig in sigs.items():
    icv = np.nanmean(ic(sig, base)); net = booknet(sig)
    row = [sh(net, yr == y) for y in ys]
    print(f"{name:>16s} {icv:>+8.4f} {sh(net, np.ones(T, bool)):>+7.2f} {row[0]:>+7.2f} {row[1]:>+7.2f} {row[2]:>+7.2f}")

print(f"\n=== 信号相关矩阵(横截面平均;低=独立=堆叠有效) ===")
names = [k for k in sigs if k != "★组合(等权)"]
arr = [sigs[k] for k in names]
print("            " + " ".join(f"{n[:6]:>7s}" for n in names))
for i, ni in enumerate(names):
    row = []
    for j in range(len(names)):
        cs = []
        for t in range(0, T, 6):
            idx = np.where(base[t] & np.isfinite(arr[i][t]) & np.isfinite(arr[j][t]))[0]
            if len(idx) > 10 and arr[i][t, idx].std() > 1e-12 and arr[j][t, idx].std() > 1e-12:
                cs.append(np.corrcoef(arr[i][t, idx], arr[j][t, idx])[0, 1])
        row.append(np.nanmean(cs))
    print(f"{ni[:10]:>11s} " + " ".join(f"{v:>+7.2f}" for v in row))

print(f"\n判读: 组合 net@5(尤其2026)> 任何单信号 = 堆叠有效=variance machine 对我们成立。相关矩阵低=信号真独立。")
