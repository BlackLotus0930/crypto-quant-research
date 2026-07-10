# -*- coding: utf-8 -*-
"""装上保护后再压测:单币杠杆分层(按价格波动率)→ 真实崩盘日还爆不爆。
裸书 uniform L=3 崩盘日 48-75% 强平(stress_crash.py)。分层:高波动薄币低杠杆、稳定币高杠杆。
问:同样的**有效书杠杆**下,分层把崩盘日强平%降多少?能不能在更高有效杠杆下还活?
强平模型同 stress_crash:逐腿 cumulative move < -(1/L_i - mm) 即爆。
跑:PYTHONUTF8=1 .venv/Scripts/python.exe research/stress_protected.py
"""
import glob, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from venues import canon
from research.xvenue_honest import fetch_intervals
from strategy import _cap_renorm
MM = 0.02

z = np.load("data/clean/crypto_tensor_60min_pit.npz", allow_pickle=True)
adj = (z["adj"] if "adj" in z else z["adj_close"]).astype(float)
slots = list(z["slots"].astype(str)); dates = z["dates"].astype(str); T, N = adj.shape
nmap = {s: i for i, s in enumerate(slots)}
valid = z["mask"] & (adj > 0)
xv = np.load("data/clean/xvenue_funding.npz", allow_pickle=True)
bi, yi = fetch_intervals()
biv = np.array([bi.get(s, 8) for s in slots]); yiv = np.array([yi.get(s, 8) for s in slots])
fb = xv["f_bin"] / biv[None, :]; fy = xv["f_byb"] / yiv[None, :]
grid = np.sort(pd.read_parquet("data/clean/crypto_60min_pit.parquet", columns=["ts"])["ts"].unique()).astype(np.int64)
gdf = pd.DataFrame({"ts": grid})
byb = np.full((T, N), np.nan)
for f in glob.glob("data/raw/bybit/kline/*.csv"):
    j = nmap.get(os.path.basename(f)[:-4])
    if j is None: continue
    df = pd.read_csv(f).sort_values("ts").drop_duplicates("ts", keep="last")
    if df.empty: continue
    byb[:, j] = pd.merge_asof(gdf, df[["ts", "close"]], on="ts", direction="backward", tolerance=3600_001)["close"].to_numpy()
day = np.array([d[:10] for d in dates])

# 每币全样本小时波动 → 日波动(分层用)
hr = np.full((T, N), np.nan); v = valid[:-1] & valid[1:]
hr[:-1][v] = (adj[1:][v] / adj[:-1][v] - 1)
dvol = np.nanstd(hr, axis=0) * np.sqrt(24)                 # 日波动
dvol = np.where(np.isfinite(dvol) & (dvol > 0), dvol, 0.10)


def tier_L(idxs, c, Lcap):
    """按 c×日波动 反比定每币杠杆:L_i = min(Lcap, 1/(c·dvol_i + mm))。"""
    return np.minimum(Lcap, 1.0 / (c * dvol[idxs] + MM))


def sim_day(d, c_tier, Lcap):
    idx = np.where(day == d)[0]
    if len(idx) < 6: return None
    t0 = idx[0]; hrs = idx
    act = np.where(valid[t0] & np.isfinite(byb[t0]) & (byb[t0] > 0) & np.isfinite(fb[t0]) & np.isfinite(fy[t0]) & (adj[t0] > 0))[0]
    if len(act) < 5: return None
    spr = np.abs(fb[t0, act] - fy[t0, act]); w = _cap_renorm(np.power(spr, 2) / max(np.power(spr, 2).sum(), 1e-12), 0.05)
    sgn = np.sign(fb[t0, act] - fy[t0, act])
    cA = np.nan_to_num(adj[hrs][:, act] / adj[t0, act] - 1); cB = np.nan_to_num(byb[hrs][:, act] / byb[t0, act] - 1)
    Li = tier_L(act, c_tier, Lcap)
    eff = (w * Li).sum() / w.sum()
    liq_w = 0.0
    for k in range(len(act)):
        dliq = 1.0 / Li[k] - MM
        legA = -sgn[k] * cA[:, k]; legB = sgn[k] * cB[:, k]
        if legA.min() < -dliq or legB.min() < -dliq:
            liq_w += w[k]
    worst = -liq_w * 1.0                                    # 被平=最坏全损那部分(保守)
    return eff, liq_w / max(w.sum(), 1e-9), worst, np.nanmedian(np.abs(cA[-1]))


CRASH = ["2021-05-19", "2022-05-11", "2025-10-10"]
print("=== 装上单币杠杆分层后,真实崩盘日强平(vs 裸 uniform)===")
print("分层:L_i=min(Lcap, 1/(c×日波动+2%)) —— 高波动薄币自动低杠杆\n")
print("【裸 uniform(对照,stress_crash)】崩盘日强平: L2~1-10% / L3~48-75% / L5~90-95%\n")
for c_tier, Lcap in [(3, 12), (4, 12), (5, 10), (6, 8)]:
    print(f"[分层 c={c_tier}(扛{c_tier}σ日动), Lcap={Lcap}]")
    print(f"  {'崩盘日':>12s} {'有效书杠杆':>9s} {'强平腿%':>7s} {'最坏损':>7s} {'当日中位动':>9s}")
    effs = []
    for d in CRASH:
        r = sim_day(d, c_tier, Lcap)
        if r is None:
            print(f"  {d:>12s} 数据不足"); continue
        eff, liq, worst, mov = r; effs.append(eff)
        print(f"  {d:>12s} {eff:>8.1f}x {liq:>6.0%} {worst:>+6.0%} {mov:>8.0%}")
    print(f"  → 平均有效杠杆 {np.mean(effs):.1f}x\n")
print("判读:")
print("  比'裸 uniform 同有效杠杆'的强平%。分层把高波动币杠杆自动压低 → 崩盘日强平应大降。")
print("  若分层在 ~3-4x 有效杠杆下崩盘日强平仍低 → 用户对:加了保护就能扛更高杠杆。")
print("  注:仍是跨所薄币书;c 越大越保守(扛更大日动);Lcap 限单币上限。真值前向证。")
