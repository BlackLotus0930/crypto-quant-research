# -*- coding: utf-8 -*-
"""尾部对冲压测:给分层书(~2.5-3x)配 BTC 看跌,看真实崩盘日能救回多少。
真实 BTC 崩盘幅度(数据) + 真实 Deribit premium(20%OTM 8.8%/yr、15% 14%、11% 25%)。
问:要把 -95% 的 2021-05-19 拉回 -30% 可恢复,需多少 put 覆盖?年化 premium 多少?划得来吗?
跑:PYTHONUTF8=1 .venv/Scripts/python.exe research/tail_hedge_crash.py
"""
import glob, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from venues import canon
from research.xvenue_honest import fetch_intervals
from strategy import _cap_renorm
MM = 0.02
PREM = {0.20: 0.088, 0.15: 0.141, 0.11: 0.246}    # OTM→年化滚动 premium/单位notional(真实Deribit)

z = np.load("data/clean/crypto_tensor_60min_pit.npz", allow_pickle=True)
adj = (z["adj"] if "adj" in z else z["adj_close"]).astype(float)
slots = list(z["slots"].astype(str)); dates = z["dates"].astype(str); T, N = adj.shape
nmap = {s: i for i, s in enumerate(slots)}; valid = z["mask"] & (adj > 0)
xv = np.load("data/clean/xvenue_funding.npz", allow_pickle=True)
bi, yi = fetch_intervals(); biv = np.array([bi.get(s, 8) for s in slots]); yiv = np.array([yi.get(s, 8) for s in slots])
fb = xv["f_bin"] / biv[None, :]; fy = xv["f_byb"] / yiv[None, :]
grid = np.sort(pd.read_parquet("data/clean/crypto_60min_pit.parquet", columns=["ts"])["ts"].unique()).astype(np.int64)
gdf = pd.DataFrame({"ts": grid}); byb = np.full((T, N), np.nan)
for f in glob.glob("data/raw/bybit/kline/*.csv"):
    j = nmap.get(os.path.basename(f)[:-4])
    if j is None: continue
    df = pd.read_csv(f).sort_values("ts").drop_duplicates("ts", keep="last")
    if not df.empty:
        byb[:, j] = pd.merge_asof(gdf, df[["ts", "close"]], on="ts", direction="backward", tolerance=3600_001)["close"].to_numpy()
day = np.array([d[:10] for d in dates])
hr = np.full((T, N), np.nan); v = valid[:-1] & valid[1:]; hr[:-1][v] = (adj[1:][v] / adj[:-1][v] - 1)
dvol = np.nanstd(hr, axis=0) * np.sqrt(24); dvol = np.where(np.isfinite(dvol) & (dvol > 0), dvol, 0.10)
BTC = nmap.get("BTCUSDT")


def book_loss(d, c=6, Lcap=8):
    """分层书(~2.5-3x)崩盘日强平损 + BTC 当日最坏跌幅。"""
    idx = np.where(day == d)[0]; t0 = idx[0]; hrs = idx
    act = np.where(valid[t0] & np.isfinite(byb[t0]) & (byb[t0] > 0) & np.isfinite(fb[t0]) & np.isfinite(fy[t0]) & (adj[t0] > 0))[0]
    spr = np.abs(fb[t0, act] - fy[t0, act]); w = _cap_renorm(np.power(spr, 2) / max(np.power(spr, 2).sum(), 1e-12), 0.05)
    sgn = np.sign(fb[t0, act] - fy[t0, act])
    cA = np.nan_to_num(adj[hrs][:, act] / adj[t0, act] - 1); cB = np.nan_to_num(byb[hrs][:, act] / byb[t0, act] - 1)
    Li = np.minimum(Lcap, 1.0 / (c * dvol[act] + MM)); eff = (w * Li).sum() / w.sum()
    liq_w = 0.0
    for k in range(len(act)):
        dliq = 1.0 / Li[k] - MM
        if (-sgn[k] * cA[:, k]).min() < -dliq or (sgn[k] * cB[:, k]).min() < -dliq:
            liq_w += w[k]
    btc_drop = (adj[hrs, BTC] / adj[t0, BTC] - 1).min() if BTC is not None else np.nan
    return eff, -liq_w, btc_drop


print("=== 尾部对冲能不能救崩盘日(分层书~2.5-3x + BTC看跌)===\n")
print(f"{'崩盘日':>12s} {'有效杠杆':>7s} {'书强平损':>7s} {'BTC最坏跌':>8s} | 救到-30%需:")
for d in ["2021-05-19", "2022-05-11", "2025-10-10"]:
    eff, loss, btc = book_loss(d)
    print(f"{d:>12s} {eff:>6.1f}x {loss:>+6.0%} {btc:>+8.0%} |")
    for otm, prem in PREM.items():
        payoff = max(0.0, -otm - btc)               # put 赔付/单位notional(BTC跌超OTM才赔)
        if payoff <= 1e-9:
            print(f"      {otm:.0%}OTM put: BTC 没跌穿行权价 → **零赔付**(救不了)"); continue
        need = max(0.0, (-loss - 0.30)) / payoff     # 需覆盖倍数(×book notional)
        ann_prem = need * prem
        print(f"      {otm:.0%}OTM put: 赔付{payoff:.0%}/notional → 需覆盖 {need:.1f}× book → **年化premium {ann_prem:.0%}**")
print("\n判读:")
print("  救崩盘日要的 put 覆盖 = (书损-30%)/单位赔付。书损被杠杆放大、put 又是OTM部分赔付 → 覆盖倍数大。")
print("  年化 premium = 覆盖 × 8.8%。若 premium 远超策略年化(~30%@3x)→ 尾部对冲救不回这种日子,经济上不成立。")
print("  → 那 3x 就是真实硬顶(对冲也突破不了),除非接受崩盘日大回撤或砍到 ≤2x。")
