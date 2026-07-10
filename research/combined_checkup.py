"""完整组合书体检:carry + 跨所(50/50 资本)。我们实际做的是组合,不是单 carry。
- carry:strategy.backtest(正侧 tilt2)。
- 跨所:驱动生产 CrossVenueStrategy(tilt2)on Bin-Byb 真实净额(spread funding + 价格腿 + 成本,真实间隔)。
- 组合 = 0.5·carry + 0.5·跨所(同 paper_live)。逐年 return/Sharpe/maxDD + 相关性 + 组合杠杆。
多年(2020-2026)Bin/Byb 代理(用户实际 HL/Gate 历史短→前向)。
跑：PYTHONUTF8=1 .venv/Scripts/python.exe research/combined_checkup.py
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import strategy as S


def fetch_intervals():
    import json
    import urllib.request
    UA = {"User-Agent": "Mozilla/5.0"}
    g = lambda u: json.loads(urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=25).read())
    bi = {d["symbol"]: d.get("fundingIntervalHours", 8) for d in g("https://fapi.binance.com/fapi/v1/fundingInfo")}
    yi = {}; cur = ""
    while True:
        res = g("https://api.bybit.com/v5/market/instruments-info?category=linear&limit=1000" + (f"&cursor={cur}" if cur else ""))["result"]
        for s in res["list"]:
            if s.get("fundingInterval"):
                yi[s["symbol"]] = int(s["fundingInterval"]) / 60.0
        cur = res.get("nextPageCursor", "")
        if not cur:
            break
    return bi, yi


def main():
    z = np.load("data/clean/crypto_tensor_60min_pit.npz", allow_pickle=True)
    adj = z["adj_close"].astype(float); tdv = z["tdv"].astype(float)
    slots = list(z["slots"].astype(str)); dates = z["dates"].astype(str); T, N = adj.shape
    nmap = {s: i for i, s in enumerate(slots)}
    xv = np.load("data/clean/xvenue_funding.npz", allow_pickle=True)
    bi, yi = fetch_intervals()
    biv = np.array([bi.get(s, 8) for s in slots]); yiv = np.array([yi.get(s, 8) for s in slots])
    fb = xv["f_bin"] / biv[None, :]; fy = xv["f_byb"] / yiv[None, :]
    grid = np.sort(pd.read_parquet("data/clean/crypto_60min_pit.parquet", columns=["ts"])["ts"].unique()).astype(np.int64)
    byb = np.full((T, N), np.nan); gdf = pd.DataFrame({"ts": grid})
    for f in glob.glob("data/raw/bybit/kline/*.csv"):
        s = os.path.basename(f)[:-4]; j = nmap.get(s)
        if j is None:
            continue
        df = pd.read_csv(f).sort_values("ts").drop_duplicates("ts", keep="last")
        if df.empty:
            continue
        m = pd.merge_asof(gdf, df[["ts", "close"]], on="ts", direction="backward", tolerance=3600_000)
        byb[:, j] = m["close"].to_numpy()
    both = (adj > 0) & np.isfinite(fb) & np.isfinite(fy) & np.isfinite(byb) & (byb > 0)
    d = np.where(both, fb - fy, 0.0); spread = np.abs(d); sgn = np.sign(d)
    binret = np.zeros((T, N)); bybret = np.zeros((T, N)); v2 = both[:-1] & both[1:]
    binret[:-1][v2] = (adj[1:] / adj[:-1] - 1)[v2]; bybret[:-1][v2] = (byb[1:] / byb[:-1] - 1)[v2]
    dret = np.clip(np.nan_to_num(binret - bybret), -0.10, 0.10)

    # 跨所:驱动生产 CrossVenueStrategy
    xc = S.XVenueConfig()
    xstrat = S.CrossVenueStrategy(N, xc)
    inc = np.zeros(T); pxl = np.zeros(T); turn = np.zeros(T); prev = np.zeros(N)
    for t in range(T):
        cur = xstrat.step(spread[t], tdv[t], both[t])
        inc[t] = (cur * spread[t]).sum()
        pxl[t] = (cur * (-sgn[t] * dret[t])).sum()
        turn[t] = np.abs(cur - prev).sum(); prev = cur
    xnet = inc + pxl - xc.cost_bps / 1e4 * turn

    bnet = S.backtest(S.CarryConfig(leverage=1.0))["net"]           # carry 正侧 tilt2
    comb = 0.5 * bnet + 0.5 * xnet                                  # 50/50 资本(同 paper_live)

    yr = np.array([dd[:4] for dd in dates]); oos = np.zeros(T, bool); oos[int(T * 0.4):] = True

    def sh(p, m):
        p = p[m]; return p.mean() / p.std() * np.sqrt(8760) if (len(p) and p.std() > 0) else 0.0

    def mdd(p, m):
        p = p[m]; c = np.cumsum(p); return (c - np.maximum.accumulate(c)).min() if len(p) else 0.0

    print("=== 完整组合书体检(carry + 跨所 50/50,多年 Bin/Byb 代理,L1)===")
    print(f"{'区间':>8s} | {'carry年化':>9s} {'跨所年化':>9s} {'组合年化':>9s} | {'组合Sh':>7s} {'组合maxDD':>9s}")
    for lab, m in [("OOS全", oos)] + [(y, yr == y) for y in ("2021", "2022", "2023", "2024", "2025", "2026")]:
        if m.sum() < 50:
            continue
        print(f"{lab:>8s} | {bnet[m].mean()*8760:>+8.1%} {xnet[m].mean()*8760:>+8.1%} {comb[m].mean()*8760:>+8.1%} | "
              f"{sh(comb,m):>+7.2f} {mdd(comb,m):>+9.1%}")
    print(f"\n相关性(OOS) corr(carry, 跨所) = {np.corrcoef(bnet[oos], xnet[oos])[0,1]:+.2f}  (低=√N 多元化真)")
    print(f"OOS Sharpe: 单carry {sh(bnet,oos):.2f} / 单跨所 {sh(xnet,oos):.2f} / **组合 {sh(comb,oos):.2f}**")
    print(f"  → 组合/单carry = {sh(comb,oos)/max(sh(bnet,oos),1e-9):.2f}× (√N 提升;这是组合的价值)")
    print("\n=== 组合杠杆 vs 回撤 ===")
    for L in [1, 2, 3]:
        c = comb * L
        print(f"  {L}x: OOS年化 {c[oos].mean()*8760:>+6.1%}  全样本maxDD {mdd(c,np.ones(T,bool)):>+6.1%}  2022maxDD {mdd(c,yr=='2022'):>+6.1%}")
    print("\n注:跨所绝对Sharpe虚高(funding近确定);**看组合/单carry比例(√N)+ 相关性=稳健**。Bin/Byb代理,用户HL/Gate前向证。")


if __name__ == "__main__":
    main()
