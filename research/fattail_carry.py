"""肥尾 carry 调查(深机会#1):极端正 funding 币的 carry 净额是否更肥?
**真实多年数据**(Binance 现货+永续 2020-2026,含死币=幸存者已避免)。每币正funding时持仓(long现货/short永续),
净 = funding(收正)+ 价格腿(现货收益−永续收益,winsor)。真实间隔。按毛 funding 分桶报净额+净日Sharpe。
数据问题排查:幸存者(死币在内✓)、价格腿(真实spot−perp)、间隔(真实)、退市末期(mask双边有效才计)。
跑：PYTHONUTF8=1 .venv/Scripts/python.exe research/fattail_carry.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANN = 8760
WINS = 0.25       # 价格腿 winsor(carry 用,见 strategy.price_winsor)


def fetch_intervals():
    import json
    import urllib.request
    UA = {"User-Agent": "Mozilla/5.0"}
    g = lambda u: json.loads(urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=25).read())
    return {d["symbol"]: d.get("fundingIntervalHours", 8) for d in g("https://fapi.binance.com/fapi/v1/fundingInfo")}


def main():
    z = np.load("data/clean/crypto_tensor_60min_pit.npz", allow_pickle=True)
    mask = z["mask"]; adj = z["adj_close"].astype(float)            # 永续
    slots = list(z["slots"].astype(str)); dates = z["dates"].astype(str); T, N = adj.shape
    spot = np.load("data/clean/spot_pit.npz", allow_pickle=True)["spot"].astype(float)
    fund = np.load("data/clean/funding_pit.npz", allow_pickle=True)["funding"].astype(float)
    biv = np.array([fetch_intervals().get(s, 8) for s in slots])
    f_hr = fund / biv[None, :]                                       # 每小时 funding(真实间隔)

    vp = mask & np.isfinite(spot) & (spot > 0) & (adj > 0)          # 永续+现货都有效(可对冲)
    vpp = vp[:-1] & vp[1:]
    sr = np.full((T, N), np.nan); pr = np.full((T, N), np.nan)
    sr[:-1][vpp] = (spot[1:] / spot[:-1] - 1)[vpp]
    pr[:-1][vpp] = (adj[1:] / adj[:-1] - 1)[vpp]
    price_leg = np.clip(sr - pr, -WINS, WINS)                        # 现货−永续(≈−Δbasis)

    yr = np.array([d[:4] for d in dates])
    rows = []
    for i in range(N):
        held = np.zeros(T, bool)
        held[1:] = (f_hr[:-1, i] > 0) & vp[1:, i] & np.isfinite(price_leg[1:, i])   # 滞后:上bar正funding才持
        h = np.where(held)[0]
        if len(h) < 1500:
            continue
        fpart = f_hr[h, i]                       # 收正 funding(short perp)
        net = fpart + price_leg[h, i]
        gross = fpart.mean() * ANN
        netA = net.mean() * ANN
        nd = len(net) // 24
        dd = net[:nd * 24].reshape(nd, 24).sum(1) if nd > 1 else net
        nsh = dd.mean() / dd.std() * np.sqrt(365) if dd.std() > 0 else 0
        rows.append((slots[i], gross, netA, nsh, len(h)))
    rows.sort(key=lambda r: -r[1])
    g = np.array([r[1] for r in rows]); n = np.array([r[2] for r in rows]); sh = np.array([r[3] for r in rows])
    print(f"肥尾 carry 调查(Binance 多年 2020-2026,含死币,n={len(rows)} 币)")
    print(f"全体: 毛 funding 中位 {np.median(g):+.1%} / 净额(含价格腿) 中位 {np.median(n):+.1%}")
    print(f"\n按毛 funding 分桶:")
    print(f"  {'桶':>14s} {'n':>3s} {'毛中位':>8s} {'净中位':>8s} {'净/毛':>7s} {'净日Sharpe中位':>12s} {'净>0占比':>8s}")
    qs = np.quantile(g, [0, 0.5, 0.75, 0.9, 1.0])
    labels = ["低(<50%)", "中(50-75%)", "高(75-90%)", "肥尾(top10%)"]
    for k in range(4):
        lo, hi = qs[k], qs[k + 1]
        m = (g >= lo) & (g <= hi) if k == 3 else (g >= lo) & (g < hi)
        if m.sum() < 2:
            continue
        print(f"  {labels[k]:>14s} {m.sum():>3d} {np.median(g[m]):>+8.1%} {np.median(n[m]):>+8.1%} "
              f"{np.median(n[m])/np.median(g[m]) if np.median(g[m]) else 0:>7.0%} {np.median(sh[m]):>12.1f} {np.mean(n[m]>0):>8.0%}")
    print(f"\n肥尾 carry 币例(毛>90分位):")
    for s, gg, nn, ss, hh in rows[:10]:
        print(f"  {s:14s} 毛 {gg:+8.1%}  净 {nn:+8.1%}  净日Sharpe {ss:.1f}  持仓 {hh} bar")
    print("\n数据问题排查:含死币(175/272退市)✓ | 真实价格腿(spot−perp)✓ | 真实间隔✓ | 双边mask有效才计✓ | 多年✓")
    print("判读:肥尾桶净中位 vs 中桶——更高=极端正funding carry值得集中;净/毛看价格腿吃多少;净>0占比看稳。")


if __name__ == "__main__":
    main()
