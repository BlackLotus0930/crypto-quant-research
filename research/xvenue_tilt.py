"""验证肥尾倾斜:权重 ∝ spread^p 扫 p,在 Bin-Byb 真实 1yr 数据(含真价格腿)看组合净额/Sharpe/集中度。
确认倾斜不伤(净不降、Sharpe不塌)再把 tilt_pow 设进生产。每币帽 cap 仍在(防过度集中)。
跑：PYTHONUTF8=1 .venv/Scripts/python.exe research/xvenue_tilt.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from strategy import _cap_renorm

ANN = 8760


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
    import glob
    z = np.load("data/clean/crypto_tensor_60min_pit.npz", allow_pickle=True)
    adj = z["adj_close"].astype(float); tdv = z["tdv"].astype(float)
    slots = list(z["slots"].astype(str)); T, N = adj.shape
    nmap = {s: i for i, s in enumerate(slots)}
    xv = np.load("data/clean/xvenue_funding.npz", allow_pickle=True)
    bi, yi = fetch_intervals()
    biv = np.array([bi.get(s, 8) for s in slots]); yiv = np.array([yi.get(s, 8) for s in slots])
    fb = xv["f_bin"] / biv[None, :]; fy = xv["f_byb"] / yiv[None, :]      # per-hr,真实间隔
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

    yr = np.array([d[:4] for d in z["dates"].astype(str)]); oos = np.zeros(T, bool); oos[int(T * 0.4):] = True
    CAP = 0.05; K = 100

    def run(p):
        held = np.zeros((T, N)); prevsgn = np.zeros(N)
        pnl = np.zeros(T)
        # 简化:每bar直接按 spread^p 选权重(lagged 用 t 的 spread 决策、t+1 realize),含帽
        S = np.zeros(N); al = 1 - 0.5 ** (1.0 / 120)
        for t in range(T):
            cand = np.where(both[t] & (spread[t] > 0))[0]
            W = np.zeros(N)
            if len(cand):
                if len(cand) > K:
                    cand = cand[np.argsort(tdv[t, cand])[::-1][:K]]
                w = np.power(spread[t, cand], p); s = w.sum()
                if s > 0:
                    W[cand] = _cap_renorm(w / s, CAP)
            S = al * W + (1 - al) * S if t else W
            g = np.abs(S).sum(); held[t] = S / g if g > 0 else S
            # P&L(t→t+1):funding=held·spread + price=held·(−sgn)·dret(下一bar realize 在 t+1 加)
        # P&L:用 held[t] 收 t+1 的 funding(spread[t+1])+ price(dret[t+1],sgn[t])
        for t in range(T - 1):
            pnl[t + 1] = (held[t] * (spread[t + 1] + (-sgn[t]) * dret[t + 1])).sum()
        return pnl, held

    print("Bybit 价格覆盖 + 真实间隔已加载。扫肥尾倾斜 tilt_pow:")
    print(f"  {'tilt_pow':>8s} {'OOS年化':>8s} {'OOS日Sharpe':>11s} {'最大单币权重':>11s} {'有效币数(中位)':>13s}")
    for p in [1.0, 1.5, 2.0, 3.0]:
        pnl, held = run(p)
        po = pnl[oos]
        nd = len(po) // 24; dd = po[:nd * 24].reshape(nd, 24).sum(1)
        sh = dd.mean() / dd.std() * np.sqrt(365) if dd.std() > 0 else 0
        maxw = np.median(np.abs(held[oos]).max(1))
        npos = np.median((np.abs(held[oos]) > 1e-9).sum(1))
        print(f"  {p:>8.1f} {po.mean()*ANN:>+8.1%} {sh:>11.2f} {maxw:>11.1%} {npos:>13.0f}")
    print("\n判读:tilt_pow↑ → 往肥尾集中(最大单币权重↑、有效币数↓)。看年化是否升 + 日Sharpe 不塌 = 倾斜值得。")
    print("(cap=5% 仍封顶;funding 腿+真实 Bin-Byb 价格腿;OOS 后60%。)")


if __name__ == "__main__":
    main()
