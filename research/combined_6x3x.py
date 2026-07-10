# -*- coding: utf-8 -*-
"""整体检验:carry 6x + 跨所 3x(50/50 资本),跨所带每对止损 overlay。
回答:止损在平时(没崩盘)会不会频繁触发→磨掉 funding→拉低整体年化/Sharpe?
- 跨所止损:某币价从入场动 > 22%(=70%×3x强平距离)→ 平该对(funding 停收 + 往返成本),下次调仓再入。
- 对比:跨所 3x 不带止损(理想但崩盘会被强平,回测测不到强平) vs 带止损(可实盘,有磨损)。
- 组合 = 0.5·6·carry_1x + 0.5·3·跨所_1x;周频 Sharpe(=6.2 口径)。Bin-Byb 代理(长样本含崩盘)。
跑:PYTHONUTF8=1 .venv/Scripts/python.exe research/combined_6x3x.py
"""
import glob, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import strategy as S
from research.xvenue_honest import fetch_intervals, ema_causal
ANN = 8760


def load():
    z = np.load("data/clean/crypto_tensor_60min_pit.npz", allow_pickle=True)
    adj = z["adj_close"].astype(float); tdv = z["tdv"].astype(float)
    slots = list(z["slots"].astype(str)); dates = z["dates"].astype(str); T, N = adj.shape
    nmap = {s: i for i, s in enumerate(slots)}
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
    return adj, byb, fb, fy, tdv, dates, T, N


def cross_bt(adj, byb, fb, fy, tdv, T, N, stop=None):
    """跨所 1x per-bar net。stop=None 不带止损;stop=0.22 带每对止损(价动>stop 平该对,调仓再入)。"""
    both = (adj > 0) & np.isfinite(fb) & np.isfinite(fy) & np.isfinite(byb) & (byb > 0)
    d = np.where(both, fb - fy, 0.0); spread = np.abs(d)
    binret = np.zeros((T, N)); bybret = np.zeros((T, N)); v = both[:-1] & both[1:]
    binret[:-1][v] = (adj[1:] / adj[:-1] - 1)[v]; bybret[:-1][v] = (byb[1:] / byb[:-1] - 1)[v]
    binret = np.nan_to_num(binret); bybret = np.nan_to_num(bybret)
    emaD = ema_causal(np.where(both, d, 0.0), 24, both)
    xc = S.XVenueConfig(); xs = S.CrossVenueStrategy(N, xc)
    net = np.zeros(T); prev = np.zeros(N)
    entry = np.full(N, np.nan); stopped = np.zeros(N, bool); nfire = 0; nbar_active = 0
    since = 0
    for t in range(T):
        lag = max(t - 1, 0)
        w = xs.step(spread[lag], tdv[lag], both[lag])
        dirn = np.sign(emaD[lag])
        held = w > 1e-9
        # 入场记录 / 调仓重置
        since += 1; rebal = since >= xc.cad
        new = held & np.isnan(entry)
        entry[new] = adj[t, new]; stopped[new] = False
        if rebal:                                    # 调仓:重置入场(被止损的重新入场)
            entry[held] = adj[t, held]; stopped[held] = False; since = 0
        entry[~held] = np.nan; stopped[~held] = False
        # 止损判定
        cost_stop = 0.0
        if stop is not None:
            move = np.abs(np.where(held & np.isfinite(entry), adj[t] / entry - 1, 0.0))
            newstop = held & (~stopped) & (move > stop)
            cost_stop = xc.cost_bps / 1e4 * 2 * w[newstop].sum()    # 平该对往返成本
            stopped |= newstop
            nfire += newstop.sum()
        eff = w * (~stopped)                          # 被止损的不收 funding/不参与
        nbar_active += held.sum()
        fund = (eff * dirn * d[t]).sum()
        price = (eff * dirn * (bybret[t] - binret[t])).sum()
        turn = np.abs(w - prev).sum(); prev = w
        net[t] = fund + np.clip(price, -0.5, 0.5) - xc.cost_bps / 1e4 * turn - cost_stop
    return net, nfire, nbar_active


def main():
    adj, byb, fb, fy, tdv, dates, T, N = load()
    carry1 = S.backtest(S.CarryConfig(leverage=1.0))["net"]
    x_nostop, _, _ = cross_bt(adj, byb, fb, fy, tdv, T, N, stop=None)
    x_stop, nfire, nact = cross_bt(adj, byb, fb, fy, tdv, T, N, stop=0.22)

    oos = np.zeros(T, bool); oos[int(T * 0.4):] = True; yr = np.array([s[:4] for s in dates])
    def wk(net, m):
        idx = np.where(m)[0]; df = pd.DataFrame({"w": [dates[i][:7] + "-" + str(int(dates[i][8:10]) // 7) for i in idx], "r": net[idx]})
        return df.groupby("w")["r"].sum().to_numpy()
    def shw(net, m): w = wk(net, m); return w.mean() / w.std() * np.sqrt(52) if w.std() > 0 else 0
    def ann(net, m): return net[m].mean() * ANN
    def mdd(net, m): c = np.cumsum(net[m]); return (c - np.maximum.accumulate(c)).min()

    # 组合 = 0.5·6·carry + 0.5·3·跨所
    comb_nostop = 3 * carry1 + 1.5 * x_nostop
    comb_stop = 3 * carry1 + 1.5 * x_stop

    print("=== 整体检验:carry 6x + 跨所 3x(50/50 资本),OOS,周频 Sharpe ===\n")
    print(f"止损触发: {nfire} 次 / {nact} 币·bar 持仓 = {nfire/max(nact,1)*100:.2f}% 的持仓bar(平时磨损源)\n")
    print(f"{'组件':>22s} {'OOS年化':>8s} {'周Sharpe':>8s} {'全样本maxDD':>11s}")
    for lab, s in [("carry 6x", 6 * carry1),
                   ("跨所 3x(无止损,理想)", 3 * x_nostop),
                   ("跨所 3x(带止损,可实盘)", 3 * x_stop),
                   ("组合(无止损,理想)", comb_nostop),
                   ("组合 carry6x+跨所3x(带止损)", comb_stop)]:
        print(f"{lab:>22s} {ann(s,oos):>+7.1%} {shw(s,oos):>8.2f} {mdd(s,np.ones(T,bool)):>+11.1%}")

    print("\n=== 止损的代价(跨所 3x:带 vs 不带)===")
    dr_a = ann(3*x_nostop, oos) - ann(3*x_stop, oos); dr_s = shw(3*x_nostop, oos) - shw(3*x_stop, oos)
    print(f"  年化磨损 {dr_a:+.1%} | Sharpe 磨损 {dr_s:+.2f}")
    print(f"  组合层面: 年化 {ann(comb_nostop,oos):+.1%}→{ann(comb_stop,oos):+.1%} | Sharpe {shw(comb_nostop,oos):.2f}→{shw(comb_stop,oos):.2f}")
    print("\n=== 逐年(组合带止损,@6x/3x)===")
    for y in ("2021","2022","2023","2024","2025","2026"):
        m = yr == y
        if m.sum() < 200: continue
        print(f"  {y}: 年化 {ann(comb_stop,m):>+6.1%} | 周Sharpe {shw(comb_stop,m):>5.2f} | maxDD {mdd(comb_stop,m):>+6.1%}")
    print("\n判读:① 止损'年化磨损'=平时没事也触发的代价。② 但无止损版崩盘会被强平(回测测不到,见E55:裸3x崩盘−45~70%)")
    print("  → 无止损的'理想'数不可实盘达到;带止损的数才是真能拿到的。③ 看磨损是否<<崩盘保护价值。")


if __name__ == "__main__":
    main()
