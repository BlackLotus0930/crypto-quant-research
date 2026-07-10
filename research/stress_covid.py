# -*- coding: utf-8 -*-
"""COVID(2020-03)最坏情况实测——按"百年危机"锚定杠杆,而非为它阉割收益。
carry(Binance 现货+永续)完整覆盖 COVID;跨所 Bin-Byb 代理价格腿 Bybit klines 从 2020-03-25 起(漏最惨几天)。
量:COVID 窗 carry/组合 的 最坏单日 + maxDD(1x)→ 各杠杆下 COVID maxDD → 定"扛得住但不阉割"的杠杆。
跑:PYTHONUTF8=1 .venv/Scripts/python.exe research/stress_covid.py
"""
import glob, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import strategy as S
from research.xvenue_honest import fetch_intervals, honest_legs
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
        if df.empty: continue
        byb[:, j] = pd.merge_asof(gdf, df[["ts", "close"]], on="ts", direction="backward", tolerance=3600_000)["close"].to_numpy()
    return adj, tdv, dates, fb, fy, byb, T, N


def daily(net, dates, m):
    idx = np.where(m)[0]; df = pd.DataFrame({"d": [dates[i][:10] for i in idx], "r": net[idx]})
    return df.groupby("d")["r"].sum()


def mdd(c): return (c / np.maximum.accumulate(c) - 1).min()


def main():
    adj, tdv, dates, fb, fy, byb, T, N = load()
    both = (adj > 0) & np.isfinite(fb) & np.isfinite(fy) & np.isfinite(byb) & (byb > 0)
    d = np.where(both, fb - fy, 0.0)
    def rets(p, m): o = np.zeros((T, N)); v = m[:-1] & m[1:]; o[:-1][v] = (p[1:]/p[:-1]-1)[v]; return np.nan_to_num(o)
    carry = S.backtest(S.CarryConfig(leverage=1.0))["net"]
    xnet = honest_legs(d, rets(adj, both), rets(byb, both), both, tdv, np.abs(d), "x", ema_hl=24)["net_h"]
    comb = 0.5 * carry + 0.5 * xnet

    ym = np.array([s[:7] for s in dates])
    covid = np.array([m in ("2020-02", "2020-03", "2020-04") for m in ym])
    full = np.ones(T, bool)
    print(f"COVID 窗覆盖: {covid.sum()} bar | 该窗 carry 活跃币中位 {np.median((np.abs(S.backtest(S.CarryConfig())['held'][covid])>1e-9).sum(1)):.0f}")
    print(f"该窗跨所 both-finite 中位币数 {np.median(both[covid].sum(1)):.0f}(Bybit klines 2020-03-25起→3月初跨所近乎空,COVID 主要测 carry)\n")

    print("=== COVID(2020-02~04)最坏实测(1x)===")
    for lab, s in [("carry", carry), ("组合(carry+跨所)", comb)]:
        dc = daily(s, dates, covid); cum = np.cumprod(1 + dc.to_numpy())
        print(f"  [{lab}] COVID 期间: 累计 {cum[-1]-1:>+6.1%} | 最坏单日 {dc.min():>+6.2%} | maxDD {mdd(cum):>+6.1%}")
        worst = dc.nsmallest(3)
        print(f"     最差3天: " + ", ".join(f"{i} {v:+.1%}" for i, v in worst.items()))
    # funding 行为
    fb_covid = fb[covid]; print(f"\n  COVID 期 Binance funding 均值 {np.nanmean(fb_covid[np.isfinite(fb_covid)])*ANN:+.0%}/yr(看是否翻负/飙)")

    print("\n=== COVID maxDD 随杠杆(carry,完整覆盖)===")
    dc = daily(carry, dates, covid).to_numpy()
    print(f"{'杠杆':>5s} {'COVID maxDD':>12s} {'COVID最坏单日':>13s} | 判读(10%高风险桶)")
    for L in [1, 3, 5, 6, 8, 10]:
        cum = np.cumprod(1 + L * dc); dmax = mdd(cum); wd = (L * dc).min()
        tag = "可接受" if dmax > -0.5 else ("痛但活" if dmax > -0.75 else "归零")
        print(f"{L:>4d}x {dmax:>+12.1%} {wd:>+13.2%} | {tag}")

    print("\n=== 对照:全样本 & 2022 的 carry maxDD(看 COVID 是不是真最坏)===")
    for lab, m in [("全样本", full), ("2022熊市", ym == "2022-06") if False else ("2022", np.array([s[:4]=='2022' for s in dates]))]:
        dc2 = daily(carry, dates, m).to_numpy(); print(f"  {lab}: carry maxDD(1x) {mdd(np.cumprod(1+dc2)):>+6.1%}")
    print("\n判读:若 COVID 1x maxDD 小 → 这书连百年危机都温和 → 杠杆可大胆;若大 → 按它锚定。")
    print("caveat:跨所价格腿漏3月初最惨;carry 完整但 2020 币少(PIT)。真值前向+实盘。")


if __name__ == "__main__":
    main()
