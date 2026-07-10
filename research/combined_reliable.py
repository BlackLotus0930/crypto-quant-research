# -*- coding: utf-8 -*-
"""可靠 Sharpe + 6.02→2.7 溯源 + 尾部。全部在同一真实窗(HL 那一年)上做,苹果对苹果。
回答:(1)组合可靠 Sharpe 多少(频率阶梯去自相关)(2)Sharpe 怎么从旧 6.02 掉到 2.7(改了什么)
(3)尾部长什么样(worst day/CVaR/峰度/杠杆 scaling)。
跑:PYTHONUTF8=1 .venv/Scripts/python.exe research/combined_reliable.py
"""
import glob, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import strategy as S
from venues import canon
from research.xvenue_honest import fetch_intervals, honest_legs
ANN = 8760


def load():
    z = np.load("data/clean/crypto_tensor_60min_pit.npz", allow_pickle=True)
    adj = z["adj_close"].astype(float); tdv = z["tdv"].astype(float)
    slots = list(z["slots"].astype(str)); dates = z["dates"].astype(str); T, N = adj.shape
    nmap = {s: i for i, s in enumerate(slots)}
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
        m = pd.merge_asof(gdf, df[["ts", "close"]], on="ts", direction="backward", tolerance=3600_000)
        byb[:, j] = m["close"].to_numpy()
    slot_canon = {canon(s.replace("USDT", "")): nmap[s] for s in slots}
    fhl = np.full((T, N), np.nan); phl = np.full((T, N), np.nan)
    for f in glob.glob("data/raw/hl/funding/*.csv"):
        j = slot_canon.get(canon(os.path.basename(f)[:-4]))
        if j is None: continue
        df = pd.read_csv(f).sort_values("ts").drop_duplicates("ts", keep="last")
        m = pd.merge_asof(gdf, df[["ts", "funding"]], on="ts", direction="backward", tolerance=3600_001)
        fhl[:, j] = m["funding"].to_numpy()
    for f in glob.glob("data/raw/hl/kline/*.csv"):
        j = slot_canon.get(canon(os.path.basename(f)[:-4]))
        if j is None: continue
        df = pd.read_csv(f).sort_values("ts").drop_duplicates("ts", keep="last")
        m = pd.merge_asof(gdf, df[["ts", "close"]], on="ts", direction="backward", tolerance=3600_000)
        phl[:, j] = m["close"].to_numpy()
    return dict(adj=adj, tdv=tdv, dates=dates, T=T, N=N, fb=fb, fy=fy, byb=byb, fhl=fhl, phl=phl)


def xv_net(fa, fb_, ra, rb, both, tdv):
    """返回 (honest_net, old_net) 两条 per-bar 序列。fa-fb_=有符号价差;ra/rb=两所收益。"""
    d = np.where(both, fa - fb_, 0.0)
    r = honest_legs(d, ra, rb, both, tdv, np.abs(d), "x", ema_hl=24)
    return r["net_h"], r["net_o"]


def rets(adj_or_p, both):
    T, N = adj_or_p.shape
    out = np.zeros((T, N)); v2 = both[:-1] & both[1:]
    out[:-1][v2] = (adj_or_p[1:] / adj_or_p[:-1] - 1)[v2]
    return np.nan_to_num(out)


def sh(p, ann):
    p = np.asarray(p); return p.mean() / p.std() * np.sqrt(ann) if p.std() > 0 else 0.0


def to_freq(net, dts, n):
    """把 per-bar(小时)net 聚合成每 n 小时一块(24=日,168=周)的非重叠 sum。"""
    k = (len(net) // n) * n
    return net[:k].reshape(-1, n).sum(1)


def mdd(p):
    c = np.cumsum(p); return (c - np.maximum.accumulate(c)).min()


def main():
    D = load()
    adj, fb, fy, byb, fhl, phl, tdv, dates = D["adj"], D["fb"], D["fy"], D["byb"], D["fhl"], D["phl"], D["tdv"], D["dates"]
    # 真实 HL 窗
    bothHL = (adj > 0) & np.isfinite(fb) & np.isfinite(fhl) & np.isfinite(phl) & (phl > 0)
    rows = np.where(bothHL.sum(1) >= 3)[0]; lo, hi = rows.min(), rows.max() + 1
    sl = slice(lo, hi); dts = dates[lo:hi]; M = hi - lo
    print(f"统一真实窗: {dates[lo]} → {dates[hi-1]}  ({M} bar ≈ {M/8760:.2f} 年)\n")

    binret = rets(adj, bothHL); hlret = rets(phl, bothHL)
    bbboth = (adj > 0) & np.isfinite(fb) & np.isfinite(fy) & np.isfinite(byb) & (byb > 0)
    bbbin = rets(adj, bbboth); bybret = rets(byb, bbboth)

    # 各跨所变体(都切到同一 HL 窗)
    hl_h, hl_o = xv_net(fb[sl], fhl[sl], binret[sl], hlret[sl], bothHL[sl], tdv[sl])
    bb_h, bb_o = xv_net(fb[sl], fy[sl], bbbin[sl], bybret[sl], bbboth[sl], tdv[sl])
    carry = S.backtest(S.CarryConfig(leverage=1.0))["net"][sl]
    comb = 0.5 * carry + 0.5 * hl_h               # 真实组合(carry + 真实 HL 跨所)

    print("=== (2) 6.02 → 2.7 溯源:同窗、同 per-bar×√8760,逐步改变 ===")
    print(f"{'变体':>34s} {'年化':>8s} {'per-bar Sh':>11s}")
    for lab, s in [("Bin-Byb 旧法(|spread|+当根sign)", bb_o),
                   ("Bin-Byb 诚实(固定方向+有符号)", bb_h),
                   ("→ 改成真实 HL 数据(诚实法)", hl_h)]:
        print(f"{lab:>34s} {s.mean()*ANN:>+7.1%} {sh(s,ANN):>11.2f}")
    print("  解读:|spread|→有符号 让 Bin-Byb 略降;**真正把 Sh 打下来的是换成真实 HL**(DEX 价格腿噪声进分母)。")
    print(f"  注:E45 的 6.02 是更早脚本(funding/8 + 不同聚合);本表同口径重算,锚点是'改了什么',非复刻 6.02。\n")

    print("=== (1) 可靠 Sharpe:频率阶梯(去 funding 自相关)===")
    print(f"{'序列':>14s} | {'per-bar×√8760':>13s} {'日×√365':>9s} {'周×√52':>8s} | {'日net自相关':>10s}")
    for lab, s in [("carry", carry), ("跨所(真实HL)", hl_h), ("组合50/50", comb)]:
        dd = to_freq(s, dts, 24); wk = to_freq(s, dts, 168)
        ac = np.corrcoef(dd[:-1], dd[1:])[0, 1] if len(dd) > 2 else 0
        print(f"{lab:>14s} | {sh(s,ANN):>13.2f} {sh(dd,365):>9.2f} {sh(wk,52):>8.2f} | {ac:>+10.2f}")
    print("  → per-bar 虚高(funding 近确定+强自相关,有效样本<<8760);**日/周 Sharpe 才是可信区**。")
    print("  → 若日 net 自相关≈0,日×√365 就是诚实 Sharpe;仍>0 说明还有持久性,看周。\n")

    print("=== (3) 尾部:组合书(日频)===")
    dd = to_freq(comb, dts, 24)
    q5 = np.percentile(dd, 5); cvar = dd[dd <= q5].mean()
    kurt = ((dd - dd.mean())**4).mean() / dd.std()**4 - 3
    print(f"  日均 {dd.mean():+.3%} | 日 std {dd.std():.3%} | 最差单日 {dd.min():+.2%} | 最好 {dd.max():+.2%}")
    print(f"  5% CVaR(最差5%日均) {cvar:+.3%} | 超额峰度 {kurt:+.1f}(0=正态;>0=肥尾)")
    print(f"{'杠杆':>6s} {'年化':>8s} {'日maxDD':>8s} {'最差单日':>8s} {'估算:一次5%基差gap':>18s}")
    for L in [1, 2, 3, 5]:
        d2 = dd * L
        print(f"{L:>5d}x {d2.sum()/(M/8760):>+7.1%} {mdd(d2):>+8.1%} {d2.min():>+8.2%} {(-0.05*L):>17.0%}")
    print("  注:这是**市场/基差尾部**(数据里有的)。对手方冻结/强平/脱锚**不在数据里**→ 真尾部更肥,靠结构控(下文)。")


if __name__ == "__main__":
    main()
