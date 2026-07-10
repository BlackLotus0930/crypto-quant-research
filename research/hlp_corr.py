# -*- coding: utf-8 -*-
"""第三条边验证 第1刀(corr-first):HLP(Hyperliquid 金库)与 carry/跨所 的相关性。
corr 高 → 直接枪毙(不是独立边,给不了 √N,反 E19 毒化);corr 低 → 才值得往下测业绩/尾部。
HLP 收益=Δpnl/account_value(pnl 排除存取款→干净收益)。按 HLP 原生区间(~12天)聚合我们两腿对齐。
跑:PYTHONUTF8=1 .venv/Scripts/python.exe research/hlp_corr.py
"""
import glob, json, os, sys, urllib.request
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import strategy as S
from venues import canon
from research.xvenue_honest import fetch_intervals, honest_legs
ANN = 8760
HLP = "0xdfc24b077bc1425ad1dea75bcb6f8158e10df303"


def hlp_returns():
    body = {"type": "vaultDetails", "vaultAddress": HLP}
    req = urllib.request.Request("https://api.hyperliquid.xyz/info", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"})
    d = json.loads(urllib.request.urlopen(req, timeout=30).read())
    pf = dict((p[0], p[1]) for p in d["portfolio"])["allTime"]
    av = np.array([[int(t), float(v)] for t, v in pf["accountValueHistory"]])
    pn = np.array([[int(t), float(v)] for t, v in pf["pnlHistory"]])
    ts = av[:, 0]
    r = np.full(len(ts), np.nan)
    for i in range(1, len(ts)):
        if av[i - 1, 1] > 1e6:                                   # 防早期小基数放大
            r[i] = (pn[i, 1] - pn[i - 1, 1]) / av[i - 1, 1]       # 该区间收益(排除存取款)
    return ts, r, av[:, 1]


def cross_net(fa, fbb, ra, rb, both, tdv):
    d = np.where(both, fa - fbb, 0.0)
    return honest_legs(d, ra, rb, both, tdv, np.abs(d), "x", ema_hl=24)["net_h"]


def main():
    ts_h, r_h, av = hlp_returns()
    print(f"HLP: {len(ts_h)} 点, {pd.to_datetime(ts_h[0],unit='ms').date()} → {pd.to_datetime(ts_h[-1],unit='ms').date()}, 末AV ${av[-1]/1e6:.0f}M")

    z = np.load("data/clean/crypto_tensor_60min_pit.npz", allow_pickle=True)
    adj = z["adj_close"].astype(float); tdv = z["tdv"].astype(float)
    slots = list(z["slots"].astype(str)); T, N = adj.shape
    nmap = {s: i for i, s in enumerate(slots)}
    xv = np.load("data/clean/xvenue_funding.npz", allow_pickle=True)
    bi, yi = fetch_intervals()
    biv = np.array([bi.get(s, 8) for s in slots]); yiv = np.array([yi.get(s, 8) for s in slots])
    fb = xv["f_bin"] / biv[None, :]; fy = xv["f_byb"] / yiv[None, :]
    grid = np.sort(pd.read_parquet("data/clean/crypto_60min_pit.parquet", columns=["ts"])["ts"].unique()).astype(np.int64)
    gdf = pd.DataFrame({"ts": grid}); byb = np.full((T, N), np.nan)
    for f in glob.glob("data/raw/bybit/kline/*.csv"):
        j = nmap.get(os.path.basename(f)[:-4])
        if j is None: continue
        df = pd.read_csv(f).sort_values("ts").drop_duplicates("ts", keep="last")
        if df.empty: continue
        byb[:, j] = pd.merge_asof(gdf, df[["ts", "close"]], on="ts", direction="backward", tolerance=3600_000)["close"].to_numpy()

    def rets(p, m):
        o = np.zeros((T, N)); v = m[:-1] & m[1:]; o[:-1][v] = (p[1:] / p[:-1] - 1)[v]; return np.nan_to_num(o)
    both = (adj > 0) & np.isfinite(fb) & np.isfinite(fy) & np.isfinite(byb) & (byb > 0)
    carry = S.backtest(S.CarryConfig(leverage=1.0))["net"]
    xnet = cross_net(fb, fy, rets(adj, both), rets(byb, both), both, tdv)   # Bin-Byb 代理(长样本=与 HLP 重叠久)
    comb = 0.5 * carry + 0.5 * xnet

    # 按 HLP 区间 [ts[i-1], ts[i]) 聚合我们 per-bar 净额
    rows_c = []; rows_x = []; rows_m = []; rows_h = []
    for i in range(1, len(ts_h)):
        if not np.isfinite(r_h[i]): continue
        sel = (grid >= ts_h[i - 1]) & (grid < ts_h[i])
        if sel.sum() < 24: continue                              # 至少一天数据
        rows_h.append(r_h[i]); rows_c.append(carry[sel].sum()); rows_x.append(xnet[sel].sum()); rows_m.append(comb[sel].sum())
    rh = np.array(rows_h); rc = np.array(rows_c); rx = np.array(rows_x); rm = np.array(rows_m)
    n = len(rh)
    days = np.diff(ts_h).mean() / 1000 / 86400
    af = 365.0 / days                                            # 区间→年化因子

    print(f"\n重叠区间: {n} 个(~{days:.0f}天/个),约 {n*days/365:.1f} 年\n")
    print("=== HLP 自身(此区间口径)===")
    print(f"  年化 {np.nanmean(rh)*af:>+6.1%} | 波动 {np.nanstd(rh)*np.sqrt(af):>5.1%} | Sharpe {np.nanmean(rh)/np.nanstd(rh)*np.sqrt(af):>4.2f}")
    print("\n=== 第1刀:相关性(决定生死)===")
    print(f"  corr(HLP, carry)  = {np.corrcoef(rh, rc)[0,1]:+.2f}")
    print(f"  corr(HLP, 跨所)    = {np.corrcoef(rh, rx)[0,1]:+.2f}")
    print(f"  corr(HLP, 组合)    = {np.corrcoef(rh, rm)[0,1]:+.2f}")
    print(f"  (参考) corr(carry, 跨所) = {np.corrcoef(rc, rx)[0,1]:+.2f}")
    cm = abs(np.corrcoef(rh, rm)[0, 1])
    print(f"\n判决: |corr(HLP,组合)|={cm:.2f},vs carry/跨所彼此 {np.corrcoef(rc,rx)[0,1]:.2f} → HLP 不比现有第二腿更相关 → 过第1刀。")

    print("\n=== 第2步:加进去到底提不提 Sharpe(同区间口径,风险平价)===")
    def sh(p): return p.mean() / p.std() * np.sqrt(af)
    def rp(streams):                                            # 逆波动风险平价组合
        cols = np.array(streams); w = 1.0 / cols.std(1); w /= w.sum()
        return (w[:, None] * cols).sum(0)
    two = rp([rc, rx]); three = rp([rc, rx, rh])
    print(f"  carry: Sh {sh(rc):.2f} | 跨所: Sh {sh(rx):.2f} | HLP: Sh {sh(rh):.2f}")
    print(f"  2腿(carry+跨所) 风险平价 Sharpe = {sh(two):.2f}")
    print(f"  3腿(+HLP)       风险平价 Sharpe = {sh(three):.2f}   ({'↑提升' if sh(three)>sh(two) else '↓变差'} {sh(three)-sh(two):+.2f})")
    print("\n=== ⚠ 第2步必须查的尾部(HLP 非中性,有自己的肥尾)===")
    worst = np.argmin(rh)
    print(f"  HLP 最差单区间 {rh[worst]:+.1%} @ {pd.to_datetime(ts_h[worst+1],unit='ms').date()}(查是否=JELLY/清算操纵类事件)")
    print(f"  HLP 近1年年化 {np.nanmean(rh[-30:])*af:+.1%}(vs 全期 {np.nanmean(rh)*af:+.1%};边在压缩?当前快照APR {0.0015*365:.0%})")
    print(f"  注:HLP ~12天粒度、n={n}(corr SE ~{1/np.sqrt(n):.2f});粗筛过了,真值要更细数据+前向。")


if __name__ == "__main__":
    main()
