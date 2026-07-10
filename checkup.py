# -*- coding: utf-8 -*-
"""★ 标准策略体检(单一把尺子)——每次改策略都跑这个,数字才可比。
回答三件事,全用诚实法(固定方向+有符号+真实数据,见 research/xvenue_honest.py):
  (1) 可靠 Sharpe:频率阶梯(per-bar/日/周)+ 日自相关 —— 自相关趋零的频率才是诚实 Sharpe。
  (2) 跨 regime 稳定性:组合日 Sharpe 逐年(2022 熊市是真考验)。
  (3) 尾部:峰度 + CVaR + 杠杆 scaling(只含市场/基差尾;对手方/强平/脱锚不在数据里)。

两个口径:
  A) 多年 Bin-Byb 代理(2020-2026,含 2022)= 长样本、可靠;但 CEX 干净→真实 HL 在此之下。
  B) 真实 HL 窗(~1年)= 真所、真噪声;但短、全上行市。
真值在 A、B 之间,且前向只会更低。

跑:PYTHONUTF8=1 .venv/Scripts/python.exe checkup.py [--lev 1]
"""
import argparse, glob, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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
    def klines(pat, key):
        out = np.full((T, N), np.nan)
        for f in glob.glob(pat):
            j = key(os.path.basename(f)[:-4])
            if j is None: continue
            df = pd.read_csv(f).sort_values("ts").drop_duplicates("ts", keep="last")
            if df.empty: continue
            col = "funding" if "funding" in df.columns else "close"
            m = pd.merge_asof(gdf, df[["ts", col]], on="ts", direction="backward", tolerance=3600_001)
            out[:, j] = m[col].to_numpy()
        return out
    byb = klines("data/raw/bybit/kline/*.csv", lambda s: nmap.get(s))
    sc = {canon(s.replace("USDT", "")): nmap[s] for s in slots}
    fhl = klines("data/raw/hl/funding/*.csv", lambda s: sc.get(canon(s)))
    phl = klines("data/raw/hl/kline/*.csv", lambda s: sc.get(canon(s)))
    return dict(adj=adj, tdv=tdv, dates=dates, T=T, N=N, fb=fb, fy=fy, byb=byb, fhl=fhl, phl=phl)


def rets(p, m, T, N):
    o = np.zeros((T, N)); v = m[:-1] & m[1:]; o[:-1][v] = (p[1:] / p[:-1] - 1)[v]; return np.nan_to_num(o)


def xnet(fa, fbb, ra, rb, both, tdv):
    d = np.where(both, fa - fbb, 0.0)
    return honest_legs(d, ra, rb, both, tdv, np.abs(d), "x", ema_hl=24)["net_h"]


def sh(p, ann): p = np.asarray(p); return p.mean() / p.std() * np.sqrt(ann) if p.std() > 0 else 0.0
def mdd(p): c = np.cumsum(p); return (c - np.maximum.accumulate(c)).min()
def by(net, dates, idx, key):
    df = pd.DataFrame({"k": [key(dates[i]) for i in idx], "r": net[idx]}); return df.groupby("k")["r"].sum().to_numpy()
def daily(net, dates, m): return by(net, dates, np.where(m)[0], lambda x: x[:10])
def weekly(net, dates, m): return by(net, dates, np.where(m)[0], lambda x: x[:7] + "-" + str(int(x[8:10]) // 7))


def ladder(lab, s, dates, m):
    dd = daily(s, dates, m); wk = weekly(s, dates, m)
    ac = np.corrcoef(dd[:-1], dd[1:])[0, 1] if len(dd) > 2 else 0.0
    print(f"{lab:>14s} | {sh(s[m], ANN):>8.2f} {sh(dd, 365):>8.2f} {sh(wk, 52):>7.2f} | {ac:>+8.2f}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--lev", type=float, default=1.0); a = ap.parse_args()
    D = load(); adj, fb, fy, byb, fhl, phl, tdv, dates, T, N = (D[k] for k in
        ("adj", "fb", "fy", "byb", "fhl", "phl", "tdv", "dates", "T", "N"))
    carry = S.backtest(S.CarryConfig(leverage=a.lev))["net"]
    bbboth = (adj > 0) & np.isfinite(fb) & np.isfinite(fy) & np.isfinite(byb) & (byb > 0)
    xbb = a.lev * xnet(fb, fy, rets(adj, bbboth, T, N), rets(byb, bbboth, T, N), bbboth, tdv)
    comb = 0.5 * carry + 0.5 * xbb
    oos = np.zeros(T, bool); oos[int(T * 0.4):] = True; yr = np.array([s[:4] for s in dates])

    print(f"================ 策略体检(杠杆 {a.lev}x)================")
    print("\n=== A) 多年 Bin-Byb 代理(2020-2026,含2022;长样本=可靠口径)===")
    print("注:CEX 代理偏干净→真实 HL 噪声更大,真值在此之下(见 B)。")
    print(f"{'序列':>14s} | {'per-bar':>8s} {'日×√365':>8s} {'周×√52':>7s} | {'日自相关':>8s}  (自相关→0 的频率才是诚实 Sharpe)")
    ladder("carry", carry, dates, oos); ladder("跨所代理", xbb, dates, oos); ladder("组合", comb, dates, oos)
    print("\n  组合 日Sharpe 逐年(跨 regime 稳定性=真可靠性):")
    for y in ("2021", "2022", "2023", "2024", "2025", "2026"):
        m = yr == y
        if m.sum() < 200: continue
        dd = daily(comb, dates, m)
        tag = " ← 熊市真考验" if y == "2022" else ""
        print(f"    {y}: 日Sharpe {sh(dd,365):>5.2f} | 年化 {comb[m].mean()*ANN:>+6.1%} | 最差单日 {dd.min():>+6.2%}{tag}")

    # B) 真实 HL 窗
    hlboth = (adj > 0) & np.isfinite(fb) & np.isfinite(fhl) & np.isfinite(phl) & (phl > 0)
    rows = np.where(hlboth.sum(1) >= 3)[0]; lo, hi = rows.min(), rows.max() + 1
    m = np.zeros(T, bool); m[lo:hi] = True
    xhl = a.lev * xnet(fb, fhl, rets(adj, hlboth, T, N), rets(phl, hlboth, T, N), hlboth, tdv)
    combHL = 0.5 * carry + 0.5 * xhl
    print(f"\n=== B) 真实 HL 窗 {dates[lo][:10]}→{dates[hi-1][:10]}(真所真噪声;短、全上行)===")
    print(f"{'序列':>14s} | {'per-bar':>8s} {'日×√365':>8s} {'周×√52':>7s} | {'日自相关':>8s}")
    ladder("跨所(真实HL)", xhl, dates, m); ladder("组合(真实HL)", combHL, dates, m)

    # C) 尾部(用真实 HL 组合,日频)
    dd = daily(combHL, dates, m); q5 = np.percentile(dd, 5)
    kurt = ((dd - dd.mean())**4).mean() / dd.std()**4 - 3
    print(f"\n=== C) 尾部(真实HL组合,日频,{a.lev}x)===")
    print(f"  最差单日 {dd.min():+.2%} | 5%CVaR {dd[dd<=q5].mean():+.3%} | 超额峰度 {kurt:+.1f}(>0=肥尾,迟早破最差日)")
    base = dd / a.lev  # 1x 日序列(线性缩放)
    print("  杠杆 scaling(线性): " + " | ".join(
        f"{L}x→年化{base.sum()/( (hi-lo)/ANN)*L:+.0%},最差日{base.min()*L:+.2%},一次5%gap{-0.05*L:+.0%}" for L in (1, 2, 3, 5)))
    print("  ⚠ 只含市场/基差尾;对手方冻结/强平/脱锚不在数据里→真尾部更肥,靠结构控(docs/风控.md)。")
    print("\n判读铁律:看周 Sharpe + 逐年(尤其2022),别看 per-bar 绝对值;前向是唯一终判。")


if __name__ == "__main__":
    main()
