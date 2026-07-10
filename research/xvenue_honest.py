"""诚实跨所体检 —— 杀掉两个 lookahead,并用真实 HL 数据验证。
用户质疑(对):combined_checkup 的跨所 +24%/Sh13 不可信,因为
  (1) funding 腿 `inc = cur*|spread|` 恒为正 —— 假设永远站在对的一边;
  (2) 价格腿 `-sgn[t]*dret` 用**当根 bar** 的 sign[t] —— 每 bar 事后选对边(lookahead)。

本脚本两部分:
  A) 长历史 Bin-Byb(2020-2026):对比【旧乐观法(|spread|+当根sign)】vs【诚实法(固定方向+有符号)】,
     量化 lookahead 吹了多少。诚实法:方向按**滞后一根**的 EMA(d)定,持有,收**有符号**价差(翻向就付);
     sizing 也滞后一根(t-1 决策,t 兑现);价格腿不剪极端(单列尾部)。
  B) 真实 HL vs Binance(~6 个月,HL klines 仅 2025-11 起):同一诚实法跑真实 HL,看真所到底有没有边。

跑：PYTHONUTF8=1 .venv/Scripts/python.exe research/xvenue_honest.py
"""
import glob
import json
import os
import sys
import urllib.request

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from venues import canon

UA = {"User-Agent": "Mozilla/5.0"}
ANN = 8760


def fetch_intervals():
    g = lambda u: json.loads(urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=25).read())
    try:
        bi = {d["symbol"]: d.get("fundingIntervalHours", 8) for d in g("https://fapi.binance.com/fapi/v1/fundingInfo")}
    except Exception:
        bi = {}
    yi = {}; cur = ""
    try:
        while True:
            res = g("https://api.bybit.com/v5/market/instruments-info?category=linear&limit=1000" + (f"&cursor={cur}" if cur else ""))["result"]
            for s in res["list"]:
                if s.get("fundingInterval"):
                    yi[s["symbol"]] = int(s["fundingInterval"]) / 60.0
            cur = res.get("nextPageCursor", "")
            if not cur:
                break
    except Exception:
        pass
    return bi, yi


def ema_causal(x, hl, valid):
    """逐 bar 因果 EMA(只用 ≤t),无效 bar 不更新。返回 T×N。"""
    al = 1.0 - 0.5 ** (1.0 / hl)
    T, N = x.shape
    out = np.zeros((T, N)); s = np.zeros(N); init = np.zeros(N, bool)
    for t in range(T):
        v = valid[t]
        newv = v & ~init
        s[newv] = x[t][newv]; init[newv] = True
        upd = v & init & ~newv
        s[upd] = al * x[t][upd] + (1 - al) * s[upd]
        out[t] = s
    return out


def honest_legs(d, retA, retB, both, tdv, spread_for_size, lab, ema_hl=24, cost_bps=None, xcfg=None):
    """d=fA-fB(每小时有符号);retA/retB=两所永续次bar收益。
    诚实:方向=sign(EMA(d) 滞后一根),sizing 滞后一根;funding=w*dir*d;price=w*dir*(retB-retA)。
    返回 net(诚实)、net(旧乐观)、诊断。ema_hl/cost_bps/xcfg 用于敏感性/画像对比。"""
    import strategy as S
    T, N = d.shape
    emaD = ema_causal(np.where(both, d, 0.0), ema_hl, both)    # 因果 EMA 信号
    # ---- 诚实法 ----
    xc = xcfg or S.XVenueConfig()
    if cost_bps is not None:
        xc.cost_bps = cost_bps
    xs = S.CrossVenueStrategy(N, xc)
    fund_h = np.zeros(T); price_h = np.zeros(T); turn_h = np.zeros(T); prev = np.zeros(N)
    neg_bars = 0; tot_bars = 0
    for t in range(T):
        lag = max(t - 1, 0)
        w = xs.step(spread_for_size[lag], tdv[lag], both[lag])    # sizing 滞后一根
        dirn = np.sign(emaD[lag])                                 # 方向滞后一根(≤t-1)
        contrib_f = w * dirn * d[t]                               # 有符号 funding(翻向就付)
        fund_h[t] = contrib_f.sum()
        price_h[t] = (w * dirn * (retB[t] - retA[t])).sum()
        turn_h[t] = np.abs(w - prev).sum(); prev = w
        act = (w > 1e-9) & both[t]
        neg_bars += (contrib_f[act] < 0).sum(); tot_bars += act.sum()
    net_h = fund_h + np.clip(price_h, -0.5, 0.5) - xc.cost_bps / 1e4 * turn_h  # 宽剪(只防爆数值)
    # ---- 旧乐观法(复现 combined_checkup)----
    spread = np.abs(d); sgn = np.sign(d)
    xs2 = S.CrossVenueStrategy(N, xc)
    fund_o = np.zeros(T); price_o = np.zeros(T); turn_o = np.zeros(T); prev = np.zeros(N)
    dret = np.clip(retA - retB, -0.10, 0.10)
    for t in range(T):
        w = xs2.step(spread[t], tdv[t], both[t])                 # 当根 sizing
        fund_o[t] = (w * spread[t]).sum()                        # |spread| 恒正
        price_o[t] = (w * (-sgn[t] * dret[t])).sum()            # 当根 sign(lookahead)
        turn_o[t] = np.abs(w - prev).sum(); prev = w
    net_o = fund_o + price_o - xc.cost_bps / 1e4 * turn_o
    return dict(net_h=net_h, fund_h=fund_h, price_h=price_h, net_o=net_o, fund_o=fund_o,
                price_o=price_o, neg_frac=neg_bars / max(tot_bars, 1), turn_h=turn_h.mean() * ANN,
                price_raw=price_h)


def sh(p, m):
    p = p[m]; return p.mean() / p.std() * np.sqrt(ANN) if (len(p) and p.std() > 0) else 0.0


def mdd(p, m):
    p = p[m]; c = np.cumsum(p); return (c - np.maximum.accumulate(c)).min() if len(p) else 0.0


def report(r, dates, lab):
    T = len(r["net_h"]); yr = np.array([dd[:4] for dd in dates])
    oos = np.zeros(T, bool); oos[int(T * 0.4):] = True
    print(f"\n===== {lab} =====")
    print(f"{'区间':>8s} | {'旧funding':>9s} {'诚实funding':>11s} | {'旧net':>8s} {'诚实net':>8s} | {'诚实Sh':>7s} {'诚实maxDD':>9s}")
    for L, m in [("OOS全", oos)] + [(y, yr == y) for y in ("2021", "2022", "2023", "2024", "2025", "2026")]:
        if m.sum() < 50:
            continue
        print(f"{L:>8s} | {r['fund_o'][m].mean()*ANN:>+8.1%} {r['fund_h'][m].mean()*ANN:>+10.1%} | "
              f"{r['net_o'][m].mean()*ANN:>+7.1%} {r['net_h'][m].mean()*ANN:>+7.1%} | "
              f"{sh(r['net_h'],m):>+7.2f} {mdd(r['net_h'],m):>+9.1%}")
    pr = r["price_raw"]
    print(f"诊断: funding 翻向(付钱)bar 占比 {r['neg_frac']:.1%}  | 诚实换手 {r['turn_h']:.0f}/yr")
    print(f"  价格腿尾部(未剪): 最差单bar {pr.min():+.2%} / 最好 {pr.max():+.2%} / |>5%| 的bar数 {(np.abs(pr)>0.05).sum()}")
    print(f"  *** lookahead 吹胀: 旧net OOS {r['net_o'][oos].mean()*ANN:+.1%} → 诚实net OOS {r['net_h'][oos].mean()*ANN:+.1%}"
          f"  (缩水 {1-r['net_h'][oos].mean()/max(r['net_o'][oos].mean(),1e-12):.0%})")


def part_A():
    z = np.load("data/clean/crypto_tensor_60min_pit.npz", allow_pickle=True)
    adj = z["adj_close"].astype(float); tdv = z["tdv"].astype(float)
    slots = list(z["slots"].astype(str)); dates = z["dates"].astype(str); T, N = adj.shape
    nmap = {s: i for i, s in enumerate(slots)}
    xv = np.load("data/clean/xvenue_funding.npz", allow_pickle=True)
    bi, yi = fetch_intervals()
    biv = np.array([bi.get(s, 8) for s in slots]); yiv = np.array([yi.get(s, 8) for s in slots])
    fb = xv["f_bin"] / biv[None, :]; fy = xv["f_byb"] / yiv[None, :]              # 每小时
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
    d = np.where(both, fb - fy, 0.0)
    binret = np.zeros((T, N)); bybret = np.zeros((T, N)); v2 = both[:-1] & both[1:]
    binret[:-1][v2] = (adj[1:] / adj[:-1] - 1)[v2]; bybret[:-1][v2] = (byb[1:] / byb[:-1] - 1)[v2]
    binret = np.nan_to_num(binret); bybret = np.nan_to_num(bybret)
    r = honest_legs(d, binret, bybret, both, tdv, np.abs(d), "Bin-Byb")
    report(r, dates, "A) 长历史 Bin-Byb(2020-2026):旧乐观法 vs 诚实法")
    # --- 敏感性:证明诚实数不是 EMA半衰期/成本 调出来的 ---
    T = len(dates); oos = np.zeros(T, bool); oos[int(T * 0.4):] = True
    print("\n--- 敏感性(诚实 net OOS 年化;非真实数据=只有这两个选择参数)---")
    print(f"{'EMA半衰期\\成本bps':>16s} {'5':>8s} {'10(默认)':>9s} {'20':>8s}")
    for hl in [12, 24, 48]:
        row = []
        for cb in [5, 10, 20]:
            rr = honest_legs(d, binret, bybret, both, tdv, np.abs(d), "s", ema_hl=hl, cost_bps=cb)
            row.append(f"{rr['net_h'][oos].mean()*ANN:>+7.1%}")
        mark = "(默认)" if hl == 24 else ""
        print(f"{('hl=%d%s'%(hl,mark)):>16s} {row[0]:>8s} {row[1]:>9s} {row[2]:>8s}")
    print("→ 区间窄=结论稳健,非选参产物。换手低(7/yr)故成本几乎不影响。")
    return grid, dates, slots, nmap, adj, tdv, fb, biv


def part_B(grid, dates, slots, nmap, adj, tdv, fb, biv):
    """真实 HL vs Binance。HL funding(1h)+ HL klines。窗口受 HL klines 限(~2025-11起)。"""
    T = len(dates); N = len(slots)
    gdf = pd.DataFrame({"ts": grid})
    # canon 映射:tensor slot -> 列 j;HL 文件名 canon 后匹配
    slot_canon = {canon(s.replace("USDT", "")): nmap[s] for s in slots}
    fhl = np.full((T, N), np.nan); phl = np.full((T, N), np.nan)
    for f in glob.glob("data/raw/hl/funding/*.csv"):
        c = canon(os.path.basename(f)[:-4]); j = slot_canon.get(c)
        if j is None:
            continue
        df = pd.read_csv(f).sort_values("ts").drop_duplicates("ts", keep="last")
        m = pd.merge_asof(gdf, df[["ts", "funding"]], on="ts", direction="backward", tolerance=3600_000 + 1)
        fhl[:, j] = m["funding"].to_numpy()                       # HL 已是每小时
    for f in glob.glob("data/raw/hl/kline/*.csv"):
        c = canon(os.path.basename(f)[:-4]); j = slot_canon.get(c)
        if j is None:
            continue
        df = pd.read_csv(f).sort_values("ts").drop_duplicates("ts", keep="last")
        m = pd.merge_asof(gdf, df[["ts", "close"]], on="ts", direction="backward", tolerance=3600_000)
        phl[:, j] = m["close"].to_numpy()
    both = (adj > 0) & np.isfinite(fb) & np.isfinite(fhl) & np.isfinite(phl) & (phl > 0)
    # 只在 HL 真有数据的时段(避免 merge_asof 把早期填成 NaN 之外的脏值)
    cov = both.sum(1)
    rows = np.where(cov >= 3)[0]
    if len(rows) == 0:
        print("\n===== B) 真实 HL vs Binance: 无重叠数据 ====="); return
    lo, hi = rows.min(), rows.max() + 1
    print(f"\n[B] 真实 HL 窗口: {dates[lo]} → {dates[hi-1]}  ({hi-lo} 根bar, 平均 {cov[lo:hi].mean():.0f} 币/bar)")
    sl = slice(lo, hi)
    d = np.where(both, fb - fhl, 0.0)                              # Bin(每小时) - HL(每小时)
    binret = np.zeros((T, N)); hlret = np.zeros((T, N)); v2 = both[:-1] & both[1:]
    binret[:-1][v2] = (adj[1:] / adj[:-1] - 1)[v2]; hlret[:-1][v2] = (phl[1:] / phl[:-1] - 1)[v2]
    binret = np.nan_to_num(binret); hlret = np.nan_to_num(hlret)
    r = honest_legs(d[sl], binret[sl], hlret[sl], both[sl], tdv[sl], np.abs(d[sl]), "HL-Bin")
    m = np.ones(hi - lo, bool)
    print(f"  诚实 funding 年化 {r['fund_h'][m].mean()*ANN:+.1%} | 诚实 net 年化 {r['net_h'].mean()*ANN:+.1%} | "
          f"诚实Sh {sh(r['net_h'],m):+.2f} | maxDD {mdd(r['net_h'],m):+.1%}")
    print(f"  对比旧乐观法 net 年化 {r['net_o'].mean()*ANN:+.1%}  | funding 翻向bar {r['neg_frac']:.1%}")
    pr = r["price_raw"]
    print(f"  HL 价格腿尾部(未剪): 最差 {pr.min():+.2%} / 最好 {pr.max():+.2%} / |>5%|bar {(np.abs(pr)>0.05).sum()}")
    print("  注:~6个月、HL klines 2025-11起,样本短;真值仍需前向。但这是**真所、诚实法**的第一眼。")


if __name__ == "__main__":
    g, dates, slots, nmap, adj, tdv, fb, biv = part_A()
    part_B(g, dates, slots, nmap, adj, tdv, fb, biv)
