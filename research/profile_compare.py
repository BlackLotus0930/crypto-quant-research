# -*- coding: utf-8 -*-
"""任务1:肥尾画像(薄币,tilt2)vs 流动画像(只 top 流动币,tilt1)。
用数据回答:Sharpe 会不会高、年化会不会低、配上各自安全杠杆后 谁更好。
- 流动限制:跨所 kx=25 / carry kpos=25(只 top-25 by tdv);肥尾 kx=100/kpos=100 tilt2。
- 尾部:分别量两画像持仓宇宙内的 单币跳幅 + 基差跳幅 → 各自安全杠杆 → 杠杆后年化/最坏。
跑:PYTHONUTF8=1 .venv/Scripts/python.exe research/profile_compare.py
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
    biv = np.array([bi.get(s, 8) for s in slots])
    fb = xv["f_bin"] / biv[None, :]
    grid = np.sort(pd.read_parquet("data/clean/crypto_60min_pit.parquet", columns=["ts"])["ts"].unique()).astype(np.int64)
    gdf = pd.DataFrame({"ts": grid})
    sc = {canon(s.replace("USDT", "")): nmap[s] for s in slots}
    phl = np.full((T, N), np.nan); fhl = np.full((T, N), np.nan)
    for f in glob.glob("data/raw/hl/kline/*.csv"):
        j = sc.get(canon(os.path.basename(f)[:-4]))
        if j is None: continue
        df = pd.read_csv(f).sort_values("ts").drop_duplicates("ts", keep="last")
        phl[:, j] = pd.merge_asof(gdf, df[["ts", "close"]], on="ts", direction="backward", tolerance=3600_000)["close"].to_numpy()
    for f in glob.glob("data/raw/hl/funding/*.csv"):
        j = sc.get(canon(os.path.basename(f)[:-4]))
        if j is None: continue
        df = pd.read_csv(f).sort_values("ts").drop_duplicates("ts", keep="last")
        fhl[:, j] = pd.merge_asof(gdf, df[["ts", "funding"]], on="ts", direction="backward", tolerance=3600_001)["funding"].to_numpy()
    return adj, tdv, dates, fb, fhl, phl, T, N


def rets(p, m, T, N):
    o = np.zeros((T, N)); v = m[:-1] & m[1:]; o[:-1][v] = (p[1:] / p[:-1] - 1)[v]; return np.nan_to_num(o)
def weekly(net, dates, idx):
    df = pd.DataFrame({"w": [dates[i][:7] + "-" + str(int(dates[i][8:10]) // 7) for i in idx], "r": net[idx]})
    return df.groupby("w")["r"].sum().to_numpy()
def shw(net, dates, m):
    wk = weekly(net, dates, np.where(m)[0]); return wk.mean() / wk.std() * np.sqrt(52) if wk.std() > 0 else 0


def main():
    adj, tdv, dates, fb, fhl, phl, T, N = load()
    both = (adj > 0) & np.isfinite(fb) & np.isfinite(fhl) & np.isfinite(phl) & (phl > 0)
    rows = np.where(both.sum(1) >= 3)[0]; lo, hi = rows.min(), rows.max() + 1
    m = np.zeros(T, bool); m[lo:hi] = True
    d = np.where(both, fb - fhl, 0.0)
    binret = rets(adj, both, T, N); hlret = rets(phl, both, T, N)

    # top-K 流动掩码(每bar 按 tdv 排名)
    def topk_mask(K):
        msk = np.zeros((T, N), bool)
        for t in range(lo, hi):
            act = np.where(both[t])[0]
            if len(act) == 0: continue
            keep = act[np.argsort(tdv[t, act])[::-1][:K]]
            msk[t, keep] = True
        return msk
    liq = topk_mask(25)

    print("=== 任务1:肥尾画像 vs 流动画像(真实 HL 窗,组合=carry+跨所,周频 Sharpe)===\n")
    profiles = {
        "肥尾(薄币,tilt2,k100)": dict(xcfg=S.XVenueConfig(tilt_pow=2.0, kx=100, cap=0.05),
                                  ccfg=S.CarryConfig(tilt_pow=2.0, cap=0.03, kpos=100), bothx=both),
        "流动(top25,tilt1,k25)": dict(xcfg=S.XVenueConfig(tilt_pow=1.0, kx=25, cap=0.05),
                                   ccfg=S.CarryConfig(tilt_pow=1.0, cap=0.03, kpos=25), bothx=liq),
    }
    res = {}
    for name, p in profiles.items():
        xnet = honest_legs(d, binret, hlret, p["bothx"], tdv, np.abs(d), "x", ema_hl=24, xcfg=p["xcfg"])["net_h"]
        carry = S.backtest(p["ccfg"])["net"]
        comb = 0.5 * carry + 0.5 * xnet
        ann = comb[m].mean() * ANN; sw = shw(comb, dates, m)
        # 尾部:该画像持仓宇宙内 单币跳幅 + 基差跳幅
        univ = p["bothx"][lo:hi]
        rr = np.abs(binret[lo:hi][univ]); rr = rr[np.isfinite(rr) & (rr > 0)]
        bdiff = np.abs((binret - hlret)[lo:hi][univ]); bdiff = bdiff[np.isfinite(bdiff) & (bdiff > 0)]
        spike = np.percentile(rr, 99.9); spike_max = rr.max()
        gap = np.percentile(bdiff, 99.9); gap_max = bdiff.max()
        res[name] = dict(ann=ann, sw=sw, comb=comb, spike999=spike, spikemax=spike_max, gap999=gap, gapmax=gap_max)
        print(f"[{name}]")
        print(f"  组合年化(1x) {ann:>+6.1%} | 周Sharpe {sw:>4.2f}")
        print(f"  单币跳幅 p99.9 {spike:>5.1%} / max {spike_max:>5.1%} | 基差跳幅 p99.9 {gap:>5.1%} / max {gap_max:>5.1%}")
        # 安全杠杆:扛住 p99.9 跳不爆腿(强平距离1/L>跳),且相关基差gap*L<30%
        L_liq = 1.0 / spike                       # 扛住 p99.9 单币跳的腿杠杆上限
        L_gap = 0.30 / max(gap * 5, 1e-9)         # 相关gap(~5×单币p99.9)*L<30%
        L_safe = max(1.0, min(L_liq, L_gap, 10))
        print(f"  → 安全杠杆 ≈ {L_safe:.1f}x(强平界{L_liq:.1f}x / 基差界{L_gap:.1f}x 取小)")
        print(f"  → 杠杆后年化 {ann*L_safe:>+6.1%} | 最坏相关gap {-gap*5*L_safe:>+5.1%}\n")

    print("=== 对比判读 ===")
    a, b = res["肥尾(薄币,tilt2,k100)"], res["流动(top25,tilt1,k25)"]
    print(f"  Sharpe: 肥尾 {a['sw']:.2f} → 流动 {b['sw']:.2f}（{'升' if b['sw']>a['sw'] else '降'}）")
    print(f"  1x年化: 肥尾 {a['ann']:+.1%} → 流动 {b['ann']:+.1%}（{'升' if b['ann']>a['ann'] else '降'}）")
    La = max(1.0, min(1/a['spike999'], 0.30/(a['gap999']*5), 10)); Lb = max(1.0, min(1/b['spike999'], 0.30/(b['gap999']*5), 10))
    print(f"  安全杠杆后年化: 肥尾 {a['ann']*La:+.1%}({La:.1f}x) vs 流动 {b['ann']*Lb:+.1%}({Lb:.1f}x)")
    print("  注:尾部=真实窗内持仓宇宙实测;杠杆界=强平距离+相关基差启发式(保守)。真值前向证。")


if __name__ == "__main__":
    main()
