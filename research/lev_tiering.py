# -*- coding: utf-8 -*-
"""① 单币杠杆分层:每币杠杆 ∝ 1/它的gap风险(流动币高杠杆,薄币低)。
现在全书一个杠杆,被最薄的币(gap大)卡死。分层=每币 tail 贡献拉平到预算 B → 没单币能炸 → 有效杠杆更高。
对比:统一杠杆(被worst币卡) vs 分层杠杆,在**同一单币tail预算**下,有效书杠杆+收益各多少。
跑:PYTHONUTF8=1 .venv/Scripts/python.exe research/lev_tiering.py
"""
import glob, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import strategy as S
from venues import canon
from research.xvenue_honest import fetch_intervals, ema_causal
ANN = 8760
B = 0.003          # 单币单bar tail 预算(占书,p99.9 gap×杠杆×权重 ≤ B)
LCAP = 12          # 单币杠杆上限


def main():
    z = np.load("data/clean/crypto_tensor_60min_pit.npz", allow_pickle=True)
    adj = z["adj_close"].astype(float); tdv = z["tdv"].astype(float)
    slots = list(z["slots"].astype(str)); T, N = adj.shape
    nmap = {s: i for i, s in enumerate(slots)}
    xv = np.load("data/clean/xvenue_funding.npz", allow_pickle=True)
    bi, yi = fetch_intervals(); biv = np.array([bi.get(s, 8) for s in slots])
    fb = xv["f_bin"] / biv[None, :]
    grid = np.sort(pd.read_parquet("data/clean/crypto_60min_pit.parquet", columns=["ts"])["ts"].unique()).astype(np.int64)
    gdf = pd.DataFrame({"ts": grid}); sc = {canon(s.replace("USDT", "")): nmap[s] for s in slots}
    phl = np.full((T, N), np.nan); fhl = np.full((T, N), np.nan)
    for f in glob.glob("data/raw/hl/kline/*.csv"):
        j = sc.get(canon(os.path.basename(f)[:-4]))
        if j is not None:
            df = pd.read_csv(f).sort_values("ts").drop_duplicates("ts", keep="last")
            phl[:, j] = pd.merge_asof(gdf, df[["ts", "close"]], on="ts", direction="backward", tolerance=3600_000)["close"].to_numpy()
    for f in glob.glob("data/raw/hl/funding/*.csv"):
        j = sc.get(canon(os.path.basename(f)[:-4]))
        if j is not None:
            df = pd.read_csv(f).sort_values("ts").drop_duplicates("ts", keep="last")
            fhl[:, j] = pd.merge_asof(gdf, df[["ts", "funding"]], on="ts", direction="backward", tolerance=3600_001)["funding"].to_numpy()
    both = (adj > 0) & np.isfinite(fb) & np.isfinite(fhl) & np.isfinite(phl) & (phl > 0)
    rows = np.where(both.sum(1) >= 3)[0]; lo, hi = rows.min(), rows.max() + 1
    d = np.where(both, fb - fhl, 0.0); emaD = ema_causal(np.where(both, d, 0.0), 24, both)
    binret = np.zeros((T, N)); hlret = np.zeros((T, N)); v2 = both[:-1] & both[1:]
    binret[:-1][v2] = (adj[1:]/adj[:-1]-1)[v2]; hlret[:-1][v2] = (phl[1:]/phl[:-1]-1)[v2]
    bchg = np.abs(np.nan_to_num(binret - hlret))

    # 跑跨所策略,累计每币:权重、funding贡献(有符号)
    xs = S.CrossVenueStrategy(N, S.XVenueConfig())
    sw = np.zeros(N); sfund = np.zeros(N); cnt = np.zeros(N)
    for t in range(lo, hi):
        lag = max(t - 1, lo)
        w = xs.step(np.abs(d[lag]), tdv[lag], both[lag])
        dirn = np.sign(emaD[lag])
        sw += w; sfund += w * dirn * d[t]; cnt += (w > 1e-9)
    held = np.where(sw > 1e-9)[0]
    wbar = sw[held] / (hi - lo)                      # 平均权重
    rate = sfund[held] / np.maximum(sw[held], 1e-12) * ANN   # 每单位权重 年化funding
    gap = np.array([np.percentile(bchg[lo:hi, i][both[lo:hi, i]], 99.9) if both[lo:hi, i].sum() > 50 else 0.05 for i in held])
    gap = np.maximum(gap, 1e-4)
    contrib = wbar * (sfund[held] / sw[held]) * ANN  # 每币年化收益贡献(1x)

    # 统一 vs 分层(同单币 tail 预算 B)
    risk = wbar * gap                                 # 每币单bar tail/单位杠杆
    L_uni = B / risk.max()                            # 统一:被worst卡
    L_i = np.minimum(B / risk, LCAP)                  # 分层:每币拉平到 B
    eff_uni = L_uni                                   # 统一有效杠杆
    eff_tier = (wbar * L_i).sum() / wbar.sum()        # 分层有效书杠杆(权重加权)
    ret_uni = contrib.sum() * L_uni
    ret_tier = (contrib * L_i).sum()

    print(f"=== ① 单币杠杆分层(真实 HL 窗,跨所腿,{len(held)}币,单币tail预算 B={B:.1%})===\n")
    print(f"  gap风险分布: 中位 {np.median(gap):.2%} / p90 {np.percentile(gap,90):.2%} / max {gap.max():.2%}")
    print(f"  → 最薄币 gap {gap.max():.1%} 把统一杠杆卡在 {L_uni:.1f}x;但中位币 gap 才 {np.median(gap):.2%}\n")
    print(f"{'方案':>10s} | {'有效书杠杆':>9s} | {'年化收益':>8s} | {'单币tail':>8s}")
    print(f"{'统一杠杆':>10s} | {eff_uni:>8.1f}x | {ret_uni:>+7.0%} | {B:>7.1%}(被worst卡)")
    print(f"{'分层杠杆':>10s} | {eff_tier:>8.1f}x | {ret_tier:>+7.0%} | {B:>7.1%}(每币拉平)")
    print(f"\n  → 分层 vs 统一: 有效杠杆 {eff_tier/max(eff_uni,1e-9):.1f}× | 收益 {ret_tier/max(ret_uni,1e-9):.1f}×")
    print(f"  → 杠杆上限{LCAP}x 下,{(L_i>=LCAP).sum()}/{len(held)} 币顶到上限(流动币本可更高)")
    print("\n判读:同一单币tail预算下,分层把有效杠杆从被薄币卡死的低值,抬到流动币能承受的高值。")
    print("  代价:薄币(肥funding)杠杆被压低→那部分收益少;但流动币杠杆抬高补回。净效果看 收益倍数。")
    print("  注:gap=p99.9单bar基差,真实窗内每币实测;tail预算B是风控选择;真值前向证。")


if __name__ == "__main__":
    main()
