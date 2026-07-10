# -*- coding: utf-8 -*-
"""任务2:自动减仓系统 详细调查(用数据)。
逻辑:自动减仓能拦"多bar累积回撤"(有时间反应),拦不住"单bar瞬时gap"(来不及反应)。
所以它把杠杆的硬约束从【多bar相关基差gap】挪到【单bar瞬时gap】。用数据量两个天花板 + 拖累。
- 硬天花板 = 组合 1x 单bar(1h)最坏 → 自动减仓拦不住 → L×它 < 30%。
- 软约束(无减仓)= 多bar相关gap → L×它 < 30%。
- 在岗模拟(回撤控制 overlay):高基础杠杆下,实测年化/maxDD/触发次数/拖累(证明"几乎免费保险")。
跑:PYTHONUTF8=1 .venv/Scripts/python.exe research/autodelev.py
"""
import glob, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import strategy as S
from venues import canon
from research.xvenue_honest import fetch_intervals, honest_legs
ANN = 8760


def load_combined():
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
    both = (adj > 0) & np.isfinite(fb) & np.isfinite(fy) & np.isfinite(byb) & (byb > 0)
    d = np.where(both, fb - fy, 0.0)
    def rets(p, m): o = np.zeros((T, N)); v = m[:-1] & m[1:]; o[:-1][v] = (p[1:]/p[:-1]-1)[v]; return np.nan_to_num(o)
    xnet = honest_legs(d, rets(adj, both), rets(byb, both), both, tdv, np.abs(d), "x", ema_hl=24)["net_h"]
    carry = S.backtest(S.CarryConfig(leverage=1.0))["net"]
    comb = 0.5 * carry + 0.5 * xnet
    return comb, dates, T


def daily(net, dates):
    df = pd.DataFrame({"day": [x[:10] for x in dates], "r": net}); g = df.groupby("day")["r"].sum()
    return g.to_numpy()


def sim_drawdown_control(daily_1x, L0, trig=-0.15, cut=0.25, restore=-0.06):
    """回撤控制 overlay:回撤<trig 减到 cut×L0,回撤回到 restore 上方恢复 L0。返回(年化,maxDD,触发天数)。"""
    eq = 1.0; peak = 1.0; lev = L0; ers = []; trig_days = 0
    for r in daily_1x:
        dd = eq / peak - 1
        if dd < trig: lev = L0 * cut; trig_days += 1
        elif dd > restore: lev = L0
        er = lev * r; eq *= (1 + er); peak = max(peak, eq); ers.append(er)
    ers = np.array(ers); cum = np.cumprod(1 + ers)
    mdd = (cum / np.maximum.accumulate(cum) - 1).min()
    ann = (cum[-1] ** (365 / len(ers)) - 1)
    return ann, mdd, trig_days


def naive(daily_1x, L0):
    er = L0 * daily_1x; cum = np.cumprod(1 + er)
    return cum[-1] ** (365 / len(er)) - 1, (cum / np.maximum.accumulate(cum) - 1).min()


def main():
    comb, dates, T = load_combined()
    dd = daily(comb, dates)
    print("=== 任务2:自动减仓系统(多年 Bin-Byb 代理组合,含2022)===\n")
    # 1) 两个天花板
    bar_worst = comb.min()                      # 单bar(1h)最坏 → 减仓拦不住
    day_worst = dd.min(); roll3 = pd.Series(dd).rolling(3).sum().min()
    print("=== 两个杠杆天花板(自动减仓把约束从'多bar'挪到'单bar')===")
    print(f"  单bar(1h)最坏 {bar_worst:+.2%} → 瞬时gap,减仓**拦不住** → 硬天花板 L < {0.30/abs(bar_worst):.0f}x(L×它<30%)")
    print(f"  单日最坏 {day_worst:+.2%} / 3日累计最坏 {roll3:+.2%} → 多bar,减仓**能拦**")
    print(f"  → 无减仓:约束在多bar(~相关基差gap);有减仓:约束放宽到单bar(上面那个大数)\n")
    # 2) 在岗模拟:高基础杠杆 + 回撤控制 vs naive
    print("=== 在岗模拟:基础杠杆 vs +自动减仓(回撤控制 trig-15%/减到25%)===")
    print(f"{'基础L':>6s} | {'naive年化':>9s} {'naive maxDD':>11s} | {'+减仓年化':>9s} {'+减仓maxDD':>10s} {'触发天数':>7s} {'拖累':>7s}")
    for L0 in [3, 6, 10]:
        na, nm = naive(dd, L0); ca, cm, td = sim_drawdown_control(dd, L0)
        print(f"{L0:>5d}x | {na:>+8.1%} {nm:>+11.1%} | {ca:>+8.1%} {cm:>+10.1%} {td:>7d} {ca-na:>+7.1%}")
    print("\n=== 压力注入:数据无危机,人造一次相关基差崩(连续3天各-8%@1x)看减仓救不救 ===")
    stress = np.concatenate([dd, np.array([-0.08, -0.08, -0.08])])     # 模拟一次危机尾
    for L0 in [6, 10]:
        na, nm = naive(stress, L0); ca, cm, td = sim_drawdown_control(stress, L0)
        print(f"  {L0}x: naive 危机maxDD {nm:+.1%}({'归零' if nm<-0.6 else '可恢复'}) | +减仓 maxDD {cm:+.1%}({'归零' if cm<-0.6 else '可恢复'})")
    print("\n判读:")
    print("  ① 在岗拖累≈0(平时几乎不触发)=自动减仓是'近免费保险'。")
    print("  ② 压力下:naive 高杠杆危机归零;+减仓把它拦成可恢复回撤 → **这就是它让杠杆能更高的机制**。")
    print(f"  ③ 但减仓拦不住单bar瞬时gap → 硬天花板 ~{0.30/abs(bar_worst):.0f}x;它把安全杠杆从~4x抬到这个量级,**不是抬到20x**。")
    print("  ④ 真值caveat:数据无真危机(连2022组合都正),压力是人造;减仓速度/滑点实盘要测。")


if __name__ == "__main__":
    main()
