"""fat-carry 机会普查:LEVER 这样的大头多久出现一次?有多少个?我们抓得住吗?
每币"可收割 funding"= Σ_{活跃且funding>0} funding/8(持单位现金套利、funding正时收)。
看:① 分布(是不是只有一个LEVER还是一串);② 逐年有几个大机会;③ 我们的流动性截顶会不会漏掉它们。
跑：python carry_census.py
"""
import numpy as np

z = np.load("data/clean/crypto_tensor_60min_pit.npz", allow_pickle=True)
mask = z["mask"]; adj = z["adj_close"].astype(float); tdv = z["tdv"].astype(float)
dates = z["dates"].astype(str); slots = np.array(z["slots"].astype(str)); T, N = mask.shape
funding = np.load("data/clean/funding_pit.npz", allow_pickle=True)["funding"].astype(float)
spot = np.load("data/clean/spot_pit.npz", allow_pickle=True)["spot"].astype(float)
yr = np.array([d[:4] for d in dates]); ann = 8760
vp = mask & np.isfinite(spot) & (spot > 0) & (adj > 0)          # 能做cash-and-carry(有现货)
fpos = np.where(vp & np.isfinite(funding), np.clip(funding, 0, None), 0.0)
harvest_cell = fpos / 8.0                                       # 每bar若持有该收的funding

# 每币:总可收割、年化funding、活跃年数、流动性、峰值funding
harv = harvest_cell.sum(0)
dur = vp.sum(0) / ann
annf = np.where(vp.sum(0) > 0, np.nansum(np.where(vp, funding, 0), 0) / np.maximum(vp.sum(0), 1) * 3 * 365, 0)
liq = np.array([np.median(tdv[vp[:, c], c]) if vp[:, c].any() else 0 for c in range(N)])
peakf = np.array([np.nanmax(np.where(vp[:, c], funding[:, c], np.nan)) if vp[:, c].any() else 0 for c in range(N)])

tot = harv.sum()
order = np.argsort(harv)[::-1]
print(f"全宇宙可做cash-and-carry的币: {int((dur>0).sum())};  总可收割funding(单位仓)= {tot:.2f}")
print(f"集中度:top-1占总 {harv[order[0]]/tot:.0%}, top-5 {harv[order[:5]].sum()/tot:.0%}, "
      f"top-10 {harv[order[:10]].sum()/tot:.0%}, top-20 {harv[order[:20]].sum()/tot:.0%}\n")

print("=== 最大的 20 个 fat-carry 机会(全期) ===")
print(f"{'币':>16s} {'可收割':>7s} {'占总':>5s} {'年化funding':>10s} {'活跃年':>6s} {'中位$量':>9s} {'峰值f(8h)':>9s}")
for c in order[:20]:
    print(f"{slots[c]:>16s} {harv[c]:>7.3f} {harv[c]/tot*100:>4.0f}% {annf[c]:>9.0%} {dur[c]:>6.2f}"
          f" {liq[c]:>9.2e} {peakf[c]:>9.4f}")

print("\n=== 逐年:有几个'大机会'(该年可收割funding 排名)& 当年总可收割 ===")
for y in ("2021", "2022", "2023", "2024", "2025", "2026"):
    m = yr == y
    if m.sum() < 50:
        continue
    hy = harvest_cell[m].sum(0); ty = hy.sum(); oy = np.argsort(hy)[::-1]
    n_big = int((hy > 0.05).sum())                             # "大机会"=该年可收割>0.05(单位仓)
    top3 = ", ".join(f"{slots[i]}({hy[i]/ty*100:.0f}%)" for i in oy[:3] if ty > 0)
    print(f"  {y}: 当年总可收割={ty:.2f}  大机会数(>0.05)={n_big:>2d}  top3: {top3}")

print("\n=== 我们抓得住吗:大机会在'当时'是否进 top-100 流动(=被我们的截顶选中) ===")
catch = 0; miss = 0
for c in order[:20]:
    act = np.where(vp[:, c] & (fpos[:, c] > 0))[0]
    if len(act) == 0:
        continue
    intop = 0
    for t in act:
        v = np.where(vp[t] & np.isfinite(funding[t]) & (funding[t] > 0))[0]
        if len(v) <= 100 or c in v[np.argsort(tdv[t, v])[::-1][:100]]:
            intop += 1
    frac = intop / len(act)
    tag = "✓抓住" if frac > 0.5 else "✗漏掉"
    if frac > 0.5:
        catch += 1
    else:
        miss += 1
    print(f"  {slots[c]:>16s}: 高funding期内进top-100流动的占比={frac:.0%}  {tag}")
print(f"\n  top-20 大机会:我们的流动截顶能抓住 {catch}/{catch+miss}")

print("\n判读:大机会若每年都有好几个(逐年top3轮动)、且多数能进top-100 → 抓大头是可重复的系统能力,非赌单个LEVER。")
