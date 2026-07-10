"""HL 验证:把 HL 历史 funding 对齐到 xvenue 网格,测三所两两跨所收割(滞后选所,可交易)+ 流间相关性(√N)。
思路(台账 E33→HL 前沿):HL 是 DEX、少被套利 → funding 分散最肥;问它能否当第 3 所、加多少独立流。
单位:Binance/Bybit funding 是 8h 率(网格上前向填的逐小时值)→ 每小时收益 = f/8;HL 是 1h 率 → 每小时 = f。
跨所每小时净 = |a_hr − b_hr|;**滞后一根用已知价差符号选所**(可交易,非完美选所),年化 ×8760。
跑：PYTHONUTF8=1 .venv/Scripts/python.exe research/xvenue_hl.py
"""
import glob
import os
import time

import numpy as np

HL_DIR = "data/raw/hl/funding"
ANN = 8760


def to_ms(date_strs):
    import calendar
    return np.array([calendar.timegm(time.strptime(d, "%Y-%m-%d %H:%M:%S")) * 1000 for d in date_strs])   # UTC(勿用 mktime 本地时区)


def load_hl_aligned(slots, grid_ms):
    """每个 HL csv → 映射回 slot → 前向填(因果,≤grid 时间的最近已知率)到网格。返回 f_hl(T×N), 命中集合。"""
    slotset = set(slots); slot_idx = {s: i for i, s in enumerate(slots)}
    T, N = len(grid_ms), len(slots)
    f_hl = np.full((T, N), np.nan)
    hit = []
    for path in glob.glob(os.path.join(HL_DIR, "*.csv")):
        coin = os.path.splitext(os.path.basename(path))[0]
        slot = None
        if coin + "USDT" in slotset:
            slot = coin + "USDT"
        elif coin.startswith("k") and "1000" + coin[1:] + "USDT" in slotset:
            slot = "1000" + coin[1:] + "USDT"
        if slot is None:
            continue
        arr = np.loadtxt(path, delimiter=",", skiprows=1)
        if arr.ndim != 2 or len(arr) < 100:
            continue
        ts, fr = arr[:, 0].astype(np.int64), arr[:, 1]
        # 前向填:每个 grid 时间取 ≤ 它的最近 HL 率
        pos = np.searchsorted(ts, grid_ms, side="right") - 1
        valid = pos >= 0
        col = np.full(T, np.nan); col[valid] = fr[pos[valid]]
        # 仅在 HL 数据覆盖期内有效(grid 早于首个 HL ts 的不填)
        f_hl[:, slot_idx[slot]] = col
        hit.append(slot)
    return f_hl, hit


def lagged_capture(a_hr, b_hr):
    """两腿每小时率 → 滞后选所可交易收割年化 + 完美上界 + 每bar序列(供相关)。仅两腿都有效的 bar。"""
    m = np.isfinite(a_hr) & np.isfinite(b_hr)
    if m.sum() < 500:
        return None
    sp = np.where(m, a_hr - b_hr, np.nan)
    siglag = np.full_like(sp, np.nan)
    # 逐bar滞后符号(用上一根有效价差)
    last = 0.0
    series = np.zeros(len(sp))
    for t in range(len(sp)):
        if not np.isfinite(sp[t]):
            continue
        series[t] = np.sign(last) * sp[t]
        last = sp[t]
    real = np.nanmean([series[t] for t in range(len(sp)) if np.isfinite(sp[t])]) * ANN
    ub = np.nanmean(np.abs(sp[m])) * ANN
    return real, ub, series, m


def main():
    xv = np.load("data/clean/xvenue_funding.npz", allow_pickle=True)
    slots = xv["slots"].astype(str); f_bin = xv["f_bin"]; f_byb = xv["f_byb"]; dates = xv["dates"].astype(str)
    grid_ms = to_ms(dates)
    f_hl, hit = load_hl_aligned(slots, grid_ms)
    print(f"HL 对齐:命中 {len(hit)} 币到 xvenue 网格(窗口与本地 Bin/Byb 重叠 ~1yr)")

    bin_hr = f_bin / 8.0; byb_hr = f_byb / 8.0; hl_hr = f_hl   # 每小时率
    T = len(dates); nd = T // 24
    rows = []
    # 只在"三所都有"的币上建可比的三条流(同币集、同 bar)→ 诚实的多元化度量
    daily = {"BB": [], "BH": [], "YH": []}
    for s in hit:
        i = list(slots).index(s)
        rBB = lagged_capture(bin_hr[:, i], byb_hr[:, i])
        rBH = lagged_capture(bin_hr[:, i], hl_hr[:, i])
        rYH = lagged_capture(byb_hr[:, i], hl_hr[:, i])
        if rBH is None:
            continue
        rows.append((s, rBB[0] if rBB else np.nan, rBH[0], rBH[1], rYH[0] if rYH else np.nan))
        if rBB and rYH:   # 三所齐全才进多元化分析(同币集)
            daily["BB"].append(rBB[2][:nd * 24].reshape(nd, 24).sum(1))
            daily["BH"].append(rBH[2][:nd * 24].reshape(nd, 24).sum(1))
            daily["YH"].append(rYH[2][:nd * 24].reshape(nd, 24).sum(1))
    rows.sort(key=lambda r: -r[2])
    print(f"\n{'币':14s} {'Bin-Byb':>9s} {'Bin-HL':>9s} {'BinHL上界':>9s} {'Byb-HL':>9s}  (年化,可交易滞后选所)")
    for s, bb, bh, ub, yh in rows[:18]:
        print(f"{s:14s} {bb:>+9.1%} {bh:>+9.1%} {ub:>9.1%} {yh:>+9.1%}")
    arr = np.array([(r[1], r[2], r[4]) for r in rows], float)
    print(f"\n中位(可交易): Bin-Byb {np.nanmedian(arr[:,0]):+.1%}  Bin-HL {np.nanmedian(arr[:,1]):+.1%}  Byb-HL {np.nanmedian(arr[:,2]):+.1%}")
    print(f"均值(可交易): Bin-Byb {np.nanmean(arr[:,0]):+.1%}  Bin-HL {np.nanmean(arr[:,1]):+.1%}  Byb-HL {np.nanmean(arr[:,2]):+.1%}")
    print(f"Bin-HL 正收益币占比 {np.mean(arr[:,1]>0):.0%}  (n={len(rows)})")

    # === 流间相关(同币集、日聚合、vol 归一)===
    print("\n=== 流间相关(同币集 n=%d,日聚合;低=独立流,√N 提 Sharpe)===" % len(daily["BB"]))
    streams = {k: np.sum(v, axis=0) for k, v in daily.items() if v}   # 等权汇总
    def vn(x):
        s = x.std(); return (x - x.mean() * 0) / s if s > 0 else x   # 仅归一 vol(保留均值符号用 raw 算 Sharpe)
    keys = ["BB", "BH", "YH"]; labels = {"BB": "Bin-Byb", "BH": "Bin-HL", "YH": "Byb-HL"}
    if all(k in streams for k in keys):
        import itertools
        cors = {}
        for a, b in itertools.combinations(keys, 2):
            c = np.corrcoef(streams[a], streams[b])[0, 1]; cors[(a, b)] = c
            print(f"  {labels[a]:8s} ↔ {labels[b]:8s} corr {c:+.2f}")
        rho = np.mean(list(cors.values()))
        N = 3
        ratio = np.sqrt(N) / np.sqrt(1 + (N - 1) * rho)
        print(f"\n  三流平均相关 ρ̄={rho:+.2f} → 解析多元化比例 组合/单 = √N/√(1+(N-1)ρ̄) = {ratio:.2f}×")
        print(f"  (对比理想 ρ=0 时 {np.sqrt(N):.2f}×;ρ={rho:.2f} 因共腿+同 HL 错位 → 提升打折)")
        def shp(x):
            return x.mean() / x.std() * np.sqrt(365) if x.std() > 0 else 0
        # vol-归一等权组合(等风险)
        comb = sum(streams[k] / streams[k].std() for k in keys)
        print(f"  毛 funding 腿 日Sharpe: " + "  ".join(f"{labels[k]} {shp(streams[k]):.1f}" for k in keys) + f"  | 等风险三流 {shp(comb):.1f}")
    print("\n注:funding 腿毛收割(未扣价格腿 haircut/双腿成本);**绝对 Sharpe 偏乐观,提升比例+相关性是稳健结论**。")
    print("解读:HL 收割更肥(均值 2-3×)=边更大;但跨所三对彼此相关(共腿/同 HL)→ HL 主要加'边的大小'+一条部分独立流,非干净 √N。")


if __name__ == "__main__":
    main()
