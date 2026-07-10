"""改进方案 #1 试验:多所**最优路由** vs 两两对(Bin-Byb)。
思路:每币每 bar,在所有有数据的所里 **long 最低 funding 所 / short 最高 funding 所**(滞后一根用已知率决策=可交易),
收两所 funding 差;比只做 Bin-Byb 一对能多吃多少边(零借币,perp-perp)。
三所有 1 年历史:Binance / Bybit(xvenue_funding.npz)+ HL(data/raw/hl,60 币)。Gate 历史只 30 天→前向加。
单位:Bin/Byb 8h 率/8、HL 1h 率 → 每小时率;年化 ×8760。
跑：PYTHONUTF8=1 .venv/Scripts/python.exe research/xvenue_route.py
"""
import glob
import os
import time

import numpy as np

HL_DIR = "data/raw/hl/funding"
ANN = 8760


def to_ms(ds):
    import calendar
    return np.array([calendar.timegm(time.strptime(d, "%Y-%m-%d %H:%M:%S")) * 1000 for d in ds])   # UTC(勿用 mktime 本地时区→klines 错位)


def load_hl(slots, grid_ms):
    slotset = set(slots); idx = {s: i for i, s in enumerate(slots)}
    T = len(grid_ms); f = np.full((T, len(slots)), np.nan); hit = []
    for path in glob.glob(os.path.join(HL_DIR, "*.csv")):
        coin = os.path.splitext(os.path.basename(path))[0]
        slot = coin + "USDT" if coin + "USDT" in slotset else \
            ("1000" + coin[1:] + "USDT" if coin.startswith("k") and "1000" + coin[1:] + "USDT" in slotset else None)
        if slot is None:
            continue
        arr = np.loadtxt(path, delimiter=",", skiprows=1)
        if arr.ndim != 2 or len(arr) < 100:
            continue
        ts, fr = arr[:, 0].astype(np.int64), arr[:, 1]
        pos = np.searchsorted(ts, grid_ms, side="right") - 1
        v = pos >= 0; col = np.full(T, np.nan); col[v] = fr[pos[v]]
        f[:, idx[slot]] = col; hit.append(slot)
    return f, hit


def route_capture(rates, names, hyst=0.0):
    """rates: (T, V) 每小时率(NaN=该所无数据)。滞后一根决策:用 t-1 已知率选 long(min)/short(max),
    在 t 实现 (f_short − f_long)。hyst>0:只在新对滞后价差超过当前对 hyst(年化)才换所(降换手)。
    返回:年化收割 + 每bar序列 + 各所被选 long/short 次数 + 换所次数。"""
    T, V = rates.shape
    cap = np.zeros(T); valid = np.zeros(T, bool)
    long_cnt = np.zeros(V); short_cnt = np.zeros(V)
    cur_pair = None; switches = 0
    for t in range(1, T):
        prev = rates[t - 1]; now = rates[t]
        ok = np.isfinite(prev) & np.isfinite(now)
        if ok.sum() < 2:
            continue
        idxs = np.where(ok)[0]
        li = idxs[np.argmin(prev[idxs])]
        si = idxs[np.argmax(prev[idxs])]
        if li == si:
            continue
        # 滞后切换:若已有当前对且仍有效,只有新对价差超出 hyst 才换
        if hyst > 0 and cur_pair is not None:
            cl, cs = cur_pair
            if ok[cl] and ok[cs] and cl != cs:
                new_sp = prev[si] - prev[li]; cur_sp = prev[cs] - prev[cl]
                if new_sp < cur_sp + hyst / ANN:
                    li, si = cl, cs        # 保持当前对
        if cur_pair != (li, si):
            switches += 1; cur_pair = (li, si)
        cap[t] = now[si] - now[li]
        valid[t] = True
        long_cnt[li] += 1; short_cnt[si] += 1
    real = cap[valid].mean() * ANN if valid.any() else np.nan
    yrs = valid.sum() / ANN
    sw_yr = switches / yrs if yrs > 0 else 0
    return real, cap, valid, long_cnt, short_cnt, sw_yr


def fetch_intervals():
    """真实 funding 间隔(小时);台账 E38:~60% 币是 4h 不是 8h。未列=默认 8h。"""
    import json
    import urllib.request
    UA = {"User-Agent": "Mozilla/5.0"}
    g = lambda u: json.loads(urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=25).read())
    bi = {d["symbol"]: d.get("fundingIntervalHours", 8) for d in g("https://fapi.binance.com/fapi/v1/fundingInfo")}
    yi = {}; cur = ""
    while True:
        res = g("https://api.bybit.com/v5/market/instruments-info?category=linear&limit=1000" + (f"&cursor={cur}" if cur else ""))["result"]
        for s in res["list"]:
            if s.get("fundingInterval"):
                yi[s["symbol"]] = int(s["fundingInterval"]) / 60.0
        cur = res.get("nextPageCursor", "")
        if not cur:
            break
    return bi, yi


def main():
    xv = np.load("data/clean/xvenue_funding.npz", allow_pickle=True)
    slots = list(xv["slots"].astype(str)); dates = xv["dates"].astype(str)
    # 真实间隔归一(E38 纠正:旧统一 /8 对 4h 币砍半)
    bin_iv, byb_iv = fetch_intervals()
    biv = np.array([bin_iv.get(s, 8) for s in slots]); yiv = np.array([byb_iv.get(s, 8) for s in slots])
    bin_hr = xv["f_bin"] / biv[None, :]; byb_hr = xv["f_byb"] / yiv[None, :]
    grid = to_ms(dates)
    hl_f, hit = load_hl(np.array(slots), grid); hl_hr = hl_f   # 已是 1h 率
    print(f"三所对齐:{len(hit)} 币(Binance/Bybit/HL 都有 1yr)\n")

    venues = ["Bin", "Byb", "HL"]
    rows = []; route_series = []; bb_series = []; hyst_series = []
    tot_long = np.zeros(3); tot_short = np.zeros(3); sw3 = []; swH = []; sw2 = []
    HYST = 0.05    # 5%/yr 滞后:新对要比当前对多 5%/yr 才换所
    for s in hit:
        i = slots.index(s)
        R3 = np.column_stack([bin_hr[:, i], byb_hr[:, i], hl_hr[:, i]])
        R2 = np.column_stack([bin_hr[:, i], byb_hr[:, i]])
        r3, c3, v3, lc, sc, s3 = route_capture(R3, venues)
        rh, ch, vh, _, _, sh = route_capture(R3, venues, hyst=HYST)
        r2, c2, v2, _, _, s2 = route_capture(R2, ["Bin", "Byb"])
        if np.isnan(r3):
            continue
        rows.append((s, r2, r3, rh))
        route_series.append(c3); bb_series.append(c2); hyst_series.append(ch)
        tot_long += lc; tot_short += sc; sw3.append(s3); swH.append(sh); sw2.append(s2)
    rows.sort(key=lambda r: -r[2])
    print(f"{'币':14s} {'两所Bin-Byb':>11s} {'三所路由':>9s} {'三所+滞后':>9s} {'增量(滞后)':>10s}")
    for s, r2, r3, rh in rows[:16]:
        print(f"{s:14s} {r2:>+11.1%} {r3:>+9.1%} {rh:>+9.1%} {rh-r2:>+10.1%}")
    a2 = np.array([r[1] for r in rows]); a3 = np.array([r[2] for r in rows]); ah = np.array([r[3] for r in rows])
    print(f"\n中位: 两所 {np.nanmedian(a2):+.1%} → 三所路由 {np.nanmedian(a3):+.1%} → 三所+滞后 {np.nanmedian(ah):+.1%}")
    print(f"均值: 两所 {np.nanmean(a2):+.1%} → 三所路由 {np.nanmean(a3):+.1%} → 三所+滞后 {np.nanmean(ah):+.1%}")
    fin = np.isfinite(a2) & np.isfinite(ah)
    print(f"滞后路由更高的币占比 {np.mean(ah[fin] > a2[fin] + 1e-6):.0%}  (n={fin.sum()})")
    print(f"\n换所次数/年(换手代价): 两所 {np.mean(sw2):.0f}  三所路由 {np.mean(sw3):.0f}  三所+滞后(5%) {np.mean(swH):.0f}")
    print(f"  → 滞后把换手从 {np.mean(sw3):.0f}→{np.mean(swH):.0f}/yr,边保住 {np.nanmean(ah)/np.nanmean(a3):.0%}")

    print(f"\n各所被选为 long(收最低 funding)/ short(收最高)的次数占比:")
    L = tot_long / tot_long.sum(); S = tot_short / tot_short.sum()
    for k, v in enumerate(venues):
        print(f"  {v:4s}  long {L[k]:5.0%}  short {S[k]:5.0%}")

    # 多元化:路由流 vs 两所流(日聚合)
    T = len(dates); nd = T // 24
    def daily_sum(series):
        S = np.sum(series, axis=0)
        return S[:nd * 24].reshape(nd, 24).sum(1)
    dR = daily_sum(route_series); dB = daily_sum(bb_series)
    def shp(x):
        x = x[np.isfinite(x)]; return x.mean() / x.std() * np.sqrt(365) if x.std() > 0 else 0
    print(f"\n毛 funding 腿 日Sharpe: 两所 {shp(dB):.1f} → 三所路由 {shp(dR):.1f}")
    print("\n注:funding 腿毛收割(未扣价格腿 haircut/成本);HL 腿价格 haircut 无历史→前向测。")
    print("结论看:三所最优路由 vs 两所,中位/均值增量 + HL 被选占比(证明多所路由确实多吃边)。")


if __name__ == "__main__":
    main()
