"""严谨测肥尾净额:高 funding 币的"毛 funding − 价格腿 haircut"净额,是否真肥(还是被 haircut/方差吃掉)。
真实 1 年数据(Bin/Byb/HL funding+klines,UTC对齐,真实间隔)。每币 lagged 路由→ 毛 funding + 价格腿 → 净。
按毛 funding 大小分桶,报每桶:净额、retention、净日Sharpe。
诚实限制:今天的肥尾新币(SKY/ZORA)不在1年数据里→测"历史高funding币"作代理;结构可迁移。
跑：PYTHONUTF8=1 .venv/Scripts/python.exe research/fattail_net.py
"""
import glob
import os
import time

import numpy as np

ANN = 8760
WINS = 0.10


def to_ms(ds):
    import calendar
    return np.array([calendar.timegm(time.strptime(d, "%Y-%m-%d %H:%M:%S")) * 1000 for d in ds])


def fetch_intervals():
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


def align_csv(path, grid_ms, T, tol=3600_000):
    arr = np.loadtxt(path, delimiter=",", skiprows=1)
    if arr.ndim != 2 or len(arr) < 100:
        return None
    ts, px = arr[:, 0].astype(np.int64), arr[:, 1]
    pos = np.searchsorted(ts, grid_ms, side="right") - 1
    col = np.full(T, np.nan); v = pos >= 0; vi = np.where(v)[0]
    fresh = grid_ms[vi] - ts[pos[vi]] <= tol
    col[vi[fresh]] = px[pos[vi[fresh]]]
    return col


def load_hl_funding(slots, grid_ms, T):
    idx = {s: i for i, s in enumerate(slots)}; ss = set(slots)
    f = np.full((T, len(slots)), np.nan); hit = []
    for path in glob.glob("data/raw/hl/funding/*.csv"):
        coin = os.path.splitext(os.path.basename(path))[0]
        slot = coin + "USDT" if coin + "USDT" in ss else ("1000" + coin[1:] + "USDT" if coin.startswith("k") and "1000" + coin[1:] + "USDT" in ss else None)
        if slot is None:
            continue
        arr = np.loadtxt(path, delimiter=",", skiprows=1)
        if arr.ndim != 2 or len(arr) < 100:
            continue
        ts, fri = arr[:, 0].astype(np.int64), arr[:, 1]
        pos = np.searchsorted(ts, grid_ms, side="right") - 1
        v = pos >= 0; col = np.full(T, np.nan); col[v] = fri[pos[v]]
        f[:, idx[slot]] = col; hit.append(slot)
    return f, hit


def main():
    xv = np.load("data/clean/xvenue_funding.npz", allow_pickle=True)
    slots = list(xv["slots"].astype(str)); dates = xv["dates"].astype(str)
    grid = to_ms(dates); T = len(dates)
    bin_iv, byb_iv = fetch_intervals()
    biv = np.array([bin_iv.get(s, 8) for s in slots]); yiv = np.array([byb_iv.get(s, 8) for s in slots])
    fr = {"Bin": xv["f_bin"] / biv[None, :], "Byb": xv["f_byb"] / yiv[None, :]}
    hl_f, hit = load_hl_funding(np.array(slots), grid, T); fr["HL"] = hl_f
    tz = np.load("data/clean/crypto_tensor_60min_pit.npz", allow_pickle=True)
    bpx = tz["adj_close"].astype(float); bpx[bpx <= 0] = np.nan
    idx = {s: i for i, s in enumerate(slots)}
    ypx = np.full((T, len(slots)), np.nan); hpx = np.full((T, len(slots)), np.nan)
    for p in glob.glob("data/raw/bybit/kline/*.csv"):
        s = os.path.splitext(os.path.basename(p))[0]
        if s in idx:
            c = align_csv(p, grid, T)
            if c is not None:
                ypx[:, idx[s]] = c
    for p in glob.glob("data/raw/hl/kline/*.csv"):
        coin = os.path.splitext(os.path.basename(p))[0]
        s = coin + "USDT" if coin + "USDT" in idx else ("1000" + coin[1:] + "USDT" if coin.startswith("k") and "1000" + coin[1:] + "USDT" in idx else None)
        if s is not None:
            c = align_csv(p, grid, T)
            if c is not None:
                hpx[:, idx[s]] = c
    px = {"Bin": bpx, "Byb": ypx, "HL": hpx}
    ven = ["Bin", "Byb", "HL"]

    def ret(col):
        r = np.full(T, np.nan); r[1:] = col[1:] / col[:-1] - 1
        return r

    COST_HR = 7e-4 / (24 * 30)   # ~7bps maker 双腿,摊到约月换手(保守近似)
    rows = []
    for s in hit:
        i = slots.index(s)
        rates = {v: fr[v][:, i] for v in ven}; rets = {v: ret(px[v][:, i]) for v in ven}
        fser = []; nser = []
        for t in range(1, T):
            ok = [v for v in ven if np.isfinite(rates[v][t - 1]) and np.isfinite(rates[v][t]) and np.isfinite(rets[v][t])]
            if len(ok) < 2:
                continue
            L = min(ok, key=lambda v: rates[v][t - 1]); S = max(ok, key=lambda v: rates[v][t - 1])
            if L == S:
                continue
            fp = rates[S][t] - rates[L][t]
            pl = max(-WINS, min(WINS, rets[L][t] - rets[S][t]))
            fser.append(fp); nser.append(fp + pl)
        if len(nser) < 1500:
            continue
        f_arr = np.array(fser); n_arr = np.array(nser)
        gross = f_arr.mean() * ANN
        net = n_arr.mean() * ANN
        # 净日Sharpe(每日聚合粗略:按 24 切块)
        nd = len(n_arr) // 24
        dd = n_arr[:nd * 24].reshape(nd, 24).sum(1) if nd > 1 else n_arr
        nsh = dd.mean() / dd.std() * np.sqrt(365) if dd.std() > 0 else 0
        rows.append((s, gross, net, nsh))
    rows.sort(key=lambda r: -r[1])
    arr = np.array([(g, n, sh) for _, g, n, sh in rows])
    g, n, sh = arr[:, 0], arr[:, 1], arr[:, 2]

    print(f"真实净额测试(Bin/Byb/HL 1yr lagged 路由,n={len(rows)} 币)")
    print(f"全体: 毛 funding 中位 {np.median(g):+.1%} / 净额(扣价格腿) 中位 {np.median(n):+.1%} / 净Sharpe 中位 {np.median(sh):.1f}")
    print(f"\n按毛 funding 分桶(肥尾在最高桶):")
    print(f"  {'桶(毛funding)':>16s} {'n':>3s} {'毛中位':>8s} {'净中位':>8s} {'净/毛':>7s} {'净Sharpe中位':>10s} {'净>0占比':>8s}")
    qs = np.quantile(g, [0, 0.5, 0.75, 0.9, 1.0])
    labels = ["低(<50%)", "中(50-75%)", "高(75-90%)", "肥尾(top10%)"]
    for k in range(4):
        lo, hi = qs[k], qs[k + 1]
        m = (g >= lo) & (g <= hi) if k == 3 else (g >= lo) & (g < hi)
        if m.sum() < 2:
            continue
        gm, nm, shm = np.median(g[m]), np.median(n[m]), np.median(sh[m])
        print(f"  {labels[k]:>16s} {m.sum():>3d} {gm:>+8.1%} {nm:>+8.1%} {nm/gm if gm else 0:>7.0%} {shm:>10.1f} {np.mean(n[m]>0):>8.0%}")
    print(f"\n肥尾(毛>90分位)币例:")
    for s, gg, nn, ss in rows[:10]:
        print(f"  {s:14s} 毛 {gg:+7.1%}  净 {nn:+7.1%}  净Sharpe {ss:.1f}")
    print("\n判读:肥尾桶 净中位 vs 中桶——若净仍明显更高 = 肥尾真值得重仓;净/毛 retention 看价格腿吃了多少;")
    print("净Sharpe 看风险调整后肥尾是否仍优(funding近确定虚高,看相对不看绝对)。限制:1yr数据非超薄新币,结构代理。")


if __name__ == "__main__":
    main()
