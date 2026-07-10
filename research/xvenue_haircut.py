"""测多所路由的**真实价格腿 haircut**(回测最大未知数)。MEASURE 不估计:
用三所真实永续价格(Binance adj_close / Bybit klines / HL klines)算每所每bar收益,
重放路由(滞后选所 long 最低 funding / short 最高),同时累计 **funding 腿 + 价格腿(haircut)**,出**净额**。
价格腿 = retL − retS(both 盯现货 → 理论≈0,实测是跨所基差噪声/漂移)。
跑：PYTHONUTF8=1 .venv/Scripts/python.exe research/xvenue_haircut.py
"""
import glob
import os
import time

import numpy as np

ANN = 8760
WINS = 0.10


def to_ms(ds):
    import calendar
    return np.array([calendar.timegm(time.strptime(d, "%Y-%m-%d %H:%M:%S")) * 1000 for d in ds])   # UTC(网格/klines 都 UTC;勿用 mktime 本地时区→错位)


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


def load_hl_funding(slots, grid_ms, T):
    """HL funding(1h 率)对齐到 grid。"""
    slotset = set(slots); idx = {s: i for i, s in enumerate(slots)}
    f = np.full((T, len(slots)), np.nan); hit = []
    for path in glob.glob("data/raw/hl/funding/*.csv"):
        coin = os.path.splitext(os.path.basename(path))[0]
        slot = coin + "USDT" if coin + "USDT" in slotset else \
            ("1000" + coin[1:] + "USDT" if coin.startswith("k") and "1000" + coin[1:] + "USDT" in slotset else None)
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


def align_csv(path, grid_ms, T, tol_ms=3600_000):
    """csv(ts_ms,close) → 对齐到 grid(**tolerance=1h:超过 1h 无数据→NaN,不陈旧填充**,防假收益)。"""
    arr = np.loadtxt(path, delimiter=",", skiprows=1)
    if arr.ndim != 2 or len(arr) < 100:
        return None
    ts, px = arr[:, 0].astype(np.int64), arr[:, 1]
    pos = np.searchsorted(ts, grid_ms, side="right") - 1
    col = np.full(T, np.nan)
    v = pos >= 0
    vi = np.where(v)[0]
    fresh = grid_ms[vi] - ts[pos[vi]] <= tol_ms       # 只在 1h 内有真实价才用
    col[vi[fresh]] = px[pos[vi[fresh]]]
    return col


def load_prices(slots, grid_ms, T):
    """三所价格数组(T×N)。Binance=tensor adj_close;Bybit/HL=klines 对齐。"""
    tz = np.load("data/clean/crypto_tensor_60min_pit.npz", allow_pickle=True)
    bin_px = tz["adj_close"].astype(float)              # 已对齐(网格同)
    bin_px[bin_px <= 0] = np.nan
    idx = {s: i for i, s in enumerate(slots)}
    byb_px = np.full((T, len(slots)), np.nan); hl_px = np.full((T, len(slots)), np.nan)
    for path in glob.glob("data/raw/bybit/kline/*.csv"):
        s = os.path.splitext(os.path.basename(path))[0]
        if s in idx:
            c = align_csv(path, grid_ms, T)
            if c is not None:
                byb_px[:, idx[s]] = c
    for path in glob.glob("data/raw/hl/kline/*.csv"):
        coin = os.path.splitext(os.path.basename(path))[0]
        s = coin + "USDT" if coin + "USDT" in idx else \
            ("1000" + coin[1:] + "USDT" if coin.startswith("k") and "1000" + coin[1:] + "USDT" in idx else None)
        if s is not None:
            c = align_csv(path, grid_ms, T)
            if c is not None:
                hl_px[:, idx[s]] = c
    return bin_px, byb_px, hl_px


def main():
    xv = np.load("data/clean/xvenue_funding.npz", allow_pickle=True)
    slots = list(xv["slots"].astype(str)); dates = xv["dates"].astype(str)
    grid = to_ms(dates); T = len(dates)
    bin_iv, byb_iv = fetch_intervals()
    biv = np.array([bin_iv.get(s, 8) for s in slots]); yiv = np.array([byb_iv.get(s, 8) for s in slots])
    fr = {"Bin": xv["f_bin"] / biv[None, :], "Byb": xv["f_byb"] / yiv[None, :]}
    hl_f, hit = load_hl_funding(np.array(slots), grid, T); fr["HL"] = hl_f
    bin_px, byb_px, hl_px = load_prices(slots, grid, T)
    px = {"Bin": bin_px, "Byb": byb_px, "HL": hl_px}
    venues = ["Bin", "Byb", "HL"]

    def ret(pxcol):
        r = np.full(T, np.nan); r[1:] = pxcol[1:] / pxcol[:-1] - 1
        return r

    rows = []
    hl_fund = []; hl_price = []     # HL 参与的腿:funding vs 价格 haircut
    for s in hit:
        i = slots.index(s)
        rates = {v: fr[v][:, i] for v in venues}
        rets = {v: ret(px[v][:, i]) for v in venues}
        f_sum = p_sum = 0.0; n = 0; hlf = hlp = 0.0; hln = 0
        for t in range(1, T):
            prev = {v: rates[v][t - 1] for v in venues}
            now = {v: rates[v][t] for v in venues}
            ok = [v for v in venues if np.isfinite(prev[v]) and np.isfinite(now[v])
                  and np.isfinite(rets[v][t])]
            if len(ok) < 2:
                continue
            L = min(ok, key=lambda v: prev[v]); S = max(ok, key=lambda v: prev[v])
            if L == S:
                continue
            fp = now[S] - now[L]
            pl = max(-WINS, min(WINS, rets[L][t] - rets[S][t]))
            f_sum += fp; p_sum += pl; n += 1
            if "HL" in (L, S):
                hlf += fp; hlp += pl; hln += 1
        if n < 1000:
            continue
        fund_a = f_sum / n * ANN; price_a = p_sum / n * ANN
        rows.append((s, fund_a, price_a, fund_a + price_a))
        if hln > 200:
            hl_fund.append(hlf / hln * ANN); hl_price.append(hlp / hln * ANN)
    rows.sort(key=lambda r: -r[3])
    print(f"多所路由 真实净额(funding 腿 + 价格腿 haircut),n={len(rows)} 币\n")
    print(f"{'币':14s} {'funding':>9s} {'价格腿':>9s} {'净额':>9s}")
    for s, f, p, net in rows[:16]:
        print(f"{s:14s} {f:>+9.1%} {p:>+9.1%} {net:>+9.1%}")
    F = np.array([r[1] for r in rows]); P = np.array([r[2] for r in rows]); NET = np.array([r[3] for r in rows])
    print(f"\n中位: funding {np.median(F):+.1%}  价格腿haircut {np.median(P):+.1%}  净额 {np.median(NET):+.1%}")
    print(f"均值: funding {np.mean(F):+.1%}  价格腿haircut {np.mean(P):+.1%}  净额 {np.mean(NET):+.1%}")
    print(f"haircut/funding 中位比例 {np.median(P)/np.median(F):+.0%}  净额为正币占比 {np.mean(NET>0):.0%}")
    if hl_fund:
        print(f"\n仅 HL 参与的腿(测 HL 价格腿 haircut,n={len(hl_fund)}):")
        print(f"  funding 中位 {np.median(hl_fund):+.1%}  HL价格腿haircut 中位 {np.median(hl_price):+.1%}  比例 {np.median(hl_price)/np.median(hl_fund):+.0%}")
    print("\n判读:价格腿 haircut 越接近 0=两所永续盯同现货、基差稳;HL 是 DEX/不同指数→看它 haircut 是否明显大于 Bin-Byb 的 −0.3%。")


if __name__ == "__main__":
    main()
