"""优化 #3:风险调整路由。MEASURE:对比 vanilla(只按 funding 价差选所) vs 风险调整
(按 funding价差 − λ·基差风险罚 选对)。HL 腿实测 σ20.7%/尾−124% 基差风险(E40)→ 罚它,
只在 funding 足够补偿时才上 HL。看 net/方差/尾部是否改善。
用 Bin/Byb/HL(有完整价格腿)验证原理,迁移到 HL/Gate/OKX 同理。
跑：PYTHONUTF8=1 .venv/Scripts/python.exe research/risk_routing.py
"""
import glob
import os
import time

import numpy as np

ANN = 8760
WINS = 0.10
# 各所基差风险等级(相对,bps/yr 罚项的尺度):CEX-CEX≈0,含 HL(DEX)高
RISK = {"Bin": 0.0, "Byb": 0.0, "HL": 1.0}    # HL 标记为高基差风险


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
    idx = {s: i for i, s in enumerate(slots)}; slotset = set(slots)
    f = np.full((T, len(slots)), np.nan); hit = []
    for path in glob.glob("data/raw/hl/funding/*.csv"):
        coin = os.path.splitext(os.path.basename(path))[0]
        slot = coin + "USDT" if coin + "USDT" in slotset else ("1000" + coin[1:] + "USDT" if coin.startswith("k") and "1000" + coin[1:] + "USDT" in slotset else None)
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
    pairs = [("Bin", "Byb"), ("Bin", "HL"), ("Byb", "HL")]

    def ret(col):
        r = np.full(T, np.nan); r[1:] = col[1:] / col[:-1] - 1
        return r

    # HL 数据窗口(只循环有 HL 的 bar,省 ~6x)
    hl_any = np.isfinite(fr["HL"]).any(1)
    t0 = max(1, int(np.argmax(hl_any)))

    def run(lam):
        """λ=基差风险罚系数(年化)。每bar在有效对里按 funding价差−λ·风险 选对;返回每币 net 序列。"""
        nets = []; hl_use = 0; tot = 0
        for s in hit:
            i = slots.index(s)
            rates = {v: fr[v][:, i] for v in ven}; rets = {v: ret(px[v][:, i]) for v in ven}
            ser = []
            for t in range(t0, T):
                best = None
                for a, b in pairs:
                    if not (np.isfinite(rates[a][t - 1]) and np.isfinite(rates[b][t - 1])
                            and np.isfinite(rets[a][t]) and np.isfinite(rets[b][t])):
                        continue
                    # long 低 funding,short 高
                    if rates[a][t - 1] <= rates[b][t - 1]:
                        L, S = a, b
                    else:
                        L, S = b, a
                    sp = rates[S][t - 1] - rates[L][t - 1]          # 滞后价差(决策用)
                    risk = max(RISK[a], RISK[b])                    # 对的基差风险(含HL=1)
                    score = sp - lam / ANN * risk
                    if best is None or score > best[0]:
                        best = (score, L, S, a, b)
                if best is None:
                    continue
                _, L, S, a, b = best
                fp = rates[S][t] - rates[L][t]                       # 实现 funding
                pl = max(-WINS, min(WINS, rets[L][t] - rets[S][t]))  # 价格腿
                ser.append(fp + pl)
                tot += 1; hl_use += ("HL" in (a, b))
            if len(ser) > 1000:
                nets.append(np.array(ser))
        return nets, hl_use / max(tot, 1)

    print(f"{'λ罚(年化)':>10s} {'net中位':>8s} {'net均值':>8s} {'净std中位':>9s} {'尾部(net5%分位)中位':>16s} {'HL使用率':>8s}")
    for lam in [0.0, 0.05, 0.10, 0.20, 0.40]:
        nets, hluse = run(lam)
        med = np.median([n.mean() * ANN for n in nets])
        mean = np.mean([n.mean() * ANN for n in nets])
        stdm = np.median([n.std() * np.sqrt(ANN) for n in nets])     # 年化波动
        tail = np.median([np.percentile(n, 5) for n in nets])        # 每bar 5%分位(尾部)
        print(f"{lam:>10.2f} {med:>+8.1%} {mean:>+8.1%} {stdm:>9.1%} {tail:>16.4f} {hluse:>8.0%}")
    print("\n判读:λ↑ → 罚 HL → HL 使用率↓;看 net 是否只小降而 std/尾部明显改善(=风险调整值得)。")


if __name__ == "__main__":
    main()
