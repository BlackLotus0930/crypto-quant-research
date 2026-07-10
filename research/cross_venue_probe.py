"""① 跨所 funding 探查:同一币在 Binance/Bybit/Hyperliquid 的 funding 差多少?
差大=可做跨所套利(多低 funding 所永续+空高 funding 所永续,同币→市场&价格中性,且免借现货)。
差≈0=已被套利掉。看当前快照的 spread 分布。
跑：python cross_venue_probe.py
"""
import json
import urllib.request

import numpy as np

UA = {"User-Agent": "Mozilla/5.0"}
A = 3 * 365     # 8h → 年化
A1 = 24 * 365   # 1h → 年化


def get(url, data=None, hdr=None):
    h = dict(UA); h.update(hdr or {})
    req = urllib.request.Request(url, data=data, headers=h)
    return json.loads(urllib.request.urlopen(req, timeout=30).read())


def binance():
    out = {}
    for d in get("https://fapi.binance.com/fapi/v1/premiumIndex"):
        s = d["symbol"]
        if s.endswith("USDT"):
            out[s[:-4]] = float(d["lastFundingRate"]) * A          # 8h→年化
    return out


def bybit():
    out = {}
    r = get("https://api.bybit.com/v5/market/tickers?category=linear")
    for d in r["result"]["list"]:
        s = d["symbol"]
        if s.endswith("USDT") and d.get("fundingRate") not in (None, ""):
            out[s[:-4]] = float(d["fundingRate"]) * A              # 8h→年化
    return out


def hyperliquid():
    out = {}
    r = get("https://api.hyperliquid.xyz/info",
            data=json.dumps({"type": "metaAndAssetCtxs"}).encode(),
            hdr={"Content-Type": "application/json"})
    meta, ctxs = r[0], r[1]
    for u, c in zip(meta["universe"], ctxs):
        if c.get("funding") is not None:
            out[u["name"]] = float(c["funding"]) * A1              # HL 1h→年化
    return out


venues = {}
for name, fn in [("Binance", binance), ("Bybit", bybit), ("Hyperliquid", hyperliquid)]:
    try:
        venues[name] = fn()
        print(f"{name}: {len(venues[name])} 个 USDT 永续 funding(年化)")
    except Exception as e:
        print(f"{name}: 取数失败 {type(e).__name__}: {e}")

names = list(venues)
common = set.intersection(*[set(v) for v in venues.values()]) if len(venues) >= 2 else set()
print(f"\n{len(names)} 所交集 {len(common)} 个币\n")

rows = []
for coin in common:
    vals = {n: venues[n][coin] for n in names}
    arr = np.array(list(vals.values()))
    spread = arr.max() - arr.min()                                # 跨所 funding 价差(年化)= 可收
    lo = min(vals, key=vals.get); hi = max(vals, key=vals.get)
    rows.append((coin, spread, lo, vals[lo], hi, vals[hi]))
rows.sort(key=lambda r: -r[1])
sp = np.array([r[1] for r in rows])
print(f"跨所 funding 价差(年化)分布: 中位 {np.median(sp):.1%}  均值 {sp.mean():.1%}  "
      f">10%:{(sp>0.10).mean()*100:.0f}%  >30%:{(sp>0.30).mean()*100:.0f}%  >50%:{(sp>0.50).mean()*100:.0f}%")
print(f"\n价差最大的 12 个币(多 {names}[低] / 空 [高]):")
print(f"{'币':>10s} {'跨所价差':>9s}  低所→高所(年化 funding)")
for coin, spread, lo, lov, hi, hiv in rows[:12]:
    print(f"{coin:>10s} {spread:>8.1%}  {lo}({lov:+.0%}) → {hi}({hiv:+.0%})")
print("\n判读:中位/均值价差大(如 >10-20%)且分布厚 = 跨所套利肉多、近独立流成立;价差≈0 = 已被套平。")
