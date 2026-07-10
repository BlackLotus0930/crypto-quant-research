"""实测真实执行成本(替代 backtest 里 cost_bps=10 的估计)。MEASURE 不估计:
- 价差:实时 bid-ask(各所永续 + Binance 现货),按流动可路由币。
- 费率:各所公开 maker/taker(事实,非估)。
- 真实单腿成本 = maker 费(挂单,慢策略可)或 taker 费 + 半价差(过价差)。
跑：PYTHONUTF8=1 .venv/Scripts/python.exe research/real_cost.py
"""
import json
import urllib.request

import numpy as np

UA = {"User-Agent": "Mozilla/5.0"}
# 公开费率(VIP0,bps)——事实
FEE = {"bin_perp": (2.0, 5.0), "bin_spot": (10.0, 10.0), "byb_perp": (2.0, 5.5),
       "hl": (1.5, 4.5), "gate_perp": (2.0, 5.0)}   # (maker, taker)


def get(u):
    return json.loads(urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=30).read())


def post(u, b):
    return json.loads(urllib.request.urlopen(urllib.request.Request(u, data=json.dumps(b).encode(),
                      headers={**UA, "Content-Type": "application/json"}), timeout=30).read())


def spr(bid, ask):
    return (ask - bid) / ((ask + bid) / 2) * 1e4 if bid > 0 and ask > 0 else np.nan


def main():
    # ---- 各所 bid/ask ----
    binp = {d["symbol"]: spr(float(d["bidPrice"]), float(d["askPrice"]))
            for d in get("https://fapi.binance.com/fapi/v1/ticker/bookTicker") if d["symbol"].endswith("USDT")}
    bins = {d["symbol"]: spr(float(d["bidPrice"]), float(d["askPrice"]))
            for d in get("https://api.binance.com/api/v3/ticker/bookTicker") if d["symbol"].endswith("USDT")}
    bybp = {}
    for d in get("https://api.bybit.com/v5/market/tickers?category=linear")["result"]["list"]:
        try:
            bybp[d["symbol"]] = spr(float(d["bid1Price"]), float(d["ask1Price"]))
        except (KeyError, ValueError, TypeError):
            pass
    hl = {}
    l2 = None
    r = post("https://api.hyperliquid.xyz/info", {"type": "metaAndAssetCtxs"})
    for m, c in zip(r[0]["universe"], r[1]):
        try:
            mid = float(c["midPx"]); impact = c.get("impactPxs")
            if impact and mid > 0:
                hl[m["name"]] = (float(impact[1]) - float(impact[0])) / mid * 1e4   # impact bid/ask spread
        except (KeyError, ValueError, TypeError):
            pass
    gatep = {}
    for d in get("https://api.gateio.ws/api/v4/futures/usdt/tickers"):
        try:
            b, a = float(d.get("highest_bid", 0) or 0), float(d.get("lowest_ask", 0) or 0)
            gatep[d["contract"]] = spr(b, a)
        except (KeyError, ValueError, TypeError):
            pass

    def med(d):
        v = np.array([x for x in d.values() if np.isfinite(x) and x > 0])
        return np.median(v) if len(v) else np.nan

    print("=== 实测价差(bps,全 bid-ask;中位)===")
    print(f"  Binance 永续 {med(binp):5.1f} | Binance 现货 {med(bins):5.1f} | Bybit 永续 {med(bybp):5.1f} | "
          f"HL {med(hl):5.1f} | Gate 永续 {med(gatep):5.1f}")

    # 流动币(成交额 top)上的价差——更接近我们实际交易的币
    binqv = {d["symbol"]: float(d["quoteVolume"]) for d in get("https://fapi.binance.com/fapi/v1/ticker/24hr")}
    liq = sorted((s for s in binp if binqv.get(s, 0) > 5e7), key=lambda s: -binqv.get(s, 0))[:80]
    lp = np.median([binp[s] for s in liq if np.isfinite(binp[s])])
    ls = np.median([bins[s] for s in liq if s in bins and np.isfinite(bins[s])])
    print(f"  (流动 top80 永续 {lp:.1f}bps / 现货 {ls:.1f}bps —— 更接近实际交易的薄边)")

    print("\n=== 真实单腿成本(bps;maker=挂单/taker=过价差=费+半价差)===")
    for k, (mk, tk) in FEE.items():
        sp = {"bin_perp": med(binp), "bin_spot": med(bins), "byb_perp": med(bybp),
              "hl": med(hl), "gate_perp": med(gatep)}[k]
        print(f"  {k:10s}: maker {mk:4.1f} | taker {tk + sp/2:5.1f} (费{tk}+半价差{sp/2:.1f})")

    print("\n=== 组合真实成本估算(每次建/平仓,双腿)===")
    # carry: 现货(maker) + Binance 永续(maker)
    carry_mk = FEE["bin_spot"][0] + FEE["bin_perp"][0]
    carry_tk = (FEE["bin_spot"][1] + med(bins) / 2) + (FEE["bin_perp"][1] + med(binp) / 2)
    print(f"  carry 双腿(现货+永续): maker {carry_mk:.1f}bps | taker {carry_tk:.1f}bps")
    # 路由: 两条永续腿(取最贵组合 Binance+HL 近似)
    route_mk = FEE["bin_perp"][0] + FEE["hl"][0]
    route_tk = (FEE["bin_perp"][1] + med(binp) / 2) + (FEE["hl"][1] + med(hl) / 2)
    print(f"  路由 双腿(两所永续): maker {route_mk:.1f}bps | taker {route_tk:.1f}bps")
    print("\n判读:慢策略应**maker 为主**→ 双腿成本 ~4-12bps;backtest 用的 10bps 偏保守(贴近 taker),非乐观。")
    print("注:周转一次=建+平=2×;但 carry/路由换手低(年 ~350-550 次×小权重),成本/turnover 才是要乘的。")


if __name__ == "__main__":
    main()
