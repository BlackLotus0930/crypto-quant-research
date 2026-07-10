"""实盘就绪体检:量我们当前书里这些币的真实执行摩擦(价差/深度)+ 借券,评估"回测的边实盘剩多少"。
关键张力:funding 肥的币常不流动 → 价差宽/深度薄/借券贵。把它对着仓位权重看。
全只读、不下单。跑：PYTHONUTF8=1 .venv/Scripts/python.exe live_readiness.py --capital 5000
"""
import argparse
import json
import urllib.request

import numpy as np

FAPI = "https://fapi.binance.com"; SAPI = "https://api.binance.com"; BYBIT = "https://api.bybit.com"
UA = {"User-Agent": "Mozilla/5.0"}


def get(u):
    return json.loads(urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=30).read())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--capital", type=float, default=5000)
    ap.add_argument("--state", default="data/paper/state.json"); a = ap.parse_args()
    st = json.load(open(a.state)); carry = st["carry"]["cur"]; xv = st["xv"]["cur"]
    binf = {d["symbol"]: float(d["lastFundingRate"]) for d in get(f"{FAPI}/fapi/v1/premiumIndex")}
    bp = {d["symbol"]: (float(d["bidPrice"]), float(d["askPrice"])) for d in get(f"{FAPI}/fapi/v1/ticker/bookTicker") if float(d["bidPrice"]) > 0}
    bs = {d["symbol"]: (float(d["bidPrice"]), float(d["askPrice"])) for d in get(f"{SAPI}/api/v3/ticker/bookTicker") if float(d["bidPrice"]) > 0}
    byb = {}
    for d in get(f"{BYBIT}/v5/market/tickers?category=linear")["result"]["list"]:
        try:
            b, ak = float(d["bid1Price"]), float(d["ask1Price"])
            if b > 0:
                byb[d["symbol"]] = (b, ak)
        except (KeyError, ValueError, TypeError):
            pass

    def spr(bidask):
        b, ak = bidask; return (ak - b) / ((ak + b) / 2) * 1e4   # 半价差? 这里是全价差 bps

    # --- carry 腿价差(Binance 永续 + 现货):权重加权 ---
    print("=== 执行价差(bps,全 bid-ask;taker 过价差≈付半价差,maker 赚半价差)===")
    rows = []
    for s, w in carry.items():
        if s in bp and s in bs:
            rows.append((s, abs(w), spr(bp[s]), spr(bs[s]), binf.get(s, 0) * 3 * 365))
    rows.sort(key=lambda r: -r[1])
    ws = np.array([r[1] for r in rows]); pp = np.array([r[2] for r in rows]); sps = np.array([r[3] for r in rows])
    wavg_perp = (ws * pp).sum() / ws.sum(); wavg_spot = (ws * sps).sum() / ws.sum()
    print(f"  carry 腿 ({len(rows)}币): 永续价差 权重加权 {wavg_perp:.0f}bps(中位{np.median(pp):.0f}) | 现货 {wavg_spot:.0f}bps(中位{np.median(sps):.0f})")
    print(f"    → 一次双腿建仓 taker 估付 ~{(wavg_perp+wavg_spot)/2:.0f}bps;maker 若成交则赚回")
    # 价差 vs funding(肥币是不是更宽)
    fa = np.array([abs(r[4]) for r in rows])
    if len(rows) > 5:
        print(f"    价差(永续) vs |funding年化| 相关={np.corrcoef(pp, fa)[0,1]:+.2f}  (正=肥币更宽=edge被friction吃)")

    rows2 = [(s, abs(w), spr(bp[s]), spr(byb[s])) for s, w in xv.items() if s in bp and s in byb]
    if rows2:
        w2 = np.array([r[1] for r in rows2]); bpp = np.array([r[2] for r in rows2]); byy = np.array([r[3] for r in rows2])
        print(f"  跨所腿 ({len(rows2)}币): Binance永续 {(w2*bpp).sum()/w2.sum():.0f}bps | Bybit永续 {(w2*byy).sum()/w2.sum():.0f}bps(中位{np.median(byy):.0f})")

    # --- 深度:top 权重 carry 币,看 $size 内能否成交 ---
    print(f"\n=== 订单簿深度(@ 资本 ${a.capital:,.0f},每币名义=权重×资本/2)===")
    print(f"  {'币':>12s} {'仓位$':>7s} {'永续买1深10bps$':>14s} {'够吗':>5s}")
    Cc = a.capital * 0.5
    thin = 0
    for s, w in sorted(carry.items(), key=lambda kv: -abs(kv[1]))[:8]:
        need = abs(w) * Cc
        try:
            dep = get(f"{FAPI}/fapi/v1/depth?symbol={s}&limit=20")
            mid = (float(dep["bids"][0][0]) + float(dep["asks"][0][0])) / 2
            within = sum(float(p) * float(q) for p, q in dep["asks"] if float(p) <= mid * 1.001)  # 卖盘 10bps 内 $
            ok = "✓" if within > need * 3 else "⚠️薄"
            thin += within <= need * 3
            print(f"  {s:>12s} {need:>7,.0f} {within:>14,.0f} {ok:>5s}")
        except Exception as e:
            print(f"  {s:>12s} {need:>7,.0f}  深度取数失败 {type(e).__name__}")
    print(f"  → 我们 size 极小($数十-数百/币);{'有薄的(看上)' if thin else '深度充足'}")

    # --- 借券(负侧):试 Binance 公共 margin 利率 ---
    print("\n=== 借券(负侧空现货依赖)===")
    try:
        cm = get(f"{SAPI}/sapi/v1/margin/crossMarginData")  # 可能需 auth
        print(f"  公共 margin 数据可取:{len(cm)} 项(含借币利率)")
    except Exception as e:
        print(f"  Binance 公共 margin 利率端点取不到({type(e).__name__})→ **需账户(只读)在实盘前手查**:")
        print("    每个负侧币:① 是否可借(cross/isolated margin 资产)② 借币利率(年化)③ 借币上限")
        negc = [s for s, w in carry.items() if w < 0]
        print(f"    当前负侧 {len(negc)} 币需查,前几:{negc[:6]}")
    print("\n判读:价差是 taker 的硬成本(maker 能赚回但要挂得上);肥币若价差宽=edge 被吃;借券利率直接减负侧收益。")


if __name__ == "__main__":
    main()
