"""多所 funding 抓取 + 最优路由(改进 #1 的实盘/前向基建)。
- 4 所:Binance / Bybit / HL / Gate,U本位永续。每所抓 funding + mark,**按真实结算间隔**归一成每小时率
  (台账 E38:~60% 币是 4h 不是 8h;HL 是 1h)。
- 规范化符号到 canonical=base(去 USDT、去 1000/k 前缀)→ 同一经济币跨所对齐(funding% 与收益率对 1000x 缩放不变)。
- 路由:每币在有数据的所里 long 最低 funding 所 / short 最高所,spread=f_short_hr−f_long_hr(≥0,零借币)。
自检:PYTHONUTF8=1 .venv/Scripts/python.exe venues.py
"""
import concurrent.futures as cf
import json
import re
import urllib.request

FAPI = "https://fapi.binance.com"; SAPI = "https://api.binance.com"
BYBIT = "https://api.bybit.com"; HLINFO = "https://api.hyperliquid.xyz/info"; GATE = "https://api.gateio.ws/api/v4"
OKX = "https://www.okx.com"
UA = {"User-Agent": "Mozilla/5.0"}
VENUES = ["bin", "byb", "hl", "gate", "okx"]
# 基差风险等级(台账 E40/E42:含 HL/DEX 腿 σ20-34% 基差风险;CEX-CEX≈0)→ 路由风险罚用
RISK = {"bin": 0.0, "byb": 0.0, "gate": 0.0, "okx": 0.0, "hl": 1.0}
RISK_LAMBDA = 0.10        # 风险罚系数(年化):E42 甜点,net 不降而方差/HL使用↓
# 加拿大可达所 → 实盘路由默认只用这些;None=全用。
# HL=DEX干净;Gate=未受监管离岸但未退出加拿大(较轻灰色);OKX 已宣布退出加拿大→剔除(冻账风险高且几乎不加宽度)。
# (用户上线前请自行核实各所当前对加拿大的状态——监管会变。)
ACCESSIBLE = {"hl", "gate"}


def get(u, retries=3):
    for i in range(retries):
        try:
            return json.loads(urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=30).read())
        except Exception:
            if i == retries - 1:
                raise
            import time; time.sleep(1.0 * (i + 1))


def post(url, body, retries=3):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={**UA, "Content-Type": "application/json"})
            return json.loads(urllib.request.urlopen(req, timeout=30).read())
        except Exception:
            if i == retries - 1:
                raise
            import time; time.sleep(1.0 * (i + 1))


def canon(base):
    """去 1000/k 前缀 → 经济币名(funding% 与收益率对缩放不变)。"""
    b = base.upper()
    b = re.sub(r"^1000000", "", b)
    b = re.sub(r"^10000", "", b)
    b = re.sub(r"^1000", "", b)
    if b.startswith("K") and len(b) > 3:        # HL kPEPE → PEPE(仅当去前缀后仍像 ticker)
        pass                                     # k 前缀单独在 hl 映射处处理
    return b


def fetch():
    """→ data[canon] = {venue: {'f_hr':每小时funding率, 'mark':标记价}}, 以及 binance 现货价 spot[canon]。"""
    data = {}
    spot = {}

    def slot(canon_key):
        return data.setdefault(canon_key, {})

    # ---- Binance 永续:funding + mark + 间隔 + 24h成交额 ----
    bin_iv = {d["symbol"]: d.get("fundingIntervalHours", 8) for d in get(f"{FAPI}/fapi/v1/fundingInfo")}
    bin_qv = {d["symbol"]: float(d["quoteVolume"]) for d in get(f"{FAPI}/fapi/v1/ticker/24hr")}
    for d in get(f"{FAPI}/fapi/v1/premiumIndex"):
        s = d["symbol"]
        if not s.endswith("USDT"):
            continue
        mark = float(d["markPrice"])
        if mark <= 0:
            continue
        iv = bin_iv.get(s, 8)
        c = canon(s[:-4])
        slot(c)["bin"] = {"f_hr": float(d["lastFundingRate"]) / iv, "mark": mark, "qv": bin_qv.get(s, 0.0)}
    # 现货(carry 正侧腿):加拿大可达 → 用 **Gate 现货**(非 Binance)。顺手抓现货成交额(carry 现货腿流动过滤用)
    try:
        for d in get(f"{GATE}/spot/tickers"):
            cp = d.get("currency_pair", "")
            if cp.endswith("_USDT") and d.get("last") not in (None, ""):
                p = float(d["last"])
                if p > 0:
                    c = canon(cp[:-5])
                    spot[c] = p
                    slot(c)["gate_spot"] = {"mark": p, "qv": float(d.get("quote_volume", 0) or 0)}
    except Exception:
        pass

    # ---- Bybit 永续 ----
    byb_iv = {}
    cur = ""
    while True:
        res = get(f"{BYBIT}/v5/market/instruments-info?category=linear&limit=1000" + (f"&cursor={cur}" if cur else ""))["result"]
        for s in res["list"]:
            if s.get("fundingInterval"):
                byb_iv[s["symbol"]] = int(s["fundingInterval"]) / 60.0
        cur = res.get("nextPageCursor", "")
        if not cur:
            break
    for d in get(f"{BYBIT}/v5/market/tickers?category=linear")["result"]["list"]:
        s = d["symbol"]
        if not s.endswith("USDT") or d.get("fundingRate") in (None, "") or d.get("markPrice") in (None, ""):
            continue
        mark = float(d["markPrice"])
        if mark <= 0:
            continue
        iv = byb_iv.get(s, 8)
        slot(canon(s[:-4]))["byb"] = {"f_hr": float(d["fundingRate"]) / iv, "mark": mark,
                                      "qv": float(d.get("turnover24h", 0.0) or 0.0)}

    # ---- HL(funding 1h;名字 base 或 kXxx)----
    r = post(HLINFO, {"type": "metaAndAssetCtxs"})
    for m, ctx in zip(r[0]["universe"], r[1]):
        nm = m["name"]; base = nm[1:] if nm.startswith("k") else nm     # kPEPE→PEPE
        mark = ctx.get("markPx")
        if ctx.get("funding") in (None, "") or mark in (None, ""):
            continue
        mark = float(mark)
        if mark <= 0:
            continue
        slot(canon(base))["hl"] = {"f_hr": float(ctx["funding"]), "mark": mark,    # 已是 1h 率
                                   "qv": float(ctx.get("dayNtlVlm", 0.0) or 0.0)}

    # ---- Gate(funding_interval 秒;名字 BASE_USDT)----
    gate_qv = {}
    try:
        for d in get(f"{GATE}/futures/usdt/tickers"):
            gate_qv[d["contract"]] = float(d.get("volume_24h_quote", 0.0) or 0.0)
    except Exception:
        pass
    gc = {d["name"]: d for d in get(f"{GATE}/futures/usdt/contracts")}
    for name, d in gc.items():
        if not name.endswith("_USDT"):
            continue
        try:
            r_ = d.get("funding_rate"); mark = d.get("mark_price")
            if r_ in (None, "") or mark in (None, ""):
                continue
            iv = float(d.get("funding_interval", 28800)) / 3600.0
            mark = float(mark)
            if mark <= 0:
                continue
            slot(canon(name[:-5]))["gate"] = {"f_hr": float(r_) / iv, "mark": mark, "qv": gate_qv.get(name, 0.0)}
        except (KeyError, ValueError, TypeError):
            continue

    # ---- OKX(funding 8h,per-inst;名字 BASE-USDT-SWAP)。仅当路由用 OKX 才拉(per-inst 慢,~360 调用)----
    if "okx" in (ACCESSIBLE if ACCESSIBLE is not None else {"okx"}):
        _fetch_okx(data, get)

    return data, spot


def _fetch_okx(data, get):
    def slot(c):
        return data.setdefault(c, {})
    try:
        okx_inst = {canon(d["instId"].replace("-USDT-SWAP", "")): d["instId"]
                    for d in get(f"{OKX}/api/v5/public/instruments?instType=SWAP")["data"]
                    if d.get("settleCcy") == "USDT"}
        okx_tk = {}
        for d in get(f"{OKX}/api/v5/market/tickers?instType=SWAP")["data"]:
            try:
                last = float(d["last"]); vol = float(d.get("volCcy24h", 0) or 0)
                okx_tk[d["instId"]] = (last, vol * last)
            except (KeyError, ValueError, TypeError):
                continue
        want = [c for c in data if c in okx_inst]      # 只拉已见经济币(限调用数)

        def okxf(c):
            try:
                fr = get(f"{OKX}/api/v5/public/funding-rate?instId={okx_inst[c]}")["data"][0]
                return c, float(fr["fundingRate"])
            except Exception:
                return c, None
        with cf.ThreadPoolExecutor(max_workers=8) as ex:
            for c, fr in ex.map(okxf, want):
                inst = okx_inst[c]
                if fr is not None and inst in okx_tk and okx_tk[inst][0] > 0:
                    slot(c)["okx"] = {"f_hr": fr / 8.0, "mark": okx_tk[inst][0], "qv": okx_tk[inst][1]}
    except Exception:
        pass


def route(venue_map, min_qv=1e6, max_spread_ann=3.0, accessible=ACCESSIBLE, lam=RISK_LAMBDA):
    # min_qv=$1M:小资金($5-10k,$200-500/单)合适阈值;大资金应调高(台账 E45 宽度扫)。
    """venue_map: {venue:{'f_hr','mark','qv'}} → (long, short, spread_hr, min_qv) 或 None。
    - 安全过滤:两腿 24h 成交额 ≥ min_qv;spread 年化 ≤ max_spread_ann(防错配/薄币幻影套利)。
    - accessible:只在可达所里选(加拿大:HL/Gate/OKX;None=全用)。
    - **风险调整(E42)**:在所有有效对里按 (spread − lam·基差风险) 选最优对,而非只挑 funding 极端两所
      → 罚 HL/DEX 腿的 σ20-34% 基差风险,偏好 CEX-CEX 干净对。"""
    vs = [(v, d["f_hr"], d.get("qv", 0.0)) for v, d in venue_map.items()
          if "f_hr" in d and d.get("qv", 0.0) >= min_qv and (accessible is None or v in accessible)]   # 'f_hr' in d:排除 gate_spot(无funding)
    if len(vs) < 2:
        return None
    best = None
    for a in range(len(vs)):
        for b in range(a + 1, len(vs)):
            va, vb = vs[a], vs[b]
            lo, hi = (va, vb) if va[1] <= vb[1] else (vb, va)   # long 低 funding、short 高
            spread = hi[1] - lo[1]
            if spread * 8760 > max_spread_ann:
                continue
            risk = max(RISK.get(va[0], 0.0), RISK.get(vb[0], 0.0))
            score = spread - lam / 8760.0 * risk                # 风险调整后得分
            if best is None or score > best[0]:
                best = (score, lo[0], hi[0], spread, min(lo[2], hi[2]))
    if best is None or best[3] <= 0:
        return None
    return best[1], best[2], best[3], best[4]


if __name__ == "__main__":
    data, spot = fetch()
    cov = {v: sum(1 for d in data.values() if v in d) for v in VENUES}
    print(f"多所快照:{len(data)} 经济币;各所覆盖 {cov};现货 {len(spot)}")
    # routing 快照:安全过滤(成交额≥$20M、spread≤300%/yr)后按 spread 排序前 15
    rows = []
    for c, vm in data.items():
        r = route(vm)
        if r:
            rows.append((c, r[0], r[1], r[2] * 8760, r[3]))    # 年化 spread + min 成交额
    rows.sort(key=lambda x: -x[3])
    print(f"\n可路由币(过滤后){len(rows)};最肥 spread(年化)前 15:")
    print(f"  {'币':12s} {'long所':>6s} {'short所':>7s} {'spread年化':>10s} {'min成交额$M':>10s}")
    for c, lo, hi, sp, qv in rows[:15]:
        print(f"  {c:12s} {lo:>6s} {hi:>7s} {sp:>+10.1%} {qv/1e6:>10.0f}")
    import numpy as np
    sps = np.array([r[3] for r in rows])
    print(f"\n中位 spread 年化 {np.median(sps):+.1%}  均值 {sps.mean():+.1%}")
    from collections import Counter
    print("各所当 short(收最高 funding)次数:", dict(Counter(r[2] for r in rows)))
    print("各所当 long(收最低 funding)次数:", dict(Counter(r[1] for r in rows)))
