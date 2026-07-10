"""dry-run 执行层(加拿大可达栈:Gate现货+Gate永续+HL+OKX)。
目标书(state.json)→ 各所有符号名义 → 净额合并(同所同币合一净仓)→ 对账(target−实际=delta)
→ 按各所 lot/min-notional 取整过滤 → maker 限价 → 风控 → **审计留痕** → 只打印不下单。
- carry(Gate):每仓 long Gate现货 + short Gate永续。
- 路由(HL/Gate/OKX):每仓 long 低funding所 + short 高funding所(route 决定)。
- 审计:每次对账整条计划 append 到 data/exec/audit_log.jsonl(时间戳+序号+链式哈希,不可变,给 prop/投资人留痕)。
独立脚本,复用 venues.py 行情;**全程 dry-run、不碰钱、不连账户。**
跑：PYTHONUTF8=1 .venv/Scripts/python.exe execution.py --capital 5000
"""
import argparse
import hashlib
import json
import math
import os
import time
from collections import defaultdict
from datetime import datetime, timezone

import urllib.request

from venues import fetch as fetch_venues, canon

UA = {"User-Agent": "Mozilla/5.0"}
OKX = "https://www.okx.com"; GATE = "https://api.gateio.ws/api/v4"; HLINFO = "https://api.hyperliquid.xyz/info"
POSITIONS = "data/exec/positions.json"; AUDIT = "data/exec/audit_log.jsonl"; RULES_CACHE = "data/exec/rules_access.json"
FEE_MAKER = {"gate_spot": 10.0, "gate_perp": 2.0, "hl": 1.5, "okx": 2.0}    # maker bps
LEV_CAP = 2.0


def get(u):
    return json.loads(urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=30).read())


def post(u, b):
    return json.loads(urllib.request.urlopen(urllib.request.Request(u, data=json.dumps(b).encode(),
                      headers={**UA, "Content-Type": "application/json"}), timeout=30).read())


def load_rules():
    """各所 lot(base 步长)/min_notional($),keyed by canonical。"""
    if os.path.exists(RULES_CACHE) and time.time() - os.path.getmtime(RULES_CACHE) < 86400:
        return json.load(open(RULES_CACHE))
    r = {"gate_spot": {}, "gate_perp": {}, "hl": {}, "okx": {}}
    for m in post(HLINFO, {"type": "meta"})["universe"]:        # HL:szDecimals→lot,最小单~$10
        r["hl"][canon(m["name"])] = {"sym": m["name"], "lot": 10 ** (-int(m["szDecimals"])), "min_notional": 10.0}
    for d in get(f"{OKX}/api/v5/public/instruments?instType=SWAP")["data"]:   # OKX:1张=ctVal base
        if d.get("settleCcy") != "USDT":
            continue
        ct = float(d["ctVal"])
        r["okx"][canon(d["instId"].replace("-USDT-SWAP", ""))] = {"sym": d["instId"], "lot": float(d["lotSz"]) * ct, "min_notional": 1.0}
    for d in get(f"{GATE}/futures/usdt/contracts"):             # Gate永续:1张=quanto_multiplier base
        if not d["name"].endswith("_USDT"):
            continue
        qm = float(d.get("quanto_multiplier", 0) or 0)
        if qm > 0:
            r["gate_perp"][canon(d["name"][:-5])] = {"sym": d["name"], "lot": qm, "min_notional": 5.0}
    for d in get(f"{GATE}/spot/currency_pairs"):                # Gate现货:min_quote_amount($)
        if d["id"].endswith("_USDT"):
            r["gate_spot"][canon(d["id"][:-5])] = {"sym": d["id"], "lot": 10 ** (-int(d.get("amount_precision", 6))),
                                                   "min_notional": float(d.get("min_quote_amount", 5) or 5)}
    os.makedirs("data/exec", exist_ok=True); json.dump(r, open(RULES_CACHE, "w"))
    return r


def round_lot(qty, lot):
    return math.floor(abs(qty) / lot) * lot * (1 if qty >= 0 else -1) if lot > 0 else qty


def build_targets(st, capital, data, spot, split=0.5):
    """目标书 → 各 (venue, canon) 有符号名义($)→ 净额合并。"""
    carry = st["carry"].get("cur", {}); xv = st["xv"].get("cur", {})
    route = {c: tuple(v) for c, v in st["xv"].get("route", {}).items()}
    Cc, Cx = capital * split, capital * (1 - split)
    notion = defaultdict(float); miss = {"carry": 0.0, "xv": 0.0}; tot = {"carry": 0.0, "xv": 0.0}
    for c, w in carry.items():
        tot["carry"] += abs(w)
        if w <= 0 or c not in spot or c not in data or "gate" not in data[c]:
            miss["carry"] += abs(w); continue
        n = w * Cc
        notion[("gate_spot", c)] += n; notion[("gate_perp", c)] += -n      # 多现货+空永续
    vmap = {"hl": "hl", "gate": "gate_perp", "okx": "okx"}
    for c, w in xv.items():
        tot["xv"] += abs(w)
        if w <= 0 or c not in route:
            miss["xv"] += abs(w); continue
        L, S = route[c]
        if L not in vmap or S not in vmap or c not in data or L not in data[c] or S not in data[c]:
            miss["xv"] += abs(w); continue
        n = w * Cx
        notion[(vmap[L], c)] += n; notion[(vmap[S], c)] += -n              # long低/short高
    return notion, miss, tot


def price_of(venue, c, data, spot):
    if venue == "gate_spot":
        return spot.get(c)
    vk = {"gate_perp": "gate", "hl": "hl", "okx": "okx"}.get(venue)    # 未知所(如旧格式)→ None
    if vk is None:
        return None
    return data.get(c, {}).get(vk, {}).get("mark")


def reconcile(notion, current, data, spot, rules):
    cur_t = {}
    for ks, q in current.items():
        venue, c = ks.split("|", 1); cur_t[(venue, c)] = q
    tgt_q = {}
    for (venue, c), nv in notion.items():
        px = price_of(venue, c, data, spot); rl = rules.get(venue, {}).get(c)
        if not px or px <= 0 or rl is None:
            continue
        q = round_lot(nv / px, rl["lot"])
        if q != 0:
            tgt_q[(venue, c)] = q
    orders = []; skipped = 0.0
    for k in set(tgt_q) | set(cur_t):
        venue, c = k; rl = rules.get(venue, {}).get(c); px = price_of(venue, c, data, spot)
        if rl is None or not px or px <= 0:
            continue
        dq = round_lot(tgt_q.get(k, 0.0) - cur_t.get(k, 0.0), rl["lot"])
        if dq == 0:
            continue
        notional = abs(dq) * px
        if notional < rl["min_notional"]:
            skipped += notional; continue
        orders.append({"venue": venue, "sym": rl["sym"], "coin": c, "side": "BUY" if dq > 0 else "SELL",
                       "qty": abs(dq), "price": px, "notional": notional})
    return orders, tgt_q, skipped


def flatten_orders(coins, st, capital, data, spot, rules, reduce_to=0.0):
    """风控触发 → reduce-only 平仓计划:把指定 coins 的仓位降到 reduce_to×当前(0=全平,0.25=降到25%)。
    用 build_targets 取当前(paper)名义;delta=目标−当前,方向永远是**减仓**(reduce_only=True)。给 risk_monitor 调。"""
    coinset = set(coins)
    notion, _, _ = build_targets(st, capital, data, spot)
    orders = []
    for (venue, c), nv in notion.items():
        if (coins != "ALL" and c not in coinset) or abs(nv) < 1e-9:
            continue
        px = price_of(venue, c, data, spot); rl = rules.get(venue, {}).get(c)
        if not px or px <= 0 or rl is None:
            continue
        dq = round_lot((nv * reduce_to - nv) / px, rl["lot"])      # 减仓方向
        if dq == 0:
            continue
        notional = abs(dq) * px
        if notional < rl["min_notional"]:
            continue
        orders.append({"venue": venue, "sym": rl["sym"], "coin": c, "side": "BUY" if dq > 0 else "SELL",
                       "qty": abs(dq), "price": px, "notional": notional, "reduce_only": True})
    return orders


def audit_write(record):
    """链式哈希 append-only 审计日志(每条含前条 hash → 不可篡改)。"""
    os.makedirs("data/exec", exist_ok=True)
    prev = ""
    if os.path.exists(AUDIT):
        lines = open(AUDIT).read().splitlines()
        if lines:
            prev = json.loads(lines[-1]).get("hash", "")
    record["prev_hash"] = prev
    record["hash"] = hashlib.sha256((prev + json.dumps(record, sort_keys=True)).encode()).hexdigest()[:16]
    with open(AUDIT, "a") as f:
        f.write(json.dumps(record) + "\n")
    return record["hash"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capital", type=float, default=5000)
    ap.add_argument("--state", default="data/paper/state.json")
    a = ap.parse_args()
    st = json.load(open(a.state)); data, spot = fetch_venues(); rules = load_rules()
    current = json.load(open(POSITIONS)) if os.path.exists(POSITIONS) else {}
    notion, miss, tot = build_targets(st, a.capital, data, spot)
    orders, tgt_q, skipped = reconcile(notion, current, data, spot, rules)

    gross = sum(abs(q) * (price_of(v, c, data, spot) or 0) for (v, c), q in tgt_q.items())
    lev = gross / a.capital
    cov_c = 1 - miss["carry"] / max(tot["carry"], 1e-9); cov_x = 1 - miss["xv"] / max(tot["xv"], 1e-9)
    cost_mk = sum(o["notional"] * FEE_MAKER[o["venue"]] / 1e4 for o in orders)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    seq = sum(1 for _ in open(AUDIT)) if os.path.exists(AUDIT) else 0
    h = audit_write({"seq": seq, "ts": ts, "capital": a.capital, "n_target": len(tgt_q), "n_orders": len(orders),
                     "gross": round(gross, 2), "leverage": round(lev, 3), "cost_maker": round(cost_mk, 2),
                     "skipped_minnotional": round(skipped, 2), "current_positions": len(current),
                     "coverage": {"carry": round(cov_c, 3), "xv": round(cov_x, 3)}, "orders": orders})

    print(f"=== dry-run 执行计划 @ ${a.capital:,.0f}  (可达栈 Gate/HL/OKX) ===")
    print(f"  目标净仓 {len(tgt_q)} 个;delta 订单 {len(orders)} 条;实际持仓 {len(current)} 个({'读自 '+POSITIONS if current else '空仓→全量建'})")
    print(f"  覆盖率 carry {cov_c:.0%} / 路由 {cov_x:.0%};毛杠杆 {lev:.2f}x" + ("  ⚠️超2x!" if lev > LEV_CAP + 1e-6 else "  ✓"))
    print(f"  maker 建仓成本 ${cost_mk:,.2f} ({cost_mk/a.capital*1e4:.1f}bps);min-notional 略过 ${skipped:,.0f}")
    vc = defaultdict(int)
    for o in orders:
        vc[o["venue"]] += 1
    print(f"  各所订单数: {dict(vc)}")
    print("  样例订单(前 10,maker 限价):")
    for o in orders[:10]:
        print(f"    {o['venue']:10s} {o['side']:4s} {o['coin']:10s} qty={o['qty']:<12.6g} @ {o['price']:<11.6g} ${o['notional']:,.0f}")
    print(f"\n  审计留痕 → {AUDIT}  (seq={seq}, hash={h}, 链式哈希不可变)")
    print(f"  实际持仓写 {POSITIONS}(只读 key 拉,key 形如 'gate_perp|BTC')即可对账增量。⚠️ DRY-RUN:只打印、未下单、未连账户。")


if __name__ == "__main__":
    main()
