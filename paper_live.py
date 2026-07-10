"""前向纸上验证(加拿大可达栈:Gate 正侧 carry + HL/Gate/OKX 风险调整路由)。实时数据、无钱无单,逐小时记账。
两腿都由 strategy.py 的 step() 驱动(backtest 同一大脑);多所抓取/路由/真实间隔在 venues.py。
- carry 腿:**只做正 funding**(Gate 多现货 + 空 Gate 永续,收 funding,零借币;Binance 退出加拿大→用 Gate)。
- 跨所腿:**可达所(HL/Gate/OKX)最优路由 + 风险调整**(long 低 funding 所/short 高,收 spread,零借币;
  按 spread−λ·基差风险选对,罚 HL/DEX 的 σ20-34% 基差风险=E42;安全过滤成交额≥$20M、spread≤300%/yr)。
目的:前向测真实**组合 Sharpe** + 多所路由的**价格腿 haircut + 执行/路由换手**(回测测不了,绕开缺 HL/Gate 历史)。
状态 data/paper/state.json;日志 log.csv。跑：PYTHONUTF8=1 .venv/Scripts/python.exe paper_live.py
"""
import json
import os
import time
from datetime import datetime, timezone

import numpy as np

from strategy import CarryConfig, CarryStrategy, XVenueConfig, CrossVenueStrategy
from venues import fetch as fetch_venues, route

STATE = "data/paper/state.json"; LOG = "data/paper/log.csv"
CC = CarryConfig(); XC = XVenueConfig()
WINS = 0.10                # 价格腿单 bar winsor(跨所/carry 基差噪声上限)


CARRY_VENUE = "gate"        # carry 腿用所(加拿大可达:Gate 现货+永续;非 Binance)
CARRY_MIN_QV = 1e6          # carry 现货腿最小日成交额($,小资金画像;台账 E51:防混进现货腿不可交易的薄币)


def settle_carry(carry_cur, prev, data, spot, dt_h):
    """在上一 carry 书上结算:funding(收正 funding)+ 价格腿(Gate现货−Gate永续)。"""
    cf = cp = 0.0
    for c, w in carry_cur.items():
        if w <= 0 or c not in prev or c not in data or c not in spot:
            continue
        p = prev[c]
        if CARRY_VENUE not in p or "spot" not in p or CARRY_VENUE not in data[c]:
            continue
        cf += w * p[CARRY_VENUE]["f_hr"] * dt_h
        sp_now = spot[c]; pm_now = data[c][CARRY_VENUE]["mark"]
        pl = (sp_now / p["spot"] - 1) - (pm_now / p[CARRY_VENUE]["mark"] - 1)
        cp += w * max(-WINS, min(WINS, pl))
    return cf, cp


def settle_xv(xv_cur, route_prev, prev, data, dt_h):
    """在上一跨所书上结算:spread funding + 价格腿(long所−short所 永续收益)。"""
    xf = xp = 0.0
    for c, w in xv_cur.items():
        if w <= 0 or c not in route_prev or c not in prev or c not in data:
            continue
        L, S = route_prev[c]
        p = prev[c]
        if L not in p or S not in p or L not in data[c] or S not in data[c]:
            continue
        xf += w * (p[S]["f_hr"] - p[L]["f_hr"]) * dt_h
        lret = data[c][L]["mark"] / p[L]["mark"] - 1
        sret = data[c][S]["mark"] / p[S]["mark"] - 1
        xp += w * max(-WINS, min(WINS, lret - sret))
    return xf, xp


def main():
    os.makedirs("data/paper", exist_ok=True)
    now = time.time()
    data, spot = fetch_venues()
    st = json.load(open(STATE)) if os.path.exists(STATE) else None
    if st is None:
        st = {"U": [], "carry": {}, "xv": {}, "basis_buf": {}, "prev": {},
              "last_time": now, "t0": now, "runs": 0,
              "cum": {k: 0.0 for k in ("c_fund", "c_price", "c_cost", "c_pnl",
                                       "x_fund", "x_price", "x_cost", "x_pnl", "comb_pnl")}}
    dt_h = (now - st["last_time"]) / 3600.0
    first = st["runs"] == 0
    prev = st["prev"]; route_prev = {c: tuple(v) for c, v in st["xv"].get("route", {}).items()}

    # basis buffer(carry mom 用 Binance 永续/现货基差)
    buf = st["basis_buf"]; keep = now - (CC.mom_w + 12) * 3600.0
    for c, vm in data.items():
        if CARRY_VENUE in vm and c in spot and spot[c] > 0:
            h = [x for x in buf.get(c, []) if x[0] >= keep]
            h.append([now, vm[CARRY_VENUE]["mark"] / spot[c] - 1]); buf[c] = h

    # ---- 结算上一书 ----
    cf, cp = settle_carry(st["carry"].get("cur", {}), prev, data, spot, dt_h)
    xf, xp = settle_xv(st["xv"].get("cur", {}), route_prev, prev, data, dt_h)

    # ---- 宇宙 + 数组 ----
    U = list(st["U"]); seen = set(U)
    for c in data:
        if c not in seen:
            U.append(c); seen.add(c)
    N = len(U)
    f_carry = np.zeros(N); tdv_c = np.zeros(N); act_c = np.zeros(N, bool); mom = np.zeros(N)
    spread = np.zeros(N); tdv_x = np.zeros(N); act_x = np.zeros(N, bool)
    route_new = {}
    tgt = now - CC.mom_w * 3600.0
    for i, c in enumerate(U):
        vm = data.get(c, {})
        if CARRY_VENUE in vm:
            f_carry[i] = vm[CARRY_VENUE]["f_hr"]
            perp_qv = vm[CARRY_VENUE].get("qv", 0.0); spot_qv = vm.get("gate_spot", {}).get("qv", 0.0)
            tdv_c[i] = min(perp_qv, spot_qv)        # 瓶颈腿流动(永续/现货取小);选币与过滤都按它
            # carry 入场要求**两腿都够流动**(现货腿曾混进 TQQQX 现货$4k=不可交易,台账 E51)
            if c in spot and spot[c] > 0 and "gate_spot" in vm and spot_qv >= CARRY_MIN_QV:
                act_c[i] = True
                old = None
                for ts, b in buf.get(c, []):
                    if ts <= tgt:
                        old = b
                    else:
                        break
                if old is not None:
                    mom[i] = (vm[CARRY_VENUE]["mark"] / spot[c] - 1) - old
        r = route(vm)
        if r is not None:
            L, S, sp_hr, qv = r
            spread[i] = sp_hr; tdv_x[i] = qv; act_x[i] = True; route_new[c] = (L, S)

    # ---- 恢复策略 state,step ----
    carry = CarryStrategy(N, CC)
    carry.S = np.array([st["carry"].get("S", {}).get(c, 0.0) for c in U])
    carry.cur = np.array([st["carry"].get("cur", {}).get(c, 0.0) for c in U])
    carry.S_init = st["carry"].get("S_init", False); carry.since_rebal = st["carry"].get("since_rebal", float(CC.cad))
    cnew = carry.step(f_carry, mom, tdv_c, act_c, dt=dt_h if not first else 1.0)

    xv = CrossVenueStrategy(N, XC)
    xv.S = np.array([st["xv"].get("S", {}).get(c, 0.0) for c in U])
    xv.cur = np.array([st["xv"].get("cur", {}).get(c, 0.0) for c in U])
    xv.S_init = st["xv"].get("S_init", False); xv.since_rebal = st["xv"].get("since_rebal", float(XC.cad))
    xnew = xv.step(spread, tdv_x, act_x, dt=dt_h if not first else 1.0)

    cnew_d = {U[i]: float(cnew[i]) for i in range(N) if cnew[i] > 1e-9}
    xnew_d = {U[i]: float(xnew[i]) for i in range(N) if xnew[i] > 1e-9}
    route_keep = {c: route_new[c] for c in xnew_d if c in route_new}

    # ---- 成本:权重换手 + 路由切换(换所=平旧对开新对,额外 2×权重)----
    cc_prev = st["carry"].get("cur", {}); xc_prev = st["xv"].get("cur", {})
    c_turn = sum(abs(cnew_d.get(c, 0) - cc_prev.get(c, 0)) for c in set(cnew_d) | set(cc_prev))
    x_turn = sum(abs(xnew_d.get(c, 0) - xc_prev.get(c, 0)) for c in set(xnew_d) | set(xc_prev))
    for c, w in xnew_d.items():                       # 路由切换额外换手
        if c in route_prev and c in route_keep and route_keep[c] != route_prev[c]:
            x_turn += 2 * w
    c_cost = CC.cost_bps / 1e4 * c_turn; x_cost = XC.cost_bps / 1e4 * x_turn
    c_bar = cf + cp - c_cost; x_bar = xf + xp - x_cost
    comb_bar = 0.5 * c_bar + 0.5 * x_bar

    if not first:
        for k, v in [("c_fund", cf), ("c_price", cp), ("c_cost", c_cost), ("c_pnl", c_bar),
                     ("x_fund", xf), ("x_price", xp), ("x_cost", x_cost), ("x_pnl", x_bar), ("comb_pnl", comb_bar)]:
            st["cum"][k] += v
    # prev 快照(只存当前书涉及的币,省空间)
    keep_coins = set(cnew_d) | set(xnew_d) | set(cc_prev) | set(xc_prev)
    prev_new = {}
    for c in keep_coins:
        vm = data.get(c)
        if not vm:
            continue
        rec = {v: {"f_hr": vm[v]["f_hr"], "mark": vm[v]["mark"]} for v in vm if "f_hr" in vm[v]}   # gate_spot 无 f_hr,跳过
        if c in spot:
            rec["spot"] = spot[c]
        prev_new[c] = rec
    st.update({"U": U, "basis_buf": buf, "prev": prev_new, "last_time": now, "runs": st["runs"] + 1,
               "carry": {"S": {U[i]: float(carry.S[i]) for i in range(N) if abs(carry.S[i]) > 1e-9},
                         "cur": cnew_d, "S_init": carry.S_init, "since_rebal": carry.since_rebal},
               "xv": {"S": {U[i]: float(xv.S[i]) for i in range(N) if abs(xv.S[i]) > 1e-9},
                      "cur": xnew_d, "S_init": xv.S_init, "since_rebal": xv.since_rebal,
                      "route": {c: list(v) for c, v in route_keep.items()}}})
    json.dump(st, open(STATE, "w"))

    ts = datetime.fromtimestamp(now, timezone.utc).strftime("%Y-%m-%d %H:%M")
    if first:
        # 路由腿用所统计
        from collections import Counter
        shorts = Counter(route_keep[c][1] for c in xnew_d if c in route_keep)
        print(f"初始化(多所组合书:carry {len(cnew_d)}仓正侧only + 多所路由 {len(xnew_d)}仓)。short所分布 {dict(shorts)}。下次起累计 P&L。")
        return
    days = (now - st["t0"]) / 86400.0
    a = lambda v: v / days * 365 if days > 0.02 else 0.0
    if not os.path.exists(LOG):
        open(LOG, "w").write("time,dt_h,c_pnl,c_cum,x_pnl,x_cum,comb_pnl,comb_cum,c_npos,x_npos\n")
    cm = st["cum"]
    open(LOG, "a").write(f"{ts},{dt_h:.2f},{c_bar:.6f},{cm['c_pnl']:.6f},{x_bar:.6f},{cm['x_pnl']:.6f},"
                         f"{comb_bar:.6f},{cm['comb_pnl']:.6f},{len(cnew_d)},{len(xnew_d)}\n")
    print(f"{ts} dt={dt_h:.1f}h | carry 累计{cm['c_pnl']:+.5f}(折年{a(cm['c_pnl']):+.0%}) | "
          f"多所路由 累计{cm['x_pnl']:+.5f}(折年{a(cm['x_pnl']):+.0%}) | 组合 累计{cm['comb_pnl']:+.5f}(折年{a(cm['comb_pnl']):+.0%}) run#{st['runs']}")


if __name__ == "__main__":
    main()
