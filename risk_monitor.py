"""自动减仓监控(DRY-RUN:只读 + 只 log,绝不下单)。设计见 docs/风控.md §6 / 台账 E54-E56。
分层触发:
  L1 回撤闸:NAV 从峰值跌 > 15% → 应降到 25%×杠杆(reduce-only)。
  L2 每对价格止损(**核心,解锁跨所 3x;E55/E56**):某跨所对的币价从入场动 > 止损线(=70%×强平距离,
     3x 时 22%)→ 应**整对一起平**(对冲完整退出,不留孤儿腿)。只需实时 mark,不需账户保证金 → 现在就能验。
  R2 对手方集中度:Gate 实际敞口 ≤ 40%。
  L3 基差闸:某仓跨所基差 > 5% → 应 flatten。
分辨率结论(E56):反应窗口 ~2 分钟 → 轮询 10-30s(或 WS markPrice ~1s)足够,**不需 tick/亚秒**。
DRY-RUN:在 paper 数据上验证触发逻辑正确,再接 execution reduce-only。
跑：单次 `python risk_monitor.py`；轮询 `python risk_monitor.py --loop 30`(每 30s)。
"""
import argparse
import json
import math
import os
import time
from collections import defaultdict
from datetime import datetime, timezone

STATE = "data/paper/state.json"
NAV = "data/paper/nav.csv"
LOG = "data/exec/risk_log.jsonl"
RISK_STATE = "data/exec/risk_state.json"     # 持久化每对入场 mark(算从入场的不利动)

# ---- 阈值(docs/风控.md §3/§6;台账 E56 定稿)----
LEV_CROSS = 3.0        # 跨所目标杠杆 → 决定止损线
MM = 0.02              # 维持保证金率(薄肥尾币)
STOP_FRAC = 0.70       # 止损线 = STOP_FRAC × 强平距离(亏到这就平,留逃逸空间)
DD_TRIG = -0.15        # L1 回撤触发
DD_RESTORE = -0.06     # L1 恢复
GATE_CAP = 0.40        # R2 单一中心化托管帽
BASIS_TRIG = 0.05      # L3 单仓跨所基差熔断


def stop_level():
    """跨所目标杠杆下的止损线(不利价动到此 → 平整对)。"""
    return STOP_FRAC * (1.0 / LEV_CROSS - MM)


def basis_dev(mL, mS):
    """尺度稳健的跨所基差:Gate 'PEPE' vs HL 'kPEPE'(1000×)同名不同倍数 → 先按最近 10 的幂归一,
    再算残差基差(canon 归一了名字但 mark 在不同 multiplier 尺度;台账:策略用收益率不受影响,仅此处对比需修)。"""
    if not (mL > 0 and mS > 0):
        return 0.0
    r = mL / mS
    scale = 10.0 ** round(math.log10(r))    # 吃掉 1000/10000/1e6 等 multiplier 错配
    return abs(r / scale - 1.0)


def nav_drawdown():
    if not os.path.exists(NAV):
        return None
    navs = []
    for ln in open(NAV, encoding="utf-8").read().splitlines()[1:]:
        p = ln.split(",")
        if len(p) >= 2:
            try:
                navs.append(float(p[1]))
            except ValueError:
                pass
    if not navs:
        return None
    return navs[-1], max(navs), navs[-1] / max(navs) - 1.0


def venue_exposure(st):
    carry = st["carry"]["cur"]; xv = st["xv"]["cur"]
    route = {c: tuple(v) for c, v in st["xv"].get("route", {}).items()}
    cw = sum(w for w in carry.values() if w > 0) or 1.0
    xw = sum(w for w in xv.values() if w > 0) or 1.0
    ven = defaultdict(float)
    for c, w in carry.items():
        if w > 0:
            ven["gate"] += 2 * 0.5 * w / cw          # carry 现货+永续两腿都在 Gate
    for c, w in xv.items():
        if w > 0 and c in route:
            L, S = route[c]
            ven[L] += 0.5 * w / xw; ven[S] += 0.5 * w / xw
    return dict(ven), route


def per_pair_stop(st, route, data, entry):
    """L2:每对从入场的不利价动。entry 持久化每对入场 mark(long 腿)。返回 (触发列表, 更新后的 entry)。"""
    stop = stop_level()
    cur = st["xv"]["cur"]
    live = set()
    fires = []
    for c, w in cur.items():
        if w <= 0 or c not in route or c not in data:
            continue
        L, S = route[c]
        if L not in data[c] or S not in data[c]:
            continue
        mL = data[c][L]["mark"]
        if not (mL > 0):
            continue
        key = f"{c}|{L}|{S}"
        live.add(key)
        if key not in entry:                          # 新仓:记入场 mark
            entry[key] = mL
            continue
        e = entry[key]
        adv = abs(mL / e - 1.0)                        # 币价从入场的幅度(任一方向都伤一条腿)
        if adv > stop:
            fires.append((c, L, S, adv))
    for k in list(entry):                              # 已离场的仓清掉入场记录(再入场重置)
        if k not in live:
            del entry[k]
    return fires, entry


def run_once(verbose=True, capital=2000.0):
    os.makedirs("data/exec", exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    if not os.path.exists(STATE):
        if verbose:
            print("无 state.json(前向未跑)。先跑 paper_live.py 生成书。")
        return None
    st = json.load(open(STATE))
    entry = json.load(open(RISK_STATE)) if os.path.exists(RISK_STATE) else {}
    actions = []

    # 实时 marks(L2/L3 共用,只 fetch 一次)
    data = None; spot = {}; ferr = None
    try:
        from venues import fetch as fetch_venues
        data, spot = fetch_venues()
    except Exception as e:
        ferr = str(e)

    # L1 回撤闸
    dd = nav_drawdown()
    if dd:
        cur, peak, draw = dd
        status = "正常"
        if draw < DD_TRIG:
            status = "**触发→降到 25%×杠杆**"; actions.append({"trigger": "L1_drawdown", "dd": round(draw, 4)})
        elif draw < DD_RESTORE:
            status = "警戒区(维持降杠杆)"
        if verbose:
            print(f"[L1 回撤] NAV ${cur:,.0f} / 峰值 ${peak:,.0f} → {draw:+.1%}  {status}")
    elif verbose:
        print("[L1 回撤] 无 nav.csv,跳过")

    ven, route = venue_exposure(st)

    # L2 每对价格止损(核心)
    if data is not None:
        fires, entry = per_pair_stop(st, route, data, entry)
        json.dump(entry, open(RISK_STATE, "w"))
        if fires:
            for c, L, S, adv in fires:
                if verbose:
                    print(f"[L2 价格止损] **{c} {L}-{S} 从入场动 {adv:.1%} > 止损线 {stop_level():.1%}(@{LEV_CROSS:.0f}x)→ 平整对**")
                actions.append({"trigger": "L2_pair_stop", "coin": c, "adverse": round(adv, 4)})
        elif verbose:
            print(f"[L2 价格止损] {len([c for c,w in st['xv']['cur'].items() if w>0])} 仓监控中,无触发(止损线 {stop_level():.1%}@{LEV_CROSS:.0f}x)")
    elif verbose:
        print(f"[L2 价格止损] fetch 失败跳过: {ferr}")

    # R2 集中度
    tot = sum(ven.values()) or 1.0
    gate_frac = ven.get("gate", 0) / tot
    if gate_frac > GATE_CAP:
        actions.append({"trigger": "R2_concentration", "gate_frac": round(gate_frac, 3)})
    if verbose:
        r2 = "**超帽→扫款/减 Gate**" if gate_frac > GATE_CAP else "OK"
        print(f"[R2 集中度] Gate {gate_frac:.0%}(帽 {GATE_CAP:.0%}) | {{{', '.join(f'{k}:{v/tot:.0%}' for k,v in ven.items())}}}  {r2}")

    # L3 基差闸
    if data is not None:
        nb = 0
        for c, w in st["xv"]["cur"].items():
            if w <= 0 or c not in route or c not in data:
                continue
            L, S = route[c]
            if L in data[c] and S in data[c]:
                b = basis_dev(data[c][L]["mark"], data[c][S]["mark"])      # 尺度稳健
                if b > BASIS_TRIG:
                    nb += 1
                    if verbose:
                        print(f"[L3 基差] **{c} {L}-{S} 基差 {b:.1%} >阈→flatten**")
                    actions.append({"trigger": "L3_basis", "coin": c, "basis": round(b, 4)})
        if verbose and nb == 0:
            print("[L3 基差] 所有仓基差 < 5%,正常")

    # ---- 触发 → 生成 reduce-only 平仓计划(第13步:risk_monitor→execution 接线;DRY-RUN)----
    plan = []
    if actions and data is not None:
        try:
            from execution import flatten_orders, load_rules
            rules = load_rules()
            flat_coins = sorted({a["coin"] for a in actions if a["trigger"] in ("L2_pair_stop", "L3_basis")})
            if flat_coins:                                        # L2/L3:平触发的对(全平)
                plan += flatten_orders(flat_coins, st, capital, data, spot, rules, reduce_to=0.0)
            if any(a["trigger"] == "L1_drawdown" for a in actions):   # L1:全书降到 25%
                plan += flatten_orders("ALL", st, capital, data, spot, rules, reduce_to=0.25)
        except Exception as e:
            if verbose:
                print(f"[平仓计划] 生成失败(不影响监控): {e}")
        if verbose and plan:
            print(f"\n[reduce-only 平仓计划] {len(plan)} 条(DRY-RUN,绝不下单):")
            for o in plan[:8]:
                print(f"    {o['venue']:10s} {o['side']:4s} {o['coin']:10s} ${o['notional']:,.0f}  reduce_only")

    json.dump({"ts": ts, "dry_run": True, "actions": actions, "flatten_orders": plan}, open(LOG, "a", encoding="utf-8"))
    open(LOG, "a", encoding="utf-8").write("\n")
    if verbose:
        print(f"{ts} | {'⚠ 有触发(DRY-RUN,未下单)' if actions else '✓ 全部正常'} → {LOG}")
    return actions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", type=int, default=0, help="轮询秒数(0=单次);E56:10-30s 足够,不需亚秒")
    ap.add_argument("--capital", type=float, default=2000.0, help="资本(算平仓名义;默认 2000)")
    a = ap.parse_args()
    if a.loop <= 0:
        run_once(capital=a.capital)
        print("注:DRY-RUN 只验证触发逻辑、绝不下单;验证正确后接 execution.py 的 reduce-only。L2 是解锁跨所 3x 的闸门。")
        return
    print(f"轮询监控每 {a.loop}s(Ctrl-C 停)。DRY-RUN,绝不下单。")
    while True:
        try:
            acts = run_once(verbose=False, capital=a.capital)
            ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
            print(f"{ts} | {'⚠ '+str(len(acts))+' 触发' if acts else '✓ 正常'}")
            time.sleep(a.loop)
        except KeyboardInterrupt:
            print("停止。"); break
        except Exception as e:
            print(f"err: {e}"); time.sleep(a.loop)


if __name__ == "__main__":
    main()
