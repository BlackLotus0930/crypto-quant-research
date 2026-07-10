"""一键日报:串起前向业绩(track_record)+ 执行对账(execution)+ 持仓 → 每天一条 data/exec/daily.csv。
内容:NAV / 累计(组合·carry·路由)/ 日Sharpe / 回撤 / 仓数 / 目标vs实际漂移单数 / 毛杠杆 / 审计哈希。
--pull:先跑只读对账拉真实持仓(需 env 凭证);否则用现有 positions.json(无则视空仓=全量建)。
跑（每天一次）：PYTHONUTF8=1 .venv/Scripts/python.exe daily_report.py --capital 5000 [--pull]
"""
import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np

from track_record import load_log, sharpe, max_dd
from execution import build_targets, load_rules, reconcile, POSITIONS, AUDIT
from venues import fetch as fetch_venues

DAILY = "data/exec/daily.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capital", type=float, default=5000)
    ap.add_argument("--leverage", type=float, default=1.0)
    ap.add_argument("--pull", action="store_true", help="先只读拉真实持仓(需 env 凭证)")
    a = ap.parse_args()

    if a.pull:
        subprocess.run([sys.executable, "reconcile_positions.py"])

    # ---- 前向业绩(from log.csv)----
    rows = load_log()
    perf = {}
    if rows:
        comb = np.array([float(r["comb_pnl"]) for r in rows]) * a.leverage
        carry = np.array([float(r["c_pnl"]) for r in rows]) * a.leverage
        rout = np.array([float(r["x_pnl"]) for r in rows]) * a.leverage
        dt_h = np.array([float(r["dt_h"]) for r in rows])
        days = dt_h.sum() / 24.0
        nav = a.capital * float(np.prod(1 + comb))
        day = [r["time"][:10] for r in rows]
        dser = {}
        for d, c in zip(day, comb):
            dser[d] = dser.get(d, 0.0) + c
        navc = a.capital * np.cumprod(1 + comb)
        perf = {"days": round(days, 2), "n_obs": len(rows), "nav": round(nav, 2),
                "comb_cum": round(float(np.prod(1 + comb) - 1), 5),
                "carry_cum": round(float(np.prod(1 + carry) - 1), 5),
                "rout_cum": round(float(np.prod(1 + rout) - 1), 5),
                "daily_sharpe": round(sharpe(list(dser.values()), 365), 2),
                "maxdd": round(max_dd(navc), 4),
                "n_carry": int(rows[-1]["c_npos"]), "n_rout": int(rows[-1]["x_npos"])}

    # ---- 执行对账(target vs 实际)----
    exe = {}
    try:
        st = json.load(open("data/paper/state.json")); data, spot = fetch_venues(); rules = load_rules()
        current = json.load(open(POSITIONS)) if os.path.exists(POSITIONS) else {}
        notion, _, _ = build_targets(st, a.capital, data, spot)
        orders, tgt_q, _ = reconcile(notion, current, data, spot, rules)
        from execution import price_of
        gross = sum(abs(q) * (price_of(v, c, data, spot) or 0) for (v, c), q in tgt_q.items())
        exe = {"n_target": len(tgt_q), "drift_orders": len(orders), "gross_lev": round(gross / a.capital, 2),
               "actual_positions": len(current)}
    except Exception as e:
        exe = {"error": f"{type(e).__name__}"}

    audit_hash = ""
    if os.path.exists(AUDIT):
        lines = open(AUDIT).read().splitlines()
        if lines:
            audit_hash = json.loads(lines[-1]).get("hash", "")

    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    rec = {"date": date, **perf, **exe, "audit_hash": audit_hash}

    os.makedirs("data/exec", exist_ok=True)
    cols = ["date", "days", "n_obs", "nav", "comb_cum", "carry_cum", "rout_cum", "daily_sharpe", "maxdd",
            "n_carry", "n_rout", "n_target", "drift_orders", "gross_lev", "actual_positions", "audit_hash"]
    new = not os.path.exists(DAILY)
    with open(DAILY, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(rec)

    print(f"=== 日报 {date} (杠杆 {a.leverage:g}x, 名义 ${a.capital:,.0f}) ===")
    if perf:
        warm = "  ⚠️暖机" if perf["days"] < 7 else ""
        print(f"  前向 {perf['days']} 天 / {perf['n_obs']} 观测{warm}")
        print(f"  NAV ${perf['nav']:,.2f}  组合累计 {perf['comb_cum']:+.2%}  日Sharpe {perf['daily_sharpe']:+.2f}  回撤 {perf['maxdd']:+.2%}")
        print(f"  分腿累计: carry {perf['carry_cum']:+.2%} / 路由 {perf['rout_cum']:+.2%};仓数 carry {perf['n_carry']} / 路由 {perf['n_rout']}")
    else:
        print("  (前向还没数据)")
    if "error" not in exe:
        print(f"  对账: 目标 {exe['n_target']} 净仓 / 实际 {exe['actual_positions']} 仓 → 漂移 {exe['drift_orders']} 单;毛杠杆 {exe['gross_lev']}x")
    print(f"  审计哈希 {audit_hash or '(无)'}")
    print(f"  → 追加 {DAILY}  (每天一条,给 prop/投资人的时间序列)")


if __name__ == "__main__":
    main()
