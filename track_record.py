"""业绩账本(投资级)。把前向验证 log.csv → 可审计的 NAV 曲线 + Sharpe + 回撤 + 分腿归因 + funding 实收。
真单上线后同一本账继续记(execution 把真实成交/funding 也写进同一 log 格式)→ 无缝业绩记录。
- Sharpe:日聚合(诚实,去 funding 近确定性虚高)+ 逐bar 对照。
- NAV:复利权益曲线(可设杠杆),输出 data/paper/nav.csv 供画图。
- 归因:carry(funding/价格/成本)、路由(funding/价格/成本),看 P&L 来源。
- 暖机期(<~7天)折年化是噪声,报表会标注。
跑：PYTHONUTF8=1 .venv/Scripts/python.exe track_record.py [--leverage 1] [--capital 5000]
"""
import argparse
import csv
import json
import os
from datetime import datetime

import numpy as np

LOG = "data/paper/log.csv"; STATE = "data/paper/state.json"; NAV = "data/paper/nav.csv"


def load_log():
    if not os.path.exists(LOG):
        return []
    with open(LOG) as f:
        return list(csv.DictReader(f))


def sharpe(returns, periods_per_year):
    r = np.asarray(returns, float)
    return r.mean() / r.std() * np.sqrt(periods_per_year) if len(r) > 1 and r.std() > 0 else 0.0


def max_dd(nav):
    nav = np.asarray(nav, float)
    if len(nav) < 2:
        return 0.0
    peak = np.maximum.accumulate(nav)
    return float((nav / peak - 1).min())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leverage", type=float, default=1.0)
    ap.add_argument("--capital", type=float, default=5000)
    a = ap.parse_args()
    rows = load_log()
    st = json.load(open(STATE)) if os.path.exists(STATE) else {}
    cm = st.get("cum", {})
    if len(rows) < 1:
        print("还没有前向数据(先让 paper_live.py 跑几次)。"); return

    times = [r["time"] for r in rows]
    dt_h = np.array([float(r["dt_h"]) for r in rows])
    comb = np.array([float(r["comb_pnl"]) for r in rows]) * a.leverage
    carry = np.array([float(r["c_pnl"]) for r in rows]) * a.leverage
    rout = np.array([float(r["x_pnl"]) for r in rows]) * a.leverage
    tot_h = dt_h.sum(); tot_d = tot_h / 24.0

    # 权益曲线(复利)+ 写 NAV csv
    nav = a.capital * np.cumprod(1 + comb)
    with open(NAV, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["time", "nav", "comb_ret", "carry_ret", "rout_ret"])
        for i in range(len(rows)):
            w.writerow([times[i], f"{nav[i]:.2f}", f"{comb[i]:.6f}", f"{carry[i]:.6f}", f"{rout[i]:.6f}"])

    # 日聚合(诚实 Sharpe):按日历日分组累加
    day = [t[:10] for t in times]
    daily = {}
    for d, c in zip(day, comb):
        daily[d] = daily.get(d, 0.0) + c
    dvals = list(daily.values())

    ppy_bar = 8760.0 / max(dt_h.mean(), 1e-9)     # 逐bar 年化因子(按平均 bar 长)
    warm = "  ⚠️暖机期(<7天),折年化/Sharpe 是噪声,仅看框架" if tot_d < 7 else ""

    print(f"{'='*56}")
    print(f" 业绩账本(前向验证)  杠杆 {a.leverage:g}x  名义资本 ${a.capital:,.0f}")
    print(f"{'='*56}")
    print(f" 区间: {times[0]} → {times[-1]}  ({tot_d:.2f} 天, {len(rows)} 观测){warm}")
    print(f" 期末 NAV: ${nav[-1]:,.2f}  (累计 {nav[-1]/a.capital-1:+.2%})")
    print()
    print(f" {'':12s} {'累计':>9s} {'折年化':>9s} {'日Sharpe':>9s} {'逐bar Sh':>9s} {'最大回撤':>9s}")
    for lab, ser in [("carry", carry), ("路由", rout), ("组合", comb)]:
        cum = np.prod(1 + ser) - 1
        ann = cum / tot_d * 365 if tot_d > 0.02 else 0.0
        dser = {}
        for d, c in zip(day, ser):
            dser[d] = dser.get(d, 0.0) + c
        navser = a.capital * np.cumprod(1 + ser)
        print(f" {lab:12s} {cum:>+9.2%} {ann:>+9.1%} {sharpe(list(dser.values()),365):>+9.2f} {sharpe(ser,ppy_bar):>+9.2f} {max_dd(navser):>+9.2%}")

    # 分腿归因(从 state cum,unlevered)
    print(f"\n 分腿归因(累计, unlevered):")
    print(f"   carry: funding {cm.get('c_fund',0):+.5f} | 价格腿 {cm.get('c_price',0):+.5f} | 成本 {cm.get('c_cost',0):.5f}")
    print(f"   路由 : funding {cm.get('x_fund',0):+.5f} | 价格腿 {cm.get('x_price',0):+.5f} | 成本 {cm.get('x_cost',0):.5f}")
    fund_tot = cm.get("c_fund", 0) + cm.get("x_fund", 0)
    fund_ann = fund_tot / tot_d * 365 if tot_d > 0.02 else 0.0
    print(f"\n 核心边验证 — funding 实收(组合两腿): {fund_tot:+.5f}  折年化 {fund_ann:+.1%}")
    print(f"   (这是'边真不真'的核心:funding 是引擎,价格腿应≈0均值,成本是摩擦)")
    print(f"\n 活动: carry {rows[-1]['c_npos']} 仓 / 路由 {rows[-1]['x_npos']} 仓")
    print(f" NAV 曲线已写 → {NAV}  (画图/给投资人看)")
    print(f"\n 诚实备注: 日聚合 Sharpe 比逐bar可信(去 funding 近确定性虚高);需 ~数周才有参考;")
    print(f"   真单上线后 execution 把真实成交写进同格式 log → 同一本账延续。")


if __name__ == "__main__":
    main()
