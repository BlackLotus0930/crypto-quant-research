"""前向验证状态汇总(多所组合书:carry 正侧 + 多所路由)。跑：.venv\\Scripts\\python.exe paper_status.py"""
import json
import os
import time
from collections import Counter

p = "data/paper/state.json"
if not os.path.exists(p):
    print("还没初始化,先跑 paper_live.py")
    raise SystemExit
s = json.load(open(p))
d = max((time.time() - s["t0"]) / 86400, 1e-6)
cm = s["cum"]
cb = s["carry"].get("cur", {}); xb = s["xv"].get("cur", {})
cneg = sum(1 for v in cb.values() if v < 0)
route = s["xv"].get("route", {})
shorts = Counter(v[1] for v in route.values()); longs = Counter(v[0] for v in route.values())


def a(v):
    return v / d * 365


print(f"前向验证(多所组合书) {d:.2f} 天 / 运行 {s['runs']} 次\n")
print(f"  carry 正侧 腿: 累计 {cm['c_pnl']:+.4f}  折年化 {a(cm['c_pnl']):+.1%}  ({len(cb)}仓, 负仓{cneg})")
print(f"     funding {cm['c_fund']:+.4f} | 价格 {cm['c_price']:+.4f} | 成本 {cm['c_cost']:.4f}")
print(f"  多所路由 腿: 累计 {cm['x_pnl']:+.4f}  折年化 {a(cm['x_pnl']):+.1%}  ({len(xb)}仓)")
print(f"     spread {cm['x_fund']:+.4f} | 价格腿 {cm['x_price']:+.4f} | 成本 {cm['x_cost']:.4f}")
print(f"     short所 {dict(shorts)} | long所 {dict(longs)}")
print(f"\n  组合(50/50): 累计 {cm['comb_pnl']:+.4f}  折年化 {a(cm['comb_pnl']):+.1%}")
print("\n(对账:① carry 折年化 vs backtest 正侧only ~+4-9%(近年);② 多所路由价格腿 haircut+换手是回测测不了的真未知数;")
print(" ③ 组合 Sharpe(从 log.csv)vs 单 carry 看提升。暖机 ~30 天后才有参考;折年化前期是噪声。)")
