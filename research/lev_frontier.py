# -*- coding: utf-8 -*-
"""杠杆前沿:carry Lc × 跨所 3x(带止损,50/50 资本)。主判据=复利(几何)年化(有Kelly内部最优)。
跨所固定 3x(硬顶);扫 carry 杠杆。报 算术年化 / 复利年化 / 周Sharpe / maxDD(全+2022) → 选最好。
跑:PYTHONUTF8=1 .venv/Scripts/python.exe research/lev_frontier.py
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import strategy as S
from research.combined_6x3x import load, cross_bt
ANN = 8760

adj, byb, fb, fy, tdv, dates, T, N = load()
carry1 = S.backtest(S.CarryConfig(leverage=1.0))["net"]
x_stop, nfire, nact = cross_bt(adj, byb, fb, fy, tdv, T, N, stop=0.22)   # 跨所 1x 带止损
oos = np.zeros(T, bool); oos[int(T * 0.4):] = True; yr = np.array([s[:4] for s in dates])


def daily(net, m):
    idx = np.where(m)[0]
    return pd.DataFrame({"d": [dates[i][:10] for i in idx], "r": net[idx]}).groupby("d")["r"].sum().to_numpy()


def wk(net, m):
    idx = np.where(m)[0]
    return pd.DataFrame({"w": [dates[i][:7] + "-" + str(int(dates[i][8:10]) // 7) for i in idx], "r": net[idx]}).groupby("w")["r"].sum().to_numpy()


def stats(net):
    d = daily(net, oos); eq = np.cumprod(1 + d)
    comp = eq[-1] ** (365 / len(d)) - 1                       # 复利年化
    arith = net[oos].mean() * ANN
    w = wk(net, oos); sh = w.mean() / w.std() * np.sqrt(52) if w.std() > 0 else 0
    mdd = (eq / np.maximum.accumulate(eq) - 1).min()
    d22 = daily(net, yr == "2022"); eq22 = np.cumprod(1 + d22)
    mdd22 = (eq22 / np.maximum.accumulate(eq22) - 1).min() if len(d22) else 0
    return arith, comp, sh, mdd, mdd22


print(f"跨所止损触发率 {nfire/max(nact,1)*100:.2f}% 持仓bar\n")
print("=== 杠杆前沿:carry Lc × 跨所 3x(50/50,带止损,OOS)===")
print(f"{'carry杠杆':>8s} {'算术年化':>8s} {'复利年化':>8s} {'周Sharpe':>8s} {'全maxDD':>8s} {'2022maxDD':>9s}")
best = None
for Lc in [1, 2, 3, 4, 5, 6, 8]:
    comb = 0.5 * Lc * carry1 + 0.5 * 3 * x_stop
    a, c, sh, mdd, m22 = stats(comb)
    print(f"  carry {Lc}x {a:>+7.1%} {c:>+7.1%} {sh:>8.2f} {mdd:>+7.1%} {m22:>+8.1%}")
    if best is None or c > best[1]:
        best = (Lc, c, a, sh, mdd, m22)
print(f"\n复利年化最高 = carry {best[0]}x(复利 {best[1]:+.1%} / 算术 {best[2]:+.1%} / Sharpe {best[3]:.2f} / maxDD {best[4]:+.1%})")
print("\n判读:复利年化见顶后回落=方差拖累(Kelly)。'最好'=复利接近顶 + maxDD 在桶预算(−60%)内 + Sharpe 不塌。")
