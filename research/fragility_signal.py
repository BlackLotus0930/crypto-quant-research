# -*- coding: utf-8 -*-
"""脆弱性信号:测"极端funding+近期波动"能否预测**下一段波动/回撤**(可行?)vs **方向**(应不可测)。
若波动可测、方向不可测 → 建"崩盘风险高→给套利书降杠杆"的前瞻触发(不对称赔付),而非择时做空。
目标用 BTC(市场代理)。信号严格 ≤t 因果,目标 >t。Spearman IC(numpy 实现,无 scipy)。
跑:PYTHONUTF8=1 .venv/Scripts/python.exe research/fragility_signal.py
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

z = np.load("data/clean/crypto_tensor_60min_pit.npz", allow_pickle=True)
adj = z["adj_close"].astype(float); slots = list(z["slots"].astype(str)); dates = z["dates"].astype(str)
nmap = {s: i for i, s in enumerate(slots)}; BTC = nmap["BTCUSDT"]
xv = np.load("data/clean/xvenue_funding.npz", allow_pickle=True)
fbin = xv["f_bin"]                                        # 8h funding rate, per coin per bar
day = np.array([d[:10] for d in dates])
udays = pd.unique(day)

# 每日序列
btc = adj[:, BTC].copy()
hr = np.full(len(btc), np.nan); v = (btc[:-1] > 0) & (btc[1:] > 0); hr[1:][v] = btc[1:][v] / btc[:-1][v] - 1
aggf = np.nanmean(np.where(np.isfinite(fbin), fbin, np.nan), axis=1)   # 截面平均 funding(过度杠杆代理)

drow = {d: i for i, d in enumerate(udays)}
D = len(udays)
d_vol = np.full(D, np.nan); d_ret = np.full(D, np.nan); d_fund = np.full(D, np.nan); d_close = np.full(D, np.nan)
for d in udays:
    idx = np.where(day == d)[0]; i = drow[d]
    h = hr[idx]; h = h[np.isfinite(h)]
    if len(h): d_vol[i] = h.std()
    d_fund[i] = np.nanmean(aggf[idx])
    c = btc[idx]; c = c[c > 0]
    if len(c): d_close[i] = c[-1]
d_ret[1:] = d_close[1:] / d_close[:-1] - 1

def rank(x):
    r = np.full(len(x), np.nan); m = np.isfinite(x); o = np.argsort(np.argsort(x[m])); r[m] = o
    return r
def IC(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 30: return np.nan
    ra, rb = rank(a[m]), rank(b[m]); return np.corrcoef(ra, rb)[0, 1]

# 信号(≤t)与目标(>t)
sig_vol = np.full(D, np.nan); sig_fund = np.full(D, np.nan)
tgt_vol = np.full(D, np.nan); tgt_dd = np.full(D, np.nan); tgt_ret = np.full(D, np.nan)
for i in range(D):
    if i >= 2: sig_vol[i] = np.nanmean(d_vol[i-2:i+1]); sig_fund[i] = np.nanmean(d_fund[i-2:i+1])
    if i + 3 < D:
        tgt_vol[i] = np.nanmean(d_vol[i+1:i+4]); tgt_ret[i] = np.nanprod(1 + np.nan_to_num(d_ret[i+1:i+4])) - 1
    if i + 7 < D:
        fwd = np.cumprod(1 + np.nan_to_num(d_ret[i+1:i+8])); tgt_dd[i] = (fwd / np.maximum.accumulate(fwd) - 1).min()

print("=== 脆弱性信号:能预测波动/回撤吗?能预测方向吗?(BTC,日频,Spearman IC)===\n")
print(f"{'信号→目标':>26s} {'IC':>7s}  判读")
print(f"{'近期波动 → 下3日波动':>26s} {IC(sig_vol, tgt_vol):>+7.2f}  {'(波动聚集,应高)':>0s}")
print(f"{'近期波动 → 下7日最大回撤':>26s} {IC(sig_vol, tgt_dd):>+7.2f}  (高波动→更大回撤?)")
print(f"{'极端funding → 下7日最大回撤':>26s} {IC(sig_fund, tgt_dd):>+7.2f}  (过度杠杆→崩?)")
print(f"{'极端funding → 下3日波动':>26s} {IC(sig_fund, tgt_vol):>+7.2f}")
print(f"\n{'--- 方向(应≈0)---':>26s}")
print(f"{'极端funding → 下3日收益':>26s} {IC(sig_fund, tgt_ret):>+7.2f}  (择时做空?应≈0)")
print(f"{'近期波动 → 下3日收益':>26s} {IC(sig_vol, tgt_ret):>+7.2f}  (应≈0)")

# 崩盘日:崩前信号在历史什么分位?
print("\n=== 崩盘前,信号是否已升高?(崩盘日前1日的信号百分位)===")
def pct(arr, i):
    h = arr[:i]; h = h[np.isfinite(h)]; return (h < arr[i]).mean() if len(h) and np.isfinite(arr[i]) else np.nan
for cd in ["2021-05-19", "2022-05-11", "2025-10-10"]:
    if cd not in drow: continue
    i = drow[cd] - 1                                      # 崩前一日
    if i < 30: continue
    print(f"  {cd}: 崩前 近期波动 {pct(sig_vol,i):.0%}分位 | 极端funding {pct(sig_fund,i):.0%}分位 | 当日跌 {d_ret[drow[cd]]:+.0%}")
print("\n判读:")
print("  波动IC高+方向IC≈0 → **能测崩盘风险(波动),不能测时机(方向)** → 做空择时不成立,降杠杆触发可行。")
print("  崩前信号分位高 → 脆弱性信号能提前预警 → 接 risk_monitor 当前瞻降杠杆(不对称:错=省点funding,对=免强平)。")
