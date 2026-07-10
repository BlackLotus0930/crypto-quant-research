# -*- coding: utf-8 -*-
"""Q1 续:15分钟分辨率重测反应窗口——有了分钟数据,止损躲得掉吗?
用 crypto_15min.parquet 的 OHLC(含 high/low,看 bar 内极值)。
每币:从当日开盘起累计极值动,测从'触 22%(止损线)'到'触 31%(强平,L=3)'的 15分钟 bar 数。
窗口=有几个 15min;'同一15min bar内击穿'=连分钟级都救不了(瞬时gap);'触22%后没到31%'=止损成功躲掉。
跑:PYTHONUTF8=1 .venv/Scripts/python.exe research/react_window_15m.py
"""
import numpy as np, pandas as pd
MM = 0.02; L = 3; DLIQ = 1.0 / L - MM; STOP = 0.70 * DLIQ   # 31% / 22%

m = pd.read_parquet("data/clean/crypto_15min.parquet")
m["d"] = pd.to_datetime(m["ts"], unit="ms").dt.strftime("%Y-%m-%d")

print(f"=== 15分钟反应窗口:L={L} 强平距离 {DLIQ:.0%} / 止损线 {STOP:.0%} ===")
print(f"{'崩盘日':>12s} {'触线币':>6s} {'窗口(分钟)中位':>13s} {'最短':>5s} {'同15min内击穿':>12s} {'触22%后躲掉(没到31%)':>18s}")
for cd in ["2021-05-19", "2022-05-11", "2025-10-10"]:
    sub = m[m["d"] == cd]
    windows = []; same_bar = 0; dodged = 0; ntouch = 0
    for sym, g in sub.groupby("symbol"):
        g = g.sort_values("ts")
        o0 = g["open"].iloc[0]
        if not (o0 > 0): continue
        lo = (g["low"].to_numpy() / o0 - 1)       # 下行极值路径
        hi = (g["high"].to_numpy() / o0 - 1)       # 上行极值路径
        for path, sign in [(lo, -1), (hi, +1)]:    # 两个方向(长腿亏 or 短腿亏)
            adv = sign * path                       # 不利方向幅度(正=亏)
            run = np.maximum.accumulate(adv) if sign == -1 else np.maximum.accumulate(adv)
            # 用累计极值(已经是 high/low,无需再 accumulate 方向;直接看何时穿)
            cross_stop = np.where(adv >= STOP)[0]
            cross_liq = np.where(adv >= DLIQ)[0]
            if len(cross_stop) == 0:
                continue
            ntouch += 1
            hs = cross_stop[0]
            if len(cross_liq) == 0:
                dodged += 1                          # 触止损但全天没到强平=躲掉
            else:
                hl = cross_liq[0]
                w = max(hl - hs, 0); windows.append(w)
                if w == 0:
                    same_bar += 1                    # 同一 15min bar 内 22%→31%
            break                                    # 一个币只算先触线的那个方向
    if ntouch == 0:
        print(f"{cd:>12s} {0:>6d} {'无币触线':>13s}"); continue
    med = np.median(windows) * 15 if windows else 0
    print(f"{cd:>12s} {ntouch:>6d} {med:>11.0f}m {(min(windows)*15 if windows else 0):>4.0f}m "
          f"{same_bar}/{len(windows)} ({same_bar/max(len(windows),1):.0%}) {dodged}/{ntouch} ({dodged/ntouch:.0%})")
print("\n判读:")
print("  '窗口分钟中位'≥15-30m → 分钟级监控+执行能从容平=止损可行。")
print("  '同15min内击穿'比例 → 连15min都救不了的瞬时部分(可下1min再细看)。")
print("  '触22%后躲掉'=止损线设 22% 已经让它在没到强平前就该平掉了(这部分纯赚)。")
