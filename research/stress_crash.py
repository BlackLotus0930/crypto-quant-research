# -*- coding: utf-8 -*-
"""群体爆压力测试:模拟历史崩盘日在 Lx 下跨所书的真实损益(含强平 + 孤儿腿)。
回测假设"扛住不被平";本脚本不假设——逐小时跑保证金,某腿不利累计 > Δ_liq=1/L−mm 就强平,
对冲断裂→survivor 变裸头寸。两情景:即时平孤儿(best)/ 扛到收盘(realistic-bad,吃反转)。
崩盘日(台账 liq_impact):2020-03-12 COVID / 2021-05-19 中国禁 / 2022-05-11 LUNA / 2025-10-10。
Bin-Byb 全历史(CEX-CEX);2025-10-10 另跑 Bin-HL(真实 DEX 崩盘基差,最坏)。
跑:PYTHONUTF8=1 .venv/Scripts/python.exe research/stress_crash.py
"""
import glob, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from venues import canon
from research.xvenue_honest import fetch_intervals
from strategy import _cap_renorm
MM = 0.02   # 维持保证金率(薄肥尾币)

z = np.load("data/clean/crypto_tensor_60min_pit.npz", allow_pickle=True)
adj = (z["adj"] if "adj" in z else z["adj_close"]).astype(float)
slots = list(z["slots"].astype(str)); dates = z["dates"].astype(str); T, N = adj.shape
nmap = {s: i for i, s in enumerate(slots)}
valid = z["mask"] & (adj > 0)
xv = np.load("data/clean/xvenue_funding.npz", allow_pickle=True)
bi, yi = fetch_intervals()
biv = np.array([bi.get(s, 8) for s in slots]); yiv = np.array([yi.get(s, 8) for s in slots])
fb = xv["f_bin"] / biv[None, :]; fy = xv["f_byb"] / yiv[None, :]
grid = np.sort(pd.read_parquet("data/clean/crypto_60min_pit.parquet", columns=["ts"])["ts"].unique()).astype(np.int64)
gdf = pd.DataFrame({"ts": grid})
def klines(pat, key, col):
    out = np.full((T, N), np.nan)
    for f in glob.glob(pat):
        j = key(os.path.basename(f)[:-4])
        if j is None: continue
        df = pd.read_csv(f).sort_values("ts").drop_duplicates("ts", keep="last")
        if df.empty: continue
        m = pd.merge_asof(gdf, df[["ts", col]], on="ts", direction="backward", tolerance=3600_001)
        out[:, j] = m[col].to_numpy()
    return out
byb = klines("data/raw/bybit/kline/*.csv", lambda s: nmap.get(s), "close")
sc = {canon(s.replace("USDT", "")): nmap[s] for s in slots}
phl = klines("data/raw/hl/kline/*.csv", lambda s: sc.get(canon(s)), "close")
fhl = klines("data/raw/hl/funding/*.csv", lambda s: sc.get(canon(s)), "funding")
day = np.array([d[:10] for d in dates])


def sim_day(d, pA, pB, fA, fB, lab):
    """pA/pB=两所价(T×N);fA/fB=两所每小时funding;在崩盘日 d 上跑书。"""
    idx = np.where(day == d)[0]
    if len(idx) < 6: print(f"  [{lab}] {d}: 数据不足"); return
    t0 = idx[0]; hrs = idx
    # 书:当日有效两所的币,权重 ∝ spread^2(cap5%),方向 = short 高funding 所
    act = np.where(valid[t0] & np.isfinite(pA[t0]) & np.isfinite(pB[t0]) & (pA[t0] > 0) & (pB[t0] > 0)
                   & np.isfinite(fA[t0]) & np.isfinite(fB[t0]))[0]
    if len(act) < 5: print(f"  [{lab}] {d}: 当日两所重叠币<5"); return
    spr = np.abs(fA[t0, act] - fB[t0, act]); w = _cap_renorm(np.power(spr, 2) / max(np.power(spr, 2).sum(), 1e-12), 0.05)
    sgnA = np.sign(fA[t0, act] - fB[t0, act])  # >0: short A long B
    # 价路径(从 t0 起的累计收益)
    cA = pA[hrs][:, act] / pA[t0, act] - 1     # H×n
    cB = pB[hrs][:, act] / pB[t0, act] - 1
    cA = np.nan_to_num(cA, nan=0.0); cB = np.nan_to_num(cB, nan=0.0)
    basisgap = np.nanmax(np.abs(cA - cB), 0)   # 当日两所最大背离(未对冲损失源)
    print(f"  [{lab}] {d}: {len(act)}币 | 截面中位|日动| {np.nanmedian(np.abs(cA[-1])):.0%} | 基差背离 中位{np.median(basisgap):.1%}/最大{basisgap.max():.0%}")
    print(f"      {'L':>3s} {'被强平腿%':>8s} {'扛得住时基差损':>12s} {'被迫平仓最坏(全反转)':>18s}")
    for L in [1, 2, 3, 5, 10]:
        dliq = 1.0 / L - MM
        liq_w = 0.0; hold_loss = 0.0
        for k in range(len(act)):
            s = sgnA[k]
            legA = -s * cA[:, k]   # short A(s>0)P&L 路径
            legB = s * cB[:, k]    # long B(s>0)
            liq = (legA.min() < -dliq) or (legB.min() < -dliq)
            if liq:
                liq_w += w[k]                                       # 被强平 → 最坏=全反转,死腿丢保证金,survivor 来回归零
            else:
                hold_loss += w[k] * L * (legA[-1] + legB[-1])       # 没被平 → 只吃当日基差(EOD)
        # 被迫平仓最坏:被平的币每个丢 ~保证金(1−mm·L);没被平的吃基差
        forced_worst = -liq_w * (1 - MM * L) + hold_loss
        print(f"      {L:>3d} {liq_w/max(w.sum(),1e-9):>7.0%} {hold_loss:>+11.1%} {forced_worst:>+17.1%}")


print("=== 群体爆压力测试(跨所书,真实强平模型,mm=2%)===\n")
print("# Bin-Byb(CEX-CEX,全历史)")
for d in ["2020-03-12", "2021-05-19", "2022-05-11", "2025-10-10"]:
    sim_day(d, adj, byb, fb, fy, "Bin-Byb")
print("\n# Bin-HL(真实 DEX 崩盘基差,仅 2025-10-10 有 HL 数据)")
sim_day("2025-10-10", adj, phl, fb, fhl, "Bin-HL")
print("""
判读:
- 「扛得住时基差损」=没被强平时只吃当日两所基差(对冲有效)→ 崩盘日近 0,说明 CEX-CEX 基差在崩盘中仍紧。
- 「被迫平仓最坏(全反转)」=被强平的腿当成全损保证金 + survivor 来回归零(脉冲反转的最坏情形)。
- **决策看「被强平腿%」**:它从哪个 L 开始飙升,那个 L 之上崩盘日就会被群体强平,最坏损失=被平比例。
- 真实数据里崩盘日单向(没当日反转)其实常能骑赢家腿赚钱;但你不能赌它单向→用「全反转最坏」定杠杆。
""")
