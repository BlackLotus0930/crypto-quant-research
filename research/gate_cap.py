# -*- coding: utf-8 -*-
"""Q1:把 Gate 敞口压下来,数据变什么样。
Gate 敞口来源:carry 两腿全 Gate(100%);跨所 HL-Gate 对,每仓一半 Gate。
→ Gate 名义占比 = (1 + w_carry)/2(w_carry=carry 资本权重)。降 carry 权重 = 降 Gate。
量:不同 w_carry 下 组合 Sharpe/年化 + Gate 敞口。看能不能到 40%、代价多大。
跑:PYTHONUTF8=1 .venv/Scripts/python.exe research/gate_cap.py
"""
import glob, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import strategy as S
from venues import canon
from research.xvenue_honest import fetch_intervals, honest_legs
ANN = 8760


def main():
    z = np.load("data/clean/crypto_tensor_60min_pit.npz", allow_pickle=True)
    adj = z["adj_close"].astype(float); tdv = z["tdv"].astype(float)
    slots = list(z["slots"].astype(str)); dates = z["dates"].astype(str); T, N = adj.shape
    nmap = {s: i for i, s in enumerate(slots)}
    xv = np.load("data/clean/xvenue_funding.npz", allow_pickle=True)
    bi, yi = fetch_intervals(); biv = np.array([bi.get(s, 8) for s in slots])
    fb = xv["f_bin"] / biv[None, :]
    grid = np.sort(pd.read_parquet("data/clean/crypto_60min_pit.parquet", columns=["ts"])["ts"].unique()).astype(np.int64)
    gdf = pd.DataFrame({"ts": grid}); sc = {canon(s.replace("USDT", "")): nmap[s] for s in slots}
    phl = np.full((T, N), np.nan); fhl = np.full((T, N), np.nan)
    for f in glob.glob("data/raw/hl/kline/*.csv"):
        j = sc.get(canon(os.path.basename(f)[:-4]))
        if j is not None:
            df = pd.read_csv(f).sort_values("ts").drop_duplicates("ts", keep="last")
            phl[:, j] = pd.merge_asof(gdf, df[["ts", "close"]], on="ts", direction="backward", tolerance=3600_000)["close"].to_numpy()
    for f in glob.glob("data/raw/hl/funding/*.csv"):
        j = sc.get(canon(os.path.basename(f)[:-4]))
        if j is not None:
            df = pd.read_csv(f).sort_values("ts").drop_duplicates("ts", keep="last")
            fhl[:, j] = pd.merge_asof(gdf, df[["ts", "funding"]], on="ts", direction="backward", tolerance=3600_001)["funding"].to_numpy()
    both = (adj > 0) & np.isfinite(fb) & np.isfinite(fhl) & np.isfinite(phl) & (phl > 0)
    rows = np.where(both.sum(1) >= 3)[0]; lo, hi = rows.min(), rows.max() + 1
    m = np.zeros(T, bool); m[lo:hi] = True
    d = np.where(both, fb - fhl, 0.0)
    def rets(p): o = np.zeros((T, N)); v = both[:-1] & both[1:]; o[:-1][v] = (p[1:]/p[:-1]-1)[v]; return np.nan_to_num(o)
    xnet = honest_legs(d, rets(adj), rets(phl), both, tdv, np.abs(d), "x", ema_hl=24)["net_h"]
    carry = S.backtest(S.CarryConfig(leverage=1.0))["net"]

    def weekly(net):
        idx = np.where(m)[0]; df = pd.DataFrame({"w": [dates[i][:7]+"-"+str(int(dates[i][8:10])//7) for i in idx], "r": net[idx]})
        return df.groupby("w")["r"].sum().to_numpy()
    def sh(net): wk = weekly(net); return wk.mean()/wk.std()*np.sqrt(52) if wk.std()>0 else 0

    print("=== Q1:压低 Gate 敞口,数据变什么样(真实 HL 窗,周频 Sharpe)===")
    print("Gate 名义占比 = (1 + w_carry)/2 —— 因为跨所每仓必有一条 Gate 腿(HL+Gate 双所结构)\n")
    print(f"{'carry权重':>8s} {'Gate敞口':>7s} {'HL敞口':>6s} | {'组合年化':>8s} {'组合周Sh':>8s} | {'vs 50/50':>9s}")
    base_sh = None
    for w in [0.5, 0.35, 0.2, 0.1, 0.0]:
        comb = w * carry + (1 - w) * xnet
        gate = (1 + w) / 2; hl = (1 - w) / 2
        ann = comb[m].mean() * ANN; s = sh(comb)
        if base_sh is None: base_sh = s; base_ann = ann
        print(f"{w:>7.0%}  {gate:>6.0%} {hl:>5.0%} | {ann:>+7.1%} {s:>8.2f} | Sh{s-base_sh:>+5.2f} 年化{ann-base_ann:>+5.1%}")
    print("\n判读:")
    print("  ① **只用 HL+Gate,Gate 名义占比下不到 50%**(全跨所 w=0 时仍 50%,因每仓一条 Gate 腿)→ 40% 需第三个所。")
    print("  ② 降 carry 权重确实降 Gate(75%→50%),但 carry⊥跨所的分散红利被削 → Sharpe/年化怎么动看上表。")
    print("  ③ 真正的 R2 是**托管**(Gate 暴雷你亏的是 Gate 上的抵押=carry现货+各腿保证金),不是名义。")
    print("     → 名义 75% ≠ 亏 75%;**扫款(R3)把 Gate 闲钱提走** = 不动策略就降托管风险,这才是不换所的解。")


if __name__ == "__main__":
    main()
