# -*- coding: utf-8 -*-
"""carry 独立画像 + 重杠杆经济性(用户问:carry 年化/Sharpe?重杠杆 carry 而非跨所?)
三块:
 1) carry 独立 年化/Sharpe(频率阶梯,多年)。
 2) carry 强平安全性:同所 现货+永续 cross-margin,价格冲击账户内抵消 → 只有**基差(现货-永续)**能爆。
    量崩盘日 同所基差跳幅 → carry 的强平距离 → 能上多少杠杆(对比跨所 2x 硬顶)。
 3) 重杠杆 carry 经济性:**杠杆要借 USDT 买现货** → net(L)=L×(funding+price)−(L−1)×borrow。
    funding 才 ~4.7%、借币 ~8% → 加杠杆是否还赚?
跑:PYTHONUTF8=1 .venv/Scripts/python.exe research/carry_lever.py
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import strategy as S
ANN = 8760


def main():
    cfg = S.CarryConfig(leverage=1.0)
    a = S.load_arrays(cfg=cfg)
    T, N, dates = a["T"], a["N"], a["dates"]
    res = S.backtest(cfg, arrays=a)
    held = res["held"]; net = res["net"]

    # 分解 funding-leg vs price-leg(重建 sr-pr)
    z = np.load("data/clean/crypto_tensor_60min_pit.npz", allow_pickle=True)
    adj = z["adj_close"].astype(float)
    sp = np.load("data/clean/spot_pit.npz", allow_pickle=True)["spot"].astype(float)
    vp = a["vp"]; f0 = a["f0"]
    pr = np.zeros((T, N)); sr = np.zeros((T, N)); vpp = vp[:-1] & vp[1:]
    pr[:-1][vpp] = adj[1:][vpp] / adj[:-1][vpp] - 1
    sr[:-1][vpp] = sp[1:][vpp] / sp[:-1][vpp] - 1
    fund_leg = (held * (f0 / 8.0)).sum(1)
    price_leg = (held * np.clip(np.nan_to_num(sr - pr), -0.25, 0.25)).sum(1)

    yr = np.array([d[:4] for d in dates]); oos = np.zeros(T, bool); oos[int(T*0.4):] = True
    def sh(p, m, ann): p = p[m]; return p.mean()/p.std()*np.sqrt(ann) if p.std()>0 else 0
    def daily(p, m): idx=np.where(m)[0]; df=pd.DataFrame({"d":[dates[i][:10] for i in idx],"r":p[idx]}); return df.groupby("d")["r"].sum().to_numpy()
    def weekly(p, m): idx=np.where(m)[0]; df=pd.DataFrame({"w":[dates[i][:7]+"-"+str(int(dates[i][8:10])//7) for i in idx],"r":p[idx]}); return df.groupby("w")["r"].sum().to_numpy()

    dd = daily(net, oos); wk = weekly(net, oos)
    print("=== 1) carry 独立 年化/Sharpe(正侧only,1x,多年)===")
    print(f"  OOS 年化 {net[oos].mean()*ANN:+.1%} | per-bar Sh {sh(net,oos,ANN):.2f} | 日Sh {dd.mean()/dd.std()*np.sqrt(365):.2f} | 周Sh {wk.mean()/wk.std()*np.sqrt(52):.2f}")
    print(f"  分解(年化): funding腿 {fund_leg[oos].mean()*ANN:+.1%} | 价格腿 {price_leg[oos].mean()*ANN:+.1%}")
    print(f"  逐年: " + " ".join(f"{y}:{net[yr==y].mean()*ANN:+.0%}" for y in ('2021','2022','2023','2024','2025','2026') if (yr==y).sum()>200))
    print(f"  → carry 是**弱腿**(funding 才 ~{fund_leg[oos].mean()*ANN:.0%},价格腿吃掉一部分);Sharpe 远低于组合 6.2。\n")

    print("=== 2) carry 强平安全性:同所基差(现货-永续)崩盘日跳幅 ===")
    basis_chg = np.abs(np.nan_to_num(sr - pr))      # 同所 现货vs永续 单bar背离=carry唯一爆仓源
    day = np.array([d[:10] for d in dates])
    for d in ["2021-05-19", "2022-05-11", "2025-10-10"]:
        idx = np.where(day == d)[0]
        if len(idx) < 6: continue
        held_d = held[idx[0]] != 0
        bc = basis_chg[idx][:, held_d]; bc = bc[np.isfinite(bc)]
        if len(bc)==0: continue
        print(f"  {d}: 同所基差跳 中位 {np.median(bc):.2%} / p99 {np.percentile(bc,99):.1%} / max {bc.max():.1%} → 强平距离1/L需>此")
    print(f"  全样本同所基差跳 p99.9 = {np.percentile(basis_chg[np.isfinite(basis_chg)&(basis_chg>0)],99.9):.1%}, max {np.nanmax(basis_chg):.0%}")
    print("  → 同所现货/永续贴得很紧(基差跳~1-3%),不像跨所(价格腿+孤儿腿)。**carry 强平角度可上很高杠杆。**\n")

    print("=== 3) 重杠杆 carry 经济性:杠杆借 USDT 买现货 ===")
    fp = (fund_leg + price_leg)[oos].mean() * ANN          # 1x 的 funding+price 年化
    cost1x = fp - net[oos].mean()*ANN                       # 交易成本年化(≈)
    print(f"  1x: funding+价格 {fp:+.1%}, 扣成本后 net {net[oos].mean()*ANN:+.1%}")
    print(f"  net(L) = L×(funding+价格) − (L−1)×借币 − L×成本\n")
    print(f"  {'借币率':>6s} | " + " ".join(f"{('L'+str(L)):>7s}" for L in [1,2,3,6,10]))
    for borrow in [0.05, 0.08, 0.12]:
        row = []
        for L in [1,2,3,6,10]:
            r = L*fp - (L-1)*borrow - L*cost1x
            row.append(f"{r:+.0%}")
        print(f"  {borrow:>5.0%} | " + " ".join(f"{x:>7s}" for x in row))
    print("\n判读:")
    print("  ① carry 强平很安全(同所基差紧)→ 不像跨所被 2x 卡。")
    print("  ② 但 carry funding 太薄(~4.7%),借币 ~8% → **加杠杆借的钱比赚的funding还贵→越加越亏**。")
    print("  ③ 只有当某币 funding > 借币时,杠杆才accretive(肥尾正funding币,但那是薄/波动币,基差也更跳)。")
    print("  → 结论:carry 能扛杠杆但**不值得重杠杆**(薄edge被借币吃光);跨所edge肥但被强平卡2x。")
    print("    这正是为什么要**组合**:carry给稳+分散,跨所给肥edge,各自的短板互补。")


if __name__ == "__main__":
    main()
