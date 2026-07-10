"""② carry 尾部探查(纯统计,不训模型):肥 funding 的币里,什么入场可见特征区分
"后来benignly付费(安全)"vs"后来逼空/基差爆裂(危险)"?若有清晰分离 → 值得上模型放松帽。
危险 = 持有期(FWD)内基差对我方最大逆向走阔 > 阈。特征全因果(入场时可见)。
跑：python carry_tail_probe.py
"""
import numpy as np

z = np.load("data/clean/crypto_tensor_60min_pit.npz", allow_pickle=True)
mask = z["mask"]; adj = z["adj_close"].astype(float); tdv = z["tdv"].astype(float)
dates = z["dates"].astype(str); slots = z["slots"].astype(str); T, N = mask.shape
funding = np.load("data/clean/funding_pit.npz", allow_pickle=True)["funding"].astype(float)
spot = np.load("data/clean/spot_pit.npz", allow_pickle=True)["spot"].astype(float)
vp = mask & np.isfinite(spot) & (spot > 0) & (adj > 0)
f0 = np.nan_to_num(funding)
basis = np.full((T, N), np.nan); basis[vp] = adj[vp] / spot[vp] - 1
pr = np.zeros((T, N)); pr[1:][vp[1:] & vp[:-1]] = (adj[1:] / adj[:-1] - 1)[vp[1:] & vp[:-1]]
# 因果 EWMA 波动(10天)
vol = np.zeros((T, N)); var = np.zeros(N); dcv = 0.5 ** (1.0 / 240)
for t in range(T):
    var = dcv * var + (1 - dcv) * np.nan_to_num(pr[t]) ** 2
    vol[t] = np.sqrt(var)
age = np.cumsum(vp, axis=0)                                        # 入场前该币累计活跃 bar(币龄代理)
bm = np.zeros((T, N)); bm[48:] = np.nan_to_num(basis[48:] - basis[:-48])
fat = np.percentile(np.abs(f0[vp]), 90)                            # "肥"funding 阈(90 分位)
FWD = 336; DANGER = 0.10                                           # 持有 14 天;逆向走阔 >10% = 危险
print(f"肥 funding 阈(|8h|>{fat:.5f}, 年化 {fat*3*365:.0%}); 危险=未来{FWD//24}天基差逆向走阔>{DANGER:.0%}\n")

rows = []   # (danger, max_adv, fwd_carry, fmag_ann, |basis|, |bm|, vol, log_tdv, age_days, persist)
for c in range(N):
    act = vp[:, c]
    cand = np.where(act & (np.abs(f0[:, c]) > fat))[0]
    cand = cand[(cand > 48) & (cand < T - FWD)][::24]             # 去重叠
    for t in cand:
        s = np.sign(f0[t, c]); fb = basis[t:t + FWD, c]
        fin = np.isfinite(fb)
        if fin.sum() < FWD * 0.5:
            continue
        adv = np.nanmax(s * (fb - basis[t, c]))                   # 最大逆向基差走阔
        basis_end = fb[fin][-1]                                   # 最后一个有效基差
        fwd_carry = np.sum(f0[t:t + FWD, c] / 8.0) * s - (basis_end - basis[t, c]) * s
        persist = (np.abs(f0[max(0, t - 168):t, c]) > fat).mean()
        rows.append((adv > DANGER, adv, fwd_carry, np.abs(f0[t, c]) * 3 * 365, abs(basis[t, c]),
                     abs(bm[t, c]), vol[t, c], np.log10(tdv[t, c] + 1), age[t, c] / 24, persist))
R = np.array(rows); dgr = R[:, 0].astype(bool)
print(f"肥入场样本 {len(R)} 个;危险 {dgr.mean()*100:.0f}% / 安全 {(~dgr).mean()*100:.0f}%")
print(f"危险组 fwd_carry 均值={R[dgr,2].mean():+.3f}  安全组={R[~dgr,2].mean():+.3f}  (危险组该明显更差/更险)\n")

feats = ["fmag_ann", "|basis|", "|basis_mom|", "vol", "log_tdv", "age_days", "persist"]
print(f"{'特征':>12s} {'安全组均值':>10s} {'危险组均值':>10s} {'AUC(辨危险)':>11s}")
for k, name in enumerate(feats):
    x = R[:, 3 + k]; sa, da = np.nanmean(x[~dgr]), np.nanmean(x[dgr])
    # AUC = P(危险的x > 安全的x):rank
    order = np.argsort(np.argsort(x)); auc = order[dgr].mean() / len(x)   # 近似
    auc = (auc - (~dgr).sum() / 2 / len(x)) / dgr.mean() if dgr.mean() > 0 else 0.5
    # 用标准 Mann-Whitney 估
    rank = np.argsort(np.argsort(x)) + 1
    auc = (rank[dgr].mean() - (dgr.sum() + 1) / 2) / (~dgr).sum()
    print(f"{name:>12s} {sa:>10.3f} {da:>10.3f} {auc:>11.2f}")

print("\n=== 简单规则的回报(若只取'安全'肥仓:高流动 + 低基差动量) ===")
liq_med = np.median(R[:, 7]); bm_med = np.median(R[:, 5])
safe_rule = (R[:, 7] > liq_med) & (R[:, 5] < bm_med)              # 流动高于中位 且 基差动量低于中位
print(f"  全部肥仓:    fwd_carry 均值 {R[:,2].mean():+.3f}, 危险率 {dgr.mean()*100:.0f}%, n={len(R)}")
print(f"  规则保留:    fwd_carry 均值 {R[safe_rule,2].mean():+.3f}, 危险率 {dgr[safe_rule].mean()*100:.0f}%, n={safe_rule.sum()}")
print(f"  规则剔除:    fwd_carry 均值 {R[~safe_rule,2].mean():+.3f}, 危险率 {dgr[~safe_rule].mean()*100:.0f}%")
print("\n判读:AUC 离 0.5 越远=该特征越能辨危险;若有特征 AUC>0.6 且'规则保留'回报↑危险率↓ → 尾部可预测,值得上模型放松帽。")
