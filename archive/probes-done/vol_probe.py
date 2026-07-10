"""plan B 探针：我们的数据预测「未来波动 / 未来下行(爆仓)」，能否打赢 naive 基线?
方向预测我们 IC 只 0.016；但报告里 spread→|收益| IC=0.21 暗示引擎在"幅度/风险"上强得多。
这里不靠神经网，先用线性横截面 IC 验证“数据里到底有没有 vol/下行 edge，且超过 naive 趋势波动”。
- 目标①未来已实现波动 fvol = 未来K根收益的std；②未来下行 fdown = 未来K根累计收益的最低点(爆仓proxy)。
- naive 基线 = 过去波动(scale_ret，张量里现成的滚动收益std)。要打赢它才算我们有 edge。
- 候选: OI/ΔOI/资金(basis)/多空比(top/acct/taker)/量。重点看“扣掉过去波动后的残差IC”。
跑：PYTHONUTF8=1 .venv/Scripts/python.exe vol_probe.py --K 24
"""
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--tensor", default="data/clean/crypto_tensor_60min_joint.npz")
ap.add_argument("--K", type=int, default=24, help="未来窗口(bar)")
ap.add_argument("--step", type=int, default=6, help="锚点采样步长")
ap.add_argument("--hold_start", default="2024-01-01")
a = ap.parse_args()

z = np.load(a.tensor, allow_pickle=True)
n, mask, sret, adj, dates = z["n"], z["mask"], z["scale_ret"], z["adj_close"], z["dates"].astype(str)
T, N, C = n.shape
CH = {"n_oi": 8, "n_oichg": 9, "n_lsr_top": 10, "n_lsr_acct": 11, "n_lsr_taker": 12, "n_basis": 13,
      "n_volume": 4, "n_count": 6, "n_tbr": 5, "n_avail": 14}

# 收益
lc = np.log(np.where(adj > 0, adj, np.nan))
r = np.diff(lc, axis=0, prepend=lc[:1])                       # [T,N] 对数收益

hold = np.array([d >= a.hold_start for d in dates])
anchors = [t for t in range(64, T - a.K - 1) if hold[t] and (t % a.step == 0)]
print(f"张量 {n.shape}  holdout 锚点 {len(anchors)}  K={a.K}bar\n")


def xs_ic(feat_t, tgt_t, valid):
    """单bar横截面 Pearson IC。"""
    f, y = feat_t[valid], tgt_t[valid]
    if len(f) < 20 or f.std() < 1e-9 or y.std() < 1e-9:
        return np.nan
    return np.corrcoef(f, y)[0, 1]


def resid_xs(feat_t, base_t, valid):
    """feat 扣掉 base 的横截面残差(对 base 做 OLS 取残差)。"""
    f, b = feat_t[valid].astype(float), base_t[valid].astype(float)
    if b.std() < 1e-9:
        return feat_t, valid
    beta = np.cov(f, b)[0, 1] / (b.var() + 1e-12)
    out = np.full_like(feat_t, np.nan, dtype=float)
    out[valid] = f - beta * b
    return out, valid


# 预计算每锚点的目标 + 特征
recs = {"fvol": [], "fdown": []}
feats = {k: [] for k in list(CH) + ["trail_vol", "abs_ret"]}
feats_resid = {k: [] for k in CH}      # 扣过去波动后的残差IC(对 fvol)
masks = []
for t in anchors:
    fut = r[t + 1:t + 1 + a.K]                              # [K,N]
    fm = mask[t + 1:t + 1 + a.K]
    cnt = fm.sum(0)
    fvol = np.where(cnt >= a.K // 2, np.nanstd(np.where(fm, fut, np.nan), axis=0), np.nan)
    cum = np.nancumsum(np.where(fm, fut, 0.0), axis=0)
    fdown = np.where(cnt >= a.K // 2, cum.min(0), np.nan)   # 未来最深累计跌幅
    valid = mask[t] & np.isfinite(fvol) & np.isfinite(fdown) & (n[t, :, CH["n_avail"]] > 0)
    masks.append(valid)
    recs["fvol"].append(fvol); recs["fdown"].append(fdown)
    feats["trail_vol"].append(sret[t]); feats["abs_ret"].append(np.abs(r[t]))
    for k, ci in CH.items():
        feats[k].append(n[t, :, ci])

# 逐锚点算 IC，汇总
def summarize(tgt_key, feat_dict, resid_base=None):
    print(f"--- 目标 = {tgt_key} ({'未来波动' if tgt_key=='fvol' else '未来最深跌幅(爆仓proxy)'}) ---")
    print(f"{'特征':>12s} {'IC':>8s} {'t值':>7s}  {'残差IC(扣过去波动)':>16s} {'残差t':>7s}")
    for k in feat_dict:
        ics, rics = [], []
        for i, t in enumerate(anchors):
            v = masks[i]
            ics.append(xs_ic(feat_dict[k][i], recs[tgt_key][i], v))
            if resid_base is not None and k in CH:
                fr, vr = resid_xs(feat_dict[k][i], feats["trail_vol"][i], v)
                # 目标也扣过去波动
                yr, _ = resid_xs(recs[tgt_key][i], feats["trail_vol"][i], v)
                rics.append(xs_ic(fr, yr, vr))
        ic = np.nanmean(ics); tv = ic / (np.nanstd(ics) + 1e-12) * np.sqrt(np.sum(np.isfinite(ics)))
        if resid_base is not None and k in CH and len(rics):
            ric = np.nanmean(rics); rt = ric / (np.nanstd(rics) + 1e-12) * np.sqrt(np.sum(np.isfinite(rics)))
            print(f"{k:>12s} {ic:>+8.4f} {tv:>+7.1f}  {ric:>+16.4f} {rt:>+7.1f}")
        else:
            print(f"{k:>12s} {ic:>+8.4f} {tv:>+7.1f}")
    print()


summarize("fvol", feats, resid_base="trail_vol")
summarize("fdown", {k: feats[k] for k in CH}, resid_base="trail_vol")
print("判读：trail_vol 的 IC 是 naive 基线强度；某特征'残差IC'(扣过去波动后)显著=它带了过去波动之外的真 vol/下行 edge → plan B 有料。")
