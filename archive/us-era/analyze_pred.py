"""解剖模型原始输出(零 GPU)：模型到底是不是纯反转？残差有没有 IC？分布(spread)校不校准？信号多快？
用 crypto_pred.npz(模型 signal/spread) + crypto_tensor(算反转+真实收益)。
跑：PYTHONUTF8=1 .venv/Scripts/python.exe analyze_pred.py
"""
import numpy as np

P = np.load("data/clean/crypto_pred.npz", allow_pickle=False)
Z = np.load("data/clean/crypto_tensor_60min.npz", allow_pickle=False)
A = P["anchors"]; SIG = P["signal"]; SPR = P["spread"]; DT = int(P["dt"])
adj = Z["adj_close"].astype(np.float64); mask = Z["mask"]; scale = Z["scale_ret"].astype(np.float64)
T, N = adj.shape
with np.errstate(divide="ignore", invalid="ignore"):
    logc = np.where(adj > 0, np.log(adj), np.nan)


def xs(a, b, valid, rank=False):
    """单行横截面相关(valid 内)。rank=True 用 Spearman。"""
    m = valid & np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return np.nan
    x, y = a[m], b[m]
    if rank:
        x = x.argsort().argsort().astype(float); y = y.argsort().argsort().astype(float)
    if x.std() < 1e-12 or y.std() < 1e-12:
        return np.nan
    return np.corrcoef(x, y)[0, 1]


def resid(sig, factors, valid):
    """sig 对 factors 做横截面 OLS，返回残差(sig 中 factors 解释不了的部分)。"""
    m = valid & np.isfinite(sig) & np.all([np.isfinite(f) for f in factors], axis=0)
    out = np.full_like(sig, np.nan)
    if m.sum() < 20:
        return out
    X = np.column_stack([np.ones(m.sum())] + [f[m] for f in factors])
    y = sig[m]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    out[m] = y - X @ beta
    return out


nvol_ch = Z["n"][:, :, 4].astype(np.float64)                   # n_volume 通道
K_MOM = 120                                                    # 动量回看 120 bar(~5 天)
ic_m, ic_r5, ic_r1, align5, ic_resid, ic_resid_full, spr_vol = [], [], [], [], [], [], []
for i, a in enumerate(A):
    if a - K_MOM < 0 or a + DT >= T:
        continue
    valid = mask[a] & mask[a - 1] & mask[a - 5] & mask[a - 6] & mask[a - K_MOM] & mask[a + DT] & (adj[a] > 0)
    rev1 = -(logc[a] - logc[a - 1])
    rev5 = -(logc[a] - logc[a - 5])
    mom = logc[a] - logc[a - K_MOM]                            # 中期动量(正)
    vol = scale[a]                                            # 因果波动率
    nvol = nvol_ch[a]                                         # 归一成交量
    dvol = nvol_ch[a] - nvol_ch[a - 1]                        # 量变化
    fwd = (logc[a + DT] - logc[a]) / np.where(scale[a] > 0, scale[a], np.nan)   # 归一 dt 收益(模型的 target)
    sig = SIG[i]
    ic_m.append(xs(sig, fwd, valid))
    ic_r5.append(xs(rev5, fwd, valid))
    ic_r1.append(xs(rev1, fwd, valid))
    align5.append(xs(sig, rev5, valid))                       # 模型和反转有多像
    ic_resid.append(xs(resid(sig, [rev1, rev5], valid), fwd, valid))               # ⊥反转
    ic_resid_full.append(xs(resid(sig, [rev1, rev5, mom, vol, nvol, dvol], valid), fwd, valid))  # ⊥全部线性因子
    spr_vol.append(xs(SPR[i], np.abs(fwd), valid))            # spread 能否预测真实波动(校准)


def stat(x):
    x = np.array([v for v in x if np.isfinite(v)])
    return x.mean(), x.mean() / x.std() * np.sqrt(8760) if x.std() > 0 else 0, len(x)


print(f"holdout {len(ic_m)} bars, N={N}, dt={DT}\n")
print(f"{'量':<28s} {'均值IC':>9s} {'年化IR':>8s}")
for name, arr in [("模型 signal → 真实", ic_m), ("反转 rev5 → 真实", ic_r5), ("反转 rev1 → 真实", ic_r1),
                  ("【模型 ⟂ rev1,rev5】→ 真实", ic_resid),
                  ("【模型 ⟂ 全因子(rev/动量/波动/量)】→ 真实", ic_resid_full)]:
    mu, ir, n = stat(arr)
    print(f"{name:<34s} {mu:>+9.4f} {ir:>+8.2f}")
am, _, _ = stat(align5)
print(f"\n模型与反转的横截面相关 corr(signal, rev5) 均值 = {am:+.3f}  (越接近1=模型越是纯反转)")
sv, svir, _ = stat(spr_vol)
print(f"分布 spread 预测真实|收益| 的 IC = {sv:+.4f} (IR {svir:+.1f})  (>0=不确定性是真的、可用于sizing)")

# 信号持续性：相邻 anchor 间，每个币 signal 的 lag-1 自相关
sig_ac = []
for j in range(N):
    s = SIG[:, j]
    if s.std() > 1e-9:
        sig_ac.append(np.corrcoef(s[1:], s[:-1])[0, 1])
print(f"\n信号 bar 间 lag-1 自相关(均值 over 币) = {np.nanmean(sig_ac):.3f}  (低=信号变得快=换手高)")

# 横截面快照：挑 2 根 bar，看模型排最前/最后的币 + 它们真实 dt 收益
print("\n--- 横截面快照(模型 top5/bottom5 vs 真实 dt 收益) ---")
syms = Z["slots"].astype(str)
for i in [len(A) // 3, 2 * len(A) // 3]:
    a = A[i]
    if a + DT >= T:
        continue
    valid = mask[a] & mask[a + DT] & (adj[a] > 0)
    fwd = np.where(valid, (logc[a + DT] - logc[a]), np.nan)
    sig = np.where(valid, SIG[i], np.nan)
    order = np.argsort(np.where(np.isfinite(sig), sig, -1e9))
    top = order[-5:][::-1]; bot = order[:5]
    d = str(Z["dates"][a])
    print(f"  {d}: top5 看多 → 真实%: " + " ".join(f"{syms[k]}:{fwd[k]*100:+.1f}" for k in top))
    print(f"  {' '*len(d)}  bot5 看空 → 真实%: " + " ".join(f"{syms[k]}:{fwd[k]*100:+.1f}" for k in bot))
