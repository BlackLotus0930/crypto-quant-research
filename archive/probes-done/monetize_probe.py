"""plan B 变现 sanity：下行/风险信号能不能直接赚钱(横截面 L/S)，扣成本。
假设:n_oi 高(该币 OI 相对自身异常高=杠杆堆积) → 未来跌得更深(探针 fdown 残差IC -0.13)。
试:做空高 n_oi / 做多低 n_oi(也试 oichg、count、组合),按 cadence 持有，报 holdout 净 Sharpe@2/5/10bps。
正=下行信号可直接变现(plan B 有钱途);负/零=要么不可交易、要么得靠神经高阶版。
跑：PYTHONUTF8=1 .venv/Scripts/python.exe monetize_probe.py
"""
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--tensor", default="data/clean/crypto_tensor_60min_joint.npz")
ap.add_argument("--hold_start", default="2024-01-01")
ap.add_argument("--ann", type=int, default=8760)
a = ap.parse_args()

z = np.load(a.tensor, allow_pickle=True)
n, mask, adj, dates = z["n"], z["mask"], z["adj_close"], z["dates"].astype(str)
T, N, C = n.shape
CH = {"n_oi": 8, "n_oichg": 9, "n_lsr_acct": 11, "n_count": 6, "n_volume": 4, "n_basis": 13, "avail": 14}
lc = np.log(np.where(adj > 0, adj, np.nan))
r = np.diff(lc, axis=0, prepend=lc[:1])
r = np.where(np.isfinite(r), r, 0.0)
hold = np.array([d >= a.hold_start for d in dates])
H = np.where(hold)[0]
H = H[(H > 64) & (H < T - 1)]
print(f"holdout {len(H)} bar, N={N}\n")


def xs_weight(sig_t, valid):
    """横截面去均值→gross 1 的多空权重。"""
    w = np.zeros(N)
    v = valid & np.isfinite(sig_t)
    if v.sum() < 20:
        return w
    s = sig_t[v] - sig_t[v].mean()
    g = np.abs(s).sum()
    if g > 0:
        w[v] = s / g
    return w


def backtest(sig, cad, bps):
    held = np.zeros(N); pnl = []; turn = []
    for i, t in enumerate(H):
        if i % cad == 0:
            valid = mask[t] & (n[t, :, CH["avail"]] > 0)
            tgt = xs_weight(sig[t], valid)
            tn = np.abs(tgt - held).sum(); held = tgt
        else:
            tn = 0.0
        pnl.append((held * r[t + 1]).sum()); turn.append(tn)
    pnl = np.array(pnl); turn = np.array(turn)
    net = pnl - bps / 1e4 * turn; sd = net.std()
    sh = net.mean() / sd * np.sqrt(a.ann) if sd > 0 else 0.0
    return sh, float(turn.mean() * a.ann)


# 候选信号(做空高风险=负号)
sigs = {
    "-n_oi(空高OI堆积)": -n[:, :, CH["n_oi"]],
    "-n_oichg(空OI暴增)": -n[:, :, CH["n_oichg"]],
    "-n_count(空高活跃)": -n[:, :, CH["n_count"]],
    "+n_oi(多高OI)": n[:, :, CH["n_oi"]],
    "-n_oi-n_count(组合)": -(n[:, :, CH["n_oi"]] + n[:, :, CH["n_count"]]),
    "-n_lsr_acct(空拥挤多头)": -n[:, :, CH["n_lsr_acct"]],
}
print(f"{'信号':>22s} {'cad':>4s} {'净@2':>7s} {'净@5':>7s} {'净@10':>7s} {'年换手':>8s}")
print("-" * 64)
for name, sig in sigs.items():
    for cad in (1, 4, 24):
        s2, to = backtest(sig, cad, 2)
        s5, _ = backtest(sig, cad, 5)
        s10, _ = backtest(sig, cad, 10)
        print(f"{name:>22s} {cad:>4d} {s2:>+7.2f} {s5:>+7.2f} {s10:>+7.2f} {to:>8.0f}")
    print()
print("判读: 任一行 净@5≥0.5 = 下行信号可直接变现; 全负/零 = 线性不行,看神经高阶或换变现方式。")
