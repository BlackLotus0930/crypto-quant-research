"""干净验证"做空拥挤多头(-n_lsr_acct)"信号：2023(val) 选 cadence，应用到 holdout，逐年看。
回答：变现探针里那个 net@5 +0.36 是真稳健,还是 regime 运气 + 频率偷看?
跑：PYTHONUTF8=1 .venv/Scripts/python.exe lsr_validate.py
"""
import numpy as np

z = np.load("data/clean/crypto_tensor_60min_joint.npz", allow_pickle=True)
n, mask, adj, dates = z["n"], z["mask"], z["adj_close"], z["dates"].astype(str)
T, N, C = n.shape
ANN = 8760
LSR, OI, AVAIL = 11, 8, 14
lc = np.log(np.where(adj > 0, adj, np.nan))
r = np.where(np.isfinite(np.diff(lc, axis=0, prepend=lc[:1])), np.diff(lc, axis=0, prepend=lc[:1]), 0.0)
yr = np.array([d[:4] for d in dates])
val = (dates >= "2023-01-01") & (dates < "2024-01-01")     # val=2023
sig = -n[:, :, LSR]                                          # 空拥挤多头


def run(idx, cad, bps):
    held = np.zeros(N); pnl = []; turn = []
    idx = idx[(idx > 64) & (idx < T - 1)]
    for i, t in enumerate(idx):
        if i % cad == 0:
            v = mask[t] & (n[t, :, AVAIL] > 0) & np.isfinite(sig[t])
            w = np.zeros(N)
            if v.sum() >= 20:
                s = sig[t][v] - sig[t][v].mean(); g = np.abs(s).sum()
                if g > 0:
                    w[v] = s / g
            tn = np.abs(w - held).sum(); held = w
        else:
            tn = 0.0
        pnl.append((held * r[t + 1]).sum()); turn.append(tn)
    pnl = np.array(pnl); turn = np.array(turn); net = pnl - bps / 1e4 * turn; sd = net.std()
    return (net.mean() / sd * np.sqrt(ANN) if sd > 0 else 0.0, float(turn.mean() * ANN))


# 1) val(2023) 选 cadence(按 net@5)
vidx = np.where(val)[0]
best = max([(run(vidx, c, 5)[0], c) for c in (1, 2, 4, 6, 12, 24, 48)])
sv, cad = best
print(f"val(2023) 选定 cadence={cad}  (val 净@5={sv:+.2f})\n")

# 2) 应用到 holdout 整体 + 逐年
print(f"{'区间':>10s} {'净@2':>7s} {'净@5':>7s} {'净@10':>7s} {'年换手':>8s}")
hidx = np.where(dates >= "2024-01-01")[0]
s2, to = run(hidx, cad, 2); s5, _ = run(hidx, cad, 5); s10, _ = run(hidx, cad, 10)
print(f"{'holdout全':>10s} {s2:>+7.2f} {s5:>+7.2f} {s10:>+7.2f} {to:>8.0f}")
for y in ("2024", "2025", "2026"):
    yi = np.where(yr == y)[0]
    if len(yi) > 100:
        a2, _ = run(yi, cad, 2); a5, _ = run(yi, cad, 5); a10, _ = run(yi, cad, 10)
        print(f"{y:>10s} {a2:>+7.2f} {a5:>+7.2f} {a10:>+7.2f}")
print("\n判读: holdout 全程 + 每年都正(尤其 net@5) = 真稳健慢信号,值得建神经高阶版; 某年大负 = regime 依赖,谨慎。")
