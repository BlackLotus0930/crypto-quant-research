"""验 E7b 张量：① tod 通道正确 ② 去季节化真把日内 U 型抹平（对比旧张量）。"""
import numpy as np

OLD = "data/clean/intraday_tensor_60min.npz"
NEW = "data/clean/intraday_tensor_60min_tod.npz"


def slot_retvol(z):
    """每个 slot 上、可交易格子的"归一次bar收益"的 std。去季节化后应≈1 且跨 slot 平。"""
    ac = z["adj_close"].astype(np.float64); m = z["mask"]; sr = z["scale_ret"]; tod = z["tod"]
    nxt = np.roll(ac, -1, axis=0)
    valid = m & np.roll(m, -1, axis=0) & (ac > 0) & (nxt > 0) & (sr > 0)
    valid[-1] = False
    with np.errstate(all="ignore"):
        yn = np.where(valid, np.log(nxt / ac) / np.where(sr > 0, sr, 1), np.nan)
    out = {}
    for s in np.unique(tod):
        rows = np.where(tod == s)[0]
        v = yn[rows][valid[rows]]
        out[int(s)] = float(np.nanstd(v)) if v.size else float("nan")
    return out


def main():
    new = np.load(NEW, allow_pickle=False)
    print(f"NEW: n.shape={new['n'].shape}  mask真实率={new['mask'].mean():.3f}  C={new['n'].shape[-1]}")

    # ① tod 通道：对每个唯一 slot，通道5/6 应 = sin/cos(2π·slot/bpd)
    bpd = int(new["bars_per_day"]); tod = new["tod"]; n = new["n"]; m = new["mask"]
    ok = True
    for s in np.unique(tod):
        t0 = np.where(tod == s)[0][0]
        cells = np.where(m[t0])[0]
        if cells.size == 0:
            continue
        exp_sin, exp_cos = np.sin(2 * np.pi * s / bpd), np.cos(2 * np.pi * s / bpd)
        got_sin = n[t0, cells, 5].mean(); got_cos = n[t0, cells, 6].mean()
        if abs(got_sin - exp_sin) > 1e-4 or abs(got_cos - exp_cos) > 1e-4:
            ok = False
            print(f"  ✗ slot={s}: sin {got_sin:.4f}≠{exp_sin:.4f} 或 cos {got_cos:.4f}≠{exp_cos:.4f}")
    print(f"① tod 通道 sin/cos: {'✓ 全部正确' if ok else '✗ 有错'}")

    # ② 成交量去季节化：per-slot n_volume 均值。旧=池化→U 型(开/收槽偏高)；新=per-slot→各槽≈0
    def slot_volmean(z):
        nv = z["n"][..., 4]; m = z["mask"]; t = z["tod"]
        return {int(s): float(nv[np.where(t == s)[0]][m[np.where(t == s)[0]]].mean())
                for s in np.unique(t)}
    try:
        old = np.load(OLD, allow_pickle=False)
        vo = slot_volmean(old)
        print(f"② 旧 per-slot n_volume 均值（池化→见 U 型偏移）: " + " ".join(f"{k}:{v:+.2f}" for k, v in sorted(vo.items())))
        print(f"   旧 跨slot |均值| 平均 = {np.mean([abs(v) for v in vo.values()]):.3f}")
    except FileNotFoundError:
        print("② 旧张量不在，跳过对比")
    vn = slot_volmean(new)
    print(f"② 新 per-slot n_volume 均值（去季节化→各槽≈0）: " + " ".join(f"{k}:{v:+.2f}" for k, v in sorted(vn.items())))
    print(f"   新 跨slot |均值| 平均 = {np.mean([abs(v) for v in vn.values()]):.3f}  (越接近0=季节性除得越干净)")


if __name__ == "__main__":
    main()
