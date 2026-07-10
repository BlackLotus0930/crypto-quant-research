"""通用正交残差探针（无 GPU，几分钟）——实现 设计.md §2.5 数据获取漏斗。
候选数据流剔掉标准线性因子(反转/动量/波动/量)后，**残差还有没有横截面预测力** = 它正不正交、值不值得喂模型。
判据：原始 IC 强 ≠ 有用(E11 订单流教训)；要看 **残差 IC** + 它和各因子的相关(冗余度)。
候选来源：① 内置(从 panel 价量算 imb/count/avgsz 等) ② 外部 (symbol,ts,value) parquet(如 funding/OI/链上)。
跑：PYTHONUTF8=1 .venv/Scripts/python.exe probe_orthogonal.py --candidate imb --freq 60
"""
import argparse
import numpy as np
import pandas as pd

K_MOM = 120 // 1                                           # 动量回看(bar)，60min→120bar≈5天


def _xs(fac, fwd, valid):
    """单 bar 横截面 Spearman IC。"""
    m = valid & np.isfinite(fac) & np.isfinite(fwd)
    if m.sum() < 10:
        return np.nan
    x, y = fac[m], fwd[m]
    if x.std() < 1e-12 or y.std() < 1e-12:
        return np.nan
    xr = x.argsort().argsort().astype(float); yr = y.argsort().argsort().astype(float)
    return np.corrcoef(xr, yr)[0, 1]


def _resid(cand, facs, valid):
    """单 bar：cand 对 facs 做横截面 OLS，返回残差(cand 中因子解释不了的部分)。"""
    m = valid & np.isfinite(cand) & np.all([np.isfinite(f) for f in facs], axis=0)
    out = np.full_like(cand, np.nan)
    if m.sum() < 20:
        return out
    X = np.column_stack([np.ones(m.sum())] + [f[m] for f in facs])
    beta, *_ = np.linalg.lstsq(X, cand[m], rcond=None)
    out[m] = cand[m] - X @ beta
    return out


def _stat(arr):
    a = np.array([v for v in arr if np.isfinite(v)])
    return (float(a.mean()), float(a.mean() / a.std() * np.sqrt(len(a)))) if len(a) > 1 and a.std() > 0 else (0.0, 0.0)


def main(a):
    panel = pd.read_parquet(f"data/clean/crypto_{a.freq}min.parquet")
    if a.cand_parquet:                                     # 外部候选(symbol,ts,value)→并到 panel
        ext = pd.read_parquet(a.cand_parquet)
        panel = panel.merge(ext, on=["symbol", "ts"], how="left")
        cand_col = [c for c in ext.columns if c not in ("symbol", "ts")][0]
    else:
        cand_col = None
    ts = np.sort(panel["ts"].unique()); tmap = {x: i for i, x in enumerate(ts)}
    syms = sorted(panel["symbol"].unique()); smap = {s: i for i, s in enumerate(syms)}
    T, N = len(ts), len(syms)
    gt = panel["ts"].map(tmap).to_numpy(); gc = panel["symbol"].map(smap).to_numpy()

    def G(series):
        g = np.full((T, N), np.nan, np.float32); g[gt, gc] = series.to_numpy(np.float32); return g

    close = G(panel["close"]); vol = G(panel["volume"])
    with np.errstate(all="ignore"):
        logc = np.where(close > 0, np.log(close), np.nan).astype(np.float32)
        logv = np.log1p(np.where(vol > 0, vol, 0)).astype(np.float32)
    # 内置候选
    if cand_col is None:
        if a.candidate == "imb":
            cand = G((2 * panel["tbv"] - panel["volume"]) / panel["volume"].clip(lower=1e-9))
        elif a.candidate == "count":
            cand = G(np.log1p(panel["count"]))
        elif a.candidate == "avgsz":
            cand = G(np.log((panel["qv"] / panel["count"].clip(lower=1)).clip(lower=1e-9)))
        else:
            raise SystemExit(f"未知内置候选 {a.candidate}（或用 --cand_parquet）")
        cname = a.candidate
    else:
        cand = G(panel[cand_col]); cname = cand_col

    print(f"freq={a.freq}min  T={T:,} × N={N}  候选={cname}  时段 {pd.to_datetime(ts[0],unit='ms')}~{pd.to_datetime(ts[-1],unit='ms')}\n", flush=True)

    # 标准线性因子(时序，per-coin)
    def shift_diff(x, k):
        out = np.full_like(x, np.nan); out[k:] = x[k:] - x[:-k]; return out
    rev1 = -shift_diff(logc, 1); rev5 = -shift_diff(logc, 5)
    mom = shift_diff(logc, K_MOM)
    volf = np.full_like(logc, np.nan)                       # 因果波动率(过去20bar 收益 std)
    ret1 = shift_diff(logc, 1)
    for t in range(20, T):
        volf[t] = np.nanstd(ret1[t - 20:t], axis=0)
    nvol = logv; dvol = shift_diff(logv, 1)
    FAC = [rev1, rev5, mom, volf, nvol, dvol]

    idx = np.unique(np.linspace(K_MOM, T - max(a.horizons) - 1, min(a.nsamp, T - K_MOM - max(a.horizons))).astype(int))
    print(f"{'horizon':>8s} {'原始IC':>9s} {'原始t':>7s} | {'残差IC(⟂因子)':>12s} {'残差t':>7s} | {'novelty占比':>9s} | {'max|corr因子|':>11s}")
    fac_names = ["rev1", "rev5", "mom", "vol", "nvol", "dvol"]
    for h in a.horizons:
        fwd = np.full_like(logc, np.nan); fwd[:-h] = logc[h:] - logc[:-h]
        raw_ic, res_ic, corrs = [], [], {n: [] for n in fac_names}
        for t in idx:
            v = np.isfinite(close[t]) & (close[t] > 0)
            raw_ic.append(_xs(cand[t], fwd[t], v))
            res_ic.append(_xs(_resid(cand[t], [f[t] for f in FAC], v), fwd[t], v))
            for n, f in zip(fac_names, FAC):
                corrs[n].append(_xs(cand[t], f[t], v))
        ric_m, ric_t = _stat(raw_ic); res_m, res_t = _stat(res_ic)
        nov = (res_m / ric_m) if abs(ric_m) > 1e-9 else 0.0
        mx = max((abs(_stat(corrs[n])[0]), n) for n in fac_names)
        print(f"{h:>6d}b {ric_m:>+9.4f} {ric_t:>+7.1f} | {res_m:>+12.4f} {res_t:>+7.1f} | {nov:>+8.0%} | {mx[1]}={mx[0]:>+.2f}")
    print("\n判读：残差IC t≥3 且 novelty 占比高 = 正交、值得喂模型；残差IC≈0 = 冗余(被某因子吸收，看 max|corr|)，丢。", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--freq", type=int, default=60)
    ap.add_argument("--candidate", default="imb", help="内置候选: imb/count/avgsz")
    ap.add_argument("--cand_parquet", default=None, help="外部候选 parquet(symbol,ts,<value>)，如 funding/OI")
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 6, 12, 24], help="预测 horizon(bar)")
    ap.add_argument("--nsamp", type=int, default=20000)
    main(ap.parse_args())
