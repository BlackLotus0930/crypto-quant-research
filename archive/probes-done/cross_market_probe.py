"""跨市场探针（无 GPU）：US 宏观状态 vs 加密。设计.md §2.5 漏斗的跨市场版。
全局宏观变量无横截面变化 → 对横截面模型的价值在【方向 / regime 调制 / coin-beta 交互】，
线性横截面探针天然盲 → 这里测两件 sanity，真正的交互价值靠把 US 资产行喂进模型让 attention 找：
  A 方向：宏观因子[t] → 加密市场聚合收益[t+1..h] 时序 IC（book 中性，方向价值有限，仅 sanity）。
  B regime：加密横截面反转 IC 在不同宏观状态(SPY 涨跌 / VIX 高低)下是否变 → 宏观是否调制我们交易的信号。
US 60min(session, ts=ns UTC) 前向填到加密 24/7 UTC 网格。
跑：PYTHONUTF8=1 .venv/Scripts/python.exe cross_market_probe.py
"""
import numpy as np
import pandas as pd

MAC = ["SPY", "VIXY", "UUP", "GLD", "TLT", "QQQ", "HYG"]


def _xs(a, b, v):
    m = v & np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return np.nan
    x, y = a[m], b[m]
    if x.std() < 1e-12 or y.std() < 1e-12:
        return np.nan
    xr = x.argsort().argsort().astype(float); yr = y.argsort().argsort().astype(float)
    return np.corrcoef(xr, yr)[0, 1]


def _ts_ic(fac, fwd):
    m = np.isfinite(fac) & np.isfinite(fwd)
    if m.sum() < 100 or fac[m].std() < 1e-12 or fwd[m].std() < 1e-12:
        return np.nan, 0
    return float(np.corrcoef(fac[m], fwd[m])[0, 1]), int(m.sum())


def main():
    cp = pd.read_parquet("data/clean/crypto_60min.parquet")
    cts = np.sort(cp["ts"].unique()); ctmap = {x: i for i, x in enumerate(cts)}
    csyms = sorted(cp["symbol"].unique()); csmap = {s: i for i, s in enumerate(csyms)}
    T, N = len(cts), len(csyms)
    close = np.full((T, N), np.nan, np.float32)
    close[cp["ts"].map(ctmap).to_numpy(), cp["symbol"].map(csmap).to_numpy()] = cp["close"].to_numpy(np.float32)
    with np.errstate(all="ignore"):
        logc = np.where(close > 0, np.log(close), np.nan)
    ret1 = np.full_like(logc, np.nan); ret1[1:] = logc[1:] - logc[:-1]      # 当根收益 [T,N]
    rev1 = -ret1
    mkt = np.nanmean(ret1, axis=1)                                          # 加密市场聚合收益 [T]
    print(f"加密网格 T={T:,} × N={N}  {pd.to_datetime(cts[0],unit='ms')}~{pd.to_datetime(cts[-1],unit='ms')}", flush=True)

    # US 宏观 → 前向填到加密网格
    d = pd.read_parquet("data/clean/intraday_60min.parquet")
    mac = d[d["ticker"].isin(MAC)][["ticker", "ts", "close"]].copy()
    mac["ms"] = (mac["ts"] // 1_000_000).astype(np.int64)                   # ns → ms
    macp = mac.pivot_table(index="ms", columns="ticker", values="close").sort_index()
    aligned = macp.reindex(cts, method="ffill")                            # 前向填到加密每个 UTC 小时
    cov = aligned.notna().mean()
    print(f"宏观对齐覆盖率(前向填后非空占比): " + " ".join(f"{t}={cov.get(t,0):.2f}" for t in MAC), flush=True)

    def gret(tk, k=1):                                                      # 宏观 k-bar log 收益(网格上)
        v = aligned[tk].to_numpy(np.float64) if tk in aligned else np.full(T, np.nan)
        r = np.full(T, np.nan); r[k:] = np.log(v[k:] / v[:-k]); return r
    facs = {"SPY_ret": gret("SPY"), "QQQ_ret": gret("QQQ"), "UUP_ret": gret("UUP"),
            "GLD_ret": gret("GLD"), "TLT_ret": gret("TLT"), "HYG_ret": gret("HYG"),
            "VIX_chg": gret("VIXY")}
    vix = aligned["VIXY"].to_numpy(np.float64) if "VIXY" in aligned else np.full(T, np.nan)

    # ---- A 方向：宏观因子 → 加密市场未来累计收益 ----
    print("\n[A 方向] 宏观因子[t] → 加密市场聚合收益[t+1..h] 时序 IC（仅 sanity，book 中性方向价值有限）")
    print(f"{'因子':>8s}" + "".join(f"{f'h={h}':>9s}" for h in (1, 3, 6, 12, 24)))
    for nm, f in facs.items():
        line = f"{nm:>8s}"
        for h in (1, 3, 6, 12, 24):
            fwd = np.full(T, np.nan)
            for t in range(T - h):
                fwd[t] = np.nansum(mkt[t + 1:t + 1 + h])
            ic, _ = _ts_ic(f, fwd)
            line += f"{ic:>+9.3f}"
        print(line, flush=True)

    # ---- B regime：加密横截面反转 IC 按宏观状态分桶 ----
    print("\n[B regime] 加密横截面反转(rev1)→未来收益 IC，按宏观状态分桶（h=3bar）")
    h = 3
    fwd = np.full_like(logc, np.nan); fwd[:-h] = logc[h:] - logc[:-h]
    rev_ic = np.array([_xs(rev1[t], fwd[t], np.isfinite(close[t]) & (close[t] > 0)) for t in range(T)])
    valid_bar = np.isfinite(rev_ic)
    print(f"  全样本反转 IC 均值={np.nanmean(rev_ic):+.4f}  有效 bar={valid_bar.sum():,}")

    def bucket(name, state, lo_hi):
        for lab, msk in lo_hi:
            m = valid_bar & np.isfinite(state) & msk
            if m.sum() > 100:
                a = rev_ic[m]
                print(f"    {name} {lab:<10s} 反转IC={a.mean():+.4f}  (t={a.mean()/a.std()*np.sqrt(len(a)):+.1f}, n={m.sum():,})")
    spy = facs["SPY_ret"]
    bucket("SPY", spy, [("跌(<0)", spy < 0), ("涨(>0)", spy > 0)])
    vlo, vhi = np.nanpercentile(vix[np.isfinite(vix)], [33, 67]) if np.isfinite(vix).any() else (np.nan, np.nan)
    bucket("VIX", vix, [("低(risk-on)", vix <= vlo), ("高(risk-off)", vix >= vhi)])
    print("\n判读：A 有时序 IC=宏观和加密方向相关(sanity)；B 反转 IC 在桶间差异大=宏观调制我们的信号→喂模型有交互价值。", flush=True)


if __name__ == "__main__":
    main()
