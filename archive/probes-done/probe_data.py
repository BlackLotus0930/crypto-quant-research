# -*- coding: utf-8 -*-
"""
Data-availability probe for akshare.
Goal: empirically measure, per (market, frequency), how deep the history goes,
how many instruments exist, what columns we get, basic cleanliness, and latency.
This is the factual basis for choosing frequency / universe / Delta-t bounds.
"""
import sys, time, io
import pandas as pd

# force utf-8 stdout on Windows
sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

import akshare as ak
print("akshare", ak.__version__, "| pandas", pd.__version__)
print("=" * 90)


def _date_col(df):
    for c in df.columns:
        cl = str(c).lower()
        if any(k in str(c) for k in ["日期", "时间"]) or any(k in cl for k in ["date", "time"]):
            return c
    return df.columns[0]


def summ(df):
    """Return a one-line summary dict for a history DataFrame."""
    if df is None or len(df) == 0:
        return {"rows": 0, "first": "-", "last": "-", "span": "-", "nan": "-", "cols": "-"}
    dc = _date_col(df)
    try:
        dt = pd.to_datetime(df[dc])
        first, last = str(dt.min())[:16], str(dt.max())[:16]
        span = f"{(dt.max() - dt.min()).days / 365.25:.1f}y"
    except Exception:
        first, last, span = str(df[dc].iloc[0])[:16], str(df[dc].iloc[-1])[:16], "?"
    nan = int(df.isna().sum().sum())
    return {"rows": len(df), "first": first, "last": last, "span": span,
            "nan": nan, "cols": list(df.columns)}


RESULTS = []

def probe(market, freq, label, fn):
    """Run one probe: time it, summarize, collect."""
    t = time.time()
    try:
        df = fn()
        s = summ(df)
        s.update(market=market, freq=freq, label=label, secs=round(time.time() - t, 1), err="")
        print(f"[OK ] {market:9s} {freq:6s} {label:22s} "
              f"rows={s['rows']:>7} {s['first']}..{s['last']} ({s['span']}) {s['secs']}s")
        print(f"        cols: {s['cols']}")
        RESULTS.append(s)
    except Exception as e:
        print(f"[ERR] {market:9s} {freq:6s} {label:22s} {type(e).__name__}: {str(e)[:90]} ({round(time.time()-t,1)}s)")
        RESULTS.append(dict(market=market, freq=freq, label=label, rows=-1,
                            secs=round(time.time() - t, 1), err=str(e)[:90]))


def count(label, fn):
    t = time.time()
    try:
        df = fn()
        print(f"[CNT] {label:30s} {len(df):>7} instruments  ({round(time.time()-t,1)}s)  "
              f"cols sample: {list(df.columns)[:6]}")
    except Exception as e:
        print(f"[CNT] {label:30s} ERR {type(e).__name__}: {str(e)[:80]}")


EARLY = "19900101"
LATE = "20991231"
EARLY_MIN = "2010-01-01 09:30:00"
LATE_MIN = "2099-01-01 15:00:00"

# ---------------------------------------------------------------- universe sizes
print("\n### UNIVERSE SIZES (how many instruments exist per market) ###")
count("A-share stocks (spot_em)",      lambda: ak.stock_zh_a_spot_em())
count("ETFs (fund_etf_spot_em)",       lambda: ak.fund_etf_spot_em())
count("Index spot (index_spot_em)",    lambda: ak.stock_zh_index_spot_em(symbol="沪深重要指数"))
count("Futures main contracts (sina)", lambda: ak.futures_display_main_sina())

# ---------------------------------------------------------------- DAILY history depth
print("\n### DAILY history depth (max lookback per market) ###")
probe("stock", "1d", "600519 茅台 hfq",
      lambda: ak.stock_zh_a_hist(symbol="600519", period="daily", start_date=EARLY, end_date=LATE, adjust="hfq"))
probe("stock", "1d", "000001 平安银行 hfq",
      lambda: ak.stock_zh_a_hist(symbol="000001", period="daily", start_date=EARLY, end_date=LATE, adjust="hfq"))
probe("index", "1d", "000300 沪深300",
      lambda: ak.index_zh_a_hist(symbol="000300", period="daily", start_date=EARLY, end_date=LATE))
probe("index", "1d", "000001 上证综指",
      lambda: ak.stock_zh_index_daily(symbol="sh000001"))
probe("etf",   "1d", "510050 50ETF hfq",
      lambda: ak.fund_etf_hist_em(symbol="510050", period="daily", start_date=EARLY, end_date=LATE, adjust="hfq"))
probe("fut",   "1d", "RB0 螺纹主连",
      lambda: ak.futures_main_sina(symbol="RB0", start_date=EARLY, end_date=LATE))
probe("fut",   "1d", "IF0 股指主连",
      lambda: ak.futures_main_sina(symbol="IF0", start_date=EARLY, end_date=LATE))
probe("fut",   "1d", "T0 国债主连",
      lambda: ak.futures_main_sina(symbol="T0", start_date=EARLY, end_date=LATE))
probe("fx",    "1d", "USDCNY 中行",
      lambda: ak.currency_boc_sina(symbol="美元", start_date="20100101", end_date="20991231"))
probe("bond",  "1d", "sh019547 国债",
      lambda: ak.bond_zh_hs_daily(symbol="sh019547"))

# ---------------------------------------------------------------- INTRADAY history depth (the critical test)
print("\n### INTRADAY history depth (THE key constraint: how far back do minute bars go?) ###")
for p in ["1", "5", "15", "60"]:
    probe("stock", p + "min", "600519 茅台",
          lambda p=p: ak.stock_zh_a_hist_min_em(symbol="600519", period=p, start_date=EARLY_MIN, end_date=LATE_MIN, adjust=""))
for p in ["5", "60"]:
    probe("etf", p + "min", "510050 50ETF",
          lambda p=p: ak.fund_etf_hist_min_em(symbol="510050", period=p, start_date=EARLY_MIN, end_date=LATE_MIN, adjust=""))
    probe("index", p + "min", "000300 沪深300",
          lambda p=p: ak.index_zh_a_hist_min_em(symbol="000300", period=p, start_date=EARLY_MIN, end_date=LATE_MIN))
for p in ["5", "60"]:
    probe("fut", p + "min", "RB0 螺纹主连",
          lambda p=p: ak.futures_zh_minute_sina(symbol="RB0", period=p))

print("\n" + "=" * 90)
print("DONE. Summary rows:", len(RESULTS))
try:
    out = pd.DataFrame([r for r in RESULTS if r.get("rows", -1) >= 0])
    out = out[["market", "freq", "label", "rows", "first", "last", "span", "nan", "secs"]]
    print(out.to_string(index=False))
    out.to_csv("c:/Users/heloy/market_foundation_model/probe_results.csv", index=False, encoding="utf-8-sig")
    print("\nsaved -> probe_results.csv")
except Exception as e:
    print("summary table error:", e)
