# -*- coding: utf-8 -*-
"""
Focused, POLITE probe to settle the one unsettled question:
how deep does FREE intraday (minute) history go on eastmoney, once we add
retries + delays so we are not throttled?  Also re-count throttled universes.
"""
import sys, time
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
import akshare as ak
print("akshare", ak.__version__)
print("=" * 80)


def retry(fn, tries=5, pause=4.0):
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            last = e
            time.sleep(pause)
    raise last


def dcol(df):
    for c in df.columns:
        if any(k in str(c) for k in ["时间", "日期"]) or str(c).lower() in ("date", "time", "datetime"):
            return c
    return df.columns[0]


def show(tag, fn):
    t = time.time()
    try:
        df = retry(fn)
        dc = dcol(df)
        dt = pd.to_datetime(df[dc])
        # bars per day -> infer granularity actually returned
        days = dt.dt.normalize().nunique()
        print(f"[OK ] {tag:26s} rows={len(df):>6}  {str(dt.min())[:16]} .. {str(dt.max())[:16]}  "
              f"days={days}  ~{len(df)/max(days,1):.0f} bars/day  {round(time.time()-t,1)}s")
    except Exception as e:
        print(f"[ERR] {tag:26s} {type(e).__name__}: {str(e)[:80]}  {round(time.time()-t,1)}s")
    time.sleep(2.0)  # be polite between probes


print("\n### Re-count throttled universes (with retry) ###")
try:
    n = len(retry(ak.stock_zh_a_spot_em))
    print(f"A-share stocks: {n}")
except Exception as e:
    print("A-share count ERR:", str(e)[:80])
time.sleep(2)

print("\n### Eastmoney intraday DEPTH (period=1 and 5), with retry+delay ###")
ES, LS = "2015-01-01 09:30:00", "2099-01-01 15:00:00"
# 1-min: expected shallow (recent days only)
show("stock 600519 1min", lambda: ak.stock_zh_a_hist_min_em(symbol="600519", period="1", start_date=ES, end_date=LS, adjust=""))
# 5-min: the candidate workhorse frequency -- how deep?
show("stock 600519 5min", lambda: ak.stock_zh_a_hist_min_em(symbol="600519", period="5", start_date=ES, end_date=LS, adjust=""))
show("stock 000001 5min", lambda: ak.stock_zh_a_hist_min_em(symbol="000001", period="5", start_date=ES, end_date=LS, adjust=""))
show("etf   510300 5min", lambda: ak.fund_etf_hist_min_em(symbol="510300", period="5", start_date=ES, end_date=LS, adjust=""))
show("index 000300 5min", lambda: ak.index_zh_a_hist_min_em(symbol="000300", period="5", start_date=ES, end_date=LS))
# 15 / 30 / 60 min depth
show("stock 600519 15min", lambda: ak.stock_zh_a_hist_min_em(symbol="600519", period="15", start_date=ES, end_date=LS, adjust=""))
show("stock 600519 60min", lambda: ak.stock_zh_a_hist_min_em(symbol="600519", period="60", start_date=ES, end_date=LS, adjust=""))
print("\nDONE")
