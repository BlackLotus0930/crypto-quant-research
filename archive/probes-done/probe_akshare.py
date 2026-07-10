"""
v0 数据可得性探测 (akshare, 免费)
和 tushare 探测对等, 看免费源在各市场日频上能给多深的历史。
"""
import datetime as dt
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import akshare as ak

OUT = Path(__file__).parent / "probe_out"
OUT.mkdir(exist_ok=True)
START = "19900101"
TODAY = dt.date.today().strftime("%Y%m%d")

DATE_CANDS = ["日期", "date", "trade_date", "时间", "datetime"]

def describe(df):
    if df is None or len(df) == 0:
        return (0, None, None, [])
    n = len(df)
    dcol = next((c for c in DATE_CANDS if c in df.columns), None)
    dmin = dmax = None
    if dcol:
        d = df[dcol].astype(str)
        dmin, dmax = d.min(), d.max()
    return (n, dmin, dmax, list(df.columns))

results = []

def test(label, name, fn, save_as=None):
    print(f"\n--- [{label}] {name} ---")
    try:
        df = fn()
    except Exception as e:
        msg = str(e).strip().replace("\n", " ")[:160]
        print(f"  x FAIL: {msg}")
        results.append((label, name, "FAIL", 0, None, None, msg))
        return None
    n, dmin, dmax, cols = describe(df)
    print(f"  ok rows={n}  range={dmin}~{dmax}")
    print(f"     cols={cols}")
    results.append((label, name, "OK", n, dmin, dmax, ""))
    if save_as and n:
        df.to_csv(OUT / f"ak_{save_as}.csv", index=False, encoding="utf-8-sig")
    return df

print("=" * 70); print("AKSHARE 日频探测"); print("=" * 70)

# 个股
test("股票", "stock_zh_a_hist(000001,hfq)", lambda: ak.stock_zh_a_hist(
    symbol="000001", period="daily", start_date=START, end_date=TODAY,
    adjust="hfq"), save_as="stock_000001")

# 指数
for code in ["000300", "000905", "399006"]:
    test("指数", f"index_zh_a_hist({code})", lambda c=code: ak.index_zh_a_hist(
        symbol=c, period="daily", start_date=START, end_date=TODAY),
        save_as=f"index_{code}")

# ETF (前复权)
for code in ["510300", "510500", "159915", "518880", "511260"]:
    test("ETF", f"fund_etf_hist_em({code})", lambda c=code: ak.fund_etf_hist_em(
        symbol=c, period="daily", start_date=START, end_date=TODAY,
        adjust="hfq"), save_as=f"etf_{code}")

# 期货主力连续 (新浪)
for sym in ["RB0", "I0", "CU0", "AU0", "M0", "IF0", "IC0", "T0"]:
    test("期货主连", f"futures_main_sina({sym})", lambda s=sym: ak.futures_main_sina(
        symbol=s, start_date=START, end_date=TODAY), save_as=f"fut_{sym}")

# 外汇 (多接口尝试)
test("外汇", "forex_hist_em(USDCNY)",
     lambda: ak.forex_hist_em(symbol="USDCNY"), save_as="fx_usdcny_em")
test("外汇", "fx_spot_quote()", lambda: ak.fx_spot_quote())

# 汇总
print("\n" + "=" * 70); print("汇总"); print("=" * 70)
summ = pd.DataFrame(results, columns=[
    "类别", "接口", "状态", "行数", "最早", "最晚", "备注"])
summ.to_csv(OUT / "_SUMMARY_akshare.csv", index=False, encoding="utf-8-sig")
with pd.option_context("display.max_rows", None, "display.width", 200,
                       "display.max_colwidth", 36):
    print(summ[["类别", "接口", "状态", "行数", "最早", "最晚"]].to_string(index=False))
print(f"\nOK={(summ.状态=='OK').sum()}  FAIL={(summ.状态=='FAIL').sum()}")
