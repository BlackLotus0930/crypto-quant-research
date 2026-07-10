"""
v0 数据可得性探测 (tushare)
目标: 实测 token 在各市场/接口上的权限、历史深度、字段、缺失率。
产出: 控制台摘要 + probe_out/*.csv
不做任何建模, 只摸数据边界。
"""
import os
import sys
import datetime as dt
from pathlib import Path

import pandas as pd

# ---- 读取 token (.env, 不入库) ----
def load_token():
    env = Path(__file__).parent / ".env"
    for line in env.read_text(encoding="utf-8").splitlines():
        if line.startswith("TUSHARE_TOKEN="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("TUSHARE_TOKEN not found in .env")

import tushare as ts
ts.set_token(load_token())
pro = ts.pro_api()

OUT = Path(__file__).parent / "probe_out"
OUT.mkdir(exist_ok=True)

WIDE_START = "19900101"
TODAY = dt.date.today().strftime("%Y%m%d")

def describe(df: pd.DataFrame, date_col="trade_date"):
    """返回 (行数, 最早, 最晚, 字段, 各列缺失率string)"""
    if df is None or len(df) == 0:
        return (0, None, None, [], "")
    n = len(df)
    dmin = dmax = None
    if date_col in df.columns:
        d = df[date_col].astype(str)
        dmin, dmax = d.min(), d.max()
    cols = list(df.columns)
    na = (df.isna().mean() * 100).round(1)
    na_str = ", ".join(f"{k}:{v}%" for k, v in na.items() if v > 0) or "无缺失"
    return (n, dmin, dmax, cols, na_str)

results = []  # (类别, 接口, 标的, 状态, 行数, 最早, 最晚, 备注)

def test(label, api_name, fn, save_as=None, date_col="trade_date"):
    print(f"\n--- [{label}] {api_name} ---")
    try:
        df = fn()
    except Exception as e:
        msg = str(e).strip().replace("\n", " ")[:160]
        print(f"  ✗ FAIL: {msg}")
        results.append((label, api_name, "", "FAIL", 0, None, None, msg))
        return None
    n, dmin, dmax, cols, na = describe(df, date_col)
    print(f"  ✓ OK  rows={n}  range={dmin}~{dmax}")
    print(f"     cols={cols}")
    print(f"     NA: {na}")
    results.append((label, api_name, save_as or "", "OK", n, dmin, dmax, na))
    if save_as and n:
        df.to_csv(OUT / f"{save_as}.csv", index=False, encoding="utf-8-sig")
    return df

# ============ Phase A: 身份/列表接口 (看代码规范 + 覆盖) ============
print("=" * 70)
print("PHASE A — 列表/身份接口")
print("=" * 70)

test("股票", "stock_basic", lambda: pro.stock_basic(
    exchange="", list_status="L",
    fields="ts_code,name,area,industry,market,list_date,delist_date"),
    save_as="A_stock_basic", date_col="list_date")

# 退市股 (幸存者偏差检验: 能否拿到已退市标的)
test("股票-退市", "stock_basic(delisted)", lambda: pro.stock_basic(
    list_status="D", fields="ts_code,name,list_date,delist_date"),
    save_as="A_stock_delisted", date_col="delist_date")

test("指数", "index_basic", lambda: pro.index_basic(market="SSE"),
     save_as="A_index_basic", date_col="list_date")

test("ETF", "fund_basic", lambda: pro.fund_basic(market="E"),
     save_as="A_fund_basic", date_col="list_date")

for exch in ["CFFEX", "SHFE", "DCE", "CZCE", "INE", "GFEX"]:
    test(f"期货-{exch}", f"fut_basic({exch})",
         lambda e=exch: pro.fut_basic(exchange=e,
             fields="ts_code,symbol,name,list_date,delist_date,fut_code"),
         save_as=f"A_fut_basic_{exch}", date_col="list_date")

test("外汇", "fx_obasic", lambda: pro.fx_obasic(
     exchange="FXCM", classify="FX"), save_as="A_fx_basic", date_col="")

# ============ Phase B: 日频价格 (历史深度 + 字段 + 缺失) ============
print("\n" + "=" * 70)
print("PHASE B — 日频价格深度")
print("=" * 70)

# 股票
test("股票", "daily(000001.SZ)", lambda: pro.daily(
    ts_code="000001.SZ", start_date=WIDE_START, end_date=TODAY),
    save_as="B_stock_daily")

# 指数
for code in ["000300.SH", "000905.SH", "399006.SZ"]:
    test("指数", f"index_daily({code})", lambda c=code: pro.index_daily(
        ts_code=c, start_date=WIDE_START, end_date=TODAY),
        save_as=f"B_index_{code}")

# ETF
for code in ["510300.SH", "510500.SH", "159915.SZ", "518880.SH", "511260.SH"]:
    test("ETF", f"fund_daily({code})", lambda c=code: pro.fund_daily(
        ts_code=c, start_date=WIDE_START, end_date=TODAY),
        save_as=f"B_etf_{code}")

# 股指/国债期货 (CFFEX) — 单合约 + 主力连续尝试
for code in ["IF2406.CFX", "IF.CFX", "IFL.CFX", "T2406.CFX", "T.CFX"]:
    test("金融期货", f"fut_daily({code})", lambda c=code: pro.fut_daily(
        ts_code=c, start_date=WIDE_START, end_date=TODAY),
        save_as=f"B_futfin_{code}")

# 商品期货主力连续尝试 (多种代码规范都试)
for code in ["RB.SHF", "RBL.SHF", "CU.SHF", "AU.SHF", "I.DCE", "M.DCE"]:
    test("商品期货", f"fut_daily({code})", lambda c=code: pro.fut_daily(
        ts_code=c, start_date=WIDE_START, end_date=TODAY),
        save_as=f"B_futcom_{code}")

# 外汇
for code in ["USDCNY.FXCM", "USDCNH.FXCM"]:
    test("外汇", f"fx_daily({code})", lambda c=code: pro.fx_daily(
        ts_code=c, start_date=WIDE_START, end_date=TODAY),
        save_as=f"B_fx_{code}")

# ============ 摘要表 ============
print("\n" + "=" * 70)
print("汇总")
print("=" * 70)
summ = pd.DataFrame(results, columns=[
    "类别", "接口", "存档", "状态", "行数", "最早", "最晚", "备注"])
summ.to_csv(OUT / "_SUMMARY.csv", index=False, encoding="utf-8-sig")
# 控制台精简打印
with pd.option_context("display.max_rows", None, "display.width", 200,
                       "display.max_colwidth", 40):
    print(summ[["类别", "接口", "状态", "行数", "最早", "最晚"]].to_string(index=False))

ok = (summ.状态 == "OK").sum()
fail = (summ.状态 == "FAIL").sum()
print(f"\nOK={ok}  FAIL={fail}  (详见 probe_out/_SUMMARY.csv)")
