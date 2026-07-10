"""
美国市场数据下载器（按 数据.md 落地）。

特性：
- 断点续传：每个标的存独立 parquet；重跑自动跳过已下好的，只补没下的/失败的。
- 重试 + 指数退避 + 限速：对 sina 友好，单个失败不影响整体。
- 自检：枚举完先试下 5 只，立刻暴露代码格式问题，不让你白跑几小时。
- 全程写日志到 data/raw/_download.log，可随时 tail 看进度。

运行（用项目 venv 的 python；脚本已自带 utf-8，不必设环境变量）：
    .venv/Scripts/python.exe download_us.py                # 全量（跨资产 + 全部美股，~1-3 小时）
    .venv/Scripts/python.exe download_us.py --cross-only   # 只下跨资产层（ETF/指数/收益率，~3 分钟）
    .venv/Scripts/python.exe download_us.py --max-stocks 800   # 个股只下前 800（快速试）
    .venv/Scripts/python.exe download_us.py --consolidate  # 把所有 parquet 合并成一张长表

断了直接重跑同一条命令即可接着下。
"""
import argparse
import random
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, date
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows 控制台中文
except Exception:
    pass

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import akshare as ak

# ---------------- 配置 ----------------
OUT = Path(__file__).parent / "data" / "raw"
LOG = OUT / "_download.log"
MANIFEST = OUT / "_manifest.csv"
UNIV_CACHE = OUT / "_universe_us_stocks.csv"
ADJUST = ""   # 不复权(raw)。新浪美股不支持 hfq；qfq 对长历史老股会算出负价(加法除权)，不可用。
              # raw 全程为正，拆股假跳价留到数据管道里检测+还原（机械数据卫生，非因子）。
WORKERS = 12          # 并发线程数（实测新浪 ~5/s 无报错；12 留余量）
SLEEP = 0.15          # 每次成功后小憩（秒）+ 抖动，礼貌限速
RETRIES = 4           # 单标的最大重试次数
BACKOFF = 2.0         # 退避基数：sleep = BACKOFF**attempt
YIELD_START = "20040101"

# 跨资产层：固定清单（数据.md），全部走 stock_us_daily（除指数/收益率）
CROSS = {
    "broad_etf":   ["SPY", "QQQ", "DIA", "IWM"],
    "sector_etf":  ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC"],
    # 风格/因子层：成长/价值/动量/质量/低波/高beta/纯价值/红利 —— 对"交易股票"的模型信息密度最高的上下文
    "factor_etf":  ["IWF", "IWD", "MTUM", "QUAL", "USMV", "SPHB", "VLUE", "VYM"],
    "vol":         ["VIXY", "VXX", "SVXY", "VXZ"],                       # +VXZ 出 VIX 期限结构
    "treasury_etf":["BIL", "SHY", "IEI", "IEF", "TLT", "TIP"],          # +IEI 补曲线中段 +TIP 实际利率/通胀
    "credit":      ["LQD", "HYG", "JNK", "AGG", "EMB"],                 # +AGG 总债(久期) +EMB EM 主权债
    "fx":          ["UUP", "FXE", "FXY", "FXB", "FXF", "FXC", "FXA"],
    "commodity":   ["GLD", "SLV", "USO", "UNG", "DBC", "CPER", "DBA"],
    "global_etf":  ["EWJ", "EWG", "EWU", "EWQ", "EWL", "EWA", "EWY", "EWT", "EWH",
                    "FXI", "MCHI", "EEM", "EFA", "EWZ", "EWW", "EWC", "INDA"],
    "crypto":      ["GBTC", "ETHE", "IBIT"],
}
INDICES = [".INX", ".DJI", ".IXIC", ".NDX"]   # 走 index_us_stock_sina
CROSS_SYMS = {s for lst in CROSS.values() for s in lst}  # 用于从个股全表里剔除，避免重复下载

# ---------------- 工具 ----------------
def log(msg):
    line = f"{datetime.now():%H:%M:%S}  {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def safe_name(sym):
    return re.sub(r"[^A-Za-z0-9]+", "_", str(sym)).strip("_")

def out_path(category, sym):
    d = OUT / category
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{safe_name(sym)}.parquet"

def std_daily(df):
    """统一日频表为 date/open/high/low/close/volume。"""
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "date" not in df.columns:
        for c in ("日期", "trade_date", "时间", "datetime"):
            if c in df.columns:
                df = df.rename(columns={c: "date"})
                break
    keep = [c for c in ["date", "open", "high", "low", "close", "volume", "amount"] if c in df.columns]
    df = df[keep]
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df.dropna(subset=["date"]).reset_index(drop=True)

def warmup_v8():
    """akshare 的 stock_us_daily 内部用 py_mini_racer(V8) 算 hash；多线程并发首次初始化
    V8 全局池会竞态崩溃(partition_address_space FATAL)。先在主线程单线程初始化一次即可避免。"""
    try:
        import py_mini_racer
        m = py_mini_racer.MiniRacer(); m.eval("1+1")
        log("V8(mini_racer) 已在主线程预热")
    except Exception as e:
        log(f"V8 预热跳过：{str(e)[:80]}")

def fetch(category, sym):
    if category == "index":
        return std_daily(ak.index_us_stock_sina(symbol=sym))
    return std_daily(ak.stock_us_daily(symbol=sym, adjust=ADJUST))

def manifest_append(rows):
    df = pd.DataFrame(rows)
    header = not MANIFEST.exists()
    df.to_csv(MANIFEST, mode="a", header=header, index=False, encoding="utf-8-sig")

def _is_transient(err: str) -> bool:
    """只对网络抖动重试；坏 ticker / 解析错是确定性失败，1 次就跳。"""
    e = err.lower()
    sigs = ("connection", "timed out", "timeout", "remotedisconnected",
            "max retries", "temporarily", "reset by peer", "chunked",
            "incompleteread", "ssl", "proxy", "aborted", "10054", "10060")
    return any(s in e for s in sigs)

def download_one(category, sym, force=False):
    """返回 (status, rows, start, end, err)。已存在则跳过。"""
    p = out_path(category, sym)
    if p.exists() and not force:
        try:
            n = len(pd.read_parquet(p))
            if n > 0:
                return ("skip", n, None, None, "")
        except Exception:
            pass  # 损坏则重下
    last_err = ""
    for attempt in range(RETRIES):
        try:
            df = fetch(category, sym)
            if df is None or len(df) == 0:
                return ("fail", 0, None, None, "空")  # 确定性，不重试
            df.to_parquet(p, index=False)
            time.sleep(SLEEP + random.uniform(0, 0.25))
            return ("ok", len(df), df["date"].min(), df["date"].max(), "")
        except Exception as e:
            last_err = str(e).replace("\n", " ")[:120]
            if not _is_transient(last_err):
                break  # 坏 ticker / 解析错：快速失败，不重试
            time.sleep(BACKOFF ** attempt + random.uniform(0, 0.5))
    return ("fail", 0, None, None, last_err)

# ---------------- 美股全表枚举 ----------------
def normalize_symbol(s):
    s = str(s).strip()
    s = re.sub(r"^\$", "", s)            # 去 $ 前缀
    s = re.sub(r"^\d+\.", "", s)          # 去 "105." 这类交易所数字前缀（保留 BRK.A 的点）
    return s.upper()

def build_stock_universe(max_stocks=None):
    if UNIV_CACHE.exists():
        syms = pd.read_csv(UNIV_CACHE)["symbol"].dropna().astype(str).tolist()
        syms = [s for s in syms if s and s.lower() != "nan"]
        log(f"美股全表（缓存）= {len(syms)} 只")
    else:
        log("枚举美股全表（首次，~15 分钟，会缓存）...")
        df = None
        for fn in ("stock_us_spot", "get_us_stock_name"):
            try:
                df = getattr(ak, fn)()
                log(f"  用 {fn} 得到 {len(df)} 行，列={list(df.columns)[:8]}")
                break
            except Exception as e:
                log(f"  {fn} 失败：{str(e)[:80]}")
        if df is None:
            raise SystemExit("美股枚举全部失败，无法继续。")
        cols = {c.lower(): c for c in df.columns}
        symcol = next((cols[k] for k in ("symbol", "代码", "ticker", "sym") if k in cols), df.columns[0])
        syms = sorted({normalize_symbol(x) for x in df[symcol] if str(x).strip()})
        syms = [s for s in syms if re.fullmatch(r"[A-Z][A-Z0-9.\-]*", s)]  # 丢掉明显非 ticker
        pd.DataFrame({"symbol": syms}).to_csv(UNIV_CACHE, index=False, encoding="utf-8-sig")
        log(f"美股全表 = {len(syms)} 只（已缓存到 {UNIV_CACHE.name}）")

        # 自检：试下 5 只，立刻暴露格式问题
        sample = random.sample(syms, min(5, len(syms)))
        ok = 0
        for s in sample:
            try:
                d = std_daily(ak.stock_us_daily(symbol=s, adjust=ADJUST))
                if len(d):
                    ok += 1
            except Exception:
                pass
            time.sleep(0.3)
        log(f"  自检：{sample} → {ok}/5 成功")
        if ok == 0:
            log("  ⚠️ 自检 0/5！代码格式可能不对，先别全量跑——把上面样例发我。")

    syms = [s for s in syms if s not in CROSS_SYMS]  # 剔除跨资产 ETF（它们单独下，避免重复）
    if max_stocks:
        syms = syms[:max_stocks]
        log(f"  --max-stocks={max_stocks} → 取前 {len(syms)} 只")
    return syms

# ---------------- 主流程 ----------------
def run_batch(name, items, force):
    """items = [(category, sym), ...] —— 多线程并发（I/O-bound）。
    download_one 各写各的 parquet（线程安全）；manifest/日志由主线程统一写（ex.map 按序在主线程消费）。"""
    log(f"=== {name}：{len(items)} 个，{WORKERS} 并发 ===")
    rows, ok, done = [], 0, 0
    t0 = time.time()
    def work(item):
        cat, sym = item
        return (cat, sym) + download_one(cat, sym, force)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for cat, sym, st, n, s, e, err in ex.map(work, items):
            done += 1
            rows.append(dict(category=cat, symbol=sym, status=st, rows=n,
                             start=s, end=e, err=err, ts=datetime.now().isoformat()))
            if st in ("ok", "skip"):
                ok += 1
            elif st == "fail":
                log(f"  ✗ {sym}: {err}")
            if done % 100 == 0 or done == len(items):
                manifest_append(rows); rows = []
                rate = done / max(1e-9, time.time() - t0)
                log(f"  {done}/{len(items)}  ok+skip={ok}  ~{rate:.1f}/s")
    if rows:
        manifest_append(rows)
    log(f"=== {name} 完成：{ok}/{len(items)} ===")

def cmd_download(args):
    OUT.mkdir(parents=True, exist_ok=True)
    log(f"开始下载。OUT={OUT}")
    warmup_v8()  # 必须在任何并发之前，否则 V8 多线程首次初始化会 FATAL 崩溃

    # 1) 美债收益率曲线（单独 schema）
    if not args.stocks_only:
        try:
            p = OUT / "macro"; p.mkdir(parents=True, exist_ok=True)
            f = p / "us_treasury_yield.parquet"
            if f.exists() and not args.force:
                log("美债收益率曲线：已存在，跳过")
            else:
                df = ak.bond_zh_us_rate(start_date=YIELD_START)
                df.to_parquet(f, index=False)
                log(f"美债收益率曲线：{len(df)} 行 → {f.name}")
        except Exception as e:
            log(f"美债收益率曲线失败：{str(e)[:100]}")

        # 2) 指数
        run_batch("美股指数", [("index", s) for s in INDICES], args.force)

        # 3) 跨资产 ETF 层
        cross_items = [(cat, s) for cat, lst in CROSS.items() for s in lst]
        run_batch("跨资产ETF层", cross_items, args.force)

    # 4) 美股个股
    if not args.cross_only:
        syms = build_stock_universe(args.max_stocks)
        run_batch("美股个股", [("stock", s) for s in syms], args.force)

    log("全部完成。汇总见 _manifest.csv，合并用 --consolidate。")

def cmd_consolidate(args):
    log("合并所有 parquet → 长表 ...")
    frames = []
    for p in OUT.rglob("*.parquet"):
        if p.parent.name == "macro":
            continue
        try:
            df = pd.read_parquet(p)
            df["category"] = p.parent.name
            df["symbol"] = p.stem
            frames.append(df)
        except Exception as e:
            log(f"  跳过 {p.name}: {str(e)[:60]}")
    if not frames:
        log("没有可合并的文件。"); return
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.drop_duplicates(["symbol", "date"]).reset_index(drop=True)  # 防跨资产/个股重复
    cols = ["symbol", "category", "date", "open", "high", "low", "close", "volume"]
    panel = panel[[c for c in cols if c in panel.columns]]
    out = OUT.parent / "panel_long.parquet"
    panel.to_parquet(out, index=False)
    log(f"长表：{len(panel):,} 行，{panel['symbol'].nunique()} 标的 → {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cross-only", action="store_true", help="只下跨资产层（不下个股）")
    ap.add_argument("--stocks-only", action="store_true", help="只下个股")
    ap.add_argument("--max-stocks", type=int, default=None, help="个股只下前 N 只")
    ap.add_argument("--force", action="store_true", help="忽略已存在文件，强制重下")
    ap.add_argument("--consolidate", action="store_true", help="合并已下数据为长表")
    ap.add_argument("--workers", type=int, default=None, help=f"并发线程数（默认 {WORKERS}）")
    args = ap.parse_args()
    if args.workers:
        WORKERS = args.workers
    if args.consolidate:
        cmd_consolidate(args)
    else:
        cmd_download(args)
