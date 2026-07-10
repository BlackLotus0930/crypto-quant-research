"""拉 HL 1h K线 close → data/raw/hl/kline/<COIN>.csv(ts_ms,close)。用于测**HL 价格腿 haircut**
(多所路由里 long/short 含 HL 腿时,价格腿=各所永续收益差;回测最大的未知数)。
candleSnapshot 每次≤5000 行(~208天),按 endTime 往回翻页。币宇宙=已下 HL funding 的币。
跑：PYTHONUTF8=1 .venv/Scripts/python.exe data_build/download_hl_klines.py --start 2025-06-01
"""
import argparse
import concurrent.futures as cf
import glob
import json
import os
import time
import urllib.error
import urllib.request

UA = {"User-Agent": "Mozilla/5.0", "Content-Type": "application/json"}
OUT = "data/raw/hl/kline"
INFO = "https://api.hyperliquid.xyz/info"


def post(body, retries=5):
    err = 0
    for _ in range(60):
        try:
            req = urllib.request.Request(INFO, data=json.dumps(body).encode(), headers=UA)
            return json.loads(urllib.request.urlopen(req, timeout=30).read())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(5.0); continue
            err += 1
            if err >= retries:
                raise
            time.sleep(1.0 * err)
        except Exception:
            err += 1
            if err >= retries:
                raise
            time.sleep(1.0 * err)
    raise RuntimeError("429 exhausted")


def hist(coin, start_ms):
    dst = os.path.join(OUT, f"{coin}.csv")
    if os.path.exists(dst) and os.path.getsize(dst) > 50:
        return ("skip", coin, 0)
    rows = {}; end = int(time.time() * 1000)
    try:
        for _ in range(20):
            r = post({"type": "candleSnapshot", "req": {"coin": coin, "interval": "1h", "startTime": start_ms, "endTime": end}})
            if not r:
                break
            for d in r:
                rows[int(d["t"])] = float(d["c"])
            first = min(int(d["t"]) for d in r)
            if first <= start_ms or len(r) < 2:
                break
            end = first - 1
            time.sleep(0.3)
    except Exception as e:
        return (f"FAIL:{type(e).__name__}", coin, 0)
    if not rows:
        return ("empty", coin, 0)
    with open(dst, "w") as fp:
        fp.write("ts,close\n")
        for ts in sorted(rows):
            fp.write(f"{ts},{rows[ts]}\n")
    return ("ok", coin, len(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-06-01")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    start_ms = int(time.mktime(time.strptime(a.start, "%Y-%m-%d")) * 1000)
    coins = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob("data/raw/hl/funding/*.csv"))
    print(f"HL K线:{len(coins)} 币(=已下 funding 的币),窗口 {a.start}→今", flush=True)
    ok = empty = fail = skip = 0
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(hist, c, start_ms): c for c in coins}
        for i, fut in enumerate(cf.as_completed(futs)):
            st, coin, n = fut.result()
            if st == "ok":
                ok += 1
            elif st == "empty":
                empty += 1
            elif st == "skip":
                skip += 1
            else:
                fail += 1; print(f"  ✗ {coin} {st}", flush=True)
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(coins)}  ok{ok} empty{empty} skip{skip} fail{fail}", flush=True)
    print(f"完成:ok{ok} empty{empty} skip{skip} fail{fail} → {OUT}", flush=True)


if __name__ == "__main__":
    main()
