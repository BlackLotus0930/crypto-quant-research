"""拉 Bybit 历史 funding(8h)→ data/raw/bybit/funding/<SYMBOL>.csv(ts_ms,funding)。
宇宙=pit 张量 slots(与 Binance 同币,便于对齐)。分页(endTime 往回),线程并发。
跑：PYTHONUTF8=1 .venv/Scripts/python.exe download_bybit_funding.py
"""
import concurrent.futures as cf
import json
import os
import time
import urllib.request

import numpy as np

UA = {"User-Agent": "Mozilla/5.0"}
OUT = "data/raw/bybit/funding"
START_MS = int(time.mktime(time.strptime("2020-01-01", "%Y-%m-%d")) * 1000)


def get(u, retries=4):
    for i in range(retries):
        try:
            return json.loads(urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=30).read())
        except Exception as e:
            if i == retries - 1:
                raise
            time.sleep(1.0 * (i + 1))


def hist(sym):
    dst = os.path.join(OUT, f"{sym}.csv")
    if os.path.exists(dst) and os.path.getsize(dst) > 50:
        return ("skip", sym, 0)
    rows = []; end = int(time.time() * 1000)
    try:
        while True:
            u = f"https://api.bybit.com/v5/market/funding/history?category=linear&symbol={sym}&limit=200&endTime={end}"
            L = get(u)["result"]["list"]
            if not L:
                break
            rows += [(int(d["fundingRateTimestamp"]), float(d["fundingRate"])) for d in L]
            oldest = min(int(d["fundingRateTimestamp"]) for d in L)
            if oldest <= START_MS or len(L) < 200:
                break
            end = oldest - 1; time.sleep(0.03)
    except Exception as e:
        return (f"FAIL:{type(e).__name__}", sym, 0)
    if not rows:
        return ("empty", sym, 0)
    rows = sorted(set(rows))
    with open(dst, "w") as fp:
        fp.write("ts,funding\n")
        for ts, fr in rows:
            fp.write(f"{ts},{fr}\n")
    return ("ok", sym, len(rows))


def main():
    os.makedirs(OUT, exist_ok=True)
    slots = list(np.load("data/clean/crypto_tensor_60min_pit.npz", allow_pickle=True)["slots"].astype(str))
    print(f"宇宙 {len(slots)} 币;拉 Bybit funding(8h)→ {OUT}", flush=True)
    ok = empty = fail = skip = 0
    with cf.ThreadPoolExecutor(max_workers=12) as ex:
        for i, (st, sym, n) in enumerate(ex.map(hist, slots)):
            if st == "ok":
                ok += 1
            elif st == "empty":
                empty += 1
            elif st == "skip":
                skip += 1
            else:
                fail += 1; print(f"  ✗ {sym} {st}", flush=True)
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(slots)}  ok{ok} empty{empty} skip{skip} fail{fail}", flush=True)
    print(f"完成:ok{ok} empty{empty}(Bybit无此币) skip{skip} fail{fail}", flush=True)


if __name__ == "__main__":
    main()
