"""拉 Bybit 1h 永续 close → data/raw/bybit/kline/<SYMBOL>.csv(ts_ms,close)。
用于跨所基差噪声(价格腿):inter-venue price leg = Binance_perp_ret − Bybit_perp_ret。
分页(end 往回,limit 1000),线程并发。跑：PYTHONUTF8=1 .venv/Scripts/python.exe download_bybit_klines.py
"""
import concurrent.futures as cf
import json
import os
import time
import urllib.request

import numpy as np

UA = {"User-Agent": "Mozilla/5.0"}
OUT = "data/raw/bybit/kline"
START_MS = int(time.mktime(time.strptime("2020-01-01", "%Y-%m-%d")) * 1000)


def get(u, retries=4):
    for i in range(retries):
        try:
            return json.loads(urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=30).read())
        except Exception:
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
            u = (f"https://api.bybit.com/v5/market/kline?category=linear&symbol={sym}"
                 f"&interval=60&limit=1000&end={end}")
            L = get(u)["result"]["list"]
            if not L:
                break
            rows += [(int(d[0]), float(d[4])) for d in L]              # [start_ms, o,h,l,close,...]
            oldest = min(int(d[0]) for d in L)
            if oldest <= START_MS or len(L) < 1000:
                break
            end = oldest - 1; time.sleep(0.03)
    except Exception as e:
        return (f"FAIL:{type(e).__name__}", sym, 0)
    if not rows:
        return ("empty", sym, 0)
    rows = sorted(set(rows))
    with open(dst, "w") as fp:
        fp.write("ts,close\n")
        for ts, c in rows:
            fp.write(f"{ts},{c}\n")
    return ("ok", sym, len(rows))


def main():
    os.makedirs(OUT, exist_ok=True)
    slots = list(np.load("data/clean/crypto_tensor_60min_pit.npz", allow_pickle=True)["slots"].astype(str))
    print(f"宇宙 {len(slots)} 币;拉 Bybit 1h close → {OUT}", flush=True)
    ok = empty = fail = skip = 0
    with cf.ThreadPoolExecutor(max_workers=10) as ex:
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
    print(f"完成:ok{ok} empty{empty} skip{skip} fail{fail}", flush=True)


if __name__ == "__main__":
    main()
