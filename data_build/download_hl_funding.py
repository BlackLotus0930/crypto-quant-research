"""拉 Hyperliquid 历史 funding(1h)→ data/raw/hl/funding/<COIN>.csv(ts_ms,funding_1h)。
定向部分拉取(验证用,不等全量):
  - 宇宙 = HL 实时币名 ∩ xvenue_funding slots 的 base(直配 + k(1000) 变体),按 |Binance 年化 funding| 取 top-N(最有意思的肥币)。
  - 窗口 = --start 起(默认 2025-06-01,与本地 Bin/Byb funding 到 2026-05-31 重叠)。
  - fundingHistory 每次 500 行(~20 天),按 startTime 分页;429 耐心退避;适度并发。
跑：PYTHONUTF8=1 .venv/Scripts/python.exe data_build/download_hl_funding.py --max-coins 60 --start 2025-06-01
"""
import argparse
import concurrent.futures as cf
import json
import os
import time
import urllib.error
import urllib.request

import numpy as np

UA = {"User-Agent": "Mozilla/5.0", "Content-Type": "application/json"}
OUT = "data/raw/hl/funding"
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


def hl_universe():
    r = post({"type": "metaAndAssetCtxs"})
    return [m["name"] for m in r[0]["universe"]]


def base_of(slot):
    b = slot[:-4] if slot.endswith("USDT") else slot
    return b


def pick_coins(max_coins, hl_names):
    """xvenue slots → base;映射到 HL 币名(直配,或 1000X→kX);按 |Binance 年化 funding| 排序取 top。"""
    xv = np.load("data/clean/xvenue_funding.npz", allow_pickle=True)
    slots = xv["slots"].astype(str); fbin = xv["f_bin"]
    ann = 3 * 365
    mean_f = np.nanmean(fbin, axis=0) * ann
    hl_set = set(hl_names)
    cands = []
    for i, s in enumerate(slots):
        b = base_of(s)
        hl = None
        if b in hl_set:
            hl = b
        elif b.startswith("1000") and ("k" + b[4:]) in hl_set:
            hl = "k" + b[4:]
        elif b.startswith("1000000") and ("k" + b[7:]) in hl_set:
            hl = "k" + b[7:]
        if hl is not None and np.isfinite(mean_f[i]):
            cands.append((hl, s, abs(mean_f[i])))
    # 去重(同 HL 币可能映射多次),按 |funding| 降序
    seen = {}
    for hl, s, m in sorted(cands, key=lambda x: -x[2]):
        if hl not in seen:
            seen[hl] = (s, m)
    out = [(hl, s) for hl, (s, m) in seen.items()]
    return out[:max_coins]


def hist(coin, start_ms):
    dst = os.path.join(OUT, f"{coin}.csv")
    if os.path.exists(dst) and os.path.getsize(dst) > 50:
        return ("skip", coin, 0)
    rows = []; t = start_ms; now = int(time.time() * 1000)
    try:
        for _ in range(120):
            r = post({"type": "fundingHistory", "coin": coin, "startTime": t})
            if not r:
                break
            rows += [(int(d["time"]), float(d["fundingRate"])) for d in r]
            last = max(int(d["time"]) for d in r)
            if last <= t or last >= now - 3600_000 or len(r) < 2:
                break
            t = last + 1
            time.sleep(0.3)
    except Exception as e:
        return (f"FAIL:{type(e).__name__}", coin, 0)
    if not rows:
        return ("empty", coin, 0)
    rows = sorted(set(rows))
    with open(dst, "w") as fp:
        fp.write("ts,funding\n")
        for ts, fr in rows:
            fp.write(f"{ts},{fr}\n")
    return ("ok", coin, len(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-coins", type=int, default=60)
    ap.add_argument("--start", default="2025-06-01")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    start_ms = int(time.mktime(time.strptime(a.start, "%Y-%m-%d")) * 1000)
    hl_names = hl_universe()
    coins = pick_coins(a.max_coins, hl_names)
    print(f"HL 实时 {len(hl_names)} 币;选中 {len(coins)} 币(与 xvenue 重叠,按 |Binance funding| top);窗口 {a.start}→今", flush=True)
    print("  样例:", [c[0] for c in coins[:12]], flush=True)
    ok = empty = fail = skip = 0
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(hist, hl, start_ms): (hl, s) for hl, s in coins}
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
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(coins)}  ok{ok} empty{empty} skip{skip} fail{fail}", flush=True)
    print(f"完成:ok{ok} empty{empty} skip{skip} fail{fail} → {OUT}", flush=True)


if __name__ == "__main__":
    main()
