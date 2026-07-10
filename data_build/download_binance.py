"""Binance Data Vision 下载器（免费、24/7、最生：永续 1m K线 + 资金费率）。
对照 download_massive.py：并发 + 断点续传 + 原子写 + SHA256 校验。

选 top-N USDT 永续(按 24h 成交额) → 每个 symbol 列出月度文件(S3 XML) → 并发镜像下载。
镜像到 data/raw/binance/<key>，磁盘路径=S3 key，便于复用。

跑：PYTHONUTF8=1 .venv/Scripts/python.exe download_binance.py --top 200 --interval 1m
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import re
import time
import urllib.request
import urllib.parse

UA = {"User-Agent": "Mozilla/5.0"}
S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"   # XML 列表端点
DL = "https://data.binance.vision"                                   # 文件下载(CDN)
FAPI = "https://fapi.binance.com"
OUT = "data/raw/binance"


def _get(url, timeout=30, retries=5):
    url = urllib.parse.quote(url, safe="%/:=&?+~.")        # 百分号编码任何非 ASCII 字符
    last = None
    for i in range(retries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout).read()
        except Exception as e:
            last = e; time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"GET 失败 {url}: {last}")


def top_symbols(n):
    """24h 成交额最大的 n 个 USDT 永续(TRADING)。"""
    info = json.loads(_get(f"{FAPI}/fapi/v1/exchangeInfo"))
    perp = {s["symbol"] for s in info["symbols"]
            if s.get("contractType") == "PERPETUAL" and s.get("quoteAsset") == "USDT" and s.get("status") == "TRADING"}
    tick = json.loads(_get(f"{FAPI}/fapi/v1/ticker/24hr"))
    rank = sorted((t for t in tick if t["symbol"] in perp), key=lambda t: float(t["quoteVolume"]), reverse=True)
    return [(t["symbol"], float(t["quoteVolume"])) for t in rank[:n]]


def all_usdt_perps():
    """从 Vision 归档枚举**所有曾存在**的 USDT 永续(含已退市)——修幸存者偏差。
    列 monthly/klines/ 下的子目录(CommonPrefixes),每个 = 一个 symbol(死币历史K线仍在)。"""
    prefix = "data/futures/um/monthly/klines/"
    out, marker = [], ""
    while True:
        url = f"{S3}?delimiter=/&prefix={prefix}" + (f"&marker={marker}" if marker else "")
        xml = _get(url).decode()
        pres = re.findall(r"<CommonPrefixes><Prefix>([^<]+)</Prefix></CommonPrefixes>", xml)
        for p in pres:
            sym = p[len(prefix):].strip("/")
            if sym.endswith("USDT"):
                out.append(sym)
        if "<IsTruncated>true</IsTruncated>" in xml and pres:
            marker = pres[-1]
        else:
            break
    return sorted(set(out))


def s3_list(prefix):
    """列 prefix 下所有对象 (key,size)，处理分页。"""
    out, marker = [], ""
    while True:
        url = f"{S3}?delimiter=/&prefix={prefix}" + (f"&marker={marker}" if marker else "")
        xml = _get(url).decode()
        for m in re.finditer(r"<Contents>.*?<Key>([^<]+)</Key>.*?<Size>(\d+)</Size>.*?</Contents>", xml, re.S):
            out.append((m.group(1), int(m.group(2))))
        if "<IsTruncated>true</IsTruncated>" in xml and out:
            marker = out[-1][0]
        else:
            break
    return [(k, s) for k, s in out if k.endswith(".zip")]


def download_one(key, size, verify=True):
    """下载单个 key → OUT/key。已存在且大小一致则跳过；原子写；SHA256 校验。返回 (status,bytes)。"""
    dst = os.path.join(OUT, key)
    if os.path.exists(dst) and os.path.getsize(dst) == size:
        return ("skip", 0)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    for attempt in range(4):
        try:
            data = _get(f"{DL}/{key}")
            if verify:
                try:
                    chk = _get(f"{DL}/{key}.CHECKSUM").decode().split()[0]
                    if hashlib.sha256(data).hexdigest() != chk:
                        raise ValueError("checksum 不符")
                except urllib.error.HTTPError:
                    pass                                    # 无 .CHECKSUM 则跳过校验
            tmp = dst + ".part"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, dst)
            return ("ok", len(data))
        except Exception as e:
            if attempt == 3:
                return (f"FAIL:{type(e).__name__}", 0)
            time.sleep(2 * (attempt + 1))


def main(a):
    os.makedirs(OUT, exist_ok=True)
    if a.symbols_npz:
        import numpy as _np
        slots = _np.load(a.symbols_npz, allow_pickle=True)["slots"].astype(str)
        syms = [(s, 0.0) for s in slots]
        print(f"从 {a.symbols_npz} 取 {len(syms)} 个 symbol（市场={a.market}）", flush=True)
    elif a.all:
        syms = [(s, 0.0) for s in all_usdt_perps()]
        print(f"全部曾存在 {len(syms)} USDT 永续(含死币,修幸存者偏差)", flush=True)
    else:
        syms = top_symbols(a.top)
        print(f"选中 top {len(syms)} USDT 永续；top5={[s for s,_ in syms[:5]]}", flush=True)

    # 收集所有月度文件任务（klines + fundingRate）
    tasks = []
    DAILY = {"metrics", "liquidationSnapshot"}                    # 这俩是 daily 路径(且 ~2021 起、历史短)
    streams = a.streams.split(",")
    for i, (sym, _) in enumerate(syms):
        for st in streams:
            if st in ("klines", "premiumIndexKlines", "markPriceKlines", "indexPriceKlines"):
                kind = f"data/{a.market}/monthly/{st}/{sym}/{a.interval}/"
            elif st in DAILY:
                kind = f"data/{a.market}/daily/{st}/{sym}/"
            else:                                                # fundingRate 等 monthly 无 interval
                kind = f"data/{a.market}/monthly/{st}/{sym}/"
            try:
                listed = s3_list(kind)
            except Exception as e:
                print(f"  ⚠️ 列举失败跳过 {kind}: {e}", flush=True); continue
            for key, size in listed:
                m = re.search(r"(\d{4}-\d{2})(?:-\d{2})?\.zip$", key)   # YYYY-MM(monthly) 或 YYYY-MM-DD(daily) 都取月
                if a.start and m and m.group(1) < a.start:
                    continue
                tasks.append((key, size))
        if (i + 1) % 25 == 0:
            print(f"  列文件 {i+1}/{len(syms)}  累计任务={len(tasks)}", flush=True)
    print(f"共 {len(tasks)} 个文件待处理（含已存在）", flush=True)

    json.dump({"symbols": [s for s, _ in syms], "interval": a.interval, "start": a.start,
               "quoteVolume": {s: v for s, v in syms}},
              open(os.path.join(OUT, "_universe.json"), "w"), indent=2)

    done = nbytes = nfail = nskip = 0
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(download_one, k, s, not a.no_verify): k for k, s in tasks}
        for fut in cf.as_completed(futs):
            st, nb = fut.result()
            done += 1; nbytes += nb
            if st == "skip":
                nskip += 1
            elif st.startswith("FAIL"):
                nfail += 1
                print(f"  ✗ {futs[fut]}  {st}", flush=True)
            if done % 200 == 0:
                gb = nbytes / 1e9
                print(f"  {done}/{len(tasks)}  下载 {gb:.2f}GB  跳过 {nskip}  失败 {nfail}  {time.time()-t0:.0f}s", flush=True)
    print(f"完成：{done} 文件，下载 {nbytes/1e9:.2f}GB，跳过 {nskip}，失败 {nfail}，{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=200, help="按 24h 成交额取前 N 个 USDT 永续")
    ap.add_argument("--all", action="store_true", help="下载所有曾存在的USDT永续(含死币)→修幸存者偏差")
    ap.add_argument("--market", default="futures/um", help="S3 市场前缀：futures/um(永续) 或 spot(现货)")
    ap.add_argument("--symbols_npz", default=None, help="从 npz 的 slots 取 symbol 列表(覆盖 --top/--all)")
    ap.add_argument("--interval", default="1m")
    ap.add_argument("--streams", default="klines,fundingRate",
                    help="逗号分隔要拉的流：klines/fundingRate/premiumIndexKlines/metrics/liquidationSnapshot")
    ap.add_argument("--start", default=None, help="起始月 YYYY-MM（含），早于此跳过；默认全量")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--no_verify", action="store_true", help="跳过 SHA256 校验(更快)")
    main(ap.parse_args())
