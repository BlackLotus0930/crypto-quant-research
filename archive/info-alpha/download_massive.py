"""Massive flat-files (S3) 下载器 —— 高并发 + 断点续传 + 原子写，纯本地镜像。

为什么这样写（速度+稳）：
  - S3 GET 是 I/O-bound：用线程池并发（默认 48）打满带宽，比串行快几十倍。
  - 一个 boto3 client 配大连接池（max_pool_connections）+ 自适应重试，线程间共享、线程安全。
  - **断点续传**：用 LIST 返回的 Size 直接比对本地文件大小，已完整的跳过（不发 HEAD，省往返）。
  - **原子写**：先下到 .part 再 rename，中断不会留半个坏文件。
  - 镜像 S3 key 结构到本地，pipeline 再按 universe 过滤/重采样（下载器不做 CPU 活，最快）。

跑（你来跑，会很久、可随时断了重跑）：
  PYTHONUTF8=1 .venv/Scripts/python.exe download_massive.py                 # minute_aggs 全历史
  ... --data day_aggs_v1                                                   # 顺手也要日频(便宜)
  ... --start-year 2015 --workers 64                                       # 只近十年 / 调并发
"""
import argparse
import concurrent.futures as cf
import os
import random
import sys
import threading
import time

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT_ROOT = "data/massive"


def load_creds(path=".massive_key"):
    c = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            c[k.strip()] = v.strip()
    return c


def make_client(c, workers):
    return boto3.client(
        "s3",
        endpoint_url=c["S3_ENDPOINT"],
        aws_access_key_id=c["S3_ACCESS_KEY"],
        aws_secret_access_key=c["S3_SECRET"],
        config=Config(max_pool_connections=workers + 8,
                      retries={"max_attempts": 10, "mode": "adaptive"},
                      read_timeout=60, connect_timeout=10),    # 短超时：挂死连接快速放弃→重试（重试能成）
    )


class RateLimiter:
    """令牌桶：把**全局请求率**钉在 rate 次/秒（线程安全，不在锁内 sleep）。
    Massive flat-files 是 HTTP 429 限流——小文件(day_aggs)并发会瞬间打爆，必须限请求率而非只限并发。"""
    def __init__(self, rate):
        self.interval = 1.0 / max(rate, 0.1)
        self.lock = threading.Lock()
        self.nxt = time.time()

    def acquire(self):
        with self.lock:
            now = time.time()
            t = max(now, self.nxt)
            self.nxt = t + self.interval
        wait = t - now
        if wait > 0:
            time.sleep(wait)


def list_keys(s3, bucket, prefix, y0, y1, min_date=None):
    """分页列全部对象，返回 [(key, size)]，按年份过滤（路径形如 .../YYYY/MM/DD.csv.gz）。
    min_date='YYYY-MM-DD'：早于此日期的直接排除——Massive 订阅是**滚动 10 年窗口**，
    窗口外的文件永久 403，不过滤掉会被当限流死重试 8 次，整个下载看起来卡死。"""
    out = []
    tok = None
    while True:
        kw = dict(Bucket=bucket, Prefix=prefix, MaxKeys=1000)
        if tok:
            kw["ContinuationToken"] = tok
        r = s3.list_objects_v2(**kw)
        for o in r.get("Contents", []):
            k = o["Key"]
            if not k.endswith(".csv.gz"):
                continue
            yr = _year_of(k)
            if yr is not None and (yr < y0 or yr > y1):
                continue
            if min_date and _date_of(k) and _date_of(k) < min_date:
                continue
            out.append((k, o["Size"]))
        if r.get("IsTruncated"):
            tok = r["NextContinuationToken"]
        else:
            break
    return out


def _year_of(key):
    for part in key.split("/"):
        if len(part) == 4 and part.isdigit():
            return int(part)
    return None


def _date_of(key):
    """从文件名取日期串 'YYYY-MM-DD'（可字典序比较），取不到返回 None。"""
    name = key.rsplit("/", 1)[-1]
    if name.endswith(".csv.gz"):
        d = name[:-7]
        if len(d) == 10 and d[4] == "-" and d[7] == "-":
            return d
    return None


def download_one(s3, bucket, key, size, limiter, max_retry=8):
    """下一个 key 到本地镜像路径。已存在且大小一致→跳过。原子写。
    限速 + 对 429/503 指数退避重试（限流冷却期靠重试等过去）。返回 (status, bytes)。"""
    dst = os.path.join(OUT_ROOT, key)
    if os.path.exists(dst) and os.path.getsize(dst) == size:
        return "skip", 0
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".part"
    last = "?"
    for attempt in range(max_retry):
        limiter.acquire()
        try:
            s3.download_file(bucket, key, tmp)
            os.replace(tmp, dst)
            return "ok", size
        except ClientError as e:
            http = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0)
            last = f"HTTP{http}"
            # 此端点限流会 429→403 升级（凭证有效=单请求可成，故 403 视为限流、退避重试，非永久错）
            if http in (429, 503, 500, 403):
                time.sleep(min(2 ** attempt, 20) + random.random())
                continue
            return f"err:{last}", 0                            # 404 等永久错 → 不重试
        except Exception as e:
            last = type(e).__name__
            time.sleep(min(2 ** attempt, 15) + random.random())
    if os.path.exists(tmp):
        try:
            os.remove(tmp)
        except OSError:
            pass
    return f"err:{last}", 0


def main(a):
    c = load_creds()
    bucket = c["S3_BUCKET"]
    prefix = f"us_stocks_sip/{a.data}/"
    s3 = make_client(c, a.workers)
    limiter = RateLimiter(a.rate)

    # 预检：1 个请求验 key。认证 403 ≠ 限流 403——认证错就立刻退出（别傻重试），用 key 尾4位帮你核对
    try:
        s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
        print(f"[预检] key(尾{c['S3_SECRET'][-4:]}) 认证 OK", flush=True)
    except ClientError as e:
        http = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0)
        print(f"[预检] ❌ 认证失败 HTTP{http}（key 尾{c['S3_SECRET'][-4:]}）：是否 regenerate 过 key？"
              f"去 Massive 控制台 Keys→Accessing Flat Files(S3) 复制最新 Secret 到 .massive_key", flush=True)
        return 1

    floor = f"（≥{a.start_date}）" if a.start_date else ""
    print(f"[列举] {bucket}/{prefix}  年份 {a.start_year}..{a.end_year}{floor} ...", flush=True)
    keys = list_keys(s3, bucket, prefix, a.start_year, a.end_year, a.start_date)
    if a.limit:
        keys = keys[:a.limit]
    total_bytes = sum(s for _, s in keys)
    print(f"  共 {len(keys):,} 个文件, {total_bytes/1e9:.1f} GB（未压缩前更大）", flush=True)

    done = {"n": 0, "bytes": 0, "ok": 0, "skip": 0, "err": 0}
    lock = threading.Lock()
    t0 = time.time()
    n_total = len(keys)

    def work(item):
        key, size = item
        st, b = download_one(s3, bucket, key, size, limiter)
        with lock:
            done["n"] += 1
            done["bytes"] += b
            done["ok"] += st == "ok"
            done["skip"] += st == "skip"
            if st.startswith("err"):
                done["err"] += 1
                if done["err"] <= 5:            # 头几个错误直接打出来（看清是节流还是真错）
                    print(f"  ⚠️ {key.split('/')[-1]}: {st}", flush=True)
            # 每 25 个完成 或 每 ~3 秒 报一次进度
            now = time.time()
            if done["n"] % 25 == 0 or done["n"] == n_total or now - done.get("last_print", 0) > 3:
                done["last_print"] = now
                el = now - t0
                rate = done["bytes"] / 1e6 / max(el, 1e-9)
                doing = done["ok"] + done["err"]
                eta = (n_total - done["n"]) / max(doing / max(el, 1e-9), 1e-9) if doing else 0
                print(f"  {done['n']:,}/{n_total:,}  下{done['ok']} 跳{done['skip']} 错{done['err']}  "
                      f"{done['bytes']/1e9:.1f}GB  {rate:.0f}MB/s  ETA {eta/60:.0f}min", flush=True)
        return st

    # as_completed：乱序报进度，跳过的立刻显示、退避中的文件不挡别人
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(work, it) for it in keys]
        errs = [f.result() for f in cf.as_completed(futs)]
        errs = [s for s in errs if s.startswith("err")]

    el = time.time() - t0
    print(f"\n完成：下 {done['ok']:,} 跳 {done['skip']:,} 错 {done['err']:,}  "
          f"{done['bytes']/1e9:.1f}GB  用时 {el/60:.1f}min", flush=True)
    if errs:
        from collections import Counter
        print("  错误类型:", dict(Counter(errs)), "→ 重跑本脚本会断点续传补齐", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="minute_aggs_v1",
                    help="minute_aggs_v1(默认) / day_aggs_v1 / trades_v1 / quotes_v1")
    ap.add_argument("--start-year", type=int, default=2003, dest="start_year")
    ap.add_argument("--end-year", type=int, default=2026, dest="end_year")
    ap.add_argument("--start-date", default=None, dest="start_date",
                    help="日期下限 YYYY-MM-DD，早于此的跳过。Massive 是滚动10年窗口，"
                         "更早的永久403。当前可访问起点约 2016-06-14。")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--rate", type=float, default=12, help="全局请求率上限(次/秒)，防 429/403 限流；干净后可调大")
    ap.add_argument("--limit", type=int, default=0, help="只下前 N 个文件（0=全部，测试/分批用）")
    sys.exit(main(ap.parse_args()))
