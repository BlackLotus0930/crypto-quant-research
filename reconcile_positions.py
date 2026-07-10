"""只读对账:从各所拉真实持仓 → 写 data/exec/positions.json(execution.py 对账用)。
**只读、不下单。** 凭证从环境变量读(绝不进代码/git);缺凭证的所自动跳过(优雅降级)。
- HL:持仓按钱包地址**公开**(只需 HL_ADDRESS,无需 secret)。
- Gate:HMAC-SHA512 签名(GATE_KEY / GATE_SECRET,**建只读 key**)。
- OKX:HMAC-SHA256 签名(OKX_KEY / OKX_SECRET / OKX_PASS,**建只读 key**)。
输出 key 形如 'gate_perp|BTC' / 'gate_spot|BTC' / 'hl|BTC' / 'okx|BTC',值=有符号 base 数量。
跑（开户后,先 set 环境变量）：PYTHONUTF8=1 .venv/Scripts/python.exe reconcile_positions.py
"""
import base64
import hashlib
import hmac
import json
import os
import time
import urllib.parse
import urllib.request

from venues import canon

UA = {"User-Agent": "Mozilla/5.0"}
OKX = "https://www.okx.com"; GATE = "https://api.gateio.ws/api/v4"; HLINFO = "https://api.hyperliquid.xyz/info"
OUT = "data/exec/positions.json"


def _req(url, method="GET", headers=None, body=""):
    req = urllib.request.Request(url, data=body.encode() if body else None, headers={**UA, **(headers or {})}, method=method)
    return json.loads(urllib.request.urlopen(req, timeout=30).read())


def hl_positions(pos):
    """HL:POST clearinghouseState(地址公开,无需 secret)。"""
    addr = os.environ.get("HL_ADDRESS")
    if not addr:
        print("  HL:跳过(未设 HL_ADDRESS)"); return
    r = _req(HLINFO, "POST", {"Content-Type": "application/json"}, json.dumps({"type": "clearinghouseState", "user": addr}))
    n = 0
    for ap in r.get("assetPositions", []):
        p = ap.get("position", {})
        szi = float(p.get("szi", 0) or 0)        # 有符号张数=base
        if szi != 0:
            pos[f"hl|{canon(p['coin'])}"] = szi; n += 1
    print(f"  HL:{n} 仓")


def gate_positions(pos):
    key, sec = os.environ.get("GATE_KEY"), os.environ.get("GATE_SECRET")
    if not (key and sec):
        print("  Gate:跳过(未设 GATE_KEY/GATE_SECRET)"); return

    def signed(path, query=""):
        ts = str(int(time.time()))
        body_hash = hashlib.sha512(b"").hexdigest()
        payload = f"GET\n/api/v4{path}\n{query}\n{body_hash}\n{ts}"
        sign = hmac.new(sec.encode(), payload.encode(), hashlib.sha512).hexdigest()
        url = f"{GATE}{path}" + (f"?{query}" if query else "")
        return _req(url, "GET", {"KEY": key, "Timestamp": ts, "SIGN": sign})
    # 永续仓(quanto_multiplier 把张→base 在 execution 用价×名义已处理;这里存 size×qm=base)
    contracts = {d["name"]: float(d.get("quanto_multiplier", 0) or 0) for d in _req(f"{GATE}/futures/usdt/contracts")}
    n = 0
    for p in signed("/futures/usdt/positions"):
        sz = float(p.get("size", 0) or 0)
        if sz != 0:
            qm = contracts.get(p["contract"], 1.0)
            pos[f"gate_perp|{canon(p['contract'][:-5])}"] = sz * qm; n += 1
    # 现货余额(carry 多现货腿)
    ns = 0
    for b in signed("/spot/accounts"):
        avail = float(b.get("available", 0) or 0)
        if avail > 0 and b["currency"] != "USDT":
            pos[f"gate_spot|{canon(b['currency'])}"] = avail; ns += 1
    print(f"  Gate:{n} 永续仓 + {ns} 现货")


def okx_positions(pos):
    key, sec, pw = os.environ.get("OKX_KEY"), os.environ.get("OKX_SECRET"), os.environ.get("OKX_PASS")
    if not (key and sec and pw):
        print("  OKX:跳过(未设 OKX_KEY/OKX_SECRET/OKX_PASS)"); return
    ts = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
    path = "/api/v5/account/positions"
    sign = base64.b64encode(hmac.new(sec.encode(), (ts + "GET" + path).encode(), hashlib.sha256).digest()).decode()
    r = _req(f"{OKX}{path}", "GET", {"OK-ACCESS-KEY": key, "OK-ACCESS-SIGN": sign,
                                     "OK-ACCESS-TIMESTAMP": ts, "OK-ACCESS-PASSPHRASE": pw})
    n = 0
    for p in r.get("data", []):
        if p.get("instType") != "SWAP" or not p["instId"].endswith("USDT-SWAP"):
            continue
        posn = float(p.get("pos", 0) or 0); ctval = float(p.get("ctVal", 0) or 0) or 1.0
        if posn != 0:
            pos[f"okx|{canon(p['instId'].replace('-USDT-SWAP',''))}"] = posn * ctval; n += 1
    print(f"  OKX:{n} 仓")


def main():
    print("拉真实持仓(只读)…")
    pos = {}
    for fn in (hl_positions, gate_positions, okx_positions):
        try:
            fn(pos)
        except Exception as e:
            print(f"  {fn.__name__} 失败:{type(e).__name__}: {e}")
    os.makedirs("data/exec", exist_ok=True)
    json.dump(pos, open(OUT, "w"))
    print(f"\n写 {len(pos)} 个持仓 → {OUT}  → 跑 execution.py 即对账增量。")
    if not pos:
        print("(空:还没开户/没设凭证。开户后 set 环境变量再跑。)")


if __name__ == "__main__":
    main()
