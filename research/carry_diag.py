"""诊断:cash-and-carry 的 2026 负 funding 腿,有多少是"book 太黏、被困在已翻负的旧仓"(可修),
多少是"市场上根本没正 funding 可收"(结构)。并预览两种退出修法的 funding 腿天花板。
跑：python carry_diag.py
"""
import numpy as np

z = np.load("data/clean/crypto_tensor_60min_pit.npz", allow_pickle=True)
mask = z["mask"]; adj = z["adj_close"].astype(float); dates = z["dates"].astype(str)
T, N = mask.shape
funding = np.load("data/clean/funding_pit.npz", allow_pickle=True)["funding"].astype(float)
spot = np.load("data/clean/spot_pit.npz", allow_pickle=True)["spot"].astype(float)
yr = np.array([d[:4] for d in dates]); cut = int(T * 0.4)
oos = np.zeros(T, bool); oos[cut:] = True
ann = 8760
vp = mask & np.isfinite(spot) & (spot > 0) & (adj > 0)
active = vp & np.isfinite(funding)
f0 = np.nan_to_num(funding)
CAD, REBAL, BAND, HL = 24, 0.3, 0.01, 72


def make_W(weightfn):
    W = np.zeros((T, N))
    for t in range(T):
        a = active[t]
        if a.sum() < 5:
            continue
        w = weightfn(t, a)
        s = np.abs(w).sum()
        if s > 0:
            W[t, np.where(a)[0]] = w / s
    return W


def smooth(W, h):
    al = 1 - 0.5 ** (1.0 / h); S = np.empty_like(W); S[0] = W[0]
    for t in range(1, len(W)):
        S[t] = al * W[t] + (1 - al) * S[t - 1]
    g = np.abs(S).sum(1, keepdims=True)
    return np.divide(S, g, out=np.zeros_like(S), where=g > 0)


def build_held(W, renorm=True, hard_exit=False):
    """重建持仓;hard_exit=True 时每bar把当前 funding<0 的仓位清零(快退);renorm 控制是否补满。"""
    Ws = smooth(W, HL); cur = np.zeros(N); held = np.zeros((T, N))
    for t in range(T):
        if t % CAD == 0:
            d = Ws[t] - cur; mv = np.abs(d) > BAND
            new = cur.copy(); new[mv] = cur[mv] + REBAL * d[mv]
            s = np.abs(new).sum(); new = new / s if s > 0 else new
            cur = new
        if hard_exit:
            cur = np.where(f0[t] < 0, 0.0, cur)              # 当前 funding 翻负→立刻清零
            if renorm:
                s = np.abs(cur).sum(); cur = cur / s if s > 0 else cur
        held[t] = cur
    return held


def fleg(held, m):
    p = (held * f0 / 8.0).sum(1)
    return p[m].mean() * ann, (p[m].mean() / p[m].std() * np.sqrt(ann) if p[m].std() > 0 else 0)


W = make_W(lambda t, a: np.clip(f0[t, a], 0, None))           # 现有构造:权重∝正funding
base = build_held(W)

print("=== 现有构造:被困在'已翻负'旧仓的比例(证明黏性) ===")
print(f"{'年':>6s} {'负funding仓占比':>14s} {'持仓加权funding(年化)':>20s}")
for y in ("2024", "2025", "2026"):
    m = oos & (yr == y)
    if m.sum() < 50:
        continue
    h = base[m]; g = np.abs(h).sum(1)
    negfrac = (np.where(f0[m] < 0, np.abs(h), 0).sum(1) / np.where(g > 0, g, 1)).mean()
    a_, _ = fleg(base, m)
    print(f"{y:>6s} {negfrac*100:>13.1f}% {a_:>19.1%}")

print("\n=== 修法预览:funding 腿年化(对比) ===")
he_renorm = build_held(W, renorm=True, hard_exit=True)        # 翻负即退+补满到正funding
he_derisk = build_held(W, renorm=False, hard_exit=True)       # 翻负即退、不补满(gross缩=regime去险)
print(f"{'年':>6s} {'现有(黏)':>10s} {'翻负退+补满':>12s} {'翻负退+缩仓':>12s} {'缩仓后均gross':>14s}")
for y in ("2024", "2025", "2026"):
    m = oos & (yr == y)
    if m.sum() < 50:
        continue
    b, _ = fleg(base, m); r, _ = fleg(he_renorm, m); d, _ = fleg(he_derisk, m)
    g = np.abs(he_derisk[m]).sum(1).mean()
    print(f"{y:>6s} {b:>10.1%} {r:>12.1%} {d:>12.1%} {g:>13.0%}")

print("\n判读:负funding仓占比2026高=黏性真存在(可修);翻负退后2026 funding腿≥0=确是构造问题,缩仓版顺带控住regime。")
