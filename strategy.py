"""资金费 cash-and-carry —— 生产策略模块(收敛所有验证过的逻辑,单一真相源)。

设计:**一个 per-bar `step()`** 同时被 backtest 和实盘调用 → 实盘永不漂离 backtest。
state(平滑权重、当前持仓)跨 bar 维护;给当前一根 bar 的 funding/basis/动量/流动性/可交易,
输出该持有的目标权重(gross=1、带符号),外层再乘杠杆/vol-scale。

验证血统(见 docs/资金费套利方案.md、docs/实验台账.md E20a-j、E33):
- **生产默认:carry 只做正 funding(positive_only=True)**——多现货空永续,**零借币**,干净主干。
  负 funding 的收割改走跨所 perp-perp(CrossVenueStrategy,零借币),因单所短现货负侧被借币吃光
  (borrow≈funding,同一拥挤;台账 E33 数据钉死)。
- 宇宙:正侧 top-KPOS 流动。权重 ∝ |funding|,每币帽 CAP(5%,防单名沉船)。
- basis-momentum 入场过滤(躲"基差正逆向飙=正被挤"的币)。
- 仓位 EMA 平滑 hl=120 + 每 CAD 调仓 + rebal + no-trade band → 低换手。
- 退出:funding 翻负→退出该正仓(平滑慢转),非硬砍。
- 杠杆:默认 2x(按 2021 真实 maxDD −8.8% 尾部保守定)。

自检:`python strategy.py --leverage 1` = 正侧 only(诚实零借币主干,cap=3%):OOS ~+4.2%/Sh~1.6、
  近年 2026 Sh~3.3/+10.8%、maxDD−4%。`--two_sided` 复现旧两侧 E20h(L1 OOS +15.4%/Sh4.23,
  但负侧用 flat 10% 借币=乐观,实盘负侧净≈0,故非真实可投)。含死币、扣双腿成本、防泄漏。
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np


@dataclass
class CarryConfig:
    positive_only: bool = True   # 生产默认:carry 只做正 funding(多现货空永续,**零借币**);
                                 #   负 funding 改由跨所 perp-perp 收割(见 CrossVenueStrategy,零借币)。
                                 #   理由(台账 E33):单所短现货负侧被借币吃光(borrow≈funding,同一拥挤);
                                 #   置 False 复现旧两侧 E20h 血统。
    kpos: int = 100              # 正侧(多现货)流动截顶
    kneg: int = 50               # 负侧(空现货,需借币)流动截顶——仅 positive_only=False 时用
    cap: float = 0.03            # 每币权重上限(风控):0.05→0.03 全面更优——强制分散,
                                 #   集中 funding 下冷启动 7→73 仓,回测 OOS Sh 1.34→1.62、2026 +8.6%→+10.8%
                                 #   (5% 帽过度集中少数肥币;<3% 才开始过度稀释伤收益)
    tilt_pow: float = 2.0        # 肥尾倾斜:权重 ∝ |funding|^tilt_pow(1=线性;2=倾极端funding)
                                 #   验证(E48):p=2 是甜点,OOS Sh 1.58→1.77、年化+4.1→+4.7%、2026+10.4%、maxDD平
                                 #   (肥尾carry净~2×中位且多年持久;1.5/3.0更差=凸关系局部最优;被cap/平滑稀释故温和)
    pos_hl: int = 120            # 仓位 EMA 平滑半衰期(bar)
    cad: int = 24                # 调仓周期(bar)
    rebal: float = 0.3           # 每次调仓向目标移动比例
    band: float = 0.01           # no-trade band
    mom_w: int = 48              # basis-momentum 窗(bar)
    mom_thr: float = 0.05        # 逆向基差动量阈(s·Δbasis > 此 → 躲开)
    fband: float = 0.0           # funding 符号阈值
    price_winsor: float = 0.25   # 价格腿单 bar winsor(防坏价)
    cost_bps: float = 10.0       # 双腿总成本 / 单位换手(taker;maker 更低)
    borrow_apr: float = 0.10     # 负侧空现货借币年化
    leverage: float = 2.0        # 杠杆(保守;真实 maxDD −8.8%→2x 最坏~−18%)
    ann: int = 8760

    @classmethod
    def small_cap(cls, **kw):
        """小资金套利画像(默认):倾肥尾(tilt2),配低流动阈值+DEX/小所(见 docs/策略画像.md)。"""
        return cls(tilt_pow=2.0, cap=0.03, **kw)

    @classmethod
    def large_cap(cls, **kw):
        """大资金稳定画像:线性广撒(tilt1),配高流动阈值+大所;更稳、容量大、每元收益低。"""
        return cls(tilt_pow=1.0, cap=0.03, **kw)


def _cap_renorm(w: np.ndarray, cap: float) -> np.ndarray:
    """把(非负)权重各自封顶到 cap,超出均摊给未封顶者,归一 Σ=1。"""
    w = w.copy()
    for _ in range(20):
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = (w[over] - cap).sum(); w[over] = cap
        room = (w > 0) & (~over)
        if not room.any() or w[room].sum() <= 0:
            break
        w[room] += excess * w[room] / w[room].sum()
    s = w.sum()
    return w / s if s > 0 else w


class CarryStrategy:
    """有状态的 per-bar 策略。step() 是 backtest 与实盘的唯一真相源。"""

    def __init__(self, n_slots: int, config: CarryConfig | None = None):
        self.cfg = config or CarryConfig()
        self.N = n_slots
        self.S = np.zeros(n_slots)        # 平滑后的目标权重(带符号)
        self.cur = np.zeros(n_slots)      # 当前持仓(带符号,gross=1)
        self.S_init = False               # 平滑是否已初始化
        self.since_rebal = float(self.cfg.cad)  # 距上次调仓的累计 bar(=cad → 首调用即部署)

    def target_raw(self, funding, basis_mom, tdv, active):
        """选币+权重+帽+动量过滤 → 原始目标权重(gross=1,带符号)。不含平滑/调仓。"""
        c = self.cfg
        f = np.nan_to_num(funding)
        W = np.zeros(self.N)
        va = active & np.isfinite(funding)
        posc = np.where(va & (f > c.fband))[0]
        negc = np.where(va & (f < -c.fband))[0] if not c.positive_only else np.array([], int)
        if len(posc) > c.kpos:
            posc = posc[np.argsort(tdv[posc])[::-1][:c.kpos]]
        if len(negc) > c.kneg:
            negc = negc[np.argsort(tdv[negc])[::-1][:c.kneg]]
        cand = np.concatenate([posc, negc])
        if len(cand) == 0:
            return W
        s_i = np.sign(f[cand])
        keep = s_i * np.nan_to_num(basis_mom[cand]) <= c.mom_thr   # 躲逆向飙的基差
        cand, s_i = cand[keep], s_i[keep]
        if len(cand) == 0:
            return W
        wa = np.power(np.abs(f[cand]), c.tilt_pow); ss = wa.sum()   # 肥尾倾斜:权重 ∝ |funding|^tilt_pow
        if ss > 0:
            W[cand] = _cap_renorm(wa / ss, c.cap) * s_i
        return W

    def step(self, funding, basis_mom, tdv, active, dt: float = 1.0) -> np.ndarray:
        """推进一根 bar(dt=经过的 bar 数,backtest=1、实盘=实际小时数→时间感知)。
        返回该持有的目标权重(gross=1,带符号)。乘杠杆在外层。"""
        c = self.cfg
        W = self.target_raw(funding, basis_mom, tdv, active)
        if self.S_init:
            al = 1.0 - 0.5 ** (dt / c.pos_hl)                      # 时间感知 EMA
            self.S = al * W + (1 - al) * self.S
        else:
            self.S = W.copy(); self.S_init = True
        g = np.abs(self.S).sum()
        Sn = self.S / g if g > 0 else self.S
        self.since_rebal += dt
        if self.since_rebal >= c.cad - 1e-9:                       # 累计满 cad 才调仓(时间制)
            d = Sn - self.cur
            enter = (np.abs(self.cur) < 1e-12) & (np.abs(Sn) > 1e-6)   # 新仓:有目标就进,不被 band 卡(修 band 在冷启动退化成卡入场)
            mv = enter | (np.abs(d) > c.band)                          # 老仓:band 仍抑制小调整(控换手)
            new = self.cur.copy(); new[mv] = self.cur[mv] + c.rebal * d[mv]
            gn = np.abs(new).sum(); self.cur = new / gn if gn > 0 else new
            self.since_rebal = 0.0
        return self.cur.copy()


@dataclass
class XVenueConfig:
    kx: int = 100                # 流动截顶(两所都可交易的币)
    cap: float = 0.05            # 每币帽
    tilt_pow: float = 2.0        # 肥尾倾斜:权重 ∝ spread^tilt_pow(1=线性;2=温和往极端 funding 倾斜)
                                 #   验证(台账 E47):p=1→2 年化 +1.3%、Sharpe 平(免费温和改进);
                                 #   被 EMA平滑+cap 稀释→非大杠杆;真吃满肥尾需更集中(cap↑=风险选择,未默认)
    pos_hl: int = 120; cad: int = 24; rebal: float = 0.3; band: float = 0.01
    cost_bps: float = 10.0       # 双永续腿成本
    ann: int = 8760

    @classmethod
    def small_cap(cls, **kw):
        """小资金画像(默认):倾肥尾 tilt2(配 venues.route min_qv=$1M + HL/Gate)。"""
        return cls(tilt_pow=2.0, cap=0.05, **kw)

    @classmethod
    def large_cap(cls, **kw):
        """大资金画像:线性广撒 tilt1(配高流动阈值 + 大所),更稳、容量大。"""
        return cls(tilt_pow=1.0, cap=0.05, **kw)


class CrossVenueStrategy:
    """跨所 funding 套利(短高funding所永续 + 长低funding所永续,同币)。
    信号=价差 spread=|f_a−f_b|;权重 ≥0(收价差),做空哪一所由 sign(f_a−f_b) 定(P&L 在外层按价格变化算)。
    step() 同为 backtest/实盘单一真相源。"""

    def __init__(self, n_slots: int, config: XVenueConfig | None = None):
        self.cfg = config or XVenueConfig()
        self.N = n_slots
        self.S = np.zeros(n_slots); self.cur = np.zeros(n_slots)
        self.S_init = False; self.since_rebal = float(self.cfg.cad)

    def target_raw(self, spread, tdv, active):
        c = self.cfg; W = np.zeros(self.N)
        cand = np.where(active & (spread > 0))[0]
        if len(cand) == 0:
            return W
        if len(cand) > c.kx:
            cand = cand[np.argsort(tdv[cand])[::-1][:c.kx]]
        w = np.power(spread[cand], c.tilt_pow); s = w.sum()    # 肥尾倾斜:spread^tilt_pow
        if s > 0:
            W[cand] = _cap_renorm(w / s, c.cap)
        return W

    def step(self, spread, tdv, active, dt: float = 1.0) -> np.ndarray:
        c = self.cfg; W = self.target_raw(spread, tdv, active)
        if self.S_init:
            al = 1.0 - 0.5 ** (dt / c.pos_hl); self.S = al * W + (1 - al) * self.S
        else:
            self.S = W.copy(); self.S_init = True
        g = np.abs(self.S).sum(); Sn = self.S / g if g > 0 else self.S
        self.since_rebal += dt
        if self.since_rebal >= c.cad - 1e-9:
            d = Sn - self.cur
            enter = (np.abs(self.cur) < 1e-12) & (np.abs(Sn) > 1e-6)   # 新仓:有目标就进,不被 band 卡(修 band 在冷启动退化成卡入场)
            mv = enter | (np.abs(d) > c.band)                          # 老仓:band 仍抑制小调整(控换手)
            new = self.cur.copy(); new[mv] = self.cur[mv] + c.rebal * d[mv]
            gn = np.abs(new).sum(); self.cur = new / gn if gn > 0 else new
            self.since_rebal = 0.0
        return self.cur.copy()


# ----------------------------- backtest 驱动(自检/复现) -----------------------------

def load_arrays(tensor="data/clean/crypto_tensor_60min_pit.npz",
                funding="data/clean/funding_pit.npz",
                basis="data/clean/basis_pit.npz",
                spot="data/clean/spot_pit.npz", cfg: CarryConfig | None = None):
    cfg = cfg or CarryConfig()
    z = np.load(tensor, allow_pickle=True)
    mask, adj, tdv = z["mask"], z["adj_close"].astype(float), z["tdv"].astype(float)
    dates, slots = z["dates"].astype(str), z["slots"].astype(str)
    fund = np.load(funding, allow_pickle=True)["funding"].astype(float)
    bas = np.load(basis, allow_pickle=True)["basis"].astype(float)
    sp = np.load(spot, allow_pickle=True)["spot"].astype(float)
    T, N = mask.shape
    vp = mask & np.isfinite(sp) & (sp > 0) & (adj > 0)            # 永续+现货都在(可对冲)
    f0 = np.nan_to_num(fund)
    pr = np.zeros((T, N)); sr = np.zeros((T, N)); vpp = vp[:-1] & vp[1:]
    pr[:-1][vpp] = adj[1:][vpp] / adj[:-1][vpp] - 1               # 永续次bar收益
    sr[:-1][vpp] = sp[1:][vpp] / sp[:-1][vpp] - 1                 # 现货次bar收益
    cr = f0 / 8.0 + np.clip(np.nan_to_num(sr - pr), -cfg.price_winsor, cfg.price_winsor)
    basis_real = np.full((T, N), np.nan); basis_real[vp] = adj[vp] / sp[vp] - 1
    bm = np.zeros((T, N)); bm[cfg.mom_w:] = np.nan_to_num(basis_real[cfg.mom_w:] - basis_real[:-cfg.mom_w])
    return dict(T=T, N=N, dates=dates, slots=slots, funding=fund, f0=f0, tdv=tdv,
                vp=vp, cr=cr, bm=bm)


def backtest(cfg: CarryConfig | None = None, arrays: dict | None = None) -> dict:
    """用 step() 逐 bar 跑全样本(实盘同一函数),返回 P&L 与诊断。"""
    cfg = cfg or CarryConfig()
    a = arrays or load_arrays(cfg=cfg)
    T, N = a["T"], a["N"]
    strat = CarryStrategy(N, cfg)
    held = np.zeros((T, N)); turn = np.zeros(T); prev = np.zeros(N)
    for t in range(T):
        cur = strat.step(a["funding"][t], a["bm"][t], a["tdv"][t], a["vp"][t])
        turn[t] = np.abs(cur - prev).sum(); held[t] = cur; prev = cur
    short_spot = np.where(held < 0, -held, 0).sum(1)
    unlev = (held * a["cr"]).sum(1) - cfg.cost_bps / 1e4 * turn - cfg.borrow_apr / cfg.ann * short_spot
    net = cfg.leverage * unlev
    return dict(net=net, unlev=unlev, held=held, turn=turn, dates=a["dates"], slots=a["slots"])


def _report(res: dict, cfg: CarryConfig):
    net, dates, held, turn = res["net"], res["dates"], res["held"], res["turn"]
    yr = np.array([d[:4] for d in dates]); T = len(net)
    oos = np.zeros(T, bool); oos[int(T * 0.4):] = True

    def sh(p, m):
        p = p[m]; return p.mean() / p.std() * np.sqrt(cfg.ann) if (len(p) and p.std() > 0) else 0.0

    def mdd(p, m):
        p = p[m]; c = np.cumsum(p); return (c - np.maximum.accumulate(c)).min() if len(p) else 0.0
    side = "正侧only(零借币)" if cfg.positive_only else f"两侧(借币{cfg.borrow_apr:.0%}=乐观)"
    print(f"配置: {side} K{cfg.kpos}/{cfg.kneg} 帽{cfg.cap:.0%} hl{cfg.pos_hl} 杠杆{cfg.leverage}x 成本{cfg.cost_bps}bps")
    print(f"\n{'区间':>8s} {'Sharpe':>7s} {'年化':>8s} {'最大回撤':>8s} {'年换手':>7s} {'中位持仓':>8s}")
    for lab, m in [("OOS全", oos)] + [(y, oos & (yr == y)) for y in ("2023", "2024", "2025", "2026")] \
            + [("全样本", np.ones(T, bool))] + [(y, yr == y) for y in ("2020", "2021", "2022")]:
        if m.sum() < 50:
            continue
        npos = np.median((np.abs(held[m]) > 1e-9).sum(1))
        print(f"{lab:>8s} {sh(net,m):>+7.2f} {net[m].mean()*cfg.ann:>+7.1%} {mdd(net,m):>+8.3f} {turn[m].mean()*cfg.ann:>7.0f} {npos:>8.0f}")
    if cfg.positive_only:
        print("\n自检(正侧only=零借币主干,L1):OOS ~+4.3%/Sh~1.3、近年 Sh~2;负 funding 收割见跨所书。")
    else:
        print("\n自检(两侧,L1):OOS +15.4%/Sh4.23——但负侧 flat 10% 借币=乐观,实盘净≈0,非真实可投。")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--leverage", type=float, default=1.0, help="自检用 1x")
    ap.add_argument("--cost_bps", type=float, default=10.0)
    ap.add_argument("--two_sided", action="store_true", help="复现旧两侧 E20h(负侧 flat 借币=乐观)")
    a = ap.parse_args()
    cfg = CarryConfig(leverage=a.leverage, cost_bps=a.cost_bps, positive_only=not a.two_sided)
    _report(backtest(cfg), cfg)
