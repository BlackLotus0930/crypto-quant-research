"""P1b / E4：多 horizon 训练（Δt∈[dt_min,dt_max]）。
**判模型(alpha 研究层)用模型价值 = 真实(IR) × 新颖(⟂全因子残差仍预测)；赚不赚钱是交易系统层，降级为成本 gate。**
horizon 按 val_IR 选(让 alpha 选,不让交易系统选)；执行参数(rebal/band)按 val+成本选；每跑必出研究报告(eval/report.py)。
跑（pod 4090）： python run_gonogo.py --dt_min 1 --dt_max 5
"""
import argparse
import sys

import numpy as np
import torch

from eval import harness, backtest, report
from eval.ic import ic_table, print_ic
from model.train import sanity_overfit, fit, TorchPredictor, Net


def main(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    HZS = tuple(h for h in (1, 2, 3, 5, 10) if a.dt_min <= h <= a.dt_max) or (a.dt_min,)
    print(f"device={dev}  tensor={a.tensor or 'default512'}  T={a.T} d={a.d} L={a.L} "
          f"Δt∈[{a.dt_min},{a.dt_max}] 评估horizon={HZS} epochs={a.epochs} bs={a.bs} accum={a.accum} λ={a.lam}", flush=True)
    if a.tensor:
        tensor, splits = harness.load(a.tensor, a.splits) if a.splits else harness.load(a.tensor)
    else:
        tensor, splits = harness.load()
    C = tensor["n"].shape[-1]                             # 通道数（日内+tod=7，日频=5）从张量取
    print(f"  N={tensor['n'].shape[1]} slot, mask 真实率={tensor['mask'].mean():.3f}  "
          f"T_total={tensor['n'].shape[0]}  C={C}", flush=True)

    if a.ckpt:                                            # 载入已训模型，跳过训练（重判用）
        net = Net(d=a.d, L=a.L, T=a.T, C=C).to(dev)
        sd = torch.load(a.ckpt, map_location=dev)
        net.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in sd.items()})  # 兼容 torch.compile 存的 _orig_mod. 前缀
        print(f"[2/3] 载入 ckpt={a.ckpt}，跳过 sanity+训练", flush=True)
    else:
        print("[2] 机器 sanity：过拟合极小子集 ...", flush=True)
        f, l = sanity_overfit(tensor, splits, dev, T=a.T, d=a.d, L=a.L, bs=a.bs, C=C, residualize=a.residualize)
        print(f"    overfit {f:.4f} -> {l:.4f}  ({'OK 在学' if l < f * 0.6 else '⚠️ 没在学'})", flush=True)

        print(f"[3] 训练（多 horizon Δt∈[{a.dt_min},{a.dt_max}]，按 val_IC 选模型）...", flush=True)
        net, best_val_ic = fit(tensor, splits, dev, T=a.T, d=a.d, L=a.L, bs=a.bs, lr=a.lr,
                               epochs=a.epochs, dt_min=a.dt_min, dt_max=a.dt_max, accum=a.accum, lam=a.lam,
                               compile_model=a.compile, C=C, workers=a.workers, residualize=a.residualize)
    model = TorchPredictor(net, dev, T=a.T)

    class _ZeroIn:
        name = "zero-input"
        def __init__(self, m): self.m = m
        def predict(self, x, xmask, dt): return self.m.predict(np.zeros_like(x), xmask, dt)

    COST = a.cost_bps    # GO 判据用的现实成本（流动大盘股半价差+冲击 ~2bps）；2/5/10 仍报敏感性
    # [4] 选 horizon：只前向 val 算 val_IR（每 horizon 仅 1 次前向）。holdout/回测不在这里跑——
    #     只为选定的那个 horizon 跑一次，省掉非赢家 horizon 的 holdout IC + val 回测 + holdout 回测（多 horizon 下约省 6 成前向）。
    print(f"\n[4] 选 horizon（按 val_IR，仅前向 val；holdout/回测只为选定 horizon 跑=省冗余前向）", flush=True)
    sel = []
    for h in HZS:
        icv = ic_table(tensor, splits, "val", model, dt=h, T=a.T)
        irV = max(v["IR"] for v in icv.values())
        sel.append({"h": h, "irV": irV})
        print(f"  Δt={h}: val_IR={irV:+.2f}", flush=True)
    win = max(sel, key=lambda r: r["irV"]); best_h, irV = win["h"], win["irV"]   # horizon 按 val_IR 选(不碰 holdout)

    # [5] 仅对选定 horizon：holdout IC + 执行参数(val) + holdout 回测 + 完整报告
    print(f"\n[5] 选定 Δt={best_h}（val_IR={irV:+.2f} 最高）→ 仅此 horizon 跑 holdout 完整报告：", flush=True)
    ictbl = ic_table(tensor, splits, "holdout", model, dt=best_h, T=a.T)
    bsig, bic = print_ic(ictbl, dt=best_h)
    bt_val = backtest.run(tensor, splits, model, split="val", T=a.T, pred_dt=best_h)
    rb, bd = backtest.best_exec_at(bt_val, COST)                        # 执行参数(rebal+band)由 val+成本定
    sV = backtest.metrics(bt_val, COST, rb, bd)["sharpe"]
    bt = backtest.run(tensor, splits, model, split="holdout", T=a.T, pred_dt=best_h)
    m = {b: backtest.metrics(bt, b, rb, bd) for b in (2, 5, 10)}
    best = {"h": best_h, "IC": bic["IC"], "IR": bic["IR"], "irV": irV, "bt": bt, "rb": rb, "bd": bd, "sV": sV,
            "sC": m[COST]["sharpe"], "s2": m[2]["sharpe"], "s5": m[5]["sharpe"], "s10": m[10]["sharpe"]}
    print(f"  Δt={best_h}: holdout IC={bic['IC']:+.4f} IR={bic['IR']:+.2f} (val_IR={irV:+.2f})  →  "
          f"holdout净@2={m[2]['sharpe']:+.2f} @5={m[5]['sharpe']:+.2f} @10={m[10]['sharpe']:+.2f}  "
          f"(val选 rebal={rb} band={bd})", flush=True)
    backtest.report(best["bt"], name=f"Model Δt={best_h}", rebal=rb, band=bd)
    abl = ic_table(tensor, splits, "holdout", _ZeroIn(model), dt=best_h, T=a.T)
    print(f"  [ablation 输入置零] best IR={max(v['IR'] for v in abl.values()):+.2f}  (应≈0)", flush=True)

    try:
        ck = f"ckpt_e4{a.tag}.pt"; bk = f"holdout_book{a.tag}.npz"            # --tag 区分输出，共享卷并行不互覆盖
        torch.save({k.replace("_orig_mod.", ""): v for k, v in net.state_dict().items()}, ck)  # 剥 compile 前缀→可直接 Net().load
        np.savez_compressed(bk, dates=best["bt"]["dates"], W=best["bt"]["W"], R=best["bt"]["R"])
        print(f"  已存 {ck} + {bk}", flush=True)
    except Exception as e:
        print(f"  存盘跳过: {e}", flush=True)

    # 研究报告先跑：它产出"模型价值"的分解(真实/新颖/校准/半衰期/成本前沿)，是主判据来源
    rep = None
    if a.report:
        try:
            rep = report.report(tensor, splits, model, a.T, best["h"], bt=best["bt"])   # 复用已算的 holdout 账本=省 1 次前向
        except Exception as e:
            print(f"  研究报告跳过: {e}", flush=True)

    # ====== VERDICT：判模型(alpha 研究层)，不判交易系统 ======
    # 第一性原理：无先验模型的价值 = 真实(IR) × 新颖(⟂全因子残差仍预测) × 可交易 horizon。
    #   "赚不赚钱"是整条栈(模型×组合×执行)的指标，会把真信号误杀(见 E9)→降级为成本 gate。
    print("\n" + "=" * 64)
    if rep is not None:                                            # 阈值=设计.md §0.2 模型毕业标准
        retain = (best["IR"] / best["irV"]) if best["irV"] > 0 else 0.0   # val→holdout IC 保留率(不塌)
        real = rep["ic_t"] >= 3.0 and retain >= 0.50               # 真实: t≥3(非运气) 且 保留≥50%
        new = rep["nov_t"] >= 3.0 and rep["nov_pct"] >= 0.50       # 新颖: ⟂全因子残差 t≥3 且占比≥50%
        trad5 = rep["front_net5"] >= 0.5                           # taker 口径有可交易区间
        trad2 = rep["front_net2"] >= 0.5                           # 仅 maker 口径可交易(逆选风险)
        print(f"[模型价值·alpha研究层] 真实: IC={rep['ic']:+.4f} t={rep['ic_t']:+.1f} 保留={retain:.0%}(线 t≥3,保留≥50%)   "
              f"新颖: ⟂全因子残差 t={rep['nov_t']:+.1f} 占比={rep['nov_pct']:.0%}(线 t≥3,占比≥50%)")
        print(f"                       分布: spread→|收益| IC={rep['spr_ic']:+.4f}   半衰期≈{rep['half_life']:.0f}bar")
        print(f"[可交易 gate·交易系统层(后打磨)] 成本前沿最优净 Sharpe @2bps={rep['front_net2']:+.2f} @5bps={rep['front_net5']:+.2f}"
              f"   (裸执行净@{COST:g}bps={best['sC']:+.2f}=交易系统指标,非模型指标)")
        if real and new:
            v = "✅ 模型层 GO：真实且新颖的 alpha(⟂全因子残差仍预测→thesis 有支撑,值得放大)"
            if trad5:
                v += "；taker 口径已可交易"
            elif trad2:
                v += "；仅 maker 口径可交易(逆选风险)→ 执行另立板块打磨"
            else:
                v += "；当前执行口径不可交易→交易系统/市场问题,非模型问题(换场/执行,勿弃模型)"
        elif real and not new:
            v = "🟡 真实但 novelty 不足(残差 t<3 或占比<50%)→ thesis 未证：加正交数据流(订单流/跨市场)逼高阶"
        else:
            v = "❌ 模型层 NO-GO：信号不真或塌缩(t<3 或 val→holdout 保留<50%)"
    else:
        sC = best["sC"]                                             # 无报告时退回旧标量口径
        v = (f"✅ 净@{COST:g}bps={sC:+.2f}≥0.5" if sC >= 0.5
             else f"🟡 IR={best['IR']:+.2f}≥1" if best["IR"] >= 1.0 else "❌ 无 alpha")
    print("VERDICT:", v)
    print("=" * 64)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--bs", type=int, default=32, help="批大小（横截面IC损失对batch线性→bs32/accum1 ≈ bs8/accum4 等价但更快）")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--dt_min", type=int, default=1)
    ap.add_argument("--dt_max", type=int, default=5)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4, help="DataLoader 进程数（pod上开，与GPU重叠；本地调试设0避僵尸）")
    ap.add_argument("--lam", type=float, default=0.2)
    ap.add_argument("--tensor", default=None, help="张量路径（放大 universe / 日内 用）；默认 512")
    ap.add_argument("--splits", default=None, help="切分 json 路径（日内用 intraday_splits_*.json）")
    ap.add_argument("--universe", default="data/clean/universe.parquet", help="universe parquet（取因子槽做中性化）")
    ap.add_argument("--cost_bps", type=float, default=2.0, help="GO 判据用的现实单边成本（流动大盘股 ~2bps）")
    ap.add_argument("--compile", action="store_true", help="开 torch.compile（~1.3-1.8× 提速；首步编译预热慢；形状须静态）")
    ap.add_argument("--ckpt", default=None, help="载入已训 state_dict 跳过训练（仅重判/换执行口径用）")
    ap.add_argument("--no_report", dest="report", action="store_false", help="关掉伴生研究报告（默认开）")
    ap.add_argument("--tag", default="", help="输出文件(ckpt/book)后缀；并行多 pod 共享卷时避免互相覆盖")
    ap.add_argument("--residualize", action="store_true", help="novelty-聚焦：训练目标对已知因子(rev/动量/波动/量)残差化")
    sys.exit(main(ap.parse_args()))
