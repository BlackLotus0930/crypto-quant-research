# Crypto Quant Research

Two parallel quantitative research tracks on crypto markets:

- **Track 1 — Market foundation model** (`archive/info-alpha/`): an any-variate transformer trained to predict cross-sectional crypto/equity returns from OHLCV. It found a real, measurable information edge (IC ~0.025) and a full pipeline/model/eval stack was built around it; work here is paused while Track 2 is in live testing (see below for why, and for the specific levers — more assets, alt data, longer horizon, capacity-aware sizing — that would reopen it).
- **Track 2 — Cash-and-carry funding arbitrage** (`research/`, `data_build/`, top-level scripts): a market-neutral production strategy, currently running in live forward paper trading. Hold spot + inverse perpetual per coin to hedge out price risk and collect the funding-rate spread, then route the residual across venues for extra carry.

## Track 1: Market foundation model

An any-variate, permutation-equivariant transformer over OHLCV panels, trained to predict cross-sectional return distributions and evaluated via information coefficient and long/short backtests.

**Where it stands:** the model surfaces a genuine cross-sectional signal (IC ~0.025) that survives standard leak/cost checks, but at current scope (single OHLCV feature set, daily rebalance) it runs into a Grossman-Stiglitz-style ceiling — the edge is thin enough that a 5bps taker fee compresses it close to breakeven at reasonable capacity. That's a precise, quantified constraint rather than a dead end: it directly names the levers that would reopen the edge (wider asset universe, alternative data beyond price/volume, longer holding horizons, capacity-aware sizing), and it's why near-term effort shifted to Track 2 (a strategy that monetizes a structural risk premium instead of a competed-down information edge — see [`docs/资金费套利方案.md`](docs/资金费套利方案.md) §1 for the full reasoning). The architecture, data pipeline, and eval harness below are a working, reusable base for resuming this line.

| Path | Role |
|---|---|
| [`archive/info-alpha/model/`](archive/info-alpha/model/) | Model definition — dual-axis (asset + time) attention encoder ([`net.py`](archive/info-alpha/model/net.py)), quantile head, training loop. |
| [`archive/info-alpha/pipeline/`](archive/info-alpha/pipeline/) | Dataset construction, cleaning, normalization, and train/val/test splitting. |
| [`archive/info-alpha/eval/`](archive/info-alpha/eval/) | Backtest harness, baselines, CRPS/IC scoring, reporting. |
| [`archive/info-alpha/*.py`](archive/info-alpha/) | Probes and diagnostics run while chasing the signal (funding/OI/netflow features, cost-aware execution, walk-forward validation, beta-neutral book construction). |
| [`archive/probes-done/`](archive/probes-done/) | Earlier exploratory probes (cross-market signals, ablations, data-source screens) that predate the info-alpha model. |
| [`archive/us-era/`](archive/us-era/) | Earliest era of the project, focused on US equities before the pivot to crypto. |
| [`docs/archive-info-alpha/`](docs/archive-info-alpha/) | Design docs for the model era (data, architecture, model design). |

## Track 2: Cash-and-carry strategy

**Idea:** perpetual funding is the fee leveraged traders pay to hold a position — a price of crowding/risk, not information. Unlike a return-prediction signal, it isn't competed away to zero, and because the position is held (not traded on noisy signals), turnover and cost sensitivity are low.

**Mechanics per coin:** `sign(funding)` determines direction — long spot + short perp when funding is positive (collect the fee longs pay), or the reverse when negative. Price risk cancels between the two legs; only the funding cash flow remains. Cross-sectional weights are proportional to `|funding|^tilt_pow`, capped per-coin, smoothed with an EMA to keep turnover low.

Backtest (dead-coin inclusive, cost-adjusted, leak-checked, OOS 2023–2026): ~+15%/yr unlevered, Sharpe ~4, max drawdown −4%, ~38 effective coins; +31–46%/yr at 2–3x leverage. See [`docs/资金费套利方案.md`](docs/资金费套利方案.md) and [`docs/实验台账.md`](docs/实验台账.md) for the full experiment ledger.

Production defaults to the **positive-funding side only** (long spot / short perp, zero borrowing). The negative-funding side is harvested separately via cross-venue perp-perp routing instead of single-venue short-spot (which requires borrowing that eats the funding edge) — see `CrossVenueStrategy` in [`strategy.py`](strategy.py).

### Layout

| Path | Role |
|---|---|
| [`strategy.py`](strategy.py) | Production strategy module — single `step()` function shared by backtest and live paper trading, so live can't drift from backtest. Defines `CarryConfig`/`CarryStrategy` (single-venue carry) and `XVenueConfig`/`CrossVenueStrategy` (cross-venue routing). |
| [`venues.py`](venues.py) | Multi-venue (Binance/Bybit/Hyperliquid/Gate/OKX) funding + mark-price fetching, normalized to a per-hour rate across venues with different settlement intervals, plus optimal-routing logic. |
| [`paper_live.py`](paper_live.py) | Forward paper-trading loop: fetches live data, drives `strategy.py`'s `step()`, logs P&L hourly. No real orders, no account access. |
| [`paper_status.py`](paper_status.py) | Quick summary of the running paper-trading state. |
| [`execution.py`](execution.py) | Dry-run execution layer: target book → per-venue orders → reconciliation against actual positions → lot/min-notional rounding → maker limit prices → immutable audit log. Print-only, never places an order. |
| [`risk_monitor.py`](risk_monitor.py) | Dry-run risk monitor: drawdown circuit breaker, per-pair stop-loss, counterparty concentration, basis-blowout gate. Read/log only. |
| [`reconcile_positions.py`](reconcile_positions.py) | Read-only position reconciliation against exchange APIs (credentials via env vars only, never committed). |
| [`track_record.py`](track_record.py) | Turns the paper-trading log into an auditable NAV curve, Sharpe, drawdown, and per-leg P&L attribution. |
| [`live_readiness.py`](live_readiness.py) | Checks real execution friction (spread/depth/borrow cost) for coins currently in the book. |
| [`checkup.py`](checkup.py) | Standard strategy health check — reliable Sharpe across bar/day/week frequencies, cross-regime stability, tail risk. Run after any strategy change. |
| [`daily_report.py`](daily_report.py) | One-shot daily report chaining `track_record.py` + `execution.py` reconciliation into a single CSV row. |
| [`data_build/`](data_build/) | Data pipeline: downloads and builds point-in-time panels for spot/funding/basis across venues. |
| [`research/`](research/) | Backtests, stress tests, and diagnostics behind every claim in the strategy doc (tail risk, leverage frontier, cross-venue routing, delisting/survivorship handling, etc.) |
| [`docs/`](docs/) | Design docs and the experiment ledger — see [`docs/资金费套利方案.md`](docs/资金费套利方案.md) (strategy design), [`docs/风控.md`](docs/风控.md) (risk policy), [`docs/实验台账.md`](docs/实验台账.md) (experiment log). |
| `run_paper.bat`, `status.ps1`, `status_remote.sh` | Scheduled-task / status helpers for keeping the paper-trading loop running. |

Everything through `execution.py` is dry-run: it prints an intended order plan and never touches an account or moves funds.

## Notes

- Python 3.12, managed with [`uv`](https://github.com/astral-sh/uv). No dependency manifest is checked in yet; core dependencies are `numpy`, `pandas`, `torch` (model era only), and the standard library `urllib`/`requests` for exchange APIs.
- Exchange/API credentials are read from environment variables only and are never committed. `data/` (raw and processed datasets, paper-trading state/logs) is gitignored and generated locally.
- This is active research code, not a packaged library — expect both tracks to keep evolving.
