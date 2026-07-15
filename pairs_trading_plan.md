# Pairs Trading Strategy — Implementation Plan

Project brief: *Pairs Trading Strategy Design & Backtest v2026*
Language: Python
Status: Steps 1–4 implemented across a multi-pair structure. This plan covers the full project (Steps 1–8) for continued implementation.

---

## 0. Project structure (current)

```
pairs_trading/
├── config.py                # PAIRS list (id, name, tickers) + all tunable parameters
│                            # (economic rationale lives as prose in the notebook)
├── main.py                  # script entry point, loops config.PAIRS, writes pair_comparison.csv
├── data/
│   └── prices_<id>.csv      # cached price data PER PAIR (commit after first --refresh-data run)
├── notebooks/
│   └── pairs_trading_analysis.ipynb   # graded write-up: one section per pair + comparison
├── src/
│   ├── data_pull.py          # Step 1 (per pair)
│   ├── stationarity.py       # Step 2 (per pair)
│   ├── eg_test.py            # Step 3 (per pair)
│   ├── pipeline.py           # orchestrates Steps 1-3 for ONE pair (run_pair_analysis),
│   │                          # loops all pairs (run_all_pairs), builds comparison_table().
│   │                          # main.py and the notebook both call into this — extend it,
│   │                          # not main.py directly, when adding new steps.
│   ├── z_optimisation.py     # Step 4  (done)
│   ├── structural_breaks.py  # Step 5  (to build)
│   ├── backtest.py           # Step 6  (to build)
│   ├── train_test.py         # Step 7  (to build)
│   └── rolling_reestimate.py # Step 8  (to build)
├── outputs/<pair_id>/         # generated plots/CSVs per pair on each run
├── outputs/pair_comparison.csv # cross-pair summary table
├── requirements.txt
└── README.md
```

**Key point for extending this:** each new step's function should take a
`pair` dict (from `config.PAIRS`) and/or a `result` dict (the output of
`run_pair_analysis`, which already holds `prices`, `stationarity_summary`,
`eg_result`, `out_dir`, etc.) and return its own result dict. Wire it into
`run_pair_analysis()` in `src/pipeline.py` so it runs automatically for
every pair via `main.py` and is available in the notebook without extra
plumbing. Shared parameters go in `config.py` (already stubbed: `Z_GRID`,
`ROLLING_WINDOW_MONTHS`, `ROLLING_STEP_DAYS`, `TRAIN_TEST_SPLIT`).

---

## Part I — Pairs trade design & cointegration analysis

### Step 1 — Data pull ✅ done
- Pull adjusted close prices for each pair in `config.PAIRS` via `yfinance`.
- Cache to `data/prices_<pair_id>.csv` so later runs and grading don't depend on network access.
- Plot normalized price series (indexed to 100 at start) to eyeball comovement.
- 2–3 pairs already scaffolded in `config.PAIRS`, each with a written economic rationale in its notebook section (brief requires this — not just a correlation screen).

### Step 2 — Stationarity pre-tests ✅ done
- ADF (lag=1, as specified) and KPSS on levels and first differences, for each ticker individually, per pair.
- Confirm the expected I(1) pattern: non-stationary in levels, stationary in first differences.
- This is a gate — if a series doesn't fit I(1), flag it before trusting downstream EG results for that pair.

### Step 3 — Engle-Granger cointegration test ✅ done
- Step A: OLS regress y on x (+intercept) → hedge ratio (beta) and residual (spread).
- Step B: ADF test on the residual. Cross-check against `statsmodels.tsa.stattools.coint()`.
- Estimate mean-reversion half-life via AR(1) fit to the residual (equivalent to discretized OU process: `theta = -ln(phi)`, `half_life = ln(2)/theta`).
- Cross-pair comparison table (`comparison_table()` in `pipeline.py`) already collects beta, R², EG p-values, and half-life side by side.
- **Extension needed:** split each pair's sample into sub-periods (e.g. yearly/half-yearly) and re-run EG on each, to check whether the EC/cointegrating relationship is stable over time. Write a short discussion of the findings per pair.
- **Optional (if using R or wanting multivariate rigor):** Johansen procedure via the `urca` package, or `statsmodels.tsa.vector_ar.vecm.coint_johansen` in Python, for >2-asset cointegration.

### Step 3b — VAR diagnostics (mentioned in brief, structural-model exercise only)
- Recode the OLS regression in matrix form (X'X)⁻¹X'y as a coding exercise.
- VAR stability check: eigenvalues of the companion matrix should lie inside the unit circle.
- Lag selection via AIC/BIC (`statsmodels.tsa.api.VAR` has `.select_order()`).
- Note: this applies to stationary structural models, not to forecasting returns in this project — it's a diagnostic exercise, not part of the trading signal itself.

### Step 4 — Optimise Z* (entry/exit threshold) ✅ done
**Built: `src/z_optimisation.py`** — `optimise_z(result)` grid-searches `config.Z_GRID`,
returns per-Z stats + selected Z* (max closed-trade P&L, overridable via a `"z_star"`
key on the pair-config dict); wired into `run_pair_analysis()` as
`result["z_optimisation"]`, chart/CSV saved to `outputs/<pair_id>/z_optimisation.{png,csv}`,
and Z*/N_trades/P&L added to `comparison_table()`.
- Function signature suggestion: `optimise_z(result: dict, z_grid=None) -> dict`, taking a `run_pair_analysis()` result (for its `eg_result['residuals']`) and returning per-Z stats.
- Do not assume Z=1. Grid-search Z over a range (e.g. 0.5 to 3.0 in steps of 0.25) — set `config.Z_GRID`.
- For each Z: generate entry/exit signals off the spread (enter at μ_e ± Z·σ_eq, exit at reversion to μ_e), compute number of trades (`N_trades`) and cumulative P&L.
- Produce a chart/table of `N_trades` and P&L vs. Z, per pair.
- Discuss the trade-off: wider Z → higher P&L per trade but fewer trades and more risk of the spread not reverting (potential structural break); narrower Z → more, smaller trades.
- Store the selected `Z*` per pair (e.g. add a `"z_star"` key to each `config.PAIRS` entry, or a separate dict keyed by pair id).
- Wire into `run_pair_analysis()` so `result["z_optimisation"]` is populated for every pair automatically.

### Step 5 — Structural break discussion
**To build: `src/structural_breaks.py`** (partly qualitative)
- Testing for structural breaks is explicitly flagged in the brief as "more art than science" and largely beyond FP study scope — a full break-detection model is not required.
- Reasonable lightweight implementation: rolling ADF/EG test over sub-windows, or a CUSUM-type plot on the residual, to visually flag where the relationship looks like it's drifting. Take `result["eg_result"]["residuals"]` as input.
- Required: a written discussion (in the notebook, per pair section) of the *economic* reasons your chosen pair's cointegrated relationship might break down (e.g. M&A deal falls through, ETF index reweighting, policy divergence for FX pairs, sector rotation) — tie it back to the rationale written in each pair's notebook section.

---

## Part II — Backtesting

> 2-year minimum backtest period, subject to the realities of the cointegrated situation. Items below are recommendations — implement what's suitable for your chosen asset classes and pairs.

### Step 6 — Systematic backtest
**To build: `src/backtest.py`**
- Take the Z*-based trading signal from Step 4 (per pair) and generate a returns series (spread P&L, position sized by ±1 unit per the cointegrating vector or scaled).
- Produce, per pair: cumulative P&L / equity curve, drawdown plot, rolling Sharpe ratio (e.g. 63-day rolling window).
- Discuss whether the P&L behaves as expected for an arb trade: number of trades generating most of the P&L, realized half-life vs. Step 3's estimate, max drawdown, volatility/VaR behavior.
- Rolling beta vs. S&P 500 / factor returns can be omitted per the brief.
- Extend `comparison_table()` in `pipeline.py` to include Sharpe/max-drawdown per pair for the cross-pair section.

### Step 7 — Train/test split
**To build: `src/train_test.py`**
- Split chronologically (no shuffling — this is time series): e.g. first 60–70% of the sample to *fit* the cointegrating relationship and choose Z*, remainder held out to *test* signal performance out-of-sample. Set `config.TRAIN_TEST_SPLIT`.
- Compare in-sample vs. out-of-sample P&L, Sharpe, and number of trades, per pair — this is the "scikit-learn inspired" cross-validation the brief references, adapted for time series (no k-fold shuffling; consider walk-forward validation if time permits).

### Step 8 — Rolling re-estimation
**To build: `src/rolling_reestimate.py`**
- Re-estimate the EG cointegrating relationship (β_coint) on a rolling window (e.g. 8 months, `config.ROLLING_WINDOW_MONTHS`), shifting forward by 10–15 days each time (`config.ROLLING_STEP_DAYS`).
- Track how β_coint drifts across windows, per pair.
- Compare fixed-β P&L (Step 6, static hedge ratio) vs. rolling-β P&L (hedge ratio updated each window).
- Discuss whether the assumption of a stable β over 3–6 months (as the brief suggests cointegration typically allows) held for your chosen pair(s).

---

## Suggested build order

1. ~~Steps 1–3, multi-pair structure~~ ✅ done
2. Finalize pair 2 and pair 3 in `config.PAIRS` (uncomment/edit, or replace with your own), duplicate the notebook's per-pair section for each, confirm all three cointegrate sensibly
3. Step 4 (Z* optimisation) + Step 6 (backtest engine) — get one full signal-to-P&L loop working for pair 1, then it applies to all pairs automatically via `pipeline.py`
4. Step 7 (train/test split) + Step 8 (rolling re-estimation)
5. Step 5 (structural break discussion) — mostly written, do last since it's qualitative and benefits from having seen the backtest results first
6. Step 3's sub-period EG stability check can be folded in alongside Step 5's discussion, since both speak to relationship stability

## Definition of done (full project)

- 2–3 pairs configured in `config.PAIRS`, each with a clear economic rationale for cointegration (not just a correlation screen)
- EG test (and optionally Johansen) with hedge ratio, half-life, and sub-period stability check for each pair
- Z* grid search with N_trades/P&L trade-off chart, per pair
- Full backtest: equity curve, drawdown, rolling Sharpe, discussion of P&L behavior, per pair
- Train/test split showing in-sample vs. out-of-sample performance, per pair
- Rolling re-estimation showing β_coint drift and fixed vs. rolling P&L comparison, per pair
- Written discussion of structural break risk specific to each chosen pair
- Cross-pair comparison table/discussion (already scaffolded in the notebook and `pipeline.comparison_table()`) extended to include Steps 4–8 metrics
- `README.md` with setup/run instructions, `requirements.txt` pinned, code runs end-to-end off committed per-pair CSVs with no network dependency required for grading
- Notebook (`notebooks/pairs_trading_analysis.ipynb`) fully filled out: all pairs, all discussion sections, executed with real data before submission

---

*Key resources per the brief: TS Project Workshop, Cointegration Lecture, FP Tutorial.*
