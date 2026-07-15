# Pairs Trading Strategy — Design & Backtest

Implementation of Steps 1–4 of the project brief across multiple
pairs: data pull, stationarity pre-tests, the Engle-Granger
cointegration test, and Z* (entry threshold) optimisation. Later steps
(structural breaks, backtesting, train/test split, rolling
re-estimation) will extend this same structure — see "Project status"
below.

## Setup

```bash
git clone <this-repo>
cd pairs_trading
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running

**Script (all pairs, headless):**
```bash
python main.py --refresh-data   # first run: downloads prices for every pair in config.PAIRS
python main.py                  # subsequent runs: reuses cached CSVs, no internet needed
```

**Notebook (for the graded write-up):**
```bash
jupyter notebook notebooks/pairs_trading_analysis.ipynb
```
Run all cells. Set `REFRESH_DATA = True` in the notebook's first data cell
on the very first run to pull prices, then switch it back to `False` so
subsequent re-runs are fast and reproducible off the cached CSVs.

Either way, run once with fresh data, then **commit the resulting
`data/prices_*.csv` files** before submitting — the grader can then run
everything with zero network dependency, regardless of whether Yahoo
Finance is reachable or rate-limiting on the day it's marked.

## Adding / editing pairs

Edit `config.PAIRS` — a list of dicts, each with an `id`, `name`, and
`tickers`:

```python
PAIRS = [
    {
        "id": "KO_PEP",
        "name": "Coca-Cola vs PepsiCo",
        "tickers": ("KO", "PEP"),
    },
    # add pair 2, pair 3 here
]
```

The economic rationale for each pair should be written in the notebook
as prose, in that pair's markdown section next to its results — not as
a data field in `config.py`.

The brief requires 2–3 pairs, each with a genuine economic rationale for
cointegration — not just a correlation screen (see the brief's own
examples: an M&A acquirer/target pair, or country ETFs during a
commodity-driven regime). Two commented-out example entries are already
in `config.py` (EWA/EWC, MAR/IHG) — uncomment and adjust, or replace with
your own choices.

Every module (`data_pull.py`, `stationarity.py`, `eg_test.py`,
`pipeline.py`) loops over `config.PAIRS` automatically — no other code
changes needed when you add a pair.

## Outputs

Running `main.py` or the notebook writes to `outputs/<pair_id>/` for each
pair:

| File | Contents |
|---|---|
| `price_series.png` | Normalized price plot of both tickers |
| `stationarity_summary.csv` | ADF + KPSS results, both tickers, levels & first differences |
| `spread.png` | Engle-Granger residual (spread) with ±1σ/±2σ bands |
| `spread.csv` | The spread series itself (input to later steps: Z* signal generation, backtesting) |
| `z_optimisation.png` | Step 4 trade-off chart: N_trades and cumulative P&L vs Z, with Z* marked |
| `z_optimisation.csv` | Per-Z grid-search stats (trades, P&L, avg P&L/trade, avg holding days) |

Plus one cross-pair file at `outputs/pair_comparison.csv`: hedge ratio,
cointegration p-values, half-life, and Z* (with its trade count and
P&L) side by side across all pairs.

## Project structure

```
pairs_trading/
├── config.py               # PAIRS list + all tunable parameters
├── main.py                 # script entry point, runs Steps 1-3 across all pairs
├── data/
│   └── prices_<id>.csv     # cached price data per pair (commit after first --refresh-data)
├── notebooks/
│   └── pairs_trading_analysis.ipynb   # graded write-up: one section per pair + comparison
├── src/
│   ├── data_pull.py        # Step 1: fetch/cache prices per pair, normalized price plot
│   ├── stationarity.py     # Step 2: ADF + KPSS on levels and first differences
│   ├── eg_test.py          # Step 3: EG regression, residual ADF test, half-life, spread plot
│   ├── z_optimisation.py   # Step 4: Z* grid search (signals, N_trades, P&L per Z)
│   └── pipeline.py         # orchestrates Steps 1-4 for one pair; also builds the
│                            # cross-pair comparison table. Both main.py and the
│                            # notebook call this so logic isn't duplicated.
└── outputs/<pair_id>/       # generated on each run
```

## Using the notebook

The notebook is structured as:
1. Intro + list of pairs, with each one's economic rationale written out in markdown
2. Run all pairs (one call into `src/pipeline.py` per pair)
3. Per-pair analysis section (price plot → stationarity → EG test → spread
   plot → discussion) — currently scaffolded for pair 1; **duplicate the
   block for pairs 2 and 3** as noted in the notebook itself
4. Cross-pair comparison table + discussion
5. Placeholder section for Part II (Steps 4–8) once those are built

The computation itself lives in `src/`, not in notebook cells — the
notebook calls `run_pair_analysis()` and presents the results, so the
same logic is testable via `main.py` and won't drift between the two.

## Project status

- [x] Step 1 — Data pull (per pair)
- [x] Step 2 — Stationarity pre-tests (ADF, KPSS on levels & diffs)
- [x] Step 3 — Engle-Granger cointegration test, hedge ratio, half-life
- [x] Multi-pair structure + notebook scaffold + cross-pair comparison
- [ ] Sub-period EG stability check (split sample, check EC term stability over time)
- [x] Step 4 — Z* optimisation (grid search over entry/exit bounds)
- [ ] Step 5 — Structural break discussion
- [ ] Step 6 — Backtest: cumulative P&L, drawdown, rolling Sharpe
- [ ] Step 7 — Train/test split
- [ ] Step 8 — Rolling re-estimation of the cointegrating relationship

These will slot into `src/` as additional modules (e.g. `z_optimisation.py`,
`backtest.py`), called from `src/pipeline.py` and surfaced in the notebook
the same way Steps 1–3 are now.
