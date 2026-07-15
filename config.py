"""
Central configuration for the pairs trading project.

Keep every tunable parameter here so later steps (Z* optimisation,
backtesting, rolling re-estimation, etc.) can import from this single
place instead of hardcoding values inside individual scripts.
"""

from pathlib import Path

# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"

# ---------------------------------------------------------------------
# Pairs under study (Part I, Steps 1-3 onward)
# ---------------------------------------------------------------------
# The brief requires 2-3 pairs, each with a genuine economic rationale for
# cointegration (not just a correlation screen). Add/remove/edit entries
# here; every downstream step (stationarity, EG test, Z* search, backtest,
# rolling re-estimation) loops over this list via src/pipeline.py.
#
# The economic rationale for each pair belongs in the notebook (as prose,
# next to that pair's results and discussion) rather than as a data field
# here — config.py stays purely mechanical.
#
# "id" is used to build cache/output file names (data/prices_<id>.csv,
# outputs/<id>/...), so keep it filesystem-safe (no spaces/slashes).

PAIRS = [
    {
        "id": "KO_PEP",
        "name": "Coca-Cola vs PepsiCo",
        "tickers": ("KO", "PEP"),
    },
    # {
    #     "id": "EWA_EWC",
    #     "name": "Australia vs Canada country ETFs",
    #     "tickers": ("EWA", "EWC"),
    # },
    # {
    #     "id": "MAR_IHG",
    #     "name": "Marriott vs IHG (hospitality sector peers)",
    #     "tickers": ("MAR", "IHG"),
    # },
]

# Convenience: currently-active single pair, for scripts/tests that only
# want to work on one pair at a time (defaults to the first in PAIRS).
ACTIVE_PAIR = PAIRS[0]

START_DATE = "2023-01-01"
END_DATE = "2025-12-31"          # >= 2 years, per brief's minimum backtest window
PRICE_FIELD = "Close"             # use adjusted close where available

# ---------------------------------------------------------------------
# Stationarity / cointegration test settings (Steps 2-3)
# ---------------------------------------------------------------------
ADF_LAG = 1          # brief specifies lag=1 for the Augmented Dickey-Fuller test
SIGNIFICANCE_LEVEL = 0.05

# ---------------------------------------------------------------------
# Step 4: Z* optimisation
# ---------------------------------------------------------------------
# Grid of entry thresholds (in multiples of the spread's equilibrium
# sigma) searched by src/z_optimisation.py. Per the brief: don't assume
# Z=1 — search 0.5 to 3.0 in steps of 0.25.
Z_GRID = [0.5 + 0.25 * i for i in range(11)]   # 0.5, 0.75, ..., 3.0

# ---------------------------------------------------------------------
# Placeholders for later steps (backtest, train/test, rolling window)
# Fill these in when those steps are implemented.
# ---------------------------------------------------------------------
ROLLING_WINDOW_MONTHS = None   # e.g. 8, for Step 8
ROLLING_STEP_DAYS = None       # e.g. 10-15, for Step 8
TRAIN_TEST_SPLIT = None        # e.g. 0.7, for Step 7


def cache_path(pair_id: str) -> Path:
    """Path to the cached price CSV for a given pair id."""
    return DATA_DIR / f"prices_{pair_id}.csv"


def pair_output_dir(pair_id: str) -> Path:
    """Per-pair output directory, e.g. outputs/KO_PEP/."""
    d = OUTPUT_DIR / pair_id
    d.mkdir(parents=True, exist_ok=True)
    return d


for _d in (DATA_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)
