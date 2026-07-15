"""
Step 1: Data pull.

Pulls adjusted close prices for a given pair via yfinance, with a
per-pair CSV cache so the rest of the pipeline (and the grader) doesn't
depend on network access or Yahoo's availability at run time.

Usage:
    python -m src.data_pull                # pulls/loads every pair in config.PAIRS
    python -m src.data_pull --refresh-data  # forces a fresh download for all pairs
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


def download_prices(tickers, start, end, price_field=config.PRICE_FIELD):
    """Download prices for `tickers` between `start` and `end` via yfinance.

    Returns a DataFrame indexed by date, one column per ticker, containing
    only dates where BOTH tickers traded (inner join).
    """
    import yfinance as yf

    raw = yf.download(list(tickers), start=start, end=end, auto_adjust=True, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw[price_field].copy()
    else:
        # Single ticker edge case
        prices = raw[[price_field]].copy()
        prices.columns = list(tickers)

    prices = prices.dropna(how="any")  # inner join on trading days
    prices.index.name = "Date"
    return prices


def load_prices(pair: dict, refresh: bool = False) -> pd.DataFrame:
    """Load prices for one pair-config dict (see config.PAIRS) from cache,
    or download and cache them if needed.
    """
    path = config.cache_path(pair["id"])

    if not refresh and path.exists():
        prices = pd.read_csv(path, index_col="Date", parse_dates=True)
        return prices

    prices = download_prices(pair["tickers"], config.START_DATE, config.END_DATE)
    prices.to_csv(path)
    return prices


def load_all_pairs(refresh: bool = False) -> dict:
    """Load prices for every pair in config.PAIRS. Returns {pair_id: DataFrame}."""
    return {pair["id"]: load_prices(pair, refresh=refresh) for pair in config.PAIRS}


def plot_prices(prices: pd.DataFrame, title: str = None, save_path: Path = None):
    import matplotlib.pyplot as plt

    normalized = prices / prices.iloc[0] * 100
    fig, ax = plt.subplots(figsize=(10, 5))
    for col in normalized.columns:
        ax.plot(normalized.index, normalized[col], label=col)
    ax.set_title(title or f"Normalized price series: {' vs '.join(prices.columns)}")
    ax.set_ylabel("Indexed to 100 at start")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-data", action="store_true",
                         help="Force a fresh download instead of using cached CSVs.")
    args = parser.parse_args()

    for pair in config.PAIRS:
        prices = load_prices(pair, refresh=args.refresh_data)
        print(f"[{pair['id']}] Loaded {len(prices)} rows for {pair['tickers']}")
        out_dir = config.pair_output_dir(pair["id"])
        plot_prices(prices, title=f"{pair['name']}: normalized prices",
                    save_path=out_dir / "price_series.png")
