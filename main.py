"""
Single entry point for Steps 1-4, run across every pair in config.PAIRS.

Run:
    python main.py                 # uses cached data/prices_<id>.csv per pair if present
    python main.py --refresh-data  # forces a fresh yfinance download for every pair

Outputs (written to outputs/<pair_id>/ for each pair):
    price_series.png          - normalized price plot
    stationarity_summary.csv  - ADF/KPSS results for both tickers, levels & diffs
    spread.png                - EG residual (spread) plot with sigma bands
    spread.csv                - the spread series itself, for later steps
    z_optimisation.png        - N_trades and P&L vs Z trade-off chart
    z_optimisation.csv        - per-Z grid-search stats (Step 4)

Also writes outputs/pair_comparison.csv, a side-by-side summary across
all pairs (hedge ratio, cointegration p-values, half-life, Z*).
"""

import argparse

import config
from src.pipeline import run_all_pairs, comparison_table


def main(refresh_data: bool = False):
    results = {}

    for pair in config.PAIRS:
        print(f"\n{'=' * 60}")
        print(f"Pair: {pair['id']} — {pair['name']}")
        print(f"{'=' * 60}")

        from src.pipeline import run_pair_analysis
        r = run_pair_analysis(pair, refresh_data=refresh_data)
        results[pair["id"]] = r

        prices = r["prices"]
        eg = r["eg_result"]

        print(f"Loaded {len(prices)} rows, {prices.index.min().date()} to {prices.index.max().date()}")
        print(f"Hedge ratio (beta): {eg['beta']:.4f}   R-squared: {eg['r_squared']:.4f}")
        print(f"EG p-value (manual / statsmodels): "
              f"{eg['manual_adf_pvalue']:.4f} / {eg['builtin_eg_pvalue']:.4f}")
        print(f"Cointegrated at 5%: {eg['builtin_cointegrated_5pct']}")
        print(f"Estimated half-life: {eg['half_life_days']:.1f} periods")

        z_opt = r["z_optimisation"]
        star_log = z_opt["best_signals"]["trade_log"]
        print(f"Z* = {z_opt['z_star']:g} ({z_opt['z_star_source']}): "
              f"{len(star_log)} trades, "
              f"cumulative P&L {star_log['pnl'].sum() if len(star_log) else 0.0:.2f} spread units")
        print(f"Outputs written to {r['out_dir']}/")

    table = comparison_table(results)
    table.to_csv(config.OUTPUT_DIR / "pair_comparison.csv", index=False)

    print(f"\n{'=' * 60}")
    print("Cross-pair comparison")
    print(f"{'=' * 60}")
    print(table.to_string(index=False))
    print(f"\nSaved to {config.OUTPUT_DIR / 'pair_comparison.csv'}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-data", action="store_true",
                         help="Force a fresh yfinance download instead of using cached CSVs.")
    args = parser.parse_args()
    main(refresh_data=args.refresh_data)
