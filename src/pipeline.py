"""
Orchestrates Steps 1-4 for a single pair. Both main.py and the analysis
notebook call `run_pair_analysis()` so the logic lives in exactly one
place, not duplicated between script and notebook.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from src.data_pull import load_prices, plot_prices
from src.stationarity import summarize_stationarity
from src.eg_test import run_eg_pipeline, plot_spread
from src.z_optimisation import optimise_z, plot_z_optimisation


def run_pair_analysis(pair: dict, refresh_data: bool = False, save_plots: bool = True) -> dict:
    """Run Steps 1-4 for one pair-config dict (see config.PAIRS).

    Returns a dict with the prices DataFrame, stationarity summary,
    EG result dict, and the resolved output directory -- everything a
    notebook cell or comparison table needs, without re-running anything.
    """
    out_dir = config.pair_output_dir(pair["id"])

    # Step 1
    prices = load_prices(pair, refresh=refresh_data)
    price_fig = plot_prices(
        prices, title=f"{pair['name']}: normalized prices",
        save_path=(out_dir / "price_series.png") if save_plots else None,
    )

    # Step 2
    stationarity_summary = summarize_stationarity(prices)
    if save_plots:
        stationarity_summary.to_csv(out_dir / "stationarity_summary.csv", index=False)

    # Step 3
    y_ticker, x_ticker = pair["tickers"]
    eg_result = run_eg_pipeline(prices, y_ticker, x_ticker)
    spread_fig = plot_spread(
        eg_result["residuals"], title=f"{pair['name']}: spread",
        save_path=(out_dir / "spread.png") if save_plots else None,
    )
    if save_plots:
        eg_result["residuals"].to_csv(out_dir / "spread.csv")

    result = {
        "pair": pair,
        "prices": prices,
        "price_fig": price_fig,
        "stationarity_summary": stationarity_summary,
        "eg_result": eg_result,
        "spread_fig": spread_fig,
        "out_dir": out_dir,
    }

    # Step 4
    z_opt = optimise_z(result)
    z_opt["fig"] = plot_z_optimisation(
        z_opt["z_stats"], z_star=z_opt["z_star"],
        title=f"{pair['name']}: Z* grid search",
        save_path=(out_dir / "z_optimisation.png") if save_plots else None,
    )
    if save_plots:
        z_opt["z_stats"].to_csv(out_dir / "z_optimisation.csv", index=False)
    result["z_optimisation"] = z_opt

    return result


def run_all_pairs(refresh_data: bool = False, save_plots: bool = True) -> dict:
    """Run Steps 1-4 for every pair in config.PAIRS. Returns {pair_id: result}."""
    return {
        pair["id"]: run_pair_analysis(pair, refresh_data=refresh_data, save_plots=save_plots)
        for pair in config.PAIRS
    }


def comparison_table(results: dict):
    """Build a side-by-side summary table across pairs (id, tickers, beta,
    cointegration p-values, half-life) -- the cross-pair comparison the
    brief's write-up should include.
    """
    import pandas as pd

    rows = []
    for pair_id, r in results.items():
        eg = r["eg_result"]
        z_opt = r["z_optimisation"]
        # best_signals covers a config-pinned z_star that isn't on the grid
        star_log = z_opt["best_signals"]["trade_log"]
        rows.append({
            "pair_id": pair_id,
            "name": r["pair"]["name"],
            "tickers": " / ".join(r["pair"]["tickers"]),
            "hedge_ratio_beta": eg["beta"],
            "r_squared": eg["r_squared"],
            "eg_pvalue_manual": eg["manual_adf_pvalue"],
            "eg_pvalue_builtin": eg["builtin_eg_pvalue"],
            "cointegrated_5pct": eg["builtin_cointegrated_5pct"],
            "half_life_periods": eg["half_life_days"],
            "z_star": z_opt["z_star"],
            "n_trades_at_z_star": len(star_log),
            "total_pnl_at_z_star": star_log["pnl"].sum() if len(star_log) else 0.0,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-data", action="store_true")
    args = parser.parse_args()

    results = run_all_pairs(refresh_data=args.refresh_data)
    table = comparison_table(results)
    print(table.to_string(index=False))
    table.to_csv(config.OUTPUT_DIR / "pair_comparison.csv", index=False)
