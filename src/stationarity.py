"""
Step 2: Stationarity pre-tests.

Before running the Engle-Granger cointegration test, confirm each price
series individually behaves like I(1): non-stationary in levels,
stationary after first-differencing. This is a sanity gate — if a series
fails this pattern, EG results downstream are not meaningful.
"""

import sys
from pathlib import Path

import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


def adf_test(series: pd.Series, lag: int = config.ADF_LAG) -> dict:
    """Augmented Dickey-Fuller test. H0: series has a unit root (non-stationary)."""
    stat, pvalue, used_lag, nobs, crit_values = adfuller(
        series.dropna(), maxlag=lag, autolag=None
    )
    return {
        "test": "ADF",
        "statistic": stat,
        "p_value": pvalue,
        "lag": used_lag,
        "n_obs": nobs,
        "critical_values": crit_values,
        "stationary": pvalue < config.SIGNIFICANCE_LEVEL,
    }


def kpss_test(series: pd.Series, regression: str = "c") -> dict:
    """KPSS test. H0: series IS stationary (opposite null to ADF) — used as a
    second opinion. Agreement between ADF and KPSS strengthens the conclusion.
    """
    stat, pvalue, lags, crit_values = kpss(series.dropna(), regression=regression, nlags="auto")
    return {
        "test": "KPSS",
        "statistic": stat,
        "p_value": pvalue,
        "lag": lags,
        "critical_values": crit_values,
        "stationary": pvalue > config.SIGNIFICANCE_LEVEL,
    }


def summarize_stationarity(prices: pd.DataFrame) -> pd.DataFrame:
    """Run ADF + KPSS on levels and first differences for every column.

    Expected I(1) pattern: level tests say "non-stationary", diff tests say
    "stationary". Flags anything that doesn't fit that pattern.
    """
    rows = []
    for col in prices.columns:
        level = prices[col]
        diff = prices[col].diff().dropna()

        for label, series in [("level", level), ("diff", diff)]:
            adf_res = adf_test(series)
            kpss_res = kpss_test(series)
            rows.append({
                "series": col,
                "form": label,
                "adf_stat": adf_res["statistic"],
                "adf_pvalue": adf_res["p_value"],
                "adf_stationary": adf_res["stationary"],
                "kpss_stat": kpss_res["statistic"],
                "kpss_pvalue": kpss_res["p_value"],
                "kpss_stationary": kpss_res["stationary"],
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from src.data_pull import load_prices

    pd.set_option("display.width", 120)

    for pair in config.PAIRS:
        prices = load_prices(pair)
        summary = summarize_stationarity(prices)
        print(f"\n=== {pair['id']} ({pair['name']}) ===")
        print(summary.to_string(index=False))

        out_dir = config.pair_output_dir(pair["id"])
        summary.to_csv(out_dir / "stationarity_summary.csv", index=False)

        # Flag anything that doesn't fit the expected I(1) pattern
        for _, row in summary.iterrows():
            if row["form"] == "level" and row["adf_stationary"]:
                print(f"NOTE: {row['series']} level series looks stationary by ADF — "
                      f"unexpected for an I(1) assumption, inspect further.")
            if row["form"] == "diff" and not row["adf_stationary"]:
                print(f"WARNING: {row['series']} first difference is NOT stationary by ADF — "
                      f"series may not be I(1).")
