"""
Step 3: Engle-Granger two-step cointegration test.

Step A: OLS regress y on x -> get hedge ratio (beta) and residual (spread).
Step B: Test the residual for stationarity (ADF). If stationary, y and x
        are cointegrated with cointegrating vector [1, -beta].

Also cross-checks against statsmodels' built-in `coint()` implementation,
and reports basic mean-reversion diagnostics (mean, std, half-life via
AR(1) fit to the residual) to feed into Step 3's trade-design bounds.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from src.stationarity import adf_test


def eg_regression(y: pd.Series, x: pd.Series) -> dict:
    """Step A: OLS of y on x (+ intercept). Returns beta, alpha, and residual series."""
    x_const = sm.add_constant(x.rename("x"))
    model = sm.OLS(y, x_const).fit()

    return {
        "alpha": model.params["const"],
        "beta": model.params["x"],
        "residuals": model.resid,
        "model_summary": model.summary(),
        "r_squared": model.rsquared,
    }


def eg_residual_test(residuals: pd.Series, lag: int = config.ADF_LAG) -> dict:
    """Step B: ADF test on the regression residual (the spread)."""
    return adf_test(residuals, lag=lag)


def eg_builtin_test(y: pd.Series, x: pd.Series) -> dict:
    """Cross-check using statsmodels' coint(), which runs the same EG
    procedure with appropriate critical values for the two-step regression.
    """
    stat, pvalue, crit_values = coint(y, x)
    return {
        "statistic": stat,
        "p_value": pvalue,
        "critical_values": {"1%": crit_values[0], "5%": crit_values[1], "10%": crit_values[2]},
        "cointegrated_5pct": pvalue < 0.05,
    }


def half_life(residuals: pd.Series) -> float:
    """Estimate mean-reversion half-life via AR(1) fit to the residual,
    equivalent to discretizing an Ornstein-Uhlenbeck process:

        e_t = phi * e_{t-1} + eps_t     =>     theta = -ln(phi)
        half_life = ln(2) / theta
    """
    e = residuals.dropna()
    e_lag = e.shift(1).dropna()
    e_curr = e.loc[e_lag.index]

    x_const = sm.add_constant(e_lag.rename("e_lag"))
    model = sm.OLS(e_curr, x_const).fit()
    phi = model.params["e_lag"]

    if phi <= 0 or phi >= 1:
        return np.nan  # not mean-reverting under this simple AR(1) fit

    theta = -np.log(phi)
    return np.log(2) / theta


def run_eg_pipeline(prices: pd.DataFrame, ticker_y: str, ticker_x: str) -> dict:
    """Full Step 1-3 pipeline for one ordering (y on x). Run twice (swap
    y/x) if you want to check which direction gives the stronger result --
    EG is not symmetric in principle, though results are often similar.
    """
    y, x = prices[ticker_y], prices[ticker_x]

    reg = eg_regression(y, x)
    resid_adf = eg_residual_test(reg["residuals"])
    builtin = eg_builtin_test(y, x)
    hl = half_life(reg["residuals"])

    return {
        "pair": (ticker_y, ticker_x),
        "alpha": reg["alpha"],
        "beta": reg["beta"],
        "r_squared": reg["r_squared"],
        "residuals": reg["residuals"],
        "manual_adf_stat": resid_adf["statistic"],
        "manual_adf_pvalue": resid_adf["p_value"],
        "manual_cointegrated": resid_adf["stationary"],
        "builtin_eg_stat": builtin["statistic"],
        "builtin_eg_pvalue": builtin["p_value"],
        "builtin_cointegrated_5pct": builtin["cointegrated_5pct"],
        "half_life_days": hl,
    }


def plot_spread(residuals: pd.Series, title: str = None, save_path: Path = None):
    import matplotlib.pyplot as plt

    mu = residuals.mean()
    sigma = residuals.std()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(residuals.index, residuals.values, label="Spread (residual)", linewidth=1)
    ax.axhline(mu, color="black", linestyle="-", linewidth=1, label="Mean")
    for z, style in [(1, "--"), (2, ":")]:
        ax.axhline(mu + z * sigma, color="red", linestyle=style, linewidth=1, alpha=0.7)
        ax.axhline(mu - z * sigma, color="red", linestyle=style, linewidth=1, alpha=0.7)
    ax.set_title(title or "Cointegration residual (spread) with +/-1sigma, +/-2sigma bands")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


if __name__ == "__main__":
    from src.data_pull import load_prices

    for pair in config.PAIRS:
        prices = load_prices(pair)
        y_ticker, x_ticker = pair["tickers"]

        result = run_eg_pipeline(prices, y_ticker, x_ticker)

        print(f"\n=== {pair['id']} ({pair['name']}) ===")
        print(f"Pair: {result['pair'][0]} ~ {result['pair'][1]}")
        print(f"  Hedge ratio (beta): {result['beta']:.4f}")
        print(f"  Intercept (alpha):  {result['alpha']:.4f}")
        print(f"  R-squared:          {result['r_squared']:.4f}")
        print("Manual EG (regress -> ADF on residual):")
        print(f"  ADF statistic: {result['manual_adf_stat']:.4f}  p-value: {result['manual_adf_pvalue']:.4f}")
        print(f"  Cointegrated at 5%: {result['manual_cointegrated']}")
        print("statsmodels.coint() cross-check:")
        print(f"  EG statistic: {result['builtin_eg_stat']:.4f}  p-value: {result['builtin_eg_pvalue']:.4f}")
        print(f"  Cointegrated at 5%: {result['builtin_cointegrated_5pct']}")
        print(f"Estimated half-life of mean reversion: {result['half_life_days']:.1f} periods")

        out_dir = config.pair_output_dir(pair["id"])
        plot_spread(result["residuals"], title=f"{pair['name']}: spread",
                    save_path=out_dir / "spread.png")
        result["residuals"].to_csv(out_dir / "spread.csv")
