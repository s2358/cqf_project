"""
Step 4: Optimise Z* (entry/exit threshold).

Do not assume Z=1. For each candidate Z in config.Z_GRID, generate
entry/exit signals off the cointegration spread (Step 3's residual):

    enter short spread when  e_t > mu_e + Z * sigma_eq
    enter long  spread when  e_t < mu_e - Z * sigma_eq
    exit on reversion to the equilibrium mean mu_e

and record the number of trades and cumulative P&L (in spread units,
i.e. per 1 unit of y hedged with beta units of x). The trade-off:
wider Z -> higher P&L per trade but fewer trades and more exposure to
the spread not reverting (structural break risk); narrower Z -> more,
smaller trades.

Z* is chosen as the grid point with the highest cumulative closed-trade
P&L, unless the pair-config dict pins one via a "z_star" key.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


def generate_signals(residuals: pd.Series, z: float,
                     mu: float = None, sigma: float = None) -> dict:
    """Run the entry/exit state machine on the spread for one threshold Z.

    Position convention: -1 = short spread (entered above the upper band),
    +1 = long spread (entered below the lower band), 0 = flat. Positions
    are held until the spread reverts to mu, then closed; re-entry is
    allowed on the next band crossing.

    Returns the position series, per-trade log, and a daily mark-to-market
    P&L series (position held into day t times the day-t spread change).
    """
    e = residuals.dropna()
    mu = e.mean() if mu is None else mu
    sigma = e.std() if sigma is None else sigma

    upper, lower = mu + z * sigma, mu - z * sigma

    position = pd.Series(0, index=e.index, dtype=int)
    trades = []
    pos, entry_date, entry_level = 0, None, None

    for date, level in e.items():
        if pos == 0:
            if level > upper:
                pos, entry_date, entry_level = -1, date, level
            elif level < lower:
                pos, entry_date, entry_level = +1, date, level
        elif (pos == -1 and level <= mu) or (pos == +1 and level >= mu):
            trades.append({
                "entry_date": entry_date,
                "exit_date": date,
                "side": "short" if pos == -1 else "long",
                "entry_level": entry_level,
                "exit_level": level,
                "pnl": pos * (level - entry_level),
                "holding_days": len(e.loc[entry_date:date]) - 1,
            })
            pos, entry_date, entry_level = 0, None, None
        position.loc[date] = pos

    trade_log = pd.DataFrame(trades)
    # Held position times daily spread change -> mark-to-market P&L,
    # including any trade still open at the end of the sample.
    daily_pnl = (position.shift(1) * e.diff()).fillna(0.0)

    return {
        "z": z,
        "mu": mu,
        "sigma": sigma,
        "position": position,
        "trade_log": trade_log,
        "daily_pnl": daily_pnl,
        "open_at_end": pos != 0,
    }


def optimise_z(result: dict, z_grid=None) -> dict:
    """Grid-search Z over config.Z_GRID for one pair.

    Takes a run_pair_analysis() result dict (uses its
    eg_result["residuals"]) and returns per-Z stats plus the selected Z*.
    A "z_star" key on the pair-config dict overrides the automatic
    (max closed-trade P&L) selection.
    """
    z_grid = config.Z_GRID if z_grid is None else z_grid
    residuals = result["eg_result"]["residuals"]

    rows, signals_by_z = [], {}
    for z in z_grid:
        sig = generate_signals(residuals, z)
        signals_by_z[z] = sig
        log = sig["trade_log"]
        n_trades = len(log)
        rows.append({
            "z": z,
            "n_trades": n_trades,
            "total_pnl": log["pnl"].sum() if n_trades else 0.0,
            "avg_pnl_per_trade": log["pnl"].mean() if n_trades else np.nan,
            "avg_holding_days": log["holding_days"].mean() if n_trades else np.nan,
            "open_at_end": sig["open_at_end"],
        })
    z_stats = pd.DataFrame(rows)

    if result["pair"].get("z_star") is not None:
        z_star = result["pair"]["z_star"]
    else:
        z_star = z_stats.loc[z_stats["total_pnl"].idxmax(), "z"]

    return {
        "z_grid": list(z_grid),
        "z_stats": z_stats,
        "z_star": z_star,
        "z_star_source": "config" if result["pair"].get("z_star") is not None else "max_total_pnl",
        "signals": signals_by_z,
        "best_signals": signals_by_z.get(z_star) or generate_signals(residuals, z_star),
    }


def plot_z_optimisation(z_stats: pd.DataFrame, z_star: float = None,
                        title: str = None, save_path: Path = None):
    """N_trades and cumulative P&L vs Z on twin axes — the Step 4 trade-off chart."""
    import matplotlib.pyplot as plt

    fig, ax_pnl = plt.subplots(figsize=(10, 5))
    ax_n = ax_pnl.twinx()

    ax_pnl.plot(z_stats["z"], z_stats["total_pnl"], color="tab:blue",
                marker="o", label="Cumulative P&L (spread units)")
    ax_n.bar(z_stats["z"], z_stats["n_trades"], width=0.15, color="tab:gray",
             alpha=0.4, label="N trades")

    if z_star is not None:
        ax_pnl.axvline(z_star, color="tab:red", linestyle="--", linewidth=1,
                       label=f"Z* = {z_star:g}")

    ax_pnl.set_xlabel("Z (entry threshold, multiples of sigma_eq)")
    ax_pnl.set_ylabel("Cumulative P&L (spread units)", color="tab:blue")
    ax_n.set_ylabel("Number of trades", color="tab:gray")
    ax_pnl.set_title(title or "Z* grid search: N_trades and P&L vs Z")

    handles1, labels1 = ax_pnl.get_legend_handles_labels()
    handles2, labels2 = ax_n.get_legend_handles_labels()
    ax_pnl.legend(handles1 + handles2, labels1 + labels2, loc="best")
    ax_pnl.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


if __name__ == "__main__":
    from src.pipeline import run_pair_analysis

    pd.set_option("display.width", 120)

    for pair in config.PAIRS:
        result = run_pair_analysis(pair, save_plots=False)
        z_opt = optimise_z(result)

        print(f"\n=== {pair['id']} ({pair['name']}) ===")
        print(z_opt["z_stats"].to_string(index=False))
        print(f"Selected Z* = {z_opt['z_star']:g} ({z_opt['z_star_source']})")

        out_dir = config.pair_output_dir(pair["id"])
        z_opt["z_stats"].to_csv(out_dir / "z_optimisation.csv", index=False)
        plot_z_optimisation(z_opt["z_stats"], z_star=z_opt["z_star"],
                            title=f"{pair['name']}: Z* grid search",
                            save_path=out_dir / "z_optimisation.png")
