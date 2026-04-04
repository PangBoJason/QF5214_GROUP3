from __future__ import annotations

import argparse
import sqlite3

import numpy as np
import pandas as pd
from numpy.polynomial import polynomial as P

from project_config import DB_PATH, TARGET_TICKERS, ensure_directories, log_pipeline_run


def load_quarterly(conn: sqlite3.Connection) -> pd.DataFrame:
    frame = pd.read_sql_query(
        """
        SELECT ticker, quarter_end, rpk_100m, load_factor
        FROM flights_quarterly_features
        ORDER BY ticker, quarter_end
        """,
        conn,
    )
    frame["quarter_end"] = pd.to_datetime(frame["quarter_end"])
    return frame


def compute_lf_momentum(quarterly: pd.DataFrame) -> pd.DataFrame:
    """
    Compute load-factor momentum per ticker as the OLS slope of the last 4
    quarters of load_factor (quarterly change per quarter).  This replaces the
    Ridge λ signal which was near-zero due to regularisation flattening the curve.

    Returns DataFrame with columns [ticker, lf_slope] where lf_slope is in
    units of load_factor / quarter.
    """
    rows = []
    for ticker, grp in quarterly.groupby("ticker"):
        lf = grp.sort_values("quarter_end")["load_factor"].dropna().tail(4).values
        if len(lf) < 2:
            rows.append({"ticker": ticker, "lf_slope": 0.0})
            continue
        x = np.arange(len(lf), dtype=float)
        # Simple OLS slope
        slope = float(np.polyfit(x, lf, 1)[0])
        rows.append({"ticker": ticker, "lf_slope": slope})
    return pd.DataFrame(rows)


def calibrate_ou(log_rpk: np.ndarray, dt: float = 1.0) -> tuple[float, float, float]:
    """
    Calibrate a log-OU (Ornstein-Uhlenbeck) process to quarterly log(RPK).

    Model: dX = κ(θ - X)dt + σ dW,  X_t = log(RPK_t)
    Discrete: X_{t+1} = a + b * X_t + ε

    κ and θ estimated via OLS on full history (including COVID — mean-reversion
    naturally absorbs the shock). σ estimated via IQR of residuals so that the
    COVID outlier quarters don't inflate the long-run volatility estimate.
    """
    x = log_rpk[:-1]
    y = log_rpk[1:]
    # OLS for mean-reversion parameters
    b = float(np.cov(x, y)[0, 1] / np.var(x, ddof=1))
    b = np.clip(b, 0.0, 0.9999)   # force mean-reversion (b < 1)
    a = float(np.mean(y) - b * np.mean(x))
    residuals = y - (a + b * x)
    kappa = float(-np.log(b) / dt)
    theta = float(a / (1.0 - b))
    # Robust σ: use IQR / 1.349 (consistent estimator for normal σ)
    # This downweights COVID crash/recovery quarters without excluding them
    q75, q25 = np.percentile(residuals, [75, 25])
    sigma_robust = float((q75 - q25) / 1.349 / np.sqrt(dt))
    # Fallback to std if IQR collapses (too few points)
    if sigma_robust < 1e-6:
        sigma_robust = float(np.std(residuals, ddof=1) / np.sqrt(dt))
    return kappa, theta, sigma_robust


def simulate_paths(
    quarterly: pd.DataFrame,
    lambda_signal: pd.DataFrame,
    ticker: str,
    paths: int,
    horizon_quarters: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ticker_quarterly = quarterly[quarterly["ticker"] == ticker].sort_values("quarter_end").copy()
    if len(ticker_quarterly) < 6:
        raise RuntimeError(f"{ticker} does not have enough quarterly history for SDE calibration.")

    log_rpk = np.log(ticker_quarterly["rpk_100m"].values)
    kappa, theta, sigma = calibrate_ou(log_rpk, dt=1.0)

    # Lambda adjustment: shift long-run mean θ by LSTM signal
    # LF momentum signal: OLS slope of last 4 quarters of load_factor (LF/quarter)
    # Convert to log-RPK drift: ΔRPK/RPK ≈ ΔLF / mean_LF  (since RPK = LF × ASK)
    lf_row = lambda_signal[lambda_signal["ticker"] == ticker]
    lf_slope = float(lf_row["lf_slope"].iloc[0]) if not lf_row.empty else 0.0
    mean_lf = float(ticker_quarterly["load_factor"].tail(8).mean())
    if pd.isna(mean_lf) or mean_lf <= 0:
        mean_lf = 0.85
    lambda_adjustment = lf_slope / mean_lf   # dimensionless quarterly log-RPK shift
    theta_adjusted = theta + lambda_adjustment

    start_log = float(np.log(ticker_quarterly["rpk_100m"].iloc[-1]))
    start_quarter = pd.Period(ticker_quarterly["quarter_end"].iloc[-1], freq="Q")

    rng = np.random.default_rng(random_seed)
    dt = 1.0
    shocks = rng.normal(0.0, np.sqrt(dt), size=(paths, horizon_quarters))

    # Simulate log(RPK) via Euler-Maruyama on OU
    log_sim = np.zeros((paths, horizon_quarters + 1), dtype=float)
    log_sim[:, 0] = start_log
    for step in range(1, horizon_quarters + 1):
        prev = log_sim[:, step - 1]
        log_sim[:, step] = (
            prev
            + kappa * (theta_adjusted - prev) * dt
            + sigma * shocks[:, step - 1]
        )
    simulated = np.exp(log_sim)

    path_rows = []
    for path_idx in range(paths):
        for step in range(horizon_quarters + 1):
            period = start_quarter + step
            path_rows.append(
                {
                    "ticker": ticker,
                    "path_id": path_idx,
                    "step": step,
                    "quarter_label": str(period),
                    "sim_rpk_100m": float(simulated[path_idx, step]),
                }
            )

    summary = pd.DataFrame(
        {
            "ticker": ticker,
            "step": np.arange(horizon_quarters + 1),
            "quarter_label": [str(start_quarter + step) for step in range(horizon_quarters + 1)],
            "mean_rpk_100m": simulated.mean(axis=0),
            "p10_rpk_100m": np.quantile(simulated, 0.10, axis=0),
            "p50_rpk_100m": np.quantile(simulated, 0.50, axis=0),
            "p90_rpk_100m": np.quantile(simulated, 0.90, axis=0),
            "kappa": kappa,
            "theta": theta_adjusted,
            "sigma": sigma,
            "lf_momentum_adjustment": lambda_adjustment,
        }
    )
    return pd.DataFrame(path_rows), summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Euler-Maruyama RPK Monte Carlo simulation (log-OU).")
    parser.add_argument("--paths", type=int, default=1000)
    parser.add_argument("--horizon-quarters", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ticker", choices=TARGET_TICKERS, default=None)
    parser.add_argument("--run-id", default="standalone")
    args = parser.parse_args()

    ensure_directories()
    with sqlite3.connect(DB_PATH) as conn:
        quarterly = load_quarterly(conn)
        lambda_signal = compute_lf_momentum(quarterly)
        tickers = [args.ticker] if args.ticker else TARGET_TICKERS
        all_paths = []
        all_summary = []
        for ticker in tickers:
            try:
                paths_frame, summary_frame = simulate_paths(quarterly, lambda_signal, ticker, args.paths, args.horizon_quarters, args.seed)
                all_paths.append(paths_frame)
                all_summary.append(summary_frame)
                print(f"[{ticker}] κ={summary_frame['kappa'].iloc[0]:.3f}  θ={summary_frame['theta'].iloc[0]:.3f}  σ={summary_frame['sigma'].iloc[0]:.3f}")
            except RuntimeError as exc:
                print(f"[{ticker}] skip — {exc}")

        path_frame = pd.concat(all_paths, ignore_index=True)
        summary_frame = pd.concat(all_summary, ignore_index=True)
        path_frame.to_sql("sde_paths", conn, if_exists="replace", index=False)
        summary_frame.to_sql("sde_summary", conn, if_exists="replace", index=False)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sde_paths_ticker ON sde_paths(ticker, path_id, step)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sde_summary_ticker ON sde_summary(ticker, step)")
        sample_start = str(summary_frame["quarter_label"].iloc[0]) if not summary_frame.empty else ""
        sample_end   = str(summary_frame["quarter_label"].iloc[-1]) if not summary_frame.empty else ""
        log_pipeline_run(conn, args.run_id, "sde_simulation", "success",
                         "sde_summary", len(summary_frame), sample_start, sample_end)
        conn.commit()

    print(f"[done] simulated rows: {len(path_frame):,}")


if __name__ == "__main__":
    main()
