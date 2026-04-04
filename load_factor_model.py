"""Standalone load-factor curve model (Ridge regression).

NOT part of the main pipeline (main.py).  Run independently to:
  - Visualise predicted load-factor curves per earnings event
  - Inspect Ridge model quality (MAE / MSE metrics)

The SDE simulation uses quarterly LF momentum directly from
flights_quarterly_features, not the Ridge lambda output.

Usage:
    python load_factor_model.py
"""
from __future__ import annotations

import argparse
import json
import pickle
import sqlite3

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

from project_config import DB_PATH, MODELS_DIR, ensure_directories


def load_monthly_features(conn: sqlite3.Connection) -> pd.DataFrame:
    frame = pd.read_sql_query(
        """
        SELECT ticker, month_end, load_factor, rpk_100m
        FROM flights_monthly_features
        ORDER BY ticker, month_end
        """,
        conn,
    )
    frame["month_end"] = pd.to_datetime(frame["month_end"])
    return frame


def load_earnings(conn: sqlite3.Connection) -> pd.DataFrame:
    frame = pd.read_sql_query(
        """
        SELECT ticker, earnings_date
        FROM earnings_calendar_master
        ORDER BY ticker, earnings_date
        """,
        conn,
    )
    frame["earnings_date"] = pd.to_datetime(frame["earnings_date"])
    return frame


def interpolate_curve(anchor_dates: pd.Series, anchor_values: pd.Series, day_grid: np.ndarray) -> np.ndarray:
    if len(anchor_dates) < 2:
        return np.repeat(anchor_values.iloc[-1], len(day_grid))
    base = anchor_dates.iloc[0]
    x = np.array([(date - base).days for date in anchor_dates], dtype=float)
    y = anchor_values.to_numpy(dtype=float)
    return np.interp(day_grid, x, y, left=y[0], right=y[-1])


def build_training_set(
    monthly: pd.DataFrame,
    earnings: pd.DataFrame,
    lookback_months: int,
    forecast_days: int,
    curve_points: int,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    X_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    meta: list[dict] = []
    day_grid = np.linspace(0, forecast_days, curve_points)

    for ticker, ticker_events in earnings.groupby("ticker"):
        monthly_ticker = monthly[monthly["ticker"] == ticker].sort_values("month_end")
        for event in ticker_events.itertuples(index=False):
            history = monthly_ticker[monthly_ticker["month_end"] < event.earnings_date].tail(lookback_months)
            future = monthly_ticker[
                (monthly_ticker["month_end"] >= event.earnings_date)
                & (monthly_ticker["month_end"] <= event.earnings_date + pd.Timedelta(days=forecast_days + 45))
            ].head(4)
            if len(history) < lookback_months or future.empty:
                continue

            sequence = history.copy()
            sequence["days_to_earnings"] = (event.earnings_date - sequence["month_end"]).dt.days
            sequence["month_sin"] = np.sin(2 * np.pi * sequence["month_end"].dt.month / 12)
            sequence["month_cos"] = np.cos(2 * np.pi * sequence["month_end"].dt.month / 12)
            # Flatten the sequence to a 1-D feature vector for Ridge
            X_rows.append(sequence[["load_factor", "rpk_100m", "days_to_earnings", "month_sin", "month_cos"]].to_numpy().ravel())

            target_dates = pd.concat(
                [pd.Series([event.earnings_date]), future["month_end"].reset_index(drop=True)],
                ignore_index=True,
            )
            target_values = pd.concat(
                [pd.Series([history["load_factor"].iloc[-1]]), future["load_factor"].reset_index(drop=True)],
                ignore_index=True,
            )
            y_rows.append(interpolate_curve(target_dates, target_values, day_grid))
            meta.append(
                {
                    "event_id": f"{ticker}_{event.earnings_date.date().isoformat()}",
                    "ticker": ticker,
                    "earnings_date": event.earnings_date.date().isoformat(),
                }
            )

    if not X_rows:
        raise RuntimeError("No training samples available. Run get_stock_data.py and feature_engineering.py first.")

    return np.stack(X_rows), np.stack(y_rows), meta


def fit_model(X: np.ndarray, y: np.ndarray) -> tuple[MultiOutputRegressor, StandardScaler, StandardScaler, dict]:
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_scaled = x_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y)

    model = MultiOutputRegressor(Ridge(alpha=1.0), n_jobs=1)
    model.fit(X_scaled, y_scaled)

    y_pred_scaled = model.predict(X_scaled)
    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    residuals = y - y_pred
    mse = float(np.mean(residuals ** 2))
    mae = float(np.mean(np.abs(residuals)))
    metrics = {"mse": mse, "mae": mae, "n_samples": len(X)}
    return model, x_scaler, y_scaler, metrics


def persist_artifacts(
    conn: sqlite3.Connection,
    model: MultiOutputRegressor,
    x_scaler: StandardScaler,
    y_scaler: StandardScaler,
    X: np.ndarray,
    meta: list[dict],
    curve_points: int,
    forecast_days: int,
) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with (MODELS_DIR / "load_factor_ridge.pkl").open("wb") as fh:
        pickle.dump({"model": model, "x_scaler": x_scaler, "y_scaler": y_scaler}, fh)

    X_scaled = x_scaler.transform(X)
    predictions = y_scaler.inverse_transform(model.predict(X_scaled))
    day_grid = np.linspace(0, forecast_days, curve_points)
    rows: list[dict] = []
    for idx, predicted_curve in enumerate(predictions):
        lambda_curve = np.gradient(predicted_curve, day_grid)
        for point_idx, (day_offset, load_factor, lam) in enumerate(zip(day_grid, predicted_curve, lambda_curve)):
            rows.append(
                {
                    "event_id": meta[idx]["event_id"],
                    "ticker": meta[idx]["ticker"],
                    "earnings_date": meta[idx]["earnings_date"],
                    "curve_idx": point_idx,
                    "day_offset": float(day_offset),
                    "pred_load_factor": float(load_factor),
                    "pred_lambda": float(lam),
                }
            )

    frame = pd.DataFrame(rows)
    frame.to_sql("lstm_lambda_forecasts", conn, if_exists="replace", index=False)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_lstm_event ON lstm_lambda_forecasts(event_id, curve_idx)")
    conn.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Ridge regression on pre-earnings load-factor sequences.")
    parser.add_argument("--lookback-months", type=int, default=12)
    parser.add_argument("--forecast-days", type=int, default=90)
    parser.add_argument("--curve-points", type=int, default=12)
    args = parser.parse_args()

    ensure_directories()
    with sqlite3.connect(DB_PATH) as conn:
        monthly = load_monthly_features(conn)
        earnings = load_earnings(conn)
        X, y, meta = build_training_set(monthly, earnings, args.lookback_months, args.forecast_days, args.curve_points)
        model, x_scaler, y_scaler, metrics = fit_model(X, y)
        persist_artifacts(conn, model, x_scaler, y_scaler, X, meta, args.curve_points, args.forecast_days)

    (MODELS_DIR / "load_factor_ridge_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[done] trained samples: {metrics['n_samples']:,}  MAE: {metrics['mae']:.4f}  MSE: {metrics['mse']:.6f}")


if __name__ == "__main__":
    main()
