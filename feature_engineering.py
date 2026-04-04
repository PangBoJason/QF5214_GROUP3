from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from project_config import CARRIER_TO_TICKER, DB_PATH, ensure_directories, log_pipeline_run


MILES_TO_KM = 1.60934
RPK_SCALE = 1e8
LOOKBACK_MONTHS = 12


def load_flights(conn: sqlite3.Connection) -> pd.DataFrame:
    carriers = tuple(CARRIER_TO_TICKER.keys())
    placeholders = ",".join("?" * len(carriers))
    frame = pd.read_sql_query(
        f"""
        SELECT YEAR, MONTH, CARRIER, DISTANCE, SEATS, PASSENGERS, AIR_TIME, DEPARTURES_PERFORMED
        FROM flights_raw
        WHERE CARRIER IN ({placeholders})
        """,
        conn,
        params=list(carriers),
    )
    frame["month_start"] = pd.to_datetime(dict(year=frame["YEAR"], month=frame["MONTH"], day=1))
    frame["month_end"] = frame["month_start"] + pd.offsets.MonthEnd(0)
    frame["ticker"] = frame["CARRIER"].map(CARRIER_TO_TICKER)
    frame["rpk_100m"] = frame["PASSENGERS"] * frame["DISTANCE"] * MILES_TO_KM / RPK_SCALE
    frame["ask_100m"] = frame["SEATS"] * frame["DISTANCE"] * MILES_TO_KM / RPK_SCALE
    return frame


def build_monthly_features(flights: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        flights.groupby(["ticker", "CARRIER", "YEAR", "MONTH", "month_end"], as_index=False)
        .agg(
            passengers=("PASSENGERS", "sum"),
            seats=("SEATS", "sum"),
            departures=("DEPARTURES_PERFORMED", "sum"),
            air_time=("AIR_TIME", "sum"),
            rpk_100m=("rpk_100m", "sum"),
            ask_100m=("ask_100m", "sum"),
        )
        .sort_values(["ticker", "month_end"])
    )
    monthly["load_factor"] = np.where(monthly["ask_100m"] > 0, monthly["rpk_100m"] / monthly["ask_100m"], np.nan)
    monthly["quarter"] = pd.PeriodIndex(monthly["month_end"], freq="Q").quarter
    monthly["quarter_label"] = pd.PeriodIndex(monthly["month_end"], freq="Q").astype(str)
    return monthly


def build_quarterly_features(monthly: pd.DataFrame) -> pd.DataFrame:
    quarterly = (
        monthly.groupby(["ticker", "CARRIER", "YEAR", "quarter", "quarter_label"], as_index=False)
        .agg(
            quarter_end=("month_end", "max"),
            passengers=("passengers", "sum"),
            seats=("seats", "sum"),
            departures=("departures", "sum"),
            air_time=("air_time", "sum"),
            rpk_100m=("rpk_100m", "sum"),
            ask_100m=("ask_100m", "sum"),
        )
        .sort_values(["ticker", "quarter_end"])
    )
    quarterly["load_factor"] = np.where(quarterly["ask_100m"] > 0, quarterly["rpk_100m"] / quarterly["ask_100m"], np.nan)
    quarterly["rpk_qoq_growth"] = quarterly.groupby("ticker")["rpk_100m"].pct_change()
    quarterly["rpk_yoy_growth"] = quarterly.groupby("ticker")["rpk_100m"].pct_change(4)
    quarterly["load_factor_qoq_change"] = quarterly.groupby("ticker")["load_factor"].diff()
    return quarterly


def build_event_sequences(conn: sqlite3.Connection, monthly: pd.DataFrame) -> pd.DataFrame:
    try:
        earnings = pd.read_sql_query(
            "SELECT ticker, earnings_date FROM earnings_calendar_master ORDER BY ticker, earnings_date",
            conn,
        )
    except Exception:
        return pd.DataFrame()

    if earnings.empty:
        return pd.DataFrame()

    earnings["earnings_date"] = pd.to_datetime(earnings["earnings_date"])
    rows: list[dict] = []
    for event in earnings.itertuples(index=False):
        history = monthly[
            (monthly["ticker"] == event.ticker) & (monthly["month_end"] < event.earnings_date)
        ].sort_values("month_end")
        if len(history) < LOOKBACK_MONTHS:
            continue

        window = history.tail(LOOKBACK_MONTHS).reset_index(drop=True)
        for seq_idx, row in window.iterrows():
            rows.append(
                {
                    "event_id": f"{event.ticker}_{event.earnings_date.date().isoformat()}",
                    "ticker": event.ticker,
                    "earnings_date": event.earnings_date.date().isoformat(),
                    "seq_idx": seq_idx,
                    "obs_date": row["month_end"].date().isoformat(),
                    "days_to_earnings": int((event.earnings_date - row["month_end"]).days),
                    "load_factor": row["load_factor"],
                    "rpk_100m": row["rpk_100m"],
                }
            )
    return pd.DataFrame(rows)


def build_event_feature_snapshot(conn: sqlite3.Connection, quarterly: pd.DataFrame) -> pd.DataFrame:
    try:
        earnings = pd.read_sql_query(
            "SELECT ticker, earnings_date, eps_estimate, reported_eps, surprise_pct FROM earnings_calendar_master ORDER BY ticker, earnings_date",
            conn,
        )
    except Exception:
        return pd.DataFrame()

    if earnings.empty:
        return pd.DataFrame()

    earnings["earnings_date"] = pd.to_datetime(earnings["earnings_date"])
    records: list[dict] = []
    for event in earnings.itertuples(index=False):
        available = quarterly[
            (quarterly["ticker"] == event.ticker) & (quarterly["quarter_end"] < event.earnings_date)
        ].sort_values("quarter_end")
        if available.empty:
            continue
        latest = available.iloc[-1]
        records.append(
            {
                "event_id": f"{event.ticker}_{event.earnings_date.date().isoformat()}",
                "ticker": event.ticker,
                "earnings_date": event.earnings_date.date().isoformat(),
                "feature_quarter_end": latest["quarter_end"].date().isoformat(),
                "rpk_100m": latest["rpk_100m"],
                "load_factor": latest["load_factor"],
                "rpk_qoq_growth": latest["rpk_qoq_growth"],
                "rpk_yoy_growth": latest["rpk_yoy_growth"],
                "load_factor_qoq_change": latest["load_factor_qoq_change"],
                "eps_estimate": event.eps_estimate,
                "reported_eps": event.reported_eps,
                "surprise_pct": event.surprise_pct,
            }
        )
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    # One event per (ticker, feature_quarter_end): keep the earliest earnings_date.
    # This drops 2025Q2+ events that would reuse the same 2024Q4 BTS features as 2025Q1.
    frame = (
        frame.sort_values("earnings_date")
             .drop_duplicates(subset=["ticker", "feature_quarter_end"], keep="first")
             .reset_index(drop=True)
    )
    return frame


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Feature engineering.")
    parser.add_argument("--run-id", default="standalone")
    args = parser.parse_args()

    ensure_directories()
    with sqlite3.connect(DB_PATH) as conn:
        flights = load_flights(conn)
        monthly = build_monthly_features(flights)
        quarterly = build_quarterly_features(monthly)
        event_sequences = build_event_sequences(conn, monthly)
        event_snapshot = build_event_feature_snapshot(conn, quarterly)

        monthly.to_sql("flights_monthly_features", conn, if_exists="replace", index=False)
        quarterly.to_sql("flights_quarterly_features", conn, if_exists="replace", index=False)
        if not event_sequences.empty:
            event_sequences.to_sql("earnings_event_sequences", conn, if_exists="replace", index=False)
        if not event_snapshot.empty:
            event_snapshot.to_sql("earnings_event_features", conn, if_exists="replace", index=False)

        conn.execute("CREATE INDEX IF NOT EXISTS idx_monthly_ticker_date ON flights_monthly_features(ticker, month_end)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_quarterly_ticker_date ON flights_quarterly_features(ticker, quarter_end)")
        if not event_sequences.empty:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_event_sequences_event ON earnings_event_sequences(event_id, seq_idx)")
        if not event_snapshot.empty:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_event_features_event ON earnings_event_features(event_id)")

        sample_start = str(quarterly["quarter_end"].min().date()) if not quarterly.empty else ""
        sample_end   = str(quarterly["quarter_end"].max().date()) if not quarterly.empty else ""
        log_pipeline_run(conn, args.run_id, "feature_engineering", "success",
                         "earnings_event_features", len(event_snapshot), sample_start, sample_end)
        conn.commit()

    print(f"[done] monthly rows: {len(monthly):,}")
    print(f"[done] quarterly rows: {len(quarterly):,}")
    print(f"[done] event sequence rows: {len(event_sequences):,}")
    print(f"[done] event snapshot rows: {len(event_snapshot):,}")


if __name__ == "__main__":
    main()
