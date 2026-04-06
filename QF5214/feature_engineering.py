"""
Module 1: Derive features from t100_segment and write engineered_features (SQLite data bus).
- TE: origin/dest target encoding from 2000–2016 train-window mean passengers only (matches current design).
- Hub features omitted per requirements.
"""
from __future__ import annotations

import os
import sqlite3
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

DEFAULT_DB_PATH: str = os.path.join("data", "quant_flights.db")
ENGINEERED_FEATURES_TABLE: str = "engineered_features"
TRAIN_YEAR_START: int = 2000
TRAIN_YEAR_END: int = 2016

READ_DTYPES: Dict[str, str] = {
    "year": "int16",
    "month": "int8",
    "origin": "string",
    "dest": "string",
    "carrier": "string",
    "distance": "float32",
    "departures_performed": "float32",
    "seats": "float32",
    "passengers": "float32",
    "cap": "float32",
}


def _build_te_from_train(
    df: pd.DataFrame,
) -> Tuple[Dict[str, float], Dict[str, float], float]:
    """Train-window mean-passengers target encoding (aligned with original train_model logic)."""
    tr: pd.DataFrame = df.loc[
        (df["year"] >= TRAIN_YEAR_START) & (df["year"] <= TRAIN_YEAR_END)
    ].copy()
    if tr.empty:
        raise RuntimeError("No rows in train window; cannot build TE.")
    global_mean: float = float(tr["passengers"].mean())
    origin_map: Dict[str, float] = {
        str(k): float(v) for k, v in tr.groupby("origin", observed=True)["passengers"].mean().items()
    }
    dest_map: Dict[str, float] = {
        str(k): float(v) for k, v in tr.groupby("dest", observed=True)["passengers"].mean().items()
    }
    return origin_map, dest_map, global_mean


def run_feature_engineering(db_path: str = DEFAULT_DB_PATH) -> Dict[str, Any]:
    """
    Read full t100_segment, engineer features, write engineered_features (if_exists='replace').
    """
    if not os.path.isfile(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")

    query: str = """
        SELECT year, month, origin, dest, carrier, distance, departures_performed, seats, passengers, cap
        FROM t100_segment
    """
    conn: sqlite3.Connection = sqlite3.connect(db_path)
    try:
        df: pd.DataFrame = pd.read_sql_query(query, conn, dtype=READ_DTYPES)
    finally:
        conn.close()

    if df.empty:
        raise RuntimeError("t100_segment is empty; run clean_data.py first")

    for col in ("origin", "dest", "carrier"):
        df[col] = df[col].astype("category")

    df = df.dropna(
        subset=["year", "month", "origin", "dest", "carrier", "seats", "passengers", "distance", "departures_performed"]
    )
    df = df.loc[df["seats"] > 0].copy()
    if "cap" not in df.columns or df["cap"].isna().any():
        df["cap"] = df["seats"] / df["departures_performed"].replace(0, np.nan)
    df = df.dropna(subset=["cap"])

    origin_map, dest_map, global_mean = _build_te_from_train(df)
    o_key: pd.Series = df["origin"].astype(str)
    d_key: pd.Series = df["dest"].astype(str)
    df["origin_te"] = o_key.map(origin_map).fillna(global_mean).astype(np.float32)
    df["dest_te"] = d_key.map(dest_map).fillna(global_mean).astype(np.float32)

    df["log_seats"] = np.log1p(df["seats"].to_numpy(dtype=np.float32)).astype(np.float32)
    df["lf"] = (
        df["passengers"].to_numpy(dtype=np.float32) / df["seats"].to_numpy(dtype=np.float32)
    )
    df["lf"] = pd.Series(df["lf"], index=df.index).replace([np.inf, -np.inf], np.nan).astype(np.float32)
    df["label_load_factor"] = np.clip(df["lf"].to_numpy(dtype=np.float32), 0.0, 1.0).astype(np.float32)

    route_tot: pd.Series = df.groupby(
        ["year", "month", "origin", "dest"], observed=True
    )["seats"].transform("sum")
    df["route_total_seats"] = route_tot.astype(np.float32)
    df["market_share"] = (
        df["seats"].to_numpy(dtype=np.float32) / df["route_total_seats"].replace(0, np.nan).to_numpy(dtype=np.float32)
    )
    df["market_share"] = pd.Series(df["market_share"], index=df.index).fillna(0.0).astype(np.float32)

    ms: pd.Series = df["market_share"]
    df["hhi"] = (
        df.groupby(["year", "month", "origin", "dest"], observed=True)[ms.name]
        .transform(lambda s: ((s.astype(float) * 100.0) ** 2).sum())
    ).astype(np.float32)

    df = df.sort_values(["carrier", "origin", "dest", "year", "month"])
    gcols: List[str] = ["carrier", "origin", "dest"]
    # Single groupby for all three lag features (avoids 3× redundant groupby)
    lag_df: pd.DataFrame = (
        df.groupby(gcols, observed=True)[["lf", "market_share", "hhi"]]
        .shift(12)
        .rename(columns={"lf": "lag_12m_lf", "market_share": "lag_12m_share", "hhi": "lag_12m_hhi"})
    )
    df = df.join(lag_df)

    # Fill NaN lags using train-window mean only (avoids leaking test-set stats)
    train_mask: pd.Series = (df["year"] >= TRAIN_YEAR_START) & (df["year"] <= TRAIN_YEAR_END)
    for c in ("lag_12m_lf", "lag_12m_share", "lag_12m_hhi"):
        fill_val: float = float(df.loc[train_mask, c].mean())
        df[c] = df[c].fillna(fill_val).astype(np.float32)

    out_cols: List[str] = [
        "year",
        "month",
        "origin",
        "dest",
        "carrier",
        "distance",
        "departures_performed",
        "seats",
        "passengers",
        "cap",
        "log_seats",
        "origin_te",
        "dest_te",
        "lf",
        "label_load_factor",
        "route_total_seats",
        "market_share",
        "hhi",
        "lag_12m_lf",
        "lag_12m_share",
        "lag_12m_hhi",
    ]
    out: pd.DataFrame = df[out_cols].copy()

    conn2: sqlite3.Connection = sqlite3.connect(db_path)
    try:
        conn2.execute("PRAGMA journal_mode=WAL")
        conn2.execute("PRAGMA synchronous=NORMAL")
        out.to_sql(ENGINEERED_FEATURES_TABLE, conn2, if_exists="replace", index=False)
        conn2.execute(
            "CREATE INDEX IF NOT EXISTS idx_ef_year_carrier "
            "ON engineered_features(year, carrier)"
        )
        conn2.execute(
            "CREATE INDEX IF NOT EXISTS idx_ef_year "
            "ON engineered_features(year)"
        )
        conn2.commit()
    finally:
        conn2.close()

    return {
        "table": ENGINEERED_FEATURES_TABLE,
        "n_rows": int(len(out)),
        "db_path": os.path.abspath(db_path),
    }


if __name__ == "__main__":
    info = run_feature_engineering()
    print(f"Wrote {info['table']}: {info['n_rows']:,} rows -> {info['db_path']}")
