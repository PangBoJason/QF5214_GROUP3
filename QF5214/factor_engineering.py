"""
Module 4: Aggregate quarterly multi-factors from sde_predictions into quarterly_factors.
- Factor_Share: carrier seats sum in quarter / market seats sum; YoY is current minus prior-year same quarter.
- Factor_LF: YoY relative change of (sum Pred_RPK / sum ASK).
- Factors are cross-sectionally z-scored within each (year, quarter) to remove market-wide bias.
- Outliers winsorized at winsorize_pct / 1 - winsorize_pct.
"""
from __future__ import annotations

import os
import sqlite3
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from backtest_model import SDE_PREDICTIONS_TABLE

DEFAULT_DB_PATH: str = os.path.join("data", "quant_flights.db")
QUARTERLY_FACTORS_TABLE: str = "quarterly_factors"


def _month_to_quarter(month: pd.Series) -> pd.Series:
    return ((month.astype(int) - 1) // 3 + 1).astype(int)


def _winsorize_series(s: pd.Series, pct: float) -> pd.Series:
    """Clip series at pct and (1-pct) quantiles."""
    lo = s.quantile(pct)
    hi = s.quantile(1.0 - pct)
    return s.clip(lo, hi)


def _zscore_within_quarter(df: pd.DataFrame, factor_cols: List[str]) -> pd.DataFrame:
    """
    Cross-sectional z-score each factor within (year, quarter).
    Makes factors relative (removes market-wide bias).
    """
    for fc in factor_cols:
        grp = df.groupby(["year", "quarter"], observed=True)[fc]
        mean = grp.transform("mean")
        std = grp.transform("std").replace(0.0, np.nan)
        df[fc] = (df[fc] - mean) / std
    return df


def run_factor_engineering(
    db_path: str = DEFAULT_DB_PATH,
    zscore_within_quarter: bool = True,
    winsorize_pct: float = 0.01,
) -> Dict[str, Any]:
    conn: sqlite3.Connection = sqlite3.connect(db_path)
    try:
        df: pd.DataFrame = pd.read_sql_query(f"SELECT * FROM {SDE_PREDICTIONS_TABLE}", conn)
    finally:
        conn.close()

    if df.empty:
        raise RuntimeError(f"{SDE_PREDICTIONS_TABLE} is empty; run backtest_model first")

    need: List[str] = ["year", "month", "carrier", "seats", "pred_rpk", "actual_rpk", "ask"]
    miss: List[str] = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"sde_predictions missing columns: {miss}")

    df["quarter"] = _month_to_quarter(df["month"])

    agg: pd.DataFrame = (
        df.groupby(["carrier", "year", "quarter"], observed=True)
        .agg(
            pred_rpk_sum=("pred_rpk", "sum"),
            actual_rpk_sum=("actual_rpk", "sum"),
            ask_sum=("ask", "sum"),
            seats_sum=("seats", "sum"),
        )
        .reset_index()
    )

    tot_seats_q: pd.DataFrame = (
        df.groupby(["year", "quarter"], observed=True)["seats"].sum().reset_index(name="total_seats_market")
    )
    agg = agg.merge(tot_seats_q, on=["year", "quarter"], how="left")
    agg["share"] = (agg["seats_sum"] / agg["total_seats_market"].replace(0, np.nan)).fillna(0.0)

    agg["pred_lf"] = (agg["pred_rpk_sum"] / agg["ask_sum"].replace(0, np.nan)).replace(
        [np.inf, -np.inf], np.nan
    )

    lag_py: pd.DataFrame = agg[["carrier", "year", "quarter", "actual_rpk_sum", "ask_sum", "share"]].copy()
    lag_py["year"] = lag_py["year"] + 1
    lag_py = lag_py.rename(
        columns={
            "actual_rpk_sum": "actual_rpk_py",
            "ask_sum": "ask_py",
            "share": "share_py",
        }
    )

    merged: pd.DataFrame = agg.merge(
        lag_py,
        on=["carrier", "year", "quarter"],
        how="left",
    )

    merged["factor_rpk"] = (merged["pred_rpk_sum"] - merged["actual_rpk_py"]) / merged["actual_rpk_py"].replace(
        0, np.nan
    )

    lf_lag: pd.DataFrame = agg[["carrier", "year", "quarter", "pred_lf"]].copy()
    lf_lag["year"] = lf_lag["year"] + 1
    lf_lag = lf_lag.rename(columns={"pred_lf": "pred_lf_py"})
    merged = merged.merge(lf_lag, on=["carrier", "year", "quarter"], how="left")
    merged["factor_lf"] = (merged["pred_lf"] - merged["pred_lf_py"]) / merged["pred_lf_py"].replace(0, np.nan)

    merged["factor_ask"] = (merged["ask_sum"] - merged["ask_py"]) / merged["ask_py"].replace(0, np.nan)
    merged["factor_share"] = merged["share"] - merged["share_py"]

    factor_cols: List[str] = ["factor_rpk", "factor_lf", "factor_ask", "factor_share"]

    out_cols: List[str] = [
        "carrier",
        "year",
        "quarter",
        "pred_rpk_sum",
        "actual_rpk_sum",
        "ask_sum",
        "share",
        "pred_lf",
    ] + factor_cols
    out: pd.DataFrame = merged[out_cols].replace([np.inf, -np.inf], np.nan).copy()

    # Winsorize outliers before z-scoring
    if winsorize_pct > 0:
        for fc in factor_cols:
            out[fc] = _winsorize_series(out[fc].dropna().reindex(out.index), winsorize_pct)

    # Cross-sectional z-score within each (year, quarter)
    if zscore_within_quarter:
        out = _zscore_within_quarter(out, factor_cols)

    conn2: sqlite3.Connection = sqlite3.connect(db_path)
    try:
        conn2.execute("PRAGMA journal_mode=WAL")
        conn2.execute("PRAGMA synchronous=NORMAL")
        out.to_sql(QUARTERLY_FACTORS_TABLE, conn2, if_exists="replace", index=False)
        conn2.execute(
            "CREATE INDEX IF NOT EXISTS idx_qf_carrier_year_q "
            "ON quarterly_factors(carrier, year, quarter)"
        )
        conn2.commit()
    finally:
        conn2.close()

    return {"table": QUARTERLY_FACTORS_TABLE, "n_rows": int(len(out)), "db_path": os.path.abspath(db_path)}


if __name__ == "__main__":
    r = run_factor_engineering()
    print(f"Wrote {r['table']}: {r['n_rows']:,} rows")
