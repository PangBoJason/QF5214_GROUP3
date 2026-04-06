"""
Module 3: Out-of-sample SDE backtest; read sde_predictions when metadata matches; else write sde_predictions.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

from feature_engineering import ENGINEERED_FEATURES_TABLE
from train_model import BUNDLE_FILENAME, DEFAULT_DB_PATH, DEFAULT_MODELS_DIR, NUMERIC_FEATURES, TOP_N_CARRIERS

TEST_YEAR_START: int = 2017
TEST_YEAR_END: int = 2025
N_MC_PATHS: int = 1000
MC_ROW_CHUNK: int = 50_000

SDE_PREDICTIONS_TABLE: str = "sde_predictions"
SDE_META_TABLE: str = "sde_run_meta"


def _load_bundle(models_dir: str) -> Dict[str, Any]:
    path: str = os.path.join(models_dir, BUNDLE_FILENAME)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model bundle not found; train first: {path}")
    return joblib.load(path)


def _engineered_row_count(db_path: str) -> int:
    conn: sqlite3.Connection = sqlite3.connect(db_path)
    try:
        cur: sqlite3.Cursor = conn.cursor()
        cur.execute(f"SELECT COUNT(*) FROM {ENGINEERED_FEATURES_TABLE}")
        return int(cur.fetchone()[0])
    except sqlite3.OperationalError:
        return -1
    finally:
        conn.close()


def compute_sde_signature(
    bundle_path: str,
    feature_columns: List[str],
    n_mc_paths: int,
    random_seed: int,
    engineered_row_count: int,
) -> str:
    st: os.stat_result = os.stat(bundle_path)
    payload: Dict[str, Any] = {
        "bundle_mtime_ns": st.st_mtime_ns,
        "bundle_size": st.st_size,
        "feature_columns": feature_columns,
        "test_year_start": TEST_YEAR_START,
        "test_year_end": TEST_YEAR_END,
        "n_mc_paths": n_mc_paths,
        "random_seed": random_seed,
        "engineered_row_count": engineered_row_count,
        "engineered_table": ENGINEERED_FEATURES_TABLE,
    }
    raw: bytes = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _read_sde_meta(conn: sqlite3.Connection) -> Dict[str, Any] | None:
    try:
        cur: sqlite3.Cursor = conn.cursor()
        cur.execute(f"SELECT signature, mae, wmape, rmse, n_test FROM {SDE_META_TABLE} WHERE id = 1")
        row = cur.fetchone()
        if row is None:
            return None
        return {
            "signature": row[0],
            "mae": row[1],
            "wmape": row[2],
            "rmse": row[3],
            "n_test": row[4],
        }
    except sqlite3.OperationalError:
        return None


def _write_sde_meta(
    conn: sqlite3.Connection,
    signature: str,
    mae: float,
    wmape: float,
    rmse: float,
    n_test: int,
) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {SDE_META_TABLE} (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            signature TEXT,
            created_at TEXT,
            mae REAL,
            wmape REAL,
            rmse REAL,
            n_test INTEGER
        )
        """
    )
    ts: str = datetime.now(timezone.utc).isoformat()
    conn.execute(
        f"INSERT OR REPLACE INTO {SDE_META_TABLE} (id, signature, created_at, mae, wmape, rmse, n_test) VALUES (1, ?, ?, ?, ?, ?, ?)",
        (signature, ts, mae, wmape, rmse, n_test),
    )


def _load_test_engineered(db_path: str) -> pd.DataFrame:
    q: str = f"""
        SELECT *
        FROM {ENGINEERED_FEATURES_TABLE}
        WHERE year >= {TEST_YEAR_START} AND year <= {TEST_YEAR_END}
    """
    conn: sqlite3.Connection = sqlite3.connect(db_path)
    try:
        return pd.read_sql_query(q, conn)
    finally:
        conn.close()


def _apply_carrier_groups(df: pd.DataFrame, top_carriers: List[str]) -> pd.DataFrame:
    out: pd.DataFrame = df.copy()
    out["carrier"] = out["carrier"].astype(str).where(out["carrier"].astype(str).isin(top_carriers), "Other")
    return out


def _apply_te_from_bundle(
    df: pd.DataFrame,
    origin_te_map: Dict[str, float],
    dest_te_map: Dict[str, float],
    global_mean: float,
) -> pd.DataFrame:
    out: pd.DataFrame = df.copy()
    o_key: pd.Series = out["origin"].astype(str)
    d_key: pd.Series = out["dest"].astype(str)
    out["origin_te"] = o_key.map(origin_te_map).fillna(global_mean).astype(float)
    out["dest_te"] = d_key.map(dest_te_map).fillna(global_mean).astype(float)
    return out


def _build_X_test(
    df: pd.DataFrame,
    top_carriers: List[str],
    feature_columns: List[str],
) -> pd.DataFrame:
    data: Dict[str, np.ndarray] = {}

    for col in NUMERIC_FEATURES:
        data[col] = df[col].to_numpy(dtype=np.float32, copy=False)

    month_vals: np.ndarray = df["month"].to_numpy(dtype=np.int16, copy=False)
    for month in sorted(pd.unique(month_vals)):
        data[f"month_{int(month)}"] = (month_vals == month).astype(np.float32, copy=False)

    carrier_grouped: np.ndarray = (
        df["carrier"].astype(str).where(df["carrier"].astype(str).isin(top_carriers), "Other").to_numpy()
    )
    for carrier in sorted(pd.unique(carrier_grouped)):
        data[f"carrier_{carrier}"] = (carrier_grouped == carrier).astype(np.float32, copy=False)

    X_raw: pd.DataFrame = pd.DataFrame(data, index=df.index, copy=False)
    return X_raw.reindex(columns=feature_columns, fill_value=np.float32(0.0)).astype(np.float32, copy=False)


def _mc_mean_pred_pax_chunked(
    lf_hat: np.ndarray,
    seats_vec: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
    n_mc_paths: int,
    row_chunk: int,
) -> np.ndarray:
    n_rows: int = int(len(lf_hat))
    mean_lf: np.ndarray = np.empty(n_rows, dtype=np.float64)
    ch: int = max(1, int(row_chunk))
    for start in range(0, n_rows, ch):
        end: int = min(start + ch, n_rows)
        lf_b: np.ndarray = lf_hat[start:end]
        noise: np.ndarray = rng.normal(loc=0.0, scale=sigma, size=(end - start, n_mc_paths))
        sim: np.ndarray = np.clip(lf_b[:, np.newaxis] + noise, 0.0, 1.0)
        mean_lf[start:end] = sim.mean(axis=1)
    return mean_lf * seats_vec


def _mae_wmape_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    err: np.ndarray = y_pred - y_true
    mae: float = float(np.mean(np.abs(err)))
    denom: float = float(np.sum(np.abs(y_true)))
    wmape: float = float(np.sum(np.abs(err)) / denom) if denom > 0 else float("nan")
    rmse: float = float(np.sqrt(np.mean(err**2)))
    return mae, wmape, rmse


def _load_predictions_from_db(db_path: str) -> pd.DataFrame:
    conn: sqlite3.Connection = sqlite3.connect(db_path)
    try:
        return pd.read_sql_query(f"SELECT * FROM {SDE_PREDICTIONS_TABLE}", conn)
    finally:
        conn.close()


def _per_carrier_wmape(pred_df: pd.DataFrame) -> List[Dict[str, float | str]]:
    """Aggregate per-carrier WMAPE once so downstream reporting/plotting can reuse it."""
    if pred_df.empty:
        return []

    carrier_col: str = "carrier_iata" if "carrier_iata" in pred_df.columns else "carrier"
    out: List[Dict[str, float | str]] = []
    for carrier, g in pred_df.groupby(carrier_col, observed=True):
        y_true: np.ndarray = g["passengers"].to_numpy(dtype=float)
        y_pred: np.ndarray = g["final_pred_pax"].to_numpy(dtype=float)
        denom: float = float(np.sum(np.abs(y_true)))
        wmape: float = float(np.sum(np.abs(y_pred - y_true)) / denom) if denom > 0 else float("nan")
        out.append(
            {
                "carrier": str(carrier),
                "wmape": wmape,
                "n_rows": int(len(g)),
            }
        )
    out.sort(key=lambda r: (float("inf") if not np.isfinite(float(r["wmape"])) else float(r["wmape"])), reverse=True)
    return out


def run_backtest(
    db_path: str = DEFAULT_DB_PATH,
    models_dir: str = DEFAULT_MODELS_DIR,
    n_mc_paths: int = N_MC_PATHS,
    random_seed: int = 42,
    mc_row_chunk: int = MC_ROW_CHUNK,
    force_sde: bool = False,
) -> Dict[str, Any]:
    bundle: Dict[str, Any] = _load_bundle(models_dir)
    model: Any = bundle["model"]
    feature_columns: List[str] = bundle["feature_columns"]
    origin_te_map: Dict[str, float] = bundle["origin_te_map"]
    dest_te_map: Dict[str, float] = bundle["dest_te_map"]
    global_mean: float = float(bundle["global_mean_passengers"])
    top_carriers: List[str] = bundle["top_carriers"]
    sigma: float = float(bundle["sigma"])

    if len(top_carriers) > TOP_N_CARRIERS:
        top_carriers = top_carriers[:TOP_N_CARRIERS]

    bundle_path: str = os.path.join(models_dir, BUNDLE_FILENAME)
    eng_n: int = _engineered_row_count(db_path)
    sig: str = compute_sde_signature(
        bundle_path, feature_columns, n_mc_paths, random_seed, eng_n
    )

    conn_chk: sqlite3.Connection = sqlite3.connect(db_path)
    try:
        meta: Dict[str, Any] | None = _read_sde_meta(conn_chk)
        if not force_sde and meta is not None and meta.get("signature") == sig:
            try:
                cur = conn_chk.cursor()
                cur.execute(f"SELECT COUNT(*) FROM {SDE_PREDICTIONS_TABLE}")
                n_cached: int = int(cur.fetchone()[0])
            except sqlite3.OperationalError:
                n_cached = 0
            if n_cached > 0:
                pred_df: pd.DataFrame = _load_predictions_from_db(db_path)
                y_t: np.ndarray = pred_df["passengers"].to_numpy(dtype=float)
                y_p: np.ndarray = pred_df["final_pred_pax"].to_numpy(dtype=float)
                mae, wmape, rmse = _mae_wmape_rmse(y_t, y_p)
                carrier_wmape: List[Dict[str, float | str]] = _per_carrier_wmape(pred_df)
                print(f"========== SDE: metadata cache hit signature={sig[:12]}... skip MC ==========")
                return {
                    "n_test": int(len(pred_df)),
                    "mae": mae,
                    "wmape": wmape,
                    "rmse": rmse,
                    "sigma": sigma,
                    "n_mc_paths": n_mc_paths,
                    "predictions_df": pred_df,
                    "per_carrier_wmape": carrier_wmape,
                    "from_cache": True,
                }
    finally:
        conn_chk.close()

    df: pd.DataFrame = _load_test_engineered(db_path)
    if df.empty:
        raise RuntimeError(
            f"Test set empty; ensure {ENGINEERED_FEATURES_TABLE} has {TEST_YEAR_START}-{TEST_YEAR_END} rows"
        )

    df = df.dropna(subset=NUMERIC_FEATURES + ["passengers", "seats", "origin", "dest", "carrier", "month", "distance"])
    df = df.loc[df["seats"] > 0].copy()
    df["carrier_iata"] = df["carrier"].astype(str)
    df = _apply_carrier_groups(df, top_carriers)
    df = _apply_te_from_bundle(df, origin_te_map, dest_te_map, global_mean)

    X_test: pd.DataFrame = _build_X_test(df, top_carriers, feature_columns)
    lf_hat: np.ndarray = np.clip(model.predict(X_test).astype(float), 0.0, 1.0)

    rng: np.random.Generator = np.random.default_rng(random_seed)
    seats_vec: np.ndarray = df["seats"].to_numpy(dtype=float)
    final_pred_pax: np.ndarray = _mc_mean_pred_pax_chunked(
        lf_hat, seats_vec, sigma, rng, n_mc_paths, mc_row_chunk
    )

    y_true: np.ndarray = df["passengers"].to_numpy(dtype=float)
    mae, wmape, rmse = _mae_wmape_rmse(y_true, final_pred_pax)

    dist: np.ndarray = df["distance"].to_numpy(dtype=float)
    pred_rpk: np.ndarray = final_pred_pax * dist
    actual_rpk: np.ndarray = y_true * dist
    ask: np.ndarray = seats_vec * dist

    out_df: pd.DataFrame = pd.DataFrame(
        {
            "year": df["year"].to_numpy(),
            "month": df["month"].to_numpy(),
            "origin": df["origin"].astype(str).to_numpy(),
            "dest": df["dest"].astype(str).to_numpy(),
            "carrier": df["carrier_iata"].astype(str).to_numpy(),
            "carrier_iata": df["carrier_iata"].to_numpy(),
            "distance": dist,
            "seats": seats_vec,
            "passengers": y_true,
            "final_pred_pax": final_pred_pax.astype(float),
            "pred_rpk": pred_rpk.astype(float),
            "actual_rpk": actual_rpk.astype(float),
            "ask": ask.astype(float),
        }
    )

    conn2: sqlite3.Connection = sqlite3.connect(db_path)
    try:
        conn2.execute("PRAGMA journal_mode=WAL")
        conn2.execute("PRAGMA synchronous=NORMAL")
        out_df.to_sql(SDE_PREDICTIONS_TABLE, conn2, if_exists="replace", index=False)
        conn2.execute(
            "CREATE INDEX IF NOT EXISTS idx_sde_year_carrier "
            "ON sde_predictions(year, carrier)"
        )
        conn2.execute(
            "CREATE INDEX IF NOT EXISTS idx_sde_year_month "
            "ON sde_predictions(year, month)"
        )
        _write_sde_meta(conn2, sig, mae, wmape, rmse, int(len(out_df)))
        conn2.commit()
    finally:
        conn2.close()

    print("========== Backtest (2017–2025, load-factor SDE → passengers) ==========")
    print(f"Test rows: {len(out_df):,}")
    print(f"MAE:       {mae:,.4f}")
    print(f"WMAPE:     {wmape:.4%}")
    print(f"RMSE:      {rmse:,.4f}")
    print(f"Wrote {SDE_PREDICTIONS_TABLE} / {SDE_META_TABLE}")

    carrier_wmape: List[Dict[str, float | str]] = _per_carrier_wmape(out_df)

    return {
        "n_test": int(len(out_df)),
        "mae": mae,
        "wmape": wmape,
        "rmse": rmse,
        "sigma": sigma,
        "n_mc_paths": n_mc_paths,
        "predictions_df": out_df,
        "per_carrier_wmape": carrier_wmape,
        "from_cache": False,
    }


if __name__ == "__main__":
    run_backtest()
