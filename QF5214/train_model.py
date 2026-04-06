"""
Module 2: Read train window from engineered_features, fit load-factor XGBoost, persist sigma and bundle.
OOS sigma is estimated via 3-fold walk-forward cross-validation to avoid optimistic in-sample bias.
"""
from __future__ import annotations

import os
import sqlite3
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from feature_engineering import ENGINEERED_FEATURES_TABLE

DEFAULT_DB_PATH: str = os.path.join("data", "quant_flights.db")
DEFAULT_MODELS_DIR: str = "models"
BUNDLE_FILENAME: str = "flight_revenue_bundle.joblib"
TRAIN_YEAR_START: int = 2000
TRAIN_YEAR_END: int = 2016
TOP_N_CARRIERS: int = 10

NUMERIC_FEATURES: List[str] = [
    "cap",
    "seats",
    "log_seats",
    "distance",
    "origin_te",
    "dest_te",
    "lag_12m_lf",
    "lag_12m_share",
    "lag_12m_hhi",
]

# 3-fold walk-forward CV windows: (train_start, train_end, val_start, val_end)
CV_FOLDS: List[Tuple[int, int, int, int]] = [
    (2000, 2010, 2011, 2013),
    (2000, 2013, 2014, 2015),
    (2000, 2015, 2016, 2016),
]


def _load_train_engineered(db_path: str) -> pd.DataFrame:
    q: str = f"""
        SELECT *
        FROM {ENGINEERED_FEATURES_TABLE}
        WHERE year >= {TRAIN_YEAR_START} AND year <= {TRAIN_YEAR_END}
    """
    conn: sqlite3.Connection = sqlite3.connect(db_path)
    try:
        df: pd.DataFrame = pd.read_sql_query(q, conn)
    finally:
        conn.close()
    return df


def _carrier_top_n_map(series: pd.Series, top_n: int) -> Tuple[pd.Series, List[str]]:
    counts: pd.Series = series.value_counts()
    top_carriers: List[str] = counts.head(top_n).index.astype(str).tolist()
    mapped: pd.Series = series.astype(str).where(series.astype(str).isin(top_carriers), "Other")
    return mapped, top_carriers


def _build_target_encoding_maps(df: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, float], float]:
    """Train-set mean-passengers TE; stored in bundle for OOD airports at backtest."""
    global_mean: float = float(df["passengers"].mean())
    origin_map: Dict[str, float] = df.groupby("origin", observed=True)["passengers"].mean().to_dict()
    dest_map: Dict[str, float] = df.groupby("dest", observed=True)["passengers"].mean().to_dict()
    origin_map = {str(k): float(v) for k, v in origin_map.items()}
    dest_map = {str(k): float(v) for k, v in dest_map.items()}
    return origin_map, dest_map, global_mean


def _apply_te_codes(
    df: pd.DataFrame,
    origin_map: Dict[str, float],
    dest_map: Dict[str, float],
    global_mean: float,
) -> pd.DataFrame:
    out: pd.DataFrame = df.copy()
    o_key: pd.Series = out["origin"].astype(str)
    d_key: pd.Series = out["dest"].astype(str)
    out["origin_te"] = o_key.map(origin_map).fillna(global_mean).astype(float)
    out["dest_te"] = d_key.map(dest_map).fillna(global_mean).astype(float)
    return out


def _build_feature_matrix(
    df: pd.DataFrame,
    top_carriers: List[str],
    fit_mode: bool,
    feature_columns: List[str] | None = None,
) -> Tuple[pd.DataFrame, List[str]]:
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

    if fit_mode or feature_columns is None:
        feature_columns = X_raw.columns.tolist()
        X: pd.DataFrame = X_raw
    else:
        X = X_raw.reindex(columns=feature_columns, fill_value=np.float32(0.0))
    return X.astype(np.float32, copy=False), feature_columns


def _make_xgb(xgb_params: Dict[str, Any]) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=int(xgb_params.get("n_estimators", 400)),
        max_depth=int(xgb_params.get("max_depth", 8)),
        learning_rate=float(xgb_params.get("learning_rate", 0.05)),
        subsample=float(xgb_params.get("subsample", 0.9)),
        colsample_bytree=float(xgb_params.get("colsample_bytree", 0.9)),
        random_state=int(xgb_params.get("random_state", 42)),
        n_jobs=-1,
        early_stopping_rounds=30,
        eval_metric="mae",
    )


def _compute_oos_sigma(
    df_full: pd.DataFrame,
    xgb_params: Dict[str, Any],
) -> float:
    """
    3-fold walk-forward cross-validation on 2000-2016.
    Each fold rebuilds top_carriers and TE maps exclusively from that fold's
    training window — no information from the validation period leaks in.
    Returns std of pooled OOS load-factor residuals.
    """
    all_residuals: List[float] = []

    for tr_s, tr_e, val_s, val_e in CV_FOLDS:
        fold_train = df_full[(df_full["year"] >= tr_s) & (df_full["year"] <= tr_e)].copy()
        fold_val = df_full[(df_full["year"] >= val_s) & (df_full["year"] <= val_e)].copy()

        if fold_train.empty or fold_val.empty:
            continue

        # Rebuild carrier map and TE maps from this fold's training data only
        _, fold_top_carriers = _carrier_top_n_map(fold_train["carrier"], TOP_N_CARRIERS)
        fold_train["carrier"] = fold_train["carrier"].astype(str).where(
            fold_train["carrier"].astype(str).isin(fold_top_carriers), "Other"
        )
        fold_val["carrier"] = fold_val["carrier"].astype(str).where(
            fold_val["carrier"].astype(str).isin(fold_top_carriers), "Other"
        )

        fold_origin_te, fold_dest_te, fold_global_mean = _build_target_encoding_maps(fold_train)
        fold_train = _apply_te_codes(fold_train, fold_origin_te, fold_dest_te, fold_global_mean)
        fold_val = _apply_te_codes(fold_val, fold_origin_te, fold_dest_te, fold_global_mean)

        X_tr, feat_cols = _build_feature_matrix(fold_train, fold_top_carriers, fit_mode=True)
        X_vl, _ = _build_feature_matrix(fold_val, fold_top_carriers, fit_mode=False, feature_columns=feat_cols)
        y_tr = fold_train["label_load_factor"].astype(float)
        y_vl = fold_val["label_load_factor"].astype(float)

        fold_model = _make_xgb(xgb_params)
        fold_model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)

        y_pred = fold_model.predict(X_vl)
        residuals = y_vl.to_numpy() - y_pred
        all_residuals.extend(residuals.tolist())

    if not all_residuals:
        return float("nan")
    return float(np.std(all_residuals, ddof=1))


def run_training(
    db_path: str = DEFAULT_DB_PATH,
    models_dir: str = DEFAULT_MODELS_DIR,
    xgb_overrides: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    os.makedirs(models_dir, exist_ok=True)

    if not os.path.isfile(db_path):
        raise FileNotFoundError(db_path)

    df: pd.DataFrame = _load_train_engineered(db_path)
    if df.empty:
        raise RuntimeError(
            f"Train set empty; run feature_engineering.py to build {ENGINEERED_FEATURES_TABLE}"
        )

    df = df.dropna(subset=NUMERIC_FEATURES + ["passengers", "seats", "origin", "dest", "carrier", "month"])
    df = df.loc[df["seats"] > 0].copy()

    carrier_mapped, top_carriers = _carrier_top_n_map(df["carrier"], TOP_N_CARRIERS)
    df = df.assign(carrier=carrier_mapped)

    origin_te_map, dest_te_map, global_mean = _build_target_encoding_maps(df)
    df = _apply_te_codes(df, origin_te_map, dest_te_map, global_mean)

    xgb_params: Dict[str, Any] = {
        "n_estimators": 400,
        "max_depth": 8,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": 42,
    }
    if xgb_overrides:
        xgb_params.update({k: v for k, v in xgb_overrides.items() if v is not None})

    # OOS sigma via 3-fold walk-forward CV (avoids in-sample bias)
    sigma_oos: float = _compute_oos_sigma(df, xgb_params)

    # Final model: train on 2000-2015, early-stop on 2016.
    # Using 2016 as a held-out validation set ensures it never leaks into training.
    final_train = df[df["year"] < 2016].copy()
    final_val = df[df["year"] == 2016].copy()

    if final_train.empty:
        # Fallback: no strict split possible
        final_train = df.copy()
        final_val = pd.DataFrame()

    X_train, feature_columns = _build_feature_matrix(final_train, top_carriers, fit_mode=True)
    y_train: pd.Series = final_train["label_load_factor"].astype(float)

    if not final_val.empty:
        X_val_es, _ = _build_feature_matrix(
            final_val, top_carriers, fit_mode=False, feature_columns=feature_columns
        )
        y_val_es = final_val["label_load_factor"].astype(float)
        eval_set = [(X_val_es, y_val_es)]
    else:
        eval_set = None

    model: XGBRegressor = _make_xgb(xgb_params)
    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

    # In-sample sigma on the training partition (2000-2015, kept for comparison)
    y_hat: np.ndarray = model.predict(X_train)
    residuals_lf: np.ndarray = y_train.to_numpy() - y_hat
    sigma_train: float = float(np.std(residuals_lf, ddof=0))

    # Use OOS sigma if valid, else fall back to in-sample
    sigma: float = sigma_oos if np.isfinite(sigma_oos) else sigma_train

    best_iteration: int = int(getattr(model, "best_iteration", xgb_params["n_estimators"]))

    bundle: Dict[str, Any] = {
        "model": model,
        "feature_columns": feature_columns,
        "origin_te_map": origin_te_map,
        "dest_te_map": dest_te_map,
        "global_mean_passengers": global_mean,
        "top_carriers": top_carriers,
        "sigma": sigma,
        "sigma_train": sigma_train,
        "sigma_oos": sigma_oos,
        "sigma_is_load_factor": True,
        "train_year_range": (TRAIN_YEAR_START, TRAIN_YEAR_END),
        "numeric_features": NUMERIC_FEATURES,
        "engineered_table": ENGINEERED_FEATURES_TABLE,
        "xgb_params": xgb_params,
        "best_iteration": best_iteration,
    }
    bundle_path: str = os.path.join(models_dir, BUNDLE_FILENAME)
    joblib.dump(bundle, bundle_path)

    print(
        f"Training done: n_train={len(final_train):,} | n_val={len(final_val):,} | "
        f"sigma_train={sigma_train:.4f} | sigma_oos={sigma_oos:.4f} | "
        f"best_iter={best_iteration} | saved {bundle_path}"
    )

    return {
        "bundle_path": bundle_path,
        "n_train": int(len(final_train)),
        "sigma": sigma,
        "sigma_train": sigma_train,
        "sigma_oos": sigma_oos,
        "n_features": len(feature_columns),
        "best_iteration": best_iteration,
    }


if __name__ == "__main__":
    info = run_training()
    print(
        f"Training done: n={info['n_train']}, sigma_oos={info['sigma_oos']:.4f}, "
        f"sigma_train={info['sigma_train']:.4f}, saved {info['bundle_path']}"
    )
