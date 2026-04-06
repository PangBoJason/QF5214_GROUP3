"""
Hyperparameter Tuning Agent — powered by GPT-5.

Runs walk-forward 3-fold CV iteratively; after each round GPT-5 suggests the next
candidate XGBoost hyperparameter set. Best params are written back to config.yaml.

Usage (via main.py):
    python main.py --tune-hyperparams
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
from typing import Any, Dict, List, Tuple

import numpy as np
import openai
import yaml

from feature_engineering import ENGINEERED_FEATURES_TABLE
from train_model import (
    NUMERIC_FEATURES,
    TOP_N_CARRIERS,
    CV_FOLDS,
    _apply_te_codes,
    _build_feature_matrix,
    _build_target_encoding_maps,
    _carrier_top_n_map,
    _make_xgb,
    TRAIN_YEAR_START,
    TRAIN_YEAR_END,
)

CONFIG_PATH: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
MAX_ITERATIONS: int = 20

_SYSTEM_PROMPT = """\
You are an expert ML engineer specialising in XGBoost hyperparameter optimisation.
You will receive a history of (params, OOS_MAE) pairs from walk-forward cross-validation
on an aviation load-factor regression task. Your goal is to minimise OOS MAE.

Respond ONLY with a JSON object of the next hyperparameter candidate — no prose, no markdown.
Valid keys: n_estimators (int 100-1000), max_depth (int 3-12), learning_rate (float 0.01-0.3),
subsample (float 0.5-1.0), colsample_bytree (float 0.5-1.0).
Example: {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8}
"""


def _load_train_df(db_path: str) -> "pd.DataFrame":
    import pandas as pd
    q = f"SELECT * FROM {ENGINEERED_FEATURES_TABLE} WHERE year >= {TRAIN_YEAR_START} AND year <= {TRAIN_YEAR_END}"
    conn = sqlite3.connect(db_path)
    try:
        return pd.read_sql_query(q, conn)
    finally:
        conn.close()


def _cv_mae(df_full, top_carriers, origin_te_map, dest_te_map, global_mean, xgb_params) -> float:
    import numpy as np
    all_errors: List[float] = []
    for tr_s, tr_e, val_s, val_e in CV_FOLDS:
        fold_train = df_full[(df_full["year"] >= tr_s) & (df_full["year"] <= tr_e)].copy()
        fold_val = df_full[(df_full["year"] >= val_s) & (df_full["year"] <= val_e)].copy()
        if fold_train.empty or fold_val.empty:
            continue
        for sub in (fold_train, fold_val):
            sub["carrier"] = sub["carrier"].astype(str).where(sub["carrier"].astype(str).isin(top_carriers), "Other")
        fold_train = _apply_te_codes(fold_train, origin_te_map, dest_te_map, global_mean)
        fold_val = _apply_te_codes(fold_val, origin_te_map, dest_te_map, global_mean)
        X_tr, feat_cols = _build_feature_matrix(fold_train, top_carriers, fit_mode=True)
        X_vl, _ = _build_feature_matrix(fold_val, top_carriers, fit_mode=False, feature_columns=feat_cols)
        y_tr = fold_train["label_load_factor"].astype(float)
        y_vl = fold_val["label_load_factor"].astype(float)
        m = _make_xgb(xgb_params)
        m.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)
        preds = m.predict(X_vl)
        all_errors.extend(np.abs(y_vl.to_numpy() - preds).tolist())
    return float(np.mean(all_errors)) if all_errors else float("nan")


def _ask_gpt(history: List[Dict[str, Any]], model: str, temperature: float) -> Dict[str, Any]:
    client = openai.OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )
    user_msg = f"Experiment history (params → OOS MAE):\n{json.dumps(history, indent=2)}\n\nSuggest next params."
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )
    raw = resp.choices[0].message.content.strip()
    # Extract JSON even if model wraps in backticks
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        raise ValueError(f"GPT did not return a JSON object: {raw!r}")
    return json.loads(m.group(0))


def _save_best_to_config(best_params: Dict[str, Any]) -> None:
    if not os.path.isfile(CONFIG_PATH):
        return
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg.setdefault("training", {}).setdefault("xgboost", {}).update(best_params)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
    print(f"[hyperparam_agent] Best params written to {CONFIG_PATH}")


def run_hyperparam_tuning(
    db_path: str,
    models_dir: str,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    agent_cfg = cfg.get("agents", {})
    model: str = agent_cfg.get("model", "gpt-5")
    temperature: float = float(agent_cfg.get("temperature", 0.2))

    print(f"[hyperparam_agent] Starting GPT-5 hyperparameter tuning (max {MAX_ITERATIONS} iterations)...")

    import pandas as pd
    df_full = _load_train_df(db_path)
    if df_full.empty:
        raise RuntimeError("Train set empty; run feature_engineering first")
    df_full = df_full.dropna(subset=NUMERIC_FEATURES + ["passengers", "seats", "origin", "dest", "carrier", "month"])
    df_full = df_full.loc[df_full["seats"] > 0].copy()

    carrier_mapped, top_carriers = _carrier_top_n_map(df_full["carrier"], TOP_N_CARRIERS)
    df_full = df_full.assign(carrier=carrier_mapped)
    origin_te_map, dest_te_map, global_mean = _build_target_encoding_maps(df_full)

    # Initial params from config
    init_params = {
        "n_estimators": int(cfg.get("training", {}).get("xgboost", {}).get("n_estimators", 400)),
        "max_depth": int(cfg.get("training", {}).get("xgboost", {}).get("max_depth", 8)),
        "learning_rate": float(cfg.get("training", {}).get("xgboost", {}).get("learning_rate", 0.05)),
        "subsample": float(cfg.get("training", {}).get("xgboost", {}).get("subsample", 0.9)),
        "colsample_bytree": float(cfg.get("training", {}).get("xgboost", {}).get("colsample_bytree", 0.9)),
        "random_state": 42,
    }
    init_mae = _cv_mae(df_full, top_carriers, origin_te_map, dest_te_map, global_mean, init_params)
    history: List[Dict[str, Any]] = [{"params": init_params, "mae": init_mae}]
    best_mae = init_mae
    best_params = init_params.copy()
    print(f"  iter 0 | params={init_params} | MAE={init_mae:.4f}")

    for i in range(1, MAX_ITERATIONS + 1):
        try:
            candidate = _ask_gpt(history, model, temperature)
        except Exception as e:
            print(f"  [hyperparam_agent] GPT call failed at iter {i}: {e}")
            break

        candidate.setdefault("random_state", 42)
        mae = _cv_mae(df_full, top_carriers, origin_te_map, dest_te_map, global_mean, candidate)
        history.append({"params": candidate, "mae": mae})
        print(f"  iter {i} | params={candidate} | MAE={mae:.4f}")

        if mae < best_mae:
            best_mae = mae
            best_params = candidate.copy()

    _save_best_to_config(best_params)
    print(f"[hyperparam_agent] Best found: params={best_params} | MAE={best_mae:.4f}")
    return {"params": best_params, "mae": best_mae, "history": history}
