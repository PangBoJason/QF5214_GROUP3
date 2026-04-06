"""
Module 6: Pipeline orchestrator — feature engineering → train → SDE → quarterly factors → MWU fundamentals.
Each stage may skip repeated work based on tables / metadata.
"""
from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import yaml

from backtest_model import run_backtest
from factor_engineering import run_factor_engineering
from feature_engineering import ENGINEERED_FEATURES_TABLE, run_feature_engineering
from fundamental_backtest import run_fundamental_backtest
from train_model import BUNDLE_FILENAME, DEFAULT_MODELS_DIR, run_training

# ── Config ────────────────────────────────────────────────────────────────────

CONFIG_PATH: str = os.path.join(os.path.dirname(__file__), "config.yaml")

DEFAULT_DB_PATH: str = os.path.join("data", "quant_flights.db")
DEFAULT_DATA_DIR: str = "data"

SEP_MAJOR: str = "=" * 72
SEP_MINOR: str = "-" * 72


def _load_config(path: str = CONFIG_PATH) -> Dict[str, Any]:
    """Load config.yaml; return empty dict if file missing."""
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _setup_logging(cfg: Dict[str, Any]) -> None:
    """Configure root logger: stdout + optional file."""
    log_cfg = cfg.get("logging", {})
    level_str: str = log_cfg.get("level", "INFO").upper()
    level: int = getattr(logging, level_str, logging.INFO)
    log_file: str | None = log_cfg.get("file")

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


logger = logging.getLogger("qf5214.pipeline")

# ── Helpers ───────────────────────────────────────────────────────────────────


def _table_exists(db_path: str, table: str) -> bool:
    conn: sqlite3.Connection = sqlite3.connect(db_path)
    try:
        cur: sqlite3.Cursor = conn.cursor()
        cur.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
            (table,),
        )
        return cur.fetchone() is not None
    finally:
        conn.close()


def ensure_directories(data_dir: str, models_dir: str) -> None:
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    # logs/ dir is created by _setup_logging


def model_bundle_path(models_dir: str) -> str:
    return os.path.join(models_dir, BUNDLE_FILENAME)


def model_cache_exists(models_dir: str) -> bool:
    return os.path.isfile(model_bundle_path(models_dir))


# ── Pipeline ──────────────────────────────────────────────────────────────────


def run_pipeline(
    db_path: str = DEFAULT_DB_PATH,
    models_dir: str = DEFAULT_MODELS_DIR,
    data_dir: str = DEFAULT_DATA_DIR,
    force_retrain: bool = False,
    force_features: bool = False,
    force_sde: bool = False,
    force_factors: bool = False,
    tune_hyperparams: bool = False,
    cfg: Dict[str, Any] | None = None,
) -> None:
    if cfg is None:
        cfg = {}

    logger.info(SEP_MAJOR)
    logger.info("Aviation quant pipeline (SQLite bus) | %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info(SEP_MAJOR)

    ensure_directories(data_dir, models_dir)

    if not os.path.isfile(db_path):
        logger.warning("Database missing %s; run clean_data.py first", db_path)

    # ---------- Hyperparameter tuning (optional) ----------
    if tune_hyperparams:
        logger.info(SEP_MINOR)
        logger.info("Hyperparameter tuning (GPT-5 agent) ...")
        logger.info(SEP_MINOR)
        try:
            from agents.hyperparam_agent import run_hyperparam_tuning
            best = run_hyperparam_tuning(db_path=db_path, models_dir=models_dir, cfg=cfg)
            logger.info("Tuning done | best_params=%s | best_mae=%.4f", best.get("params"), best.get("mae"))
            # Update cfg with best params so subsequent training uses them
            if "params" in best:
                cfg.setdefault("training", {}).setdefault("xgboost", {}).update(best["params"])
        except Exception as exc:
            logger.error("Hyperparameter tuning failed: %s", exc)

    # ---------- Phase 0: Feature engineering ----------
    logger.info(SEP_MINOR)
    logger.info("Phase 0 · Feature engineering → %s", ENGINEERED_FEATURES_TABLE)
    logger.info(SEP_MINOR)
    t0: float = time.perf_counter()
    if force_features or not _table_exists(db_path, ENGINEERED_FEATURES_TABLE):
        fe: dict = run_feature_engineering(db_path)
        logger.info("Feature table refreshed | %s rows | %.2fs", f"{fe['n_rows']:,}", time.perf_counter() - t0)
    else:
        logger.info("Table %s exists, skip (use --force-features to rebuild)", ENGINEERED_FEATURES_TABLE)

    # ---------- Phase 1: Training ----------
    logger.info(SEP_MINOR)
    logger.info("Phase 1 · Model training")
    logger.info(SEP_MINOR)
    t1: float = time.perf_counter()
    train_info: dict | None = None
    if model_cache_exists(models_dir) and not force_retrain:
        logger.info("Model bundle found, skip training (use --force-retrain to retrain)")
    else:
        xgb_cfg = cfg.get("training", {}).get("xgboost", {})
        train_info = run_training(db_path=db_path, models_dir=models_dir, xgb_overrides=xgb_cfg)
        logger.info(
            "Training done | %.2fs | n=%s | sigma_train=%.4f | sigma_oos=%.4f",
            time.perf_counter() - t1,
            f"{train_info['n_train']:,}",
            train_info.get("sigma_train", float("nan")),
            train_info.get("sigma", float("nan")),
        )

    if not os.path.isfile(model_bundle_path(models_dir)):
        raise FileNotFoundError("Model bundle missing; cannot continue SDE.")

    # ---------- Phase 2: SDE ----------
    logger.info(SEP_MINOR)
    logger.info("Phase 2 · SDE backtest / cache")
    logger.info(SEP_MINOR)
    t2: float = time.perf_counter()
    bt_cfg = cfg.get("backtest", {})
    bt: dict = run_backtest(
        db_path=db_path,
        models_dir=models_dir,
        force_sde=force_sde,
        n_mc_paths=bt_cfg.get("n_mc_paths", 1000),
        random_seed=bt_cfg.get("random_seed", 42),
        mc_row_chunk=bt_cfg.get("mc_row_chunk", 50000),
    )
    logger.info(
        "SDE done | %.2fs | cache=%s | MAE=%.4f | WMAPE=%.4f",
        time.perf_counter() - t2,
        bt.get("from_cache", False),
        bt["mae"],
        bt["wmape"],
    )

    # ---------- Anomaly detection agent ----------
    try:
        from agents.anomaly_agent import run_anomaly_detection
        run_anomaly_detection(bt=bt, db_path=db_path, cfg=cfg)
    except Exception as exc:
        logger.warning("[anomaly_agent] failed (non-fatal): %s", exc, exc_info=True)

    # ---------- Phase 3: Quarterly factors ----------
    logger.info(SEP_MINOR)
    logger.info("Phase 3 · Quarterly factor table")
    logger.info(SEP_MINOR)
    t3: float = time.perf_counter()
    sde_refreshed: bool = not bool(bt.get("from_cache", False))
    fac_cfg = cfg.get("factor", {})
    if force_factors or not _table_exists(db_path, "quarterly_factors") or sde_refreshed:
        fr: dict = run_factor_engineering(
            db_path,
            zscore_within_quarter=fac_cfg.get("zscore_within_quarter", True),
            winsorize_pct=fac_cfg.get("winsorize_pct", 0.01),
        )
        logger.info("quarterly_factors | %s rows | %.2fs", f"{fr['n_rows']:,}", time.perf_counter() - t3)
    else:
        logger.info("quarterly_factors exists and SDE was not recomputed, skip (use --force-factors to rebuild)")

    # ---------- Phase 4: MWU fundamentals ----------
    logger.info(SEP_MINOR)
    logger.info("Phase 4 · MWU fundamental backtest")
    logger.info(SEP_MINOR)
    t4: float = time.perf_counter()
    fund_cfg = cfg.get("fundamental", {})
    yf_cfg = fund_cfg.get("yfinance", {})
    yf_end: str = datetime.now().strftime("%Y-%m-%d") if yf_cfg.get("end_auto", True) else yf_cfg.get("end", "2026-03-02")
    fund_result: dict = run_fundamental_backtest(
        db_path=db_path,
        yf_start=yf_cfg.get("start", "2017-01-01"),
        yf_end=yf_end,
        mwu_eta=fund_cfg.get("mwu_eta", 2.0),
        transaction_cost_bps=fund_cfg.get("transaction_cost_bps", 50.0),
        post_event_bdays=fund_cfg.get("event_window_days", 5),
    )
    logger.info("Fundamentals done | %.2fs", time.perf_counter() - t4)

    # ---------- Report agent ----------
    try:
        from agents.report_agent import run_report
        run_report(
            bt=bt,
            train_info=train_info,
            fund_result=fund_result,
            cfg=cfg,
        )
    except Exception as exc:
        logger.warning("[report_agent] failed (non-fatal): %s", exc, exc_info=True)

    # ---------- Plots ----------
    try:
        from plotting import run_all_plots
        run_all_plots(bt=bt, fund_result=fund_result, cfg=cfg)
    except Exception as exc:
        logger.warning("[plotting] failed (non-fatal): %s", exc, exc_info=True)

    logger.info(SEP_MAJOR)
    logger.info("Pipeline finished | %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info(SEP_MAJOR)


# ── CLI ───────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aviation quant SQLite bus pipeline")
    p.add_argument("--force-retrain", action="store_true")
    p.add_argument("--force-features", action="store_true")
    p.add_argument("--force-sde", action="store_true", help="Ignore SDE metadata cache; rerun Monte Carlo")
    p.add_argument("--force-factors", action="store_true")
    p.add_argument("--tune-hyperparams", action="store_true", help="Run GPT-5 hyperparameter tuning agent before training")
    p.add_argument("--db", default=None)
    # Default None so config.yaml values take precedence over hardcoded fallbacks
    p.add_argument("--models-dir", default=None)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--config", default=CONFIG_PATH, help="Path to config.yaml")
    return p.parse_args()


if __name__ == "__main__":
    # Load .env for OPENAI_API_KEY etc.
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
    except ImportError:
        pass

    args = _parse_args()
    cfg = _load_config(args.config)
    _setup_logging(cfg)

    db = args.db or cfg.get("data", {}).get("db_path", DEFAULT_DB_PATH)
    models_dir = args.models_dir or cfg.get("models", {}).get("dir", DEFAULT_MODELS_DIR)
    data_dir = args.data_dir or cfg.get("data", {}).get("data_dir", DEFAULT_DATA_DIR)

    run_pipeline(
        db_path=db,
        models_dir=models_dir,
        data_dir=data_dir,
        force_retrain=args.force_retrain,
        force_features=args.force_features,
        force_sde=args.force_sde,
        force_factors=args.force_factors,
        tune_hyperparams=args.tune_hyperparams,
        cfg=cfg,
    )
