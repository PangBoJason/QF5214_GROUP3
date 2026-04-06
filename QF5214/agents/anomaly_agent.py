"""
Anomaly Detection Agent — powered by GPT-5.

Runs after each SDE backtest. Checks for:
- global error metrics
- target-airline WMAPE above threshold (>20%)
- monthly data volume anomalies

GPT-5 summarises findings in natural language and writes a Markdown report.
"""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import openai

from backtest_model import SDE_PREDICTIONS_TABLE

REPORTS_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
WMAPE_ALERT_THRESHOLD: float = 0.20  # 20%
TARGET_CARRIER_MAP: Dict[str, str] = {
    "DL": "DAL",
    "AA": "AAL",
    "UA": "UAL",
    "WN": "LUV",
    "AS": "ALK",
    "B6": "JBLU",
}

_SYSTEM_PROMPT = """\
You are a data quality analyst for an aviation quantitative pipeline.
You will receive a JSON summary of anomaly checks. Write a concise Markdown report:
- Title: "## Anomaly Detection Report — <date>"
- A bullet list of findings (flag issues with ⚠️ , ok items with ✅)
- A short recommendation section
Keep it under 400 words.
"""


def _compute_checks(bt: Dict[str, Any], db_path: str) -> Dict[str, Any]:
    checks: Dict[str, Any] = {}

    # 1. Global WMAPE
    checks["global_wmape"] = bt.get("wmape", float("nan"))
    checks["global_mae"] = bt.get("mae", float("nan"))
    checks["from_cache"] = bt.get("from_cache", False)

    # 2. Per-carrier WMAPE from predictions table (focus on target listed airlines)
    try:
        conn = sqlite3.connect(db_path)
        import pandas as pd
        pred_df = pd.read_sql_query(
            f"SELECT carrier, carrier_iata, passengers, final_pred_pax FROM {SDE_PREDICTIONS_TABLE}",
            conn,
        )
        conn.close()
        raw_carrier: pd.Series
        if "carrier_iata" in pred_df.columns:
            raw_carrier = pred_df["carrier_iata"].fillna(pred_df["carrier"])
        else:
            raw_carrier = pred_df["carrier"]
        pred_df["target_ticker"] = raw_carrier.astype(str).str.strip().map(TARGET_CARRIER_MAP)
        pred_df = pred_df[pred_df["target_ticker"].notna()].copy()

        carrier_wmape: Dict[str, float] = {}
        for carrier, g in pred_df.groupby("target_ticker", observed=True):
            err = np.abs(g["final_pred_pax"].to_numpy() - g["passengers"].to_numpy())
            denom = np.sum(np.abs(g["passengers"].to_numpy()))
            carrier_wmape[str(carrier)] = float(np.sum(err) / denom) if denom > 0 else float("nan")
        checks["carrier_wmape"] = carrier_wmape
        checks["carrier_scope"] = list(TARGET_CARRIER_MAP.values())
        alerts = [c for c, w in carrier_wmape.items() if np.isfinite(w) and w > WMAPE_ALERT_THRESHOLD]
        checks["carrier_wmape_alerts"] = alerts
    except Exception as e:
        checks["carrier_wmape_error"] = str(e)

    # 3. Monthly row counts: flag months with < 50% of median row count
    try:
        conn = sqlite3.connect(db_path)
        import pandas as pd
        monthly = pd.read_sql_query(
            f"SELECT year, month, COUNT(*) as cnt FROM {SDE_PREDICTIONS_TABLE} GROUP BY year, month",
            conn,
        )
        conn.close()
        median_cnt = float(monthly["cnt"].median())
        low_months = monthly[monthly["cnt"] < median_cnt * 0.5][["year", "month", "cnt"]].to_dict("records")
        checks["low_volume_months"] = low_months
    except Exception as e:
        checks["volume_check_error"] = str(e)

    return checks


def _ask_gpt(checks: Dict[str, Any], model: str, temperature: float) -> str:
    import json
    client = openai.OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )
    user_msg = f"Anomaly check results:\n{json.dumps(checks, indent=2, default=str)}"
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )
    return resp.choices[0].message.content.strip()


def run_anomaly_detection(
    bt: Dict[str, Any],
    db_path: str,
    cfg: Dict[str, Any],
) -> None:
    agent_cfg = cfg.get("agents", {})
    model: str = agent_cfg.get("model", "gpt-5")
    temperature: float = float(agent_cfg.get("temperature", 0.2))

    print("[anomaly_agent] Running anomaly checks...")
    checks = _compute_checks(bt, db_path)

    # Print quick summary
    wmape = checks.get("global_wmape", float("nan"))
    alerts = checks.get("carrier_wmape_alerts", [])
    low_months = checks.get("low_volume_months", [])
    print(f"  Global WMAPE: {wmape:.2%}")
    if alerts:
        print(f"  ⚠️  High-WMAPE carriers (>{WMAPE_ALERT_THRESHOLD:.0%}): {alerts}")
    if low_months:
        print(f"  ⚠️  Low-volume months detected: {len(low_months)}")

    # GPT-5 narrative
    try:
        report_md = _ask_gpt(checks, model, temperature)
        os.makedirs(REPORTS_DIR, exist_ok=True)
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(REPORTS_DIR, f"anomaly_report_{date_str}.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_md)
        print(f"[anomaly_agent] Report saved → {report_path}")
    except Exception as e:
        print(f"[anomaly_agent] GPT call failed: {e}")
