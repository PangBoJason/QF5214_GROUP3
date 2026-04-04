from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3

import numpy as np
import pandas as pd
from openai import OpenAI

import charts
import sde_simulation
from project_config import DB_PATH, REPORTS_DIR, TARGET_TICKERS, ensure_directories, load_dotenv


# ── Structured fact summary (Python-side, no LLM) ────────────────────────────

def _safe_float(val) -> float | None:
    try:
        v = float(val)
        return None if np.isnan(v) else v
    except (TypeError, ValueError):
        return None


def build_fact_summary() -> dict:
    """Query the DB and return a structured summary dict.

    This is the single source of truth injected into the LLM system prompt.
    The LLM must only reference values present here; it must not query the DB
    independently for report-writing purposes.
    """
    with sqlite3.connect(DB_PATH) as conn:
        # ── Latest operating metrics per ticker ───────────────────────────────
        qf = pd.read_sql_query(
            """
            SELECT ticker, quarter_end, rpk_100m, load_factor,
                   rpk_yoy_growth, rpk_qoq_growth
            FROM flights_quarterly_features
            ORDER BY ticker, quarter_end
            """, conn,
        )
        operating: dict = {}
        latest_quarter = ""
        for ticker, grp in qf.groupby("ticker"):
            row = grp.sort_values("quarter_end").iloc[-1]
            latest_quarter = str(pd.Period(row["quarter_end"], freq="Q"))
            operating[ticker] = {
                "quarter": latest_quarter,
                "rpk_yoy_growth": _safe_float(row["rpk_yoy_growth"]),
                "load_factor": _safe_float(row["load_factor"]),
                "rpk_qoq_growth": _safe_float(row["rpk_qoq_growth"]),
            }

        # ── SDE simulation 2025Q4 (step=4) ───────────────────────────────────
        try:
            sde = pd.read_sql_query(
                "SELECT ticker, mean_rpk_100m, p10_rpk_100m, p50_rpk_100m, p90_rpk_100m "
                "FROM sde_summary WHERE step = 4", conn,
            )
            simulation: dict = {
                row["ticker"]: {
                    "mean": _safe_float(row["mean_rpk_100m"]),
                    "p10":  _safe_float(row["p10_rpk_100m"]),
                    "p50":  _safe_float(row["p50_rpk_100m"]),
                    "p90":  _safe_float(row["p90_rpk_100m"]),
                }
                for _, row in sde.iterrows()
            }
        except Exception:
            simulation = {}

        # ── Factor IC — last 4 quarters ───────────────────────────────────────
        try:
            ic = pd.read_sql_query(
                "SELECT * FROM factor_ic_results ORDER BY fiscal_quarter DESC LIMIT 4", conn,
            )
            factor_ic_last4q = ic.to_dict(orient="records")
        except Exception:
            factor_ic_last4q = []

        # ── Backtest summary ──────────────────────────────────────────────────
        try:
            bs = pd.read_sql_query("SELECT metric, value FROM factor_backtest_summary", conn)
            backtest_summary = dict(zip(bs["metric"], bs["value"].map(_safe_float)))
        except Exception:
            backtest_summary = {}

        # ── Sentiment coverage ────────────────────────────────────────────────
        try:
            sf = pd.read_sql_query("SELECT sentiment_compound FROM sentiment_factors", conn)
            sentiment_coverage_pct = float(sf["sentiment_compound"].notna().mean() * 100)
        except Exception:
            sentiment_coverage_pct = None

        # ── Per-factor constant quarters (each factor is INDEPENDENT) ─────────
        per_factor_constant_quarters: dict[str, list[str]] = {}
        if factor_ic_last4q:
            ic_df = pd.DataFrame(factor_ic_last4q)
            flag_cols = [c for c in ic_df.columns if c.endswith("_constant_flag")]
            for fc in flag_cols:
                factor_name = fc.replace("_constant_flag", "")
                quarters = ic_df.loc[ic_df[fc] == 1, "fiscal_quarter"].tolist()
                if quarters:
                    per_factor_constant_quarters[factor_name] = [str(q) for q in quarters]

        # ── Relative ranking (Python-side) ────────────────────────────────────
        tickers = [t for t in TARGET_TICKERS if t in operating]

        def _rank(key: str, ascending: bool = False) -> dict[str, int]:
            vals = [(t, operating[t].get(key) or 0) for t in tickers]
            vals.sort(key=lambda x: x[1], reverse=not ascending)
            return {t: i + 1 for i, (t, _) in enumerate(vals)}

        def _sim_rank() -> dict[str, int]:
            vals = [(t, (simulation.get(t) or {}).get("mean") or 0) for t in tickers]
            vals.sort(key=lambda x: x[1], reverse=True)
            return {t: i + 1 for i, (t, _) in enumerate(vals)}

        rpk_rank  = _rank("rpk_yoy_growth", ascending=False)
        lf_rank   = _rank("load_factor",    ascending=False)
        sim_rank  = _sim_rank()

        relative_rank: dict = {}
        for t in tickers:
            score = rpk_rank.get(t, 6) + lf_rank.get(t, 6) + sim_rank.get(t, 6)
            relative_rank[t] = {
                "rpk_yoy_rank":   rpk_rank.get(t),
                "load_factor_rank": lf_rank.get(t),
                "sim_mean_rank":  sim_rank.get(t),
                "overall_score":  score,
            }
        # overall_rank: 1 = best (lowest score)
        sorted_tickers = sorted(tickers, key=lambda t: relative_rank[t]["overall_score"])
        for rank_pos, t in enumerate(sorted_tickers, 1):
            relative_rank[t]["overall_rank"] = rank_pos

    return {
        "universe": TARGET_TICKERS,
        "latest_quarter": latest_quarter,
        "operating": operating,
        "simulation_2025q4": simulation,
        "factor_ic_last4q": factor_ic_last4q,
        "backtest_summary": backtest_summary,
        "surprise_coverage_pct": backtest_summary.get("surprise_pct_coverage_pct"),
        "momentum_coverage_pct": backtest_summary.get("momentum_coverage_pct"),
        "per_factor_constant_quarters": per_factor_constant_quarters,
        "sentiment_coverage_pct": sentiment_coverage_pct,
        "relative_rank": relative_rank,
    }


# ── Report validation ─────────────────────────────────────────────────────────

def validate_report(text: str, summary: dict) -> list[str]:
    issues = []
    # 1. All 6 tickers must appear
    for t in summary["universe"]:
        if t not in text:
            issues.append(f"Missing ticker: {t}")
    # 2. No stale universe references
    for phrase in ("four carriers", "four airline carriers", "4 carriers"):
        if phrase.lower() in text.lower():
            issues.append(f"Stale universe reference: '{phrase}'")
    # 3. No wrong sentiment label
    for phrase in ("finbert", "news sentiment"):
        if phrase.lower() in text.lower():
            issues.append(f"Incorrect sentiment label: '{phrase}'")
    # 4. All 5 required section headings must be present
    required_sections = [
        "Latest Operating Snapshot",
        "Factor Efficacy",
        "Simulation Outlook",
        "Relative Ranking",
        "Risks",
    ]
    for sec in required_sections:
        if sec.lower() not in text.lower():
            issues.append(f"Missing required section: '{sec}'")
    # 5. IC values should not appear immediately after a single ticker (per-ticker IC smell)
    for t in summary["universe"]:
        if re.search(rf"{t}[^\n]{{0,40}}\bIC\b[^\n]{{0,20}}\b0\.\d+", text):
            issues.append(f"Possible per-ticker IC assignment near {t}")
    return issues


# ── Tool implementations ──────────────────────────────────────────────────────

def query_data(table: str, ticker: str | None = None, limit: int = 20) -> list[dict]:
    allowed_tables = {
        "flights_quarterly_features",
        "flights_monthly_features",
        "earnings_event_features",
        "sentiment_factors",
        "factor_ic_results",
        "sde_summary",
        "lstm_lambda_forecasts",
    }
    if table not in allowed_tables:
        return [{"error": f"Table '{table}' not available. Choose from: {sorted(allowed_tables)}"}]

    order_by_map = {
        "flights_quarterly_features": "quarter_end DESC",
        "flights_monthly_features": "month_end DESC",
        "earnings_event_features": "earnings_date DESC",
        "sentiment_factors": "earnings_date DESC",
        "factor_ic_results": "fiscal_quarter DESC",
        "sde_summary": "step ASC",
        "lstm_lambda_forecasts": "earnings_date DESC, curve_idx ASC",
    }
    _no_ticker_tables = {"factor_ic_results"}
    sql = f"SELECT * FROM {table}"
    params: list = []
    if ticker and table not in _no_ticker_tables:
        sql += " WHERE ticker = ?"
        params.append(ticker)
    sql += f" ORDER BY {order_by_map[table]} LIMIT ?"
    params.append(limit)
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(sql, params)
        rows = cursor.fetchall()
        columns = [col[0] for col in cursor.description]
    return [dict(zip(columns, row)) for row in rows]


def run_simulation(ticker: str, paths: int = 1000, horizon_quarters: int = 4) -> dict:
    with sqlite3.connect(DB_PATH) as conn:
        quarterly = sde_simulation.load_quarterly(conn)
        lf_momentum = sde_simulation.compute_lf_momentum(quarterly)
        _, summary = sde_simulation.simulate_paths(quarterly, lf_momentum, ticker, paths, horizon_quarters, 42)
        return summary.to_dict(orient="records")[-1]


_TICKER_REQUIRED_CHARTS = {"rpk_load_factor", "sentiment", "simulation_fan"}


def generate_chart(chart_name: str, ticker: str | None = None) -> str:
    if chart_name in _TICKER_REQUIRED_CHARTS and not ticker:
        raise ValueError(f"chart_name '{chart_name}' requires a ticker argument")
    with sqlite3.connect(DB_PATH) as conn:
        if chart_name == "rpk_load_factor":
            return str(charts.plot_rpk_and_load_factor(conn, ticker))
        if chart_name == "sentiment":
            return str(charts.plot_sentiment(conn, ticker))
        if chart_name == "simulation_fan":
            return str(charts.plot_simulation_fan(conn, ticker))
        if chart_name == "factor_ic":
            return str(charts.plot_factor_ic(conn))
    raise ValueError(f"Unsupported chart_name: {chart_name}")


def generate_report(ticker: str | None, question: str, answer: str) -> str:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_ticker = (ticker or "market").lower()
    report_path = REPORTS_DIR / f"{safe_ticker}_analysis_report.md"
    report_path.write_text(answer, encoding="utf-8")
    return str(report_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Structured-summary agent for airline earnings analysis.")
    parser.add_argument("--question", required=True)
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--model", default="gpt-4o")
    args = parser.parse_args()

    ensure_directories()
    load_dotenv()
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url="https://api.chatanywhere.org/v1")

    # ── Phase 1: Build structured fact summary (Python-side, no LLM) ─────────
    print("[agent] building structured fact summary ...")
    summary = build_fact_summary()
    summary_json = json.dumps(summary, ensure_ascii=False, default=str, indent=2)

    # ── Phase 2: LLM generates report from summary ────────────────────────────
    system_prompt = (
        "You are an airline earnings quant analyst writing a structured investment report.\n\n"
        "## Fact Summary (authoritative — use ONLY these values)\n"
        f"{summary_json}\n\n"
        "## Report Requirements\n"
        "Write a report with EXACTLY these 5 sections in this order:\n"
        "## 1. Latest Operating Snapshot\n"
        "## 2. Factor Efficacy\n"
        "## 3. Simulation Outlook (2025)\n"
        "## 4. Relative Ranking (all 6 carriers)\n"
        "## 5. Risks & Caveats\n\n"
        "## Mandatory rules\n"
        "(1) IC values from factor_ic_results are cross-sectional statistics for the 6-carrier universe per quarter. "
        "NEVER write IC as a per-ticker metric.\n"
        "(2) Sentiment is EDGAR 8-K filing tone (VADER compound score). "
        "NEVER call it FinBERT or news sentiment.\n"
        "(3) Include ALL 6 carriers: DAL, UAL, AAL, LUV, JBLU, ALK.\n"
        "(4) per_factor_constant_quarters maps each factor name to the quarters where ONLY THAT factor's IC is undefined. "
        "Each factor is independent: sentiment being constant does NOT affect rpk_ic, surprise_ic, momentum_ic, or composite_ic. "
        "Only mark a specific factor's IC as undefined for the quarters listed under that factor's key.\n"
        "(5) Use relative_rank.overall_rank for the ranking section — do not re-derive rankings yourself.\n"
        "(6) surprise_pct and momentum_60d are partial-coverage factors — report their coverage percentages explicitly.\n"
    )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "query_data",
                "description": (
                    "Query supplementary detail from a SQLite table. Use only to retrieve additional context "
                    "not already in the fact summary. "
                    "Tables: flights_quarterly_features, flights_monthly_features, earnings_event_features, "
                    "sentiment_factors (EDGAR 8-K VADER sentiment_compound [-1,1] — NOT FinBERT), "
                    "factor_ic_results (cross-sectional IC per fiscal_quarter, ONE row per quarter NOT per ticker), "
                    "sde_summary (Monte Carlo RPK fan: mean/p10/p50/p90 per step)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "table": {
                            "type": "string",
                            "enum": [
                                "flights_quarterly_features", "flights_monthly_features",
                                "earnings_event_features", "sentiment_factors",
                                "factor_ic_results", "sde_summary", "lstm_lambda_forecasts",
                            ],
                        },
                        "ticker": {"type": "string", "enum": ["AAL", "DAL", "UAL", "LUV", "JBLU", "ALK"]},
                        "limit": {"type": "integer", "default": 20},
                    },
                    "required": ["table"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "generate_report",
                "description": "Persist the final markdown report to the reports directory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string"},
                        "question": {"type": "string"},
                        "answer": {"type": "string"},
                    },
                    "required": ["question", "answer"],
                },
            },
        },
    ]

    tool_impls = {
        "query_data": query_data,
        "generate_report": generate_report,
    }

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": args.question if not args.ticker else f"{args.question}\nTicker focus: {args.ticker}",
        },
    ]

    final_answer = ""
    for _ in range(10):
        response = client.chat.completions.create(
            model=args.model, messages=messages, tools=tools, tool_choice="auto"
        )
        message = response.choices[0].message
        if not message.tool_calls:
            final_answer = message.content or ""
            break

        messages.append(message)
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments or "{}")
            result = tool_impls[tool_name](**tool_args)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_name,
                "content": json.dumps(result, ensure_ascii=False),
            })
    else:
        raise RuntimeError("Agent reached tool loop limit without a final answer.")

    # ── Phase 3: Validate report, append warnings if any ─────────────────────
    issues = validate_report(final_answer, summary)
    if issues:
        warning_block = "\n\n## ⚠ Report Warnings\n" + "\n".join(f"- {i}" for i in issues)
        final_answer += warning_block

    report_path = generate_report(args.ticker, args.question, final_answer)
    print(final_answer)
    print(f"\n[report_saved] {report_path}")
    if issues:
        print(f"[warn] {len(issues)} report validation issue(s) — see '## ⚠ Report Warnings' in report")


if __name__ == "__main__":
    main()
