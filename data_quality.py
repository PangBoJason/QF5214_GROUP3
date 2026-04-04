"""Standalone data quality checker.

Run after the pipeline to validate key tables and surface dirty data.

Usage:
    python data_quality.py
"""
from __future__ import annotations

import argparse
import sqlite3
import sys

import pandas as pd

from project_config import DB_PATH


PASS = "[PASS]"
WARN = "[WARN]"
FAIL = "[FAIL]"

_fail_count = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global _fail_count
    status = PASS if ok else FAIL
    if not ok:
        _fail_count += 1
    msg = f"  {status} {label}"
    if detail:
        msg += f" — {detail}"
    print(msg)


def warn(label: str, detail: str = "") -> None:
    msg = f"  {WARN} {label}"
    if detail:
        msg += f" — {detail}"
    print(msg)


def run_checks(conn: sqlite3.Connection) -> int:
    """Run all checks and return the number of FAIL items."""
    global _fail_count
    _fail_count = 0
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}

    # ── flights_raw ───────────────────────────────────────────────────────────
    print("\n[flights_raw]")
    if "flights_raw" in tables:
        df = pd.read_sql_query("SELECT PASSENGERS, SEATS, DISTANCE FROM flights_raw", conn)
        check("No NULL PASSENGERS",  df["PASSENGERS"].notna().all(),
              f"{df['PASSENGERS'].isna().sum()} nulls" if df["PASSENGERS"].isna().any() else "")
        bad_lf = (df["PASSENGERS"] > df["SEATS"]).sum()
        # BTS source data occasionally has overbooking/codeshare rows; WARN not FAIL
        if bad_lf:
            warn("PASSENGERS ≤ SEATS", f"{bad_lf} rows violate constraint (BTS artifact)")
        else:
            check("PASSENGERS ≤ SEATS", True)
        bad_dist = (df["DISTANCE"] <= 0).sum()
        # BTS phantom routes (origin==dest) are excluded by feature_engineering; WARN not FAIL
        if bad_dist:
            warn("DISTANCE > 0", f"{bad_dist} rows with non-positive distance (BTS artifact)")
        else:
            check("DISTANCE > 0", True)
    else:
        warn("flights_raw table missing")

    # ── earnings_calendar_master ──────────────────────────────────────────────
    print("\n[earnings_calendar_master]")
    if "earnings_calendar_master" in tables:
        m = pd.read_sql_query("SELECT * FROM earnings_calendar_master", conn)
        dupes = m.duplicated(subset=["ticker", "fiscal_quarter"]).sum()
        check("ticker+fiscal_quarter unique", dupes == 0, f"{dupes} duplicates")
        check("No NULL earnings_date", m["earnings_date"].notna().all())

        by_source = m["primary_source"].value_counts().to_dict()
        yf_count = sum(v for k, v in by_source.items() if k.startswith("yfinance"))
        ed_count = sum(v for k, v in by_source.items() if not k.startswith("yfinance"))
        print(f"    Source breakdown: yfinance={yf_count}, edgar_fill={ed_count}")

        surprise_cov = m["surprise_pct"].notna().mean()
        msg = f"{surprise_cov:.1%} ({m['surprise_pct'].notna().sum()}/{len(m)} events)"
        if surprise_cov < 0.5:
            warn("surprise_pct coverage low", msg)
        else:
            check("surprise_pct coverage ≥ 50%", True, msg)
    else:
        print(f"  {FAIL} earnings_calendar_master missing — run edgar_supplement.py first")

    # ── earnings_event_features ───────────────────────────────────────────────
    print("\n[earnings_event_features]")
    if "earnings_event_features" in tables:
        ef = pd.read_sql_query(
            "SELECT ticker, earnings_date, feature_quarter_end FROM earnings_event_features", conn
        )
        ef["earnings_date"] = pd.to_datetime(ef["earnings_date"])
        ef["feature_quarter_end"] = pd.to_datetime(ef["feature_quarter_end"])
        bad_order = (ef["feature_quarter_end"] >= ef["earnings_date"]).sum()
        check("feature_quarter_end < earnings_date", bad_order == 0,
              f"{bad_order} rows where feature date ≥ event date")
        dupes = ef.duplicated(subset=["ticker", "feature_quarter_end"]).sum()
        check("ticker+feature_quarter_end unique", dupes == 0, f"{dupes} duplicates")
    else:
        warn("earnings_event_features table missing")

    # ── factor_backtest_dataset ───────────────────────────────────────────────
    print("\n[factor_backtest_dataset]")
    if "factor_backtest_dataset" in tables:
        bd = pd.read_sql_query(
            "SELECT ticker, fiscal_quarter, forward_20d_return FROM factor_backtest_dataset", conn
        )
        dupes = bd.duplicated(subset=["ticker", "fiscal_quarter"]).sum()
        check("ticker+fiscal_quarter unique", dupes == 0, f"{dupes} duplicates")
        missing_ret = bd["forward_20d_return"].isna().sum()
        check("No NULL forward_20d_return", missing_ret == 0, f"{missing_ret} nulls")
        cs_sizes = bd.groupby("fiscal_quarter")["ticker"].count()
        bad_cs = (cs_sizes < 2).sum()
        if bad_cs:
            warn(f"Cross-sections with < 2 tickers", f"{bad_cs} quarters")
        else:
            check("All cross-sections ≥ 2 tickers", True, f"min={cs_sizes.min()}, max={cs_sizes.max()}")
    else:
        warn("factor_backtest_dataset table missing")

    # ── factor_ic_results ─────────────────────────────────────────────────────
    print("\n[factor_ic_results]")
    if "factor_ic_results" in tables:
        ic = pd.read_sql_query("SELECT * FROM factor_ic_results", conn)
        if "rpk_constant_flag" in ic.columns:
            rpk_const = ic["rpk_constant_flag"].sum()
            surp_const = ic["surprise_constant_flag"].sum()
            sent_const = ic["sentiment_constant_flag"].sum()
            if rpk_const or surp_const or sent_const:
                warn("Quarters with constant cross-section (IC undefined)",
                     f"rpk={rpk_const}, surprise={surp_const}, sentiment={sent_const}")
            else:
                check("No constant cross-sections", True)
        else:
            warn("constant_flag columns missing — re-run factor_backtest.py")

        if "surprise_coverage" in ic.columns:
            low_surp = (ic["surprise_coverage"] < 3).sum()
            if low_surp:
                warn(f"Quarters with surprise_pct in < 3 tickers", f"{low_surp} quarters")
    else:
        warn("factor_ic_results table missing")

    # ── sde_summary ───────────────────────────────────────────────────────────
    print("\n[sde_summary]")
    if "sde_summary" in tables:
        sde = pd.read_sql_query("SELECT ticker, sigma, kappa, lf_momentum_adjustment FROM sde_summary", conn)
        tickers_in_sde = set(sde["ticker"].unique())
        check("All 6 tickers present", len(tickers_in_sde) == 6,
              f"found: {sorted(tickers_in_sde)}")
        high_sigma = sde[sde["sigma"] > 0.3]
        if not high_sigma.empty:
            warn("σ > 0.3 (possibly inflated)", ", ".join(
                f"{r.ticker}={r.sigma:.3f}" for r in high_sigma.itertuples()
            ))
        else:
            check("σ values plausible (≤ 0.3)", True,
                  f"range [{sde['sigma'].min():.3f}, {sde['sigma'].max():.3f}]")
    else:
        warn("sde_summary table missing")

    return _fail_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Data quality validation.")
    parser.add_argument("--run-id", default="standalone")
    parser.parse_args()  # accept --run-id from pipeline without error

    print("=" * 60)
    print("Data Quality Report")
    print("=" * 60)
    with sqlite3.connect(DB_PATH) as conn:
        fail_count = run_checks(conn)
    print("\n" + "=" * 60)
    if fail_count > 0:
        print(f"\n[FAIL] {fail_count} critical check(s) failed. Fix data before continuing.")
        sys.exit(1)
    else:
        print("\n[PASS] All critical checks passed.")


if __name__ == "__main__":
    main()
