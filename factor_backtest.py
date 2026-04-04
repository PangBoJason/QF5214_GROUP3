from __future__ import annotations

import argparse
import sqlite3

import numpy as np
import pandas as pd

from project_config import DB_PATH, ensure_directories, log_pipeline_run


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    result = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?",
        (table_name,),
    ).fetchone()
    return result is not None


def load_table(conn: sqlite3.Connection, query: str) -> pd.DataFrame:
    frame = pd.read_sql_query(query, conn)
    for column in ["earnings_date", "feature_quarter_end", "date"]:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column])
    return frame


def next_trading_windows(prices: pd.DataFrame, event_date: pd.Timestamp, horizon: int) -> tuple[float | None, float | None]:
    window = prices[prices["date"] > event_date].sort_values("date")
    if len(window) <= horizon:
        return None, None
    return float(window.iloc[0]["adj_close"]), float(window.iloc[horizon]["adj_close"])


def zscore_cross_section(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.mean()) / std


def main() -> None:
    parser = argparse.ArgumentParser(description="Four-factor cross-sectional backtest.")
    parser.add_argument("--run-id", default="standalone")
    args = parser.parse_args()

    ensure_directories()
    with sqlite3.connect(DB_PATH) as conn:
        required_tables = ["earnings_event_features", "stock_prices", "sentiment_factors"]
        missing = [table for table in required_tables if not table_exists(conn, table)]
        if missing:
            raise RuntimeError(f"Missing required tables for backtest: {', '.join(missing)}")

        events = load_table(
            conn,
            """
            SELECT event_id, ticker, earnings_date, feature_quarter_end, rpk_yoy_growth, surprise_pct
            FROM earnings_event_features
            ORDER BY earnings_date, ticker
            """,
        )
        # sentiment_factors now contains VADER compound scores from SEC 8-K filings
        # Column is sentiment_compound [-1, 1]; fall back to sentiment_mean if present
        try:
            sf_cols = [r[1] for r in conn.execute("PRAGMA table_info(sentiment_factors)").fetchall()]
            if "sentiment_compound" in sf_cols:
                sentiment = load_table(conn, "SELECT event_id, sentiment_compound AS sentiment_mean FROM sentiment_factors")
            else:
                sentiment = load_table(conn, "SELECT event_id, sentiment_mean FROM sentiment_factors")
        except Exception:
            sentiment = pd.DataFrame(columns=["event_id", "sentiment_mean"])
        prices = load_table(
            conn,
            """
            SELECT ticker, date, adj_close
            FROM stock_prices
            ORDER BY ticker, date
            """,
        )

        dataset = events.merge(sentiment, on="event_id", how="left")
        # Keep sentiment_mean as NaN where missing — composite will average available factors only

        forward_returns = []
        for row in dataset.itertuples(index=False):
            ticker_prices = prices[prices["ticker"] == row.ticker]
            entry, exit_ = next_trading_windows(ticker_prices, row.earnings_date, horizon=20)
            forward_returns.append(np.nan if entry is None or exit_ is None else exit_ / entry - 1.0)
        dataset["forward_20d_return"] = forward_returns

        dataset = dataset.dropna(subset=["forward_20d_return"]).copy()
        # Group by fiscal quarter (from feature_quarter_end) so all 6 carriers align into the same cross-section
        dataset["fiscal_quarter"] = pd.to_datetime(dataset["feature_quarter_end"]).dt.to_period("Q").astype(str)
        dataset["rpk_growth_factor"] = dataset.groupby("fiscal_quarter")["rpk_yoy_growth"].transform(zscore_cross_section)
        dataset["earnings_surprise_factor"] = dataset.groupby("fiscal_quarter")["surprise_pct"].transform(zscore_cross_section)
        # Only z-score sentiment where it exists; leave NaN rows as NaN
        has_sentiment = dataset["sentiment_mean"].notna().any()
        if has_sentiment:
            dataset["sentiment_factor"] = dataset.groupby("fiscal_quarter")["sentiment_mean"].transform(zscore_cross_section)
        else:
            dataset["sentiment_factor"] = np.nan

        # ── Price momentum baseline ───────────────────────────────────────────
        # momentum_60d: cumulative return over the 60 trading days ending on the
        # last pre-event trading day (iloc[-1] relative to iloc[-61]).
        # Requires at least 61 pre-event trading days; otherwise NaN.
        mom_rows = []
        for row in dataset.itertuples(index=False):
            tp = prices[prices["ticker"] == row.ticker].sort_values("date")
            before = tp[tp["date"] < row.earnings_date]
            if len(before) < 61:
                mom_rows.append(np.nan)
                continue
            p_end   = float(before.iloc[-1]["adj_close"])
            p_start = float(before.iloc[-61]["adj_close"])
            mom_rows.append(p_end / p_start - 1.0)
        dataset["momentum_60d"] = mom_rows
        dataset["momentum_factor"] = dataset.groupby("fiscal_quarter")["momentum_60d"].transform(zscore_cross_section)

        # Composite = mean of all available (non-NaN) factors per row (skipna=True)
        factor_cols = ["rpk_growth_factor", "earnings_surprise_factor", "sentiment_factor", "momentum_factor"]
        dataset["composite_factor"] = dataset[factor_cols].mean(axis=1, skipna=True)

        ic_rows = []
        for period, event_frame in dataset.groupby("fiscal_quarter"):
            if len(event_frame) < 2:
                continue
            surprise_coverage  = int(event_frame["surprise_pct"].notna().sum())
            momentum_coverage  = int(event_frame["momentum_60d"].notna().sum())
            # Flag quarters where a factor is constant (IC undefined)
            rpk_constant       = event_frame["rpk_growth_factor"].nunique(dropna=True) <= 1
            surprise_constant  = event_frame["earnings_surprise_factor"].nunique(dropna=True) <= 1
            sentiment_constant = event_frame["sentiment_factor"].nunique(dropna=True) <= 1
            momentum_constant  = event_frame["momentum_factor"].nunique(dropna=True) <= 1
            ic_rows.append(
                {
                    "fiscal_quarter": period,
                    "rpk_ic": float(event_frame["rpk_growth_factor"].corr(event_frame["forward_20d_return"], method="spearman")),
                    "earnings_surprise_ic": float(event_frame["earnings_surprise_factor"].corr(event_frame["forward_20d_return"], method="spearman")),
                    "sentiment_ic": float(event_frame["sentiment_factor"].corr(event_frame["forward_20d_return"], method="spearman")),
                    "momentum_ic": float(event_frame["momentum_factor"].corr(event_frame["forward_20d_return"], method="spearman")),
                    "composite_ic": float(event_frame["composite_factor"].corr(event_frame["forward_20d_return"], method="spearman")),
                    "cross_section_size": int(len(event_frame)),
                    "surprise_coverage": surprise_coverage,
                    "momentum_coverage": momentum_coverage,
                    "rpk_constant_flag": int(rpk_constant),
                    "surprise_constant_flag": int(surprise_constant),
                    "sentiment_constant_flag": int(sentiment_constant),
                    "momentum_constant_flag": int(momentum_constant),
                }
            )

        ic_frame = pd.DataFrame(ic_rows)

        # Long-short portfolio: long top composite-ranked ticker, short bottom per fiscal quarter
        ls_rows = []
        for period, grp in dataset.groupby("fiscal_quarter"):
            grp = grp.dropna(subset=["composite_factor", "forward_20d_return"])
            if len(grp) < 2:
                continue
            ranked = grp.sort_values("composite_factor", ascending=False).reset_index(drop=True)
            ls_rows.append({
                "fiscal_quarter": period,
                "long_ticker":  ranked.iloc[0]["ticker"],
                "short_ticker": ranked.iloc[-1]["ticker"],
                "long_ret":  float(ranked.iloc[0]["forward_20d_return"]),
                "short_ret": float(ranked.iloc[-1]["forward_20d_return"]),
                "ls_return": float(ranked.iloc[0]["forward_20d_return"]) - float(ranked.iloc[-1]["forward_20d_return"]),
            })
        ls_frame = pd.DataFrame(ls_rows)
        sharpe = (
            float(ls_frame["ls_return"].mean() / (ls_frame["ls_return"].std(ddof=1) + 1e-8) * np.sqrt(4))
            if not ls_frame.empty else np.nan
        )

        # IC averages: each factor excludes quarters where it was constant (IC undefined)
        # composite_ic: average over all quarters where composite_ic is non-null
        def _valid_ic_mean(col: str, flag_col: str) -> float:
            if ic_frame.empty:
                return np.nan
            valid = ic_frame[ic_frame[flag_col] == 0][col].dropna()
            return float(valid.mean()) if not valid.empty else np.nan

        def _composite_ic_mean() -> float:
            if ic_frame.empty:
                return np.nan
            valid = ic_frame["composite_ic"].dropna()
            return float(valid.mean()) if not valid.empty else np.nan

        surprise_events_with_data = int(dataset["surprise_pct"].notna().sum())
        surprise_coverage_pct     = float(dataset["surprise_pct"].notna().mean() * 100)
        momentum_events_with_data = int(dataset["momentum_60d"].notna().sum())
        momentum_coverage_pct     = float(dataset["momentum_60d"].notna().mean() * 100)

        summary = pd.DataFrame(
            [
                {"metric": "avg_rpk_ic",                  "value": _valid_ic_mean("rpk_ic", "rpk_constant_flag")},
                {"metric": "avg_earnings_surprise_ic",     "value": _valid_ic_mean("earnings_surprise_ic", "surprise_constant_flag")},
                {"metric": "avg_sentiment_ic",             "value": _valid_ic_mean("sentiment_ic", "sentiment_constant_flag")},
                {"metric": "avg_momentum_ic",              "value": _valid_ic_mean("momentum_ic", "momentum_constant_flag")},
                {"metric": "avg_composite_ic",             "value": _composite_ic_mean()},
                {"metric": "ls_sharpe_annualized",         "value": sharpe},
                {"metric": "surprise_pct_coverage_events", "value": float(surprise_events_with_data)},
                {"metric": "surprise_pct_coverage_pct",    "value": surprise_coverage_pct},
                {"metric": "momentum_coverage_events",     "value": float(momentum_events_with_data)},
                {"metric": "momentum_coverage_pct",        "value": momentum_coverage_pct},
            ]
        )

        dataset.to_sql("factor_backtest_dataset", conn, if_exists="replace", index=False)
        ic_frame.to_sql("factor_ic_results", conn, if_exists="replace", index=False)
        summary.to_sql("factor_backtest_summary", conn, if_exists="replace", index=False)
        if not ls_frame.empty:
            ls_frame.to_sql("ls_portfolio_returns", conn, if_exists="replace", index=False)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_backtest_event ON factor_backtest_dataset(event_id)")

        sample_start = str(dataset["earnings_date"].min().date()) if not dataset.empty else ""
        sample_end   = str(dataset["earnings_date"].max().date()) if not dataset.empty else ""
        log_pipeline_run(conn, args.run_id, "factor_backtest", "success",
                         "factor_backtest_dataset", len(dataset), sample_start, sample_end)
        conn.commit()

    print(f"[done] factor dataset rows: {len(dataset):,}")
    print(f"[done] IC rows: {len(ic_frame):,}")
    print(f"[done] surprise_pct coverage: {surprise_events_with_data}/{len(dataset)} events ({surprise_coverage_pct:.1f}%)")
    print(f"[done] momentum coverage: {momentum_events_with_data}/{len(dataset)} events ({momentum_coverage_pct:.1f}%)")
    if not ls_frame.empty:
        print(f"[done] L/S Sharpe (annualized ×√4): {sharpe:.3f}")


if __name__ == "__main__":
    main()
