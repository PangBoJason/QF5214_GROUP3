from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from project_config import CHARTS_DIR, DB_PATH, REPORTS_DIR, TARGET_TICKERS, ensure_directories


sns.set_theme(style="whitegrid", palette="deep")


def _compute_composite_rank(df: pd.DataFrame) -> pd.Series:
    """Composite overall rank = rpk_yoy_rank + load_factor_rank + sim_mean_rank.

    All three sub-ranks are descending (higher value = rank 1).
    Ties use min method. Returns integer rank Series (1 = best).
    Matches the ranking logic in agent.py build_fact_summary().
    """
    def rank_desc(col: str) -> pd.Series:
        return df[col].rank(ascending=False, method="min").astype(int)

    score = rank_desc("rpk_yoy_growth") + rank_desc("load_factor") + rank_desc("mean_rpk_100m")
    return score.rank(method="min", ascending=True).astype(int)


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    result = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?",
        (table_name,),
    ).fetchone()
    return result is not None


def save_figure(fig: plt.Figure, name: str) -> Path:
    path = CHARTS_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_rpk_and_load_factor(conn: sqlite3.Connection, ticker: str) -> Path:
    frame = pd.read_sql_query(
        """
        SELECT quarter_end, rpk_100m, load_factor
        FROM flights_quarterly_features
        WHERE ticker = ?
        ORDER BY quarter_end
        """,
        conn,
        params=[ticker],
    )
    frame["quarter_end"] = pd.to_datetime(frame["quarter_end"])
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    ax1.plot(frame["quarter_end"], frame["rpk_100m"], marker="o", label="Quarterly RPK (100m pkm)")
    ax2.plot(frame["quarter_end"], frame["load_factor"], color="tab:red", marker="s", label="Load factor")
    ax1.set_title(f"{ticker} Quarterly RPK and Load Factor")
    ax1.set_ylabel("RPK (100m passenger-km)")
    ax2.set_ylabel("Load factor")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    return save_figure(fig, f"{ticker.lower()}_rpk_load_factor.png")


def plot_sentiment(conn: sqlite3.Connection, ticker: str) -> Path:
    frame = pd.read_sql_query(
        """
        SELECT earnings_date, sentiment_compound
        FROM sentiment_factors
        WHERE ticker = ?
        ORDER BY earnings_date
        """,
        conn,
        params=[ticker],
    )
    frame["earnings_date"] = pd.to_datetime(frame["earnings_date"])
    frame = frame.dropna(subset=["sentiment_compound"])
    fig, ax = plt.subplots(figsize=(10, 4))
    if frame.empty:
        ax.text(0.5, 0.5, "No sentiment data available", ha="center", va="center", transform=ax.transAxes)
    else:
        colors = ["tab:green" if v >= 0 else "tab:red" for v in frame["sentiment_compound"]]
        ax.bar(frame["earnings_date"], frame["sentiment_compound"], color=colors, alpha=0.75)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(f"{ticker} Pre-Earnings 8-K VADER Sentiment")
    ax.set_ylabel("Sentiment compound score")
    return save_figure(fig, f"{ticker.lower()}_sentiment.png")


def plot_simulation_fan(conn: sqlite3.Connection, ticker: str) -> Path:
    frame = pd.read_sql_query(
        """
        SELECT quarter_label, mean_rpk_100m, p10_rpk_100m, p90_rpk_100m
        FROM sde_summary
        WHERE ticker = ?
        ORDER BY step
        """,
        conn,
        params=[ticker],
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(frame["quarter_label"], frame["mean_rpk_100m"], color="tab:blue", label="Mean")
    ax.fill_between(frame["quarter_label"], frame["p10_rpk_100m"], frame["p90_rpk_100m"], alpha=0.2, color="tab:blue")
    ax.set_title(f"{ticker} SDE RPK Simulation Fan")
    ax.set_ylabel("RPK (100m passenger-km)")
    ax.tick_params(axis="x", rotation=30)
    return save_figure(fig, f"{ticker.lower()}_sde_fan.png")


def plot_factor_ic(conn: sqlite3.Connection) -> Path:
    frame = pd.read_sql_query(
        """
        SELECT fiscal_quarter, rpk_ic, earnings_surprise_ic, sentiment_ic,
               momentum_ic, composite_ic
        FROM factor_ic_results
        ORDER BY fiscal_quarter
        """,
        conn,
    )
    ic_cols = [c for c in ["rpk_ic", "earnings_surprise_ic", "sentiment_ic", "momentum_ic", "composite_ic"]
               if c in frame.columns]
    melted = frame.melt(id_vars=["fiscal_quarter"], value_vars=ic_cols, var_name="factor", value_name="ic")
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.lineplot(data=melted, x="fiscal_quarter", y="ic", hue="factor", marker="o", ax=ax)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Factor Information Coefficient Over Time (6-carrier cross-section)")
    ax.tick_params(axis="x", rotation=30)
    return save_figure(fig, "factor_ic_timeseries.png")


def plot_carrier_scorecard(conn: sqlite3.Connection) -> Path:
    """Carrier scorecard: RPK YoY, load factor, 2025Q4 sim mean+P90.

    Sorted by RPK YoY descending so the best-performing carrier is at top.
    """
    # Latest quarter operating metrics
    qf = pd.read_sql_query(
        """
        SELECT ticker, rpk_yoy_growth, load_factor
        FROM flights_quarterly_features
        WHERE (ticker, quarter_end) IN (
            SELECT ticker, MAX(quarter_end) FROM flights_quarterly_features GROUP BY ticker
        )
        """, conn,
    )
    # 2025Q4 simulation (step=4)
    sde = pd.read_sql_query(
        "SELECT ticker, mean_rpk_100m, p90_rpk_100m FROM sde_summary WHERE step = 4", conn,
    )
    merged = qf.merge(sde, on="ticker", how="left")
    # Sort by composite rank (same 3-dimension formula as agent.py relative_rank)
    merged["overall_rank"] = _compute_composite_rank(merged)
    merged = merged.sort_values("overall_rank", ascending=True).reset_index(drop=True)
    tickers = merged["ticker"].tolist()
    n = len(tickers)

    fig, axes = plt.subplots(1, 3, figsize=(15, max(4, n * 0.7 + 1)))
    fig.suptitle("Carrier Scorecard — Latest Quarter vs 2025Q4 Simulation", fontsize=13)

    # Panel 1: RPK YoY growth
    colors1 = ["tab:green" if v >= 0 else "tab:red" for v in merged["rpk_yoy_growth"]]
    axes[0].barh(tickers, merged["rpk_yoy_growth"] * 100, color=colors1, alpha=0.8)
    axes[0].axvline(0, color="black", linewidth=0.8, linestyle="--")
    axes[0].set_title("RPK YoY Growth (%)")
    axes[0].set_xlabel("%")
    axes[0].invert_yaxis()

    # Panel 2: Load factor
    axes[1].barh(tickers, merged["load_factor"] * 100, color="tab:blue", alpha=0.8)
    axes[1].set_title("Load Factor (%)")
    axes[1].set_xlabel("%")
    axes[1].invert_yaxis()

    # Panel 3: 2025Q4 mean RPK + P90 overlay
    y_pos = np.arange(n)
    axes[2].barh(y_pos, merged["mean_rpk_100m"], color="tab:blue", alpha=0.6, label="Mean RPK")
    axes[2].barh(y_pos, merged["p90_rpk_100m"] - merged["mean_rpk_100m"],
                 left=merged["mean_rpk_100m"], color="tab:orange", alpha=0.4, label="P90 upside")
    axes[2].set_yticks(y_pos)
    axes[2].set_yticklabels(tickers)
    axes[2].set_title("2025Q4 Sim RPK (100m pkm)")
    axes[2].set_xlabel("RPK 100m")
    axes[2].invert_yaxis()
    axes[2].legend(fontsize=8)

    return save_figure(fig, "carrier_scorecard.png")


def generate_summary_table(conn: sqlite3.Connection) -> tuple[Path, Path]:
    """Generate a summary table with key metrics per carrier.

    Sorted by RPK YoY descending. Outputs CSV and Markdown.
    """
    qf = pd.read_sql_query(
        """
        SELECT ticker, rpk_yoy_growth, rpk_qoq_growth, load_factor
        FROM flights_quarterly_features
        WHERE (ticker, quarter_end) IN (
            SELECT ticker, MAX(quarter_end) FROM flights_quarterly_features GROUP BY ticker
        )
        """, conn,
    )
    sde = pd.read_sql_query(
        "SELECT ticker, mean_rpk_100m, p90_rpk_100m FROM sde_summary WHERE step = 4", conn,
    )
    try:
        bs = pd.read_sql_query("SELECT metric, value FROM factor_backtest_summary", conn)
        avg_composite_ic = bs.loc[bs["metric"] == "avg_composite_ic", "value"].values
        avg_composite_ic = float(avg_composite_ic[0]) if len(avg_composite_ic) else None
    except Exception:
        avg_composite_ic = None

    merged = qf.merge(sde, on="ticker", how="left")
    # Composite rank: same 3-dimension formula as agent.py relative_rank
    merged["overall_rank"] = _compute_composite_rank(merged)
    merged = merged.sort_values("overall_rank", ascending=True).reset_index(drop=True)
    merged["avg_composite_ic"] = avg_composite_ic  # same for all rows (portfolio-level)

    # Format for display
    display = pd.DataFrame({
        "ticker":           merged["ticker"],
        "latest_rpk_yoy_%": (merged["rpk_yoy_growth"] * 100).round(1),
        "latest_lf_%":      (merged["load_factor"] * 100).round(1),
        "latest_qoq_%":     (merged["rpk_qoq_growth"] * 100).round(1),
        "sim_2025q4_mean":  merged["mean_rpk_100m"].round(1),
        "sim_2025q4_p90":   merged["p90_rpk_100m"].round(1),
        "avg_composite_ic": merged["avg_composite_ic"].round(3) if avg_composite_ic is not None else None,
        "overall_rank":     merged["overall_rank"],
    })

    csv_path = REPORTS_DIR / "summary_table.csv"
    md_path  = REPORTS_DIR / "summary_table.md"
    display.to_csv(csv_path, index=False)
    md_path.write_text(display.to_markdown(index=False), encoding="utf-8")
    return csv_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate project charts.")
    parser.add_argument("--ticker", choices=TARGET_TICKERS, default=None)
    parser.add_argument("--run-id", default="standalone")
    args = parser.parse_args()

    ensure_directories()
    outputs: list[Path] = []
    with sqlite3.connect(DB_PATH) as conn:
        tickers = [args.ticker] if args.ticker else TARGET_TICKERS
        for ticker in tickers:
            outputs.append(plot_rpk_and_load_factor(conn, ticker))
            if table_exists(conn, "sde_summary") and not pd.read_sql_query(
                "SELECT 1 FROM sde_summary WHERE ticker = ? LIMIT 1", conn, params=[ticker]
            ).empty:
                outputs.append(plot_simulation_fan(conn, ticker))
        if table_exists(conn, "factor_ic_results") and not pd.read_sql_query(
            "SELECT 1 FROM factor_ic_results LIMIT 1", conn
        ).empty:
            outputs.append(plot_factor_ic(conn))
        # Scorecard and summary table (full universe only)
        if not args.ticker and table_exists(conn, "sde_summary"):
            outputs.append(plot_carrier_scorecard(conn))
            csv_path, md_path = generate_summary_table(conn)
            outputs.extend([csv_path, md_path])

    print("[done] charts generated:")
    for path in outputs:
        print(f" - {path}")


if __name__ == "__main__":
    main()
