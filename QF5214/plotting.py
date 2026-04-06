"""
Plotting module — generates 6 standard research charts from pipeline outputs.

All figures are saved to reports/figures/ (configurable via config.yaml plots.dir).
Failures are non-fatal: each chart is wrapped in its own try/except.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

import matplotlib
matplotlib.use("Agg")  # headless backend — no display required
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger("qf5214.plotting")


def _out_path(figures_dir: str, name: str, fmt: str) -> str:
    os.makedirs(figures_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(figures_dir, f"{name}_{stamp}.{fmt}")


def _save(fig: plt.Figure, path: str, dpi: int) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot → %s", path)


# ── 1. SDE: predicted vs actual scatter ───────────────────────────────────────

def plot_sde_scatter(
    bt: Dict[str, Any],
    figures_dir: str,
    dpi: int = 140,
    fmt: str = "png",
) -> None:
    pred_df: Optional[pd.DataFrame] = bt.get("predictions_df")
    if pred_df is None or pred_df.empty:
        logger.warning("plot_sde_scatter: predictions_df missing, skipping")
        return

    actual = pred_df["passengers"].to_numpy(dtype=float)
    pred = pred_df["final_pred_pax"].to_numpy(dtype=float)

    # Sample for readability if very large
    max_pts = 20_000
    if len(actual) > max_pts:
        idx = np.random.default_rng(42).choice(len(actual), size=max_pts, replace=False)
        actual = actual[idx]
        pred = pred[idx]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(actual, pred, alpha=0.15, s=6, color="steelblue", rasterized=True)
    lim = max(actual.max(), pred.max()) * 1.05
    ax.plot([0, lim], [0, lim], "r--", linewidth=1.2, label="y = x")
    ax.set_xlabel("Actual Passengers")
    ax.set_ylabel("Predicted Passengers (SDE)")
    ax.set_title("SDE: Predicted vs Actual Passengers")
    ax.legend(fontsize=9)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    _save(fig, _out_path(figures_dir, "sde_pred_vs_actual_scatter", fmt), dpi)


# ── 2. Per-carrier WMAPE bar ───────────────────────────────────────────────────

def plot_carrier_wmape(
    bt: Dict[str, Any],
    figures_dir: str,
    dpi: int = 140,
    fmt: str = "png",
) -> None:
    records = bt.get("per_carrier_wmape")
    if not records:
        pred_df: Optional[pd.DataFrame] = bt.get("predictions_df")
        if pred_df is None or pred_df.empty:
            logger.warning("plot_carrier_wmape: per_carrier_wmape and predictions_df missing, skipping")
            return

        carrier_col = "carrier_iata" if "carrier_iata" in pred_df.columns else "carrier"
        records = []
        for carrier, g in pred_df.groupby(carrier_col, observed=True):
            err = np.abs(g["final_pred_pax"].to_numpy(dtype=float) - g["passengers"].to_numpy(dtype=float))
            denom = float(np.sum(np.abs(g["passengers"].to_numpy(dtype=float))))
            if denom > 0:
                records.append({"carrier": str(carrier), "wmape": float(np.sum(err) / denom)})

    if not records:
        logger.warning("plot_carrier_wmape: no per-carrier data, skipping")
        return

    df_w = pd.DataFrame(records)
    df_w["wmape"] = pd.to_numeric(df_w["wmape"], errors="coerce")
    df_w = df_w[np.isfinite(df_w["wmape"])].copy()
    if df_w.empty:
        logger.warning("plot_carrier_wmape: all wmape values are non-finite, skipping")
        return

    df_w = df_w.sort_values("wmape", ascending=False)
    colors = ["tomato" if w > 0.20 else "steelblue" for w in df_w["wmape"]]

    fig, ax = plt.subplots(figsize=(max(6, len(df_w) * 0.7), 5))
    bars = ax.bar(df_w["carrier"], df_w["wmape"], color=colors)
    ax.axhline(0.20, color="red", linestyle="--", linewidth=1.0, label="20% alert threshold")
    ax.set_xlabel("Carrier (IATA)")
    ax.set_ylabel("WMAPE")
    ax.set_title("Per-Carrier WMAPE (SDE Backtest)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))
    ax.legend(fontsize=9)
    for bar, val in zip(bars, df_w["wmape"]):
        if not np.isfinite(val):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{val:.1%}",
            ha="center", va="bottom", fontsize=8,
        )
    fig.tight_layout()
    _save(fig, _out_path(figures_dir, "carrier_wmape_bar", fmt), dpi)


# ── 3. Quarterly cumulative returns (gross vs net) ─────────────────────────────

def plot_cumulative_returns(
    fund_result: Dict[str, Any],
    figures_dir: str,
    dpi: int = 140,
    fmt: str = "png",
) -> None:
    detail: Optional[pd.DataFrame] = fund_result.get("quarterly_detail")
    if detail is None or detail.empty:
        logger.warning("plot_cumulative_returns: quarterly_detail missing, skipping")
        return

    detail = detail.copy()
    detail["period"] = detail["year"].astype(str) + "-Q" + detail["quarter"].astype(str)
    gross = (1 + detail["portfolio_return_gross"]).cumprod() - 1
    net = (1 + detail["portfolio_return_net"]).cumprod() - 1

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(detail["period"], gross * 100, label="Gross", color="steelblue", linewidth=1.6)
    ax.plot(detail["period"], net * 100, label="Net (after TC)", color="darkorange",
            linewidth=1.6, linestyle="--")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Cumulative Excess Return (%)")
    ax.set_title("MWU Portfolio: Cumulative Excess Return (Gross vs Net)")
    ax.legend(fontsize=9)
    # Thin out x-ticks for readability
    n = len(detail)
    step = max(1, n // 16)
    ax.set_xticks(range(0, n, step))
    ax.set_xticklabels(detail["period"].iloc[::step], rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    _save(fig, _out_path(figures_dir, "quarterly_cumulative_return_gross_net", fmt), dpi)


# ── 4. Factor IC bar ───────────────────────────────────────────────────────────

def plot_factor_ic(
    fund_result: Dict[str, Any],
    figures_dir: str,
    dpi: int = 140,
    fmt: str = "png",
) -> None:
    ic_by_factor: Optional[Dict] = fund_result.get("ic_by_factor")
    ic_pvalues: Optional[Dict] = fund_result.get("ic_pvalues", {})
    if not ic_by_factor:
        logger.warning("plot_factor_ic: ic_by_factor missing, skipping")
        return

    factors = list(ic_by_factor.keys())
    ics = [ic_by_factor[f] for f in factors]
    pvals = [ic_pvalues.get(f, float("nan")) for f in factors]
    colors = ["steelblue" if v >= 0 else "tomato" for v in ics]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(factors, ics, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Factor")
    ax.set_ylabel("Mean Spearman IC")
    ax.set_title("Factor IC vs Tradable Event Return (6-Airline Cross-Section)")
    for bar, ic_v, pv in zip(bars, ics, pvals):
        label = f"{ic_v:.3f}\np={pv:.2f}" if np.isfinite(pv) else f"{ic_v:.3f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            ic_v + (0.002 if ic_v >= 0 else -0.008),
            label,
            ha="center", va="bottom" if ic_v >= 0 else "top", fontsize=8,
        )
    fig.tight_layout()
    _save(fig, _out_path(figures_dir, "factor_ic_bar", fmt), dpi)


# ── 5. Factor correlation heatmap ─────────────────────────────────────────────

def plot_factor_corr(
    fund_result: Dict[str, Any],
    figures_dir: str,
    dpi: int = 140,
    fmt: str = "png",
) -> None:
    factor_corr: Optional[pd.DataFrame] = fund_result.get("factor_corr")
    if factor_corr is None or factor_corr.empty:
        logger.warning("plot_factor_corr: factor_corr missing, skipping")
        return

    corr = factor_corr.to_numpy(dtype=float)
    labels = [c.replace("factor_", "") for c in factor_corr.columns]
    n = len(labels)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdYlGn")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=9,
                    color="black" if abs(corr[i, j]) < 0.7 else "white")
    ax.set_title("Factor Spearman Correlation Matrix")
    fig.tight_layout()
    _save(fig, _out_path(figures_dir, "factor_corr_heatmap", fmt), dpi)


# ── 6. Regime performance bar ─────────────────────────────────────────────────

def plot_regime_performance(
    fund_result: Dict[str, Any],
    figures_dir: str,
    dpi: int = 140,
    fmt: str = "png",
) -> None:
    regime_stats: Optional[Dict] = fund_result.get("regime_stats")
    if not regime_stats:
        logger.warning("plot_regime_performance: regime_stats missing, skipping")
        return

    regimes = list(regime_stats.keys())
    sharpes = [regime_stats[r].get("sharpe", float("nan")) for r in regimes]
    cum_rets = [regime_stats[r].get("cum_ret", float("nan")) * 100 for r in regimes]

    x = np.arange(len(regimes))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax2 = ax1.twinx()

    bars1 = ax1.bar(x - width / 2, sharpes, width, label="Ann. Sharpe", color="steelblue")
    bars2 = ax2.bar(x + width / 2, cum_rets, width, label="Cum. Return (%)", color="darkorange", alpha=0.8)

    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_xlabel("Regime")
    ax1.set_ylabel("Annualized Sharpe", color="steelblue")
    ax2.set_ylabel("Cumulative Excess Return (%)", color="darkorange")
    ax1.set_xticks(x)
    ax1.set_xticklabels([r.replace("_", " ").title() for r in regimes])
    ax1.set_title("Regime Performance: Sharpe & Cumulative Return")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper right")

    fig.tight_layout()
    _save(fig, _out_path(figures_dir, "regime_performance_bar", fmt), dpi)


# ── Main entry point ──────────────────────────────────────────────────────────

def run_all_plots(
    bt: Optional[Dict[str, Any]],
    fund_result: Optional[Dict[str, Any]],
    cfg: Dict[str, Any],
) -> None:
    plot_cfg = cfg.get("plots", {})
    if not plot_cfg.get("enabled", True):
        logger.info("Plotting disabled (plots.enabled=false in config.yaml)")
        return

    figures_dir: str = plot_cfg.get("dir", "reports/figures")
    dpi: int = int(plot_cfg.get("dpi", 140))
    fmt: str = str(plot_cfg.get("format", "png"))

    logger.info("Generating research plots → %s", figures_dir)

    if bt:
        for fn in (plot_sde_scatter, plot_carrier_wmape):
            try:
                fn(bt, figures_dir, dpi, fmt)
            except Exception as exc:
                logger.warning("%s failed: %s", fn.__name__, exc, exc_info=True)

    if fund_result and fund_result.get("n_quarters", 0) > 0:
        for fn in (
            plot_cumulative_returns,
            plot_factor_ic,
            plot_factor_corr,
            plot_regime_performance,
        ):
            try:
                fn(fund_result, figures_dir, dpi, fmt)
            except Exception as exc:
                logger.warning("%s failed: %s", fn.__name__, exc, exc_info=True)
