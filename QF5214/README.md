# QF5214 Aviation Quant Pipeline

> **QF5214** is a postgraduate quantitative finance course. This repository contains the Group 3 course project.

## Overview

This project is an end-to-end aviation quantitative research pipeline. Starting from raw U.S. BTS T-100 segment data, it engineers airline operating features, trains an XGBoost load-factor regression model, generates out-of-sample passenger forecasts, constructs quarterly airline equity factors, and backtests an event-driven long-short strategy around earnings releases for six listed U.S. carriers.

On each full run the pipeline also produces:
- a structured Markdown run report with delta comparison to the previous run
- an anomaly detection report flagging data quality issues
- six research figures covering model accuracy, factor quality, and portfolio performance

## Pipeline stages

| Step | Module | Primary output |
|------|--------|---------------|
| 1 | `clean_data.py` | `t100_segment`, `earnings_day` |
| 2 | `feature_engineering.py` | `engineered_features` |
| 3 | `train_model.py` | `models/flight_revenue_bundle.joblib` |
| 4 | `backtest_model.py` | `sde_predictions`, `sde_run_meta` |
| 5 | `factor_engineering.py` | `quarterly_factors` |
| 6 | `fundamental_backtest.py` | Equity backtest results |
| 7 | `plotting.py` | `reports/figures/*.png` |

All stages communicate through a single SQLite database at `data/quant_flights.db`. Completed stages are cached and skipped on subsequent runs unless a force flag is passed.

## Design decisions

- **Single SQLite bus** — one database file serves as the shared data layer between all pipeline stages, keeping the setup dependency-free and fully reproducible.
- **OOS sigma estimation** — load-factor uncertainty is estimated via 3-fold walk-forward cross-validation rather than in-sample residuals, avoiding downward-biased noise estimates.
- **No look-ahead bias** — the portfolio forms at quarter-end + 5 business days; realised P&L is computed from that formation date forward, not from the earnings event window.
- **Cross-sectional factor standardisation** — each factor is winsorised and z-scored within each quarter before entering the backtest.
- **Transaction costs** — 50 bps one-way slippage is modelled via quarterly portfolio turnover.
- **AI agents** — optional GPT-5 powered agents generate narrative anomaly and run reports. The core pipeline completes successfully without them.

## How to run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full pipeline:

```bash
python main.py
```

Force-rebuild specific stages as needed:

```bash
# Rebuild everything from scratch (raw data files must already be in data/)
python main.py --force-features --force-retrain --force-sde --force-factors

# Retrain model and all downstream stages
python main.py --force-retrain --force-sde --force-factors

# Re-run SDE backtest and factor construction only
python main.py --force-sde --force-factors

# Run GPT-5 hyperparameter tuning before training
python main.py --tune-hyperparams --force-retrain --force-sde --force-factors
```

To rebuild the database from raw `.asc` files:

```bash
python clean_data.py
python main.py
```

## Configuration

All tunable parameters live in `config.yaml`. The most commonly adjusted sections are:

```yaml
training:
  xgboost:
    n_estimators: 400
    max_depth: 8
    learning_rate: 0.05

backtest:
  n_mc_paths: 1000

fundamental:
  transaction_cost_bps: 50.0
  mwu_eta: 2.0

plots:
  enabled: true
  dir: reports/figures
  dpi: 140
```

## Output files

| Path | Content |
|------|---------|
| `reports/run_YYYYMMDD_HHMMSS.md` | Full pipeline run report with delta vs previous run |
| `reports/_last_run_metrics.json` | Metrics snapshot used for next-run comparison |
| `logs/anomaly_report_YYYYMMDD_HHMMSS.md` | Per-carrier and data-volume anomaly report |
| `logs/pipeline.log` | Structured timestamped pipeline log |
| `reports/figures/*.png` | Six research figures (see below) |

The six figures generated on each run:

| Figure | Description |
|--------|-------------|
| `sde_pred_vs_actual_scatter` | Predicted vs actual passenger volume with y = x reference line |
| `carrier_wmape_bar` | Per-carrier WMAPE bar chart with 20% alert threshold |
| `quarterly_cumulative_return_gross_net` | Gross and net cumulative excess return curves |
| `factor_ic_bar` | Mean Spearman IC per factor with p-values |
| `factor_corr_heatmap` | Factor cross-correlation heatmap |
| `regime_performance_bar` | Sharpe ratio and cumulative return by market regime |

## Latest run metrics

| Metric | Value |
|--------|-------|
| SDE WMAPE | 11.34% |
| Portfolio quarters | 31 |
| Gross cumulative return | 91.47% |
| Net cumulative return | 68.15% |
| Gross annualised Sharpe | 1.29 |
| Net annualised Sharpe | 1.03 |

## AI agents (optional)

The report and anomaly agents require an OpenAI API key. To enable them, create a `.env` file in the `QF5214/` directory:

```
OPENAI_API_KEY=your_key_here
```

The pipeline runs normally without this file — agent steps are logged as warnings and skipped gracefully.

## Notes

- This is a course research project and should not be treated as a production trading system.
- Equity price data is downloaded from `yfinance` on first run and cached in the database for subsequent runs.
- `data/`, `models/`, `logs/`, and `reports/` are excluded from version control.
