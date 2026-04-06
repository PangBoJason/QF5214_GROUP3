# QF5214 Aviation Quant Pipeline

## What this project does

QF5214 is an end-to-end aviation quantitative research pipeline built for a student group project.

It takes U.S. BTS T-100 segment data, engineers airline operating features, trains an XGBoost model to predict load factor, converts the prediction layer into quarterly airline factors, and runs an event-driven equity backtest around earnings releases for six listed U.S. airlines.

The pipeline generates:
- a structured Markdown run report
- an anomaly detection report
- six research figures

## Pipeline flow

| Step | Module | Output table |
|------|--------|-------------|
| 1 | `clean_data.py` | `t100_segment`, `earnings_day` |
| 2 | `feature_engineering.py` | `engineered_features` |
| 3 | `train_model.py` | `models/flight_revenue_bundle.joblib` |
| 4 | `backtest_model.py` | `sde_predictions`, `sde_run_meta` |
| 5 | `factor_engineering.py` | `quarterly_factors` |
| 6 | `fundamental_backtest.py` | equity backtest results |
| 7 | `plotting.py` | `reports/figures/*.png` |

## Key design decisions

- **Single SQLite bus** — all stages communicate through `data/quant_flights.db`
- **OOS sigma** — load-factor uncertainty is estimated via 3-fold walk-forward CV, not in-sample residuals
- **No look-ahead bias** — portfolio forms at quarter-end + 5 business days; P&L uses forward return from that date
- **Cross-sectional factor standardization** — z-score + winsorize per quarter before backtesting
- **Transaction costs** — 50 bps one-way slippage modeled via quarterly turnover
- **AI agents** — optional GPT-powered anomaly and report agents (non-blocking; pipeline completes without them)

## How to run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full pipeline:

```bash
python main.py
```

Force-rebuild specific stages:

```bash
# Rebuild everything from scratch (keeps raw data)
python main.py --force-features --force-retrain --force-sde --force-factors

# Retrain model + downstream
python main.py --force-retrain --force-sde --force-factors

# Re-run SDE backtest + factors only
python main.py --force-sde --force-factors

# GPT-5 hyperparameter tuning before training
python main.py --tune-hyperparams --force-retrain --force-sde --force-factors
```

If you need to rebuild the database from raw `.asc` files first:

```bash
python clean_data.py
python main.py
```

## Configuration

All parameters are in `config.yaml`. Key sections:

```yaml
training:
  xgboost:
    n_estimators: 400
    max_depth: 8

backtest:
  n_mc_paths: 1000

fundamental:
  transaction_cost_bps: 50.0

plots:
  enabled: true
  dir: reports/figures
  dpi: 140
```

## Output files

After a successful run:

| Path | Content |
|------|---------|
| `reports/run_YYYYMMDD_HHMMSS.md` | Full pipeline run report |
| `reports/_last_run_metrics.json` | Latest metrics snapshot for delta comparison |
| `logs/anomaly_report_YYYYMMDD_HHMMSS.md` | Anomaly detection report |
| `logs/pipeline.log` | Structured pipeline log |
| `reports/figures/*.png` | Six research figures |

The six figures are:
- `sde_pred_vs_actual_scatter` — predicted vs actual passengers
- `carrier_wmape_bar` — per-carrier WMAPE with 20% alert line
- `quarterly_cumulative_return_gross_net` — gross and net cumulative return curves
- `factor_ic_bar` — mean Spearman IC per factor with p-values
- `factor_corr_heatmap` — factor cross-correlation heatmap
- `regime_performance_bar` — Sharpe and cumulative return by market regime

## Latest run metrics

| Metric | Value |
|--------|-------|
| SDE WMAPE | 11.34% |
| Portfolio quarters | 31 |
| Gross cumulative return | 91.47% |
| Net cumulative return | 68.15% |
| Gross annualized Sharpe | 1.29 |
| Net annualized Sharpe | 1.03 |

## Notes

- This is a research project, not a production trading system.
- Equity price data is fetched via `yfinance` on first run and cached in the database.
- The AI agent layer is optional and non-blocking — the pipeline runs normally without an OpenAI API key.
- `data/`, `models/`, `logs/`, and `reports/` are excluded from version control.
