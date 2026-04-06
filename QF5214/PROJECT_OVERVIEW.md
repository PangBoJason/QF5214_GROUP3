# QF5214 Project Overview

## Project summary

QF5214 is an aviation quant research pipeline built around a single SQLite database, `data/quant_flights.db`.

The project starts from raw BTS T-100 segment data, builds airline operating features, trains a load-factor prediction model, generates out-of-sample passenger and capacity forecasts, converts them into quarterly factors, and finally runs an airline equity backtest around earnings events.

This is a research and presentation project for a student group assignment, so the focus is on:
- an end-to-end workflow
- reproducible outputs
- clear reports and charts
- lightweight automation

## Pipeline stages

### 1. Data cleaning

`clean_data.py` reads the raw `.asc` files in `data/` and writes cleaned aviation segment data to:
- `t100_segment`

It also imports the earnings calendar from `data/day.csv` into:
- `earnings_day`

### 2. Feature engineering

`feature_engineering.py` reads `t100_segment` and creates:
- capacity-related features
- market-share features
- HHI concentration features
- lagged operating features
- encoded airport effects

The output table is:
- `engineered_features`

### 3. Model training

`train_model.py` trains an XGBoost regression model to predict load factor.

The saved model bundle is:
- `models/flight_revenue_bundle.joblib`

### 4. Prediction backtest

`backtest_model.py` runs the out-of-sample prediction layer for 2017–2025 and stores:
- predicted passengers
- predicted RPK
- actual RPK
- ASK

The main output table is:
- `sde_predictions`

Cache metadata is stored in:
- `sde_run_meta`

### 5. Quarterly factor construction

`factor_engineering.py` aggregates monthly prediction outputs into quarterly airline factors and writes:
- `quarterly_factors`

### 6. Fundamental backtest

`fundamental_backtest.py` combines:
- `quarterly_factors`
- `earnings_day`
- `equity_adj_close_daily`

It then runs a six-airline MWU-based event-driven backtest using a tradable post-formation return window.

## Main files

| File | Role |
|------|------|
| `main.py` | Runs the full pipeline |
| `clean_data.py` | Loads raw aviation data into SQLite |
| `feature_engineering.py` | Builds modeling features |
| `train_model.py` | Trains and saves the XGBoost model |
| `backtest_model.py` | Runs the prediction backtest |
| `factor_engineering.py` | Builds quarterly factors |
| `fundamental_backtest.py` | Runs the equity backtest |
| `plotting.py` | Generates figures |
| `agents/report_agent.py` | Writes a summary report for the current run |
| `agents/anomaly_agent.py` | Writes a current-run anomaly report |

## Core database tables

| Table | Meaning |
|------|---------|
| `t100_segment` | Cleaned BTS segment-level aviation data |
| `earnings_day` | Quarterly earnings dates |
| `engineered_features` | Model-ready feature table |
| `sde_predictions` | Prediction backtest output |
| `sde_run_meta` | Cache metadata for prediction runs |
| `quarterly_factors` | Airline-level quarterly factors |
| `equity_adj_close_daily` | Cached adjusted-close equity prices |

## Current checked outputs

Checked from `data/quant_flights.db`:

- `t100_segment`: 6,724,354 rows
- `engineered_features`: 6,723,959 rows
- `sde_predictions`: 2,656,668 rows
- `quarterly_factors`: 2,550 rows
- `earnings_day`: 216 rows
- `equity_adj_close_daily`: 16,107 rows

Latest checked metrics snapshot:

- SDE `WMAPE`: `11.34%`
- Portfolio quarters: `31`
- Gross cumulative return: `91.47%`
- Net cumulative return: `68.15%`
- Gross Sharpe: `1.287`
- Net Sharpe: `1.027`

## Generated outputs

After a successful run, the project writes:

- run report: `reports/run_YYYYMMDD_HHMMSS.md`
- latest metrics snapshot: `reports/_last_run_metrics.json`
- anomaly report: `logs/anomaly_report_YYYYMMDD_HHMMSS.md`
- figures: `reports/figures/*.png`
- logs: `logs/pipeline.log`

The main figures are:

- predicted vs actual passengers
- carrier WMAPE bar chart
- gross/net cumulative return curve
- factor IC bar chart
- factor correlation heatmap
- regime performance bar chart

## Practical notes

- This project is designed to be run from `main.py`.
- If the database already contains the required tables, later runs will mostly use cached outputs.
- The current agent layer is intentionally simple: one report agent and one anomaly agent.
- The project is suitable for coursework presentation and portfolio use, but it should not be presented as a production trading system.
