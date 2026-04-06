# QF5214 Aviation Quant Pipeline

## What this project does

This project is a small end-to-end aviation quant research pipeline built for a student group project.

It takes U.S. BTS T-100 segment data, engineers airline operating features, trains an XGBoost model to predict load factor, converts the prediction layer into quarterly airline factors, and then runs an event-driven equity backtest around earnings releases for six listed U.S. airlines.

The pipeline also generates:
- a Markdown run report
- an anomaly report
- six research plots

## Pipeline flow

1. `clean_data.py`
   Loads raw `.asc` aviation data into SQLite and imports earnings dates from `data/day.csv`.
2. `feature_engineering.py`
   Builds model-ready features such as market share, HHI, lag features, and encoded airport information.
3. `train_model.py`
   Trains the load-factor model and saves `models/flight_revenue_bundle.joblib`.
4. `backtest_model.py`
   Runs the out-of-sample SDE-style prediction layer and stores results in `sde_predictions`.
5. `factor_engineering.py`
   Aggregates monthly prediction outputs into quarterly factors.
6. `fundamental_backtest.py`
   Runs the MWU-based airline equity backtest using quarterly factors, earnings dates, and cached equity prices.
7. `plotting.py`
   Writes six PNG figures into `reports/figures/`.

## Main files

| File | Purpose |
|------|---------|
| `main.py` | Orchestrates the full pipeline |
| `clean_data.py` | Builds `t100_segment` and `earnings_day` |
| `feature_engineering.py` | Builds `engineered_features` |
| `train_model.py` | Trains the XGBoost model |
| `backtest_model.py` | Runs prediction backtest and stores `sde_predictions` |
| `factor_engineering.py` | Builds `quarterly_factors` |
| `fundamental_backtest.py` | Runs the airline stock backtest |
| `plotting.py` | Generates figures |
| `agents/report_agent.py` | Summarizes the current run |
| `agents/anomaly_agent.py` | Checks current-run anomalies |

## Data and outputs

Main SQLite database:
- `data/quant_flights.db`

Core tables:
- `t100_segment`
- `earnings_day`
- `engineered_features`
- `sde_predictions`
- `sde_run_meta`
- `quarterly_factors`
- `equity_adj_close_daily`

Output folders:
- `reports/` for run reports and latest metrics snapshot
- `reports/figures/` for PNG charts
- `logs/` for anomaly reports and pipeline logs
- `models/` for the trained model bundle

## How to run

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

If you want agent-generated reports, set these in `.env`:

```env
OPENAI_API_KEY=your_key
OPENAI_BASE_URL=https://api.chatanywhere.tech/v1
```

Run the full pipeline:

```bash
python main.py
```

Useful variants:

```bash
python clean_data.py
python main.py --force-features --force-retrain --force-sde --force-factors
python main.py --force-retrain --force-sde --force-factors
python main.py --force-sde --force-factors
```

## What the project outputs

After a successful run, you should see:

1. Updated tables inside `data/quant_flights.db`
2. A run report in `reports/run_YYYYMMDD_HHMMSS.md`
3. A metrics snapshot in `reports/_last_run_metrics.json`
4. An anomaly report in `logs/anomaly_report_YYYYMMDD_HHMMSS.md`
5. Six figures in `reports/figures/`

The figures are:
- `sde_pred_vs_actual_scatter_*.png`
- `carrier_wmape_bar_*.png`
- `quarterly_cumulative_return_gross_net_*.png`
- `factor_ic_bar_*.png`
- `factor_corr_heatmap_*.png`
- `regime_performance_bar_*.png`

## Current checked project state

Latest checked database:
- `data/quant_flights.db`

Checked table sizes:
- `t100_segment`: 6,724,354
- `engineered_features`: 6,723,959
- `sde_predictions`: 2,656,668
- `quarterly_factors`: 2,550
- `earnings_day`: 216
- `equity_adj_close_daily`: 16,107

Latest checked run snapshot:
- SDE `WMAPE`: about `11.34%`
- Portfolio quarters: `31`
- Gross cumulative return: about `91.47%`
- Net cumulative return: about `68.15%`

## Notes

- This is a research project, not a production trading system.
- Equity data is expected to come from the DB cache when available.
- The current agent setup is intentionally lightweight: report summary plus anomaly checks only.
