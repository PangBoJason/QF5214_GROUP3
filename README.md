# QF5214 Group 3 — Airline Earnings Quant Pipeline

A quantitative research pipeline for U.S. airline carriers, combining BTS flight operations data, EDGAR filings, and equity prices to produce factor backtests, Monte Carlo RPK simulations, and structured investment reports.

## Universe

**AAL · DAL · UAL · LUV · JBLU · ALK**

## Pipeline Overview

```
main.py
│
├── Phase 1b  get_stock_data.py       Stock prices + earnings dates (yfinance)
├── Phase 1c  edgar_supplement.py     EDGAR EPS + 8-K VADER sentiment scores
├── Phase 2   feature_engineering.py  Monthly / quarterly RPK + load-factor features
├── Phase 3   sde_simulation.py       Log-OU Monte Carlo (1 000 paths, 4-quarter horizon)
├── Phase 4   factor_backtest.py      Four-factor IC / L-S Sharpe backtest
├── Phase 4b  data_quality.py         Data quality gate (hard stop on FAIL)
├── Phase 5   charts.py               PNG charts + carrier scorecard + summary table
└──           agent.py                Structured-summary LLM report (GPT-4o)
```

## Four Factors

| Factor | Source | Coverage |
|---|---|---|
| RPK YoY growth | BTS T-100 segment data | 100% |
| EPS surprise | yfinance earnings history | ~47% |
| 8-K tone (VADER) | EDGAR full-text search | ~100% |
| Price momentum 60d | yfinance daily prices | 100% |

## Latest Results (2024Q4)

| Ticker | RPK YoY % | Load Factor % | Sim 2025Q4 Mean | Overall Rank |
|:---|---:|---:|---:|---:|
| AAL | +2.4 | 85.4 | 440.8 | 1 |
| DAL | +3.5 | 84.2 | 412.4 | 2 |
| UAL | +4.7 | 83.8 | 352.3 | 3 |
| ALK | +1.4 | 84.6 | 151.1 | 4 |
| LUV | −3.3 | 79.1 | 437.9 | 5 |
| JBLU | −3.7 | 83.9 | 121.6 | 6 |

Overall rank = composite of RPK YoY rank + load factor rank + simulation mean rank (1 = best).

## Project Structure

```
flight_project/
├── main.py                     Pipeline entry point
├── project_config.py           Paths, tickers, shared utilities
├── data_pipeline.py            BTS raw data ingestion
├── download_bts.py             BTS T-100 file downloader
├── get_stock_data.py           yfinance prices + earnings calendar
├── edgar_supplement.py         EDGAR EPS + 8-K sentiment (master calendar)
├── feature_engineering.py      Monthly / quarterly feature tables
├── sde_simulation.py           Ornstein-Uhlenbeck Monte Carlo
├── factor_backtest.py          IC / L-S Sharpe four-factor backtest
├── load_factor_model.py        Ridge regression load-factor model
├── data_quality.py             Automated data quality checks
├── charts.py                   Matplotlib / seaborn visualisations
├── agent.py                    LLM report agent (OpenAI-compatible)
├── models/                     Serialised model artefacts
├── reports/
│   ├── charts/                 Generated PNG charts
│   ├── summary_table.md        Carrier summary table
│   └── market_analysis_report.md  LLM-generated investment report
├── requirements.txt
└── .env.example
```

## Setup

```bash
conda create -n airline_project python=3.12
conda activate airline_project
pip install -r requirements.txt
python -m nltk.downloader vader_lexicon
cp .env.example .env        # fill in your OpenAI-compatible API key
```

## Running

```bash
# Full pipeline
python main.py

# Resume from a specific step
python main.py --from-step feature_engineering

# Single step
python main.py --only factor_backtest

# Data quality check only
python data_quality.py
```

## Data Sources

- **BTS T-100 Segment Data** — [transtats.bts.gov](https://www.transtats.bts.gov/) (downloaded via `download_bts.py`)
- **yfinance** — stock prices and earnings history
- **SEC EDGAR** — 8-K filings and EPS data via full-text search API

## Database

All intermediate and output tables are stored in `data/airline.db` (SQLite). Key tables:

| Table | Description |
|---|---|
| `flights_raw` | BTS raw segment data (~870k rows) |
| `flights_quarterly_features` | Aggregated quarterly RPK / load factor |
| `earnings_calendar_master` | Canonical one-row-per-(ticker, fiscal_quarter) event table |
| `factor_backtest_dataset` | Event-level factor dataset |
| `factor_ic_results` | Cross-sectional IC per quarter (4 factors) |
| `sde_summary` | Monte Carlo fan statistics (mean / P10 / P50 / P90) |
| `pipeline_run_log` | Run metadata with `run_id` for traceability |
