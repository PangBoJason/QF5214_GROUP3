from __future__ import annotations

import argparse
import sqlite3
import time
from dataclasses import dataclass

import pandas as pd
import yfinance as yf

from project_config import DB_PATH, TARGET_TICKERS, ensure_directories


@dataclass
class DownloadConfig:
    start: str = "2018-01-01"
    end: str | None = None
    interval: str = "1d"


def _with_retry(fn, retries: int = 5, base_delay: float = 15.0):
    """Call fn(); on rate-limit errors retry with exponential back-off."""
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:
            if "rate" in str(exc).lower() and attempt < retries - 1:
                wait = base_delay * (2 ** attempt)
                print(f"    [rate-limit] waiting {wait:.0f}s (retry {attempt + 1}/{retries - 1})...")
                time.sleep(wait)
            else:
                raise


def download_prices(ticker: str, cfg: DownloadConfig) -> pd.DataFrame:
    try:
        history = _with_retry(lambda: yf.download(
            tickers=ticker,
            start=cfg.start,
            end=cfg.end,
            interval=cfg.interval,
            auto_adjust=False,
            progress=False,
            actions=False,
            threads=False,
        ))
    except Exception as exc:
        print(f"[warn] {ticker} price download failed after retries: {exc}")
        return pd.DataFrame()

    if history.empty:
        return pd.DataFrame()

    if isinstance(history.columns, pd.MultiIndex):
        history.columns = history.columns.get_level_values(0)

    frame = history.rename(columns=str.lower).reset_index()
    frame.columns = [col.lower().replace(" ", "_") for col in frame.columns]
    frame["date"] = pd.to_datetime(frame["date"]).dt.date.astype(str)
    frame["ticker"] = ticker
    frame["return_1d"] = frame["adj_close"].pct_change()
    return frame[
        ["ticker", "date", "open", "high", "low", "close", "adj_close", "volume", "return_1d"]
    ]


def extract_earnings_dates(ticker: str) -> pd.DataFrame:
    tk = yf.Ticker(ticker)
    rows: list[dict] = []

    try:
        earnings_dates = _with_retry(lambda: tk.earnings_dates)
        if earnings_dates is not None and not earnings_dates.empty:
            frame = earnings_dates.reset_index()
            frame.columns = [str(col).strip().lower().replace(" ", "_") for col in frame.columns]
            rename_map = {
                frame.columns[0]: "earnings_date",
                "eps_estimate": "eps_estimate",
                "reported_eps": "reported_eps",
                "surprise(%)": "surprise_pct",
            }
            frame = frame.rename(columns=rename_map)
            for _, row in frame.iterrows():
                rows.append(
                    {
                        "ticker": ticker,
                        "earnings_date": pd.to_datetime(row["earnings_date"]).date().isoformat(),
                        "eps_estimate": pd.to_numeric(row.get("eps_estimate"), errors="coerce"),
                        "reported_eps": pd.to_numeric(row.get("reported_eps"), errors="coerce"),
                        "surprise_pct": pd.to_numeric(row.get("surprise_pct"), errors="coerce"),
                        "source": "yfinance_earnings_dates",
                    }
                )
    except Exception as exc:
        print(f"[warn] {ticker} earnings_dates failed: {exc}")

    if not rows:
        try:
            calendar = _with_retry(lambda: tk.calendar)
            if isinstance(calendar, pd.DataFrame) and not calendar.empty:
                idx_name = str(calendar.index[0]).lower()
                value = calendar.iloc[0, 0]
                if "earnings" in idx_name or "earning" in idx_name:
                    rows.append(
                        {
                            "ticker": ticker,
                            "earnings_date": pd.to_datetime(value).date().isoformat(),
                            "eps_estimate": None,
                            "reported_eps": None,
                            "surprise_pct": None,
                            "source": "yfinance_calendar",
                        }
                    )
        except Exception as exc:
            print(f"[warn] {ticker} calendar fallback failed: {exc}")

    if not rows:
        return pd.DataFrame(columns=["ticker", "earnings_date", "eps_estimate", "reported_eps", "surprise_pct", "source"])

    frame = pd.DataFrame(rows).drop_duplicates(subset=["ticker", "earnings_date"]).sort_values("earnings_date")
    # Drop future earnings dates (yfinance returns forward estimates)
    today = pd.Timestamp.today().normalize()
    frame = frame[pd.to_datetime(frame["earnings_date"]) <= today]
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Download stock prices and earnings dates into SQLite.")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default=None)
    args = parser.parse_args()

    ensure_directories()
    cfg = DownloadConfig(start=args.start, end=args.end)

    price_frames = []
    earnings_frames = []
    for i, ticker in enumerate(TARGET_TICKERS):
        if i > 0:
            time.sleep(5)  # inter-ticker pause to avoid rate limiting
        print(f"[info] downloading {ticker} price history")
        price_frames.append(download_prices(ticker, cfg))
        time.sleep(3)
        print(f"[info] fetching {ticker} earnings dates")
        earnings_frames.append(extract_earnings_dates(ticker))

    valid_prices   = [f for f in price_frames   if not f.empty]
    valid_earnings = [f for f in earnings_frames if not f.empty]

    if not valid_prices:
        print("[warn] No price data downloaded — yfinance still rate-limiting. Wait a few minutes and retry.")
        return

    prices   = pd.concat(valid_prices,   ignore_index=True)
    earnings = pd.concat(valid_earnings, ignore_index=True) if valid_earnings else pd.DataFrame()

    with sqlite3.connect(DB_PATH) as conn:
        if not prices.empty:
            prices.to_sql("stock_prices", conn, if_exists="replace", index=False)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_stock_prices_ticker_date ON stock_prices(ticker, date)")
        if not earnings.empty:
            earnings.to_sql("earnings_calendar", conn, if_exists="replace", index=False)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_earnings_ticker_date ON earnings_calendar(ticker, earnings_date)")
        conn.commit()

    print(f"[done] stock_prices rows: {len(prices):,}")
    print(f"[done] earnings_calendar rows: {len(earnings):,}")


if __name__ == "__main__":
    main()
