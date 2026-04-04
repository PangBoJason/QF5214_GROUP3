"""Phase 1c: SEC EDGAR supplement — historical earnings dates + reported EPS + 8-K sentiment.

Fills coverage gaps left by yfinance (typically pre-2022 quarters).
Free, no API key required.  EDGAR rate limit: ≤ 10 req/s.

Design (two-layer calendar):
  earnings_calendar      — raw table, all sources preserved as-is
                           yfinance writes here first (get_stock_data.py)
                           EDGAR appends here only for quarters not already covered
  earnings_calendar_master — canonical one-row-per-(ticker, fiscal_quarter) table
                           built by build_master_calendar() at the end of this script
                           ALL downstream modules (feature_engineering, factor_backtest)
                           read ONLY from this master table, never from earnings_calendar

EPS / surprise priority:
  - eps_estimate, reported_eps, surprise_pct come exclusively from yfinance
  - EDGAR reported_eps is stored separately as reported_eps_edgar (supplementary only)

Data flow:
  SEC company_tickers.json  → ticker → CIK
  /api/xbrl/companyfacts/   → EarningsPerShareDiluted (quarterly, filed_date)
  /submissions/             → 8-K filing dates + accession numbers
  /Archives/edgar/          → 8-K HTML text → VADER compound sentiment score

Output tables:
  earnings_calendar        (raw, append-only for EDGAR rows)
  earnings_calendar_master (canonical, replaced each run)
  sentiment_factors        (event_id, sentiment_compound per 8-K filing)
"""
from __future__ import annotations

import re
import sqlite3
import time
from datetime import timedelta

import pandas as pd
import requests
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from project_config import DB_PATH, TARGET_TICKERS, ensure_directories, log_pipeline_run

EDGAR_BASE = "https://data.sec.gov"
EDGAR_HEADERS = {"User-Agent": "AirlineQuantResearch contact@research.local"}
REQUEST_DELAY = 0.15
DATE_START    = "2018-01-01"
DATE_END      = "2024-12-31"


# ── CIK lookup ────────────────────────────────────────────────────────────────

def fetch_cik_map(tickers: list[str]) -> dict[str, str]:
    resp = requests.get(
        "https://www.sec.gov/files/company_tickers.json",
        headers=EDGAR_HEADERS,
        timeout=30,
    )
    resp.raise_for_status()
    lookup = {v["ticker"].upper(): str(v["cik_str"]).zfill(10) for v in resp.json().values()}
    found = {t: lookup[t] for t in tickers if t in lookup}
    missing = [t for t in tickers if t not in lookup]
    if missing:
        print(f"    [warn] CIK not found for: {missing}")
    return found


# ── EDGAR company-facts EPS ───────────────────────────────────────────────────

def fetch_eps_records(cik: str, ticker: str) -> list[dict]:
    url = f"{EDGAR_BASE}/api/xbrl/companyfacts/CIK{cik}.json"
    resp = requests.get(url, headers=EDGAR_HEADERS, timeout=30)
    if resp.status_code != 200:
        print(f"    [warn] company-facts HTTP {resp.status_code}")
        return []

    try:
        eps_units = (
            resp.json()["facts"]["us-gaap"]["EarningsPerShareDiluted"]["units"]["USD/shares"]
        )
    except KeyError:
        print(f"    [warn] EarningsPerShareDiluted missing in facts for {ticker}")
        return []

    rows = []
    for item in eps_units:
        if item.get("form") not in ("10-Q", "10-K"):
            continue
        fp = item.get("fp", "")
        if fp not in ("Q1", "Q2", "Q3", "Q4", "FY"):
            continue
        if not item.get("end") or not item.get("filed"):
            continue
        if item["filed"] < DATE_START or item["filed"] > DATE_END:
            continue
        rows.append({
            "ticker":        ticker,
            "fiscal_period": fp,
            "period_end":    item["end"],
            "filed_date":    item["filed"],
            "reported_eps":  float(item["val"]),
        })
    return rows


# ── EDGAR 8-K filing dates + accession numbers ───────────────────────────────

def fetch_8k_filings(cik: str) -> list[dict]:
    """Return list of {date, accession_no} for all 8-K filings."""
    url = f"{EDGAR_BASE}/submissions/CIK{cik}.json"
    resp = requests.get(url, headers=EDGAR_HEADERS, timeout=30)
    if resp.status_code != 200:
        return []

    recent = resp.json().get("filings", {}).get("recent", {})
    forms        = recent.get("form", [])
    dates        = recent.get("filingDate", [])
    accessions   = recent.get("accessionNumber", [])

    return [
        {"date": d, "accession": a}
        for f, d, a in zip(forms, dates, accessions)
        if f == "8-K"
    ]


def fetch_8k_dates(cik: str) -> list[str]:
    return sorted(f["date"] for f in fetch_8k_filings(cik))


# ── 8-K text extraction + VADER scoring ──────────────────────────────────────

_HTML_TAG = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"\s+")

def _clean_html(html: str) -> str:
    text = _HTML_TAG.sub(" ", html)
    return _WHITESPACE.sub(" ", text).strip()


def fetch_8k_text(cik: str, accession: str, max_chars: int = 8000) -> str:
    """Download the primary 8-K document and return cleaned plain text."""
    acc_nodash = accession.replace("-", "")
    index_url = (
        f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
        f"{acc_nodash}/{accession}-index.htm"
    )
    try:
        resp = requests.get(index_url, headers=EDGAR_HEADERS, timeout=20)
        if resp.status_code != 200:
            return ""
        # Find the primary document link (usually .htm/.html, not the index itself)
        links = re.findall(r'href="([^"]+\.htm[l]?)"', resp.text, re.IGNORECASE)
        primary = next(
            (l for l in links if "index" not in l.lower() and acc_nodash in l.replace("-", "")),
            links[0] if links else None,
        )
        if not primary:
            return ""
        doc_url = f"https://www.sec.gov{primary}" if primary.startswith("/") else primary
        time.sleep(REQUEST_DELAY)
        doc_resp = requests.get(doc_url, headers=EDGAR_HEADERS, timeout=20)
        if doc_resp.status_code != 200:
            return ""
        return _clean_html(doc_resp.text)[:max_chars]
    except Exception:
        return ""


def score_8k_sentiment(vader: SentimentIntensityAnalyzer, text: str) -> float | None:
    """Return VADER compound score [-1, 1] or None if text is empty."""
    if not text.strip():
        return None
    return float(vader.polarity_scores(text)["compound"])


# ── Match EPS periods → announcement dates ────────────────────────────────────

def build_earnings_rows(eps_records: list[dict], filings_8k: list[dict]) -> pd.DataFrame:
    if not eps_records:
        return pd.DataFrame()

    df = pd.DataFrame(eps_records)
    df["filed_date"] = pd.to_datetime(df["filed_date"])
    df["period_end"] = pd.to_datetime(df["period_end"])

    filings_8k_df = pd.DataFrame(filings_8k) if filings_8k else pd.DataFrame(columns=["date", "accession"])
    filings_8k_df["date"] = pd.to_datetime(filings_8k_df["date"])

    ann_dates: list[str] = []
    sources:   list[str] = []
    accessions: list[str | None] = []

    for _, row in df.iterrows():
        window_start = row["filed_date"] - timedelta(days=30)
        candidates = filings_8k_df[
            (filings_8k_df["date"] >= window_start) & (filings_8k_df["date"] <= row["filed_date"])
        ]
        if not candidates.empty:
            best = candidates.loc[candidates["date"].idxmax()]
            ann_dates.append(best["date"].strftime("%Y-%m-%d"))
            sources.append("edgar_8k")
            accessions.append(best["accession"])
        else:
            proxy = (row["filed_date"] - timedelta(days=14)).strftime("%Y-%m-%d")
            ann_dates.append(proxy)
            sources.append("edgar_10q_proxy")
            accessions.append(None)

    df["earnings_date"] = ann_dates
    df["source"]        = sources
    df["accession"]     = accessions
    df["eps_estimate"]  = None
    df["surprise_pct"]  = None

    df["earnings_date"] = pd.to_datetime(df["earnings_date"])
    df["_fiscal_q"] = df["earnings_date"].dt.to_period("Q")
    df = (
        df.sort_values("filed_date")
          .drop_duplicates(subset=["ticker", "_fiscal_q"], keep="last")
          .drop(columns=["_fiscal_q"])
          .reset_index(drop=True)
    )
    df["earnings_date"] = df["earnings_date"].dt.strftime("%Y-%m-%d")
    return df


# ── Upsert earnings_calendar ──────────────────────────────────────────────────

def upsert_calendar(conn: sqlite3.Connection, new_rows: pd.DataFrame) -> int:
    if new_rows.empty:
        return 0

    try:
        existing = pd.read_sql_query(
            "SELECT ticker, earnings_date FROM earnings_calendar", conn
        )
        existing["_fiscal_q"] = pd.to_datetime(existing["earnings_date"]).dt.to_period("Q")
        existing_keys = set(zip(existing["ticker"], existing["_fiscal_q"]))
    except Exception:
        existing_keys = set()

    new_rows = new_rows.copy()
    new_rows["_fiscal_q"] = pd.to_datetime(new_rows["earnings_date"]).dt.to_period("Q")
    to_insert = new_rows[
        ~new_rows.apply(lambda r: (r["ticker"], r["_fiscal_q"]) in existing_keys, axis=1)
    ].drop(columns=["_fiscal_q"]).copy()

    if to_insert.empty:
        return 0

    to_insert[
        ["ticker", "earnings_date", "eps_estimate", "reported_eps", "surprise_pct", "source"]
    ].to_sql("earnings_calendar", conn, if_exists="append", index=False)

    return len(to_insert)


# ── Build sentiment_factors from 8-K VADER scores ────────────────────────────

def build_sentiment_factors(
    conn: sqlite3.Connection,
    vader: SentimentIntensityAnalyzer,
    ticker: str,
    cik: str,
    rows_df: pd.DataFrame,
) -> list[dict]:
    """
    For rows that have a matched 8-K accession, download the filing text
    and compute VADER compound sentiment.  Returns list of sentiment_factor rows.
    """
    factor_rows = []
    has_accession = rows_df[rows_df["accession"].notna()]

    for _, row in has_accession.iterrows():
        event_id = f"{ticker}_{row['earnings_date']}"
        time.sleep(REQUEST_DELAY)
        text = fetch_8k_text(cik, row["accession"])
        score = score_8k_sentiment(vader, text)
        factor_rows.append({
            "event_id":           event_id,
            "ticker":             ticker,
            "earnings_date":      row["earnings_date"],
            "sentiment_source":   "edgar_8k_vader",
            "sentiment_compound": score,
            "article_count":      1 if score is not None else 0,
        })
    return factor_rows


# ── Build earnings_calendar_master ───────────────────────────────────────────

def build_master_calendar(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Collapse earnings_calendar into one canonical event per (ticker, fiscal_quarter).

    Priority rules:
      1. yfinance row is always primary when present for that fiscal_quarter.
         - eps_estimate, reported_eps, surprise_pct come exclusively from yfinance.
      2. EDGAR row fills the slot only when yfinance has NO event for that quarter.
      3. EDGAR EPS is stored in reported_eps_edgar for reference but never merged
         into the primary EPS fields.

    Output table: earnings_calendar_master
      ticker, fiscal_quarter, earnings_date,
      eps_estimate, reported_eps, surprise_pct,   ← yfinance-only
      primary_source,
      edgar_event_date,                            ← supplementary (may be NULL)
      reported_eps_edgar                           ← supplementary (may be NULL)
    """
    raw = pd.read_sql_query("SELECT * FROM earnings_calendar", conn)
    if raw.empty:
        return pd.DataFrame()

    raw["earnings_date"] = pd.to_datetime(raw["earnings_date"])
    raw["fiscal_quarter"] = raw["earnings_date"].dt.to_period("Q").astype(str)

    master_rows: list[dict] = []
    for (ticker, fq), grp in raw.groupby(["ticker", "fiscal_quarter"]):
        yf = grp[grp["source"].str.startswith("yfinance")]
        ed = grp[~grp["source"].str.startswith("yfinance")]

        if not yf.empty:
            primary = yf.iloc[0]
            primary_source = primary["source"]
            edgar_event_date = ed.iloc[0]["earnings_date"].strftime("%Y-%m-%d") if not ed.empty else None
            reported_eps_edgar = (
                float(ed.iloc[0]["reported_eps"])
                if not ed.empty and pd.notna(ed.iloc[0]["reported_eps"])
                else None
            )
        else:
            primary = ed.iloc[0]
            primary_source = primary["source"]
            edgar_event_date = primary["earnings_date"].strftime("%Y-%m-%d")
            reported_eps_edgar = (
                float(primary["reported_eps"]) if pd.notna(primary["reported_eps"]) else None
            )

        master_rows.append({
            "ticker":             ticker,
            "fiscal_quarter":     fq,
            "earnings_date":      primary["earnings_date"].strftime("%Y-%m-%d"),
            "eps_estimate":       float(primary["eps_estimate"]) if pd.notna(primary.get("eps_estimate")) else None,
            "reported_eps":       float(primary["reported_eps"]) if primary_source.startswith("yfinance") and pd.notna(primary.get("reported_eps")) else None,
            "surprise_pct":       float(primary["surprise_pct"]) if primary_source.startswith("yfinance") and pd.notna(primary.get("surprise_pct")) else None,
            "primary_source":     primary_source,
            "edgar_event_date":   edgar_event_date if primary_source.startswith("yfinance") else None,
            "reported_eps_edgar": reported_eps_edgar,
        })

    master = pd.DataFrame(master_rows).sort_values(["ticker", "fiscal_quarter"]).reset_index(drop=True)
    master.to_sql("earnings_calendar_master", conn, if_exists="replace", index=False)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_master_ticker_fq "
        "ON earnings_calendar_master(ticker, fiscal_quarter)"
    )
    return master


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="EDGAR supplement.")
    parser.add_argument("--run-id", default="standalone")
    args = parser.parse_args()

    ensure_directories()

    print("[step] Loading VADER sentiment analyser ...")
    vader = SentimentIntensityAnalyzer()

    print("[step] Fetching CIK map from SEC company_tickers.json ...")
    cik_map = fetch_cik_map(TARGET_TICKERS)

    total_added = 0
    all_sentiment: list[dict] = []

    with sqlite3.connect(DB_PATH) as conn:
        for ticker in TARGET_TICKERS:
            cik = cik_map.get(ticker)
            if not cik:
                print(f"\n  [{ticker}] skipped (no CIK)")
                continue

            print(f"\n  [{ticker}]  CIK {cik}")

            time.sleep(REQUEST_DELAY)
            eps_records = fetch_eps_records(cik, ticker)
            print(f"    EPS records  : {len(eps_records)}")

            time.sleep(REQUEST_DELAY)
            filings_8k = fetch_8k_filings(cik)
            print(f"    8-K filings  : {len(filings_8k)}")

            rows_df = build_earnings_rows(eps_records, filings_8k)
            added   = upsert_calendar(conn, rows_df)
            total_added += added
            print(f"    Calendar rows added: {added}")

            print(f"    Scoring 8-K sentiment via VADER ...")
            sentiment_rows = build_sentiment_factors(conn, vader, ticker, cik, rows_df)
            scored = sum(1 for r in sentiment_rows if r["sentiment_compound"] is not None)
            print(f"    Sentiment rows: {len(sentiment_rows)} ({scored} scored)")
            all_sentiment.extend(sentiment_rows)

        # Write sentiment_factors table (replace to get fresh scores)
        if all_sentiment:
            sf = pd.DataFrame(all_sentiment)
            sf.to_sql("sentiment_factors", conn, if_exists="replace", index=False)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_sentiment_event "
                "ON sentiment_factors(event_id)"
            )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_earnings_ticker_date "
            "ON earnings_calendar(ticker, earnings_date)"
        )

        master = build_master_calendar(conn)
        log_pipeline_run(conn, args.run_id, "edgar_supplement", "success",
                         "earnings_calendar_master", len(master), DATE_START, DATE_END)
        conn.commit()

    print(f"\n[done] Total calendar rows added : {total_added}")
    print(f"[done] Total sentiment rows      : {len(all_sentiment)}")
    print(f"[done] Master calendar rows      : {len(master)}")


if __name__ == "__main__":
    main()
