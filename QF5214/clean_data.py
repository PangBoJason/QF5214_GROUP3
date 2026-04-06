import glob
import os
import sqlite3
from typing import List, Set

import pandas as pd

# Matches headered .asc first row (29 fields; last is trailing empty column)
ASC_FULL_COLUMNS: List[str] = [
    "year",
    "month",
    "origin",
    "origin_city_market_id",
    "origin_wac",
    "origin_city_name",
    "dest",
    "dest_city_market_id",
    "dest_wac",
    "dest_city_name",
    "Carrier",
    "Carrier_Entity",
    "carrier_group",
    "distance",
    "Svc_Class",
    "Aircraft_Group",
    "Aircraft_type",
    "Aircraft_Config",
    "departures_performed",
    "departures_scheduled",
    "payload",
    "seats",
    "passengers",
    "freight",
    "mail",
    "ramp_to_ramp",
    "air_time",
    "Wac",
    "_pad",
]

# Columns used by cleaning (names match historical headered files)
TARGET_COLUMNS: List[str] = [
    "year",
    "month",
    "origin",
    "origin_city_market_id",
    "dest",
    "dest_city_market_id",
    "Carrier",
    "distance",
    "Aircraft_type",
    "departures_performed",
    "seats",
    "passengers",
    "Svc_Class",
]

_TARGET_COL_SET: Set[str] = set(TARGET_COLUMNS)

# Earnings calendar in quant_flights.db (from data/day.csv; used by fundamental backtest SQL)
EARNINGS_TABLE_NAME: str = "earnings_day"
DAY_CSV_FILENAME: str = "day.csv"


def _read_first_line(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        return f.readline()


def _asc_has_header_line(first_line: str) -> bool:
    """Treat first column 'year' as header row; else headerless (e.g. some 2023–2025 slices)."""
    col0: str = first_line.split("|")[0].strip()
    return col0.lower() == "year"


def _read_asc_to_dataframe(file_path: str) -> pd.DataFrame:
    """Read pipe-delimited .asc with/without header; keep TARGET_COLUMNS only."""
    first_line: str = _read_first_line(file_path)
    has_header: bool = _asc_has_header_line(first_line)

    if has_header:
        return pd.read_csv(
            file_path,
            sep="|",
            header=0,
            usecols=lambda c: str(c).strip() in _TARGET_COL_SET,
            engine="c",
            low_memory=False,
        )

    return pd.read_csv(
        file_path,
        sep="|",
        header=None,
        names=ASC_FULL_COLUMNS,
        usecols=TARGET_COLUMNS,
        engine="c",
        low_memory=False,
    )


def process_t100_segment_data(file_path: str) -> pd.DataFrame:
    """
    Robust T-100 Segment cleaning.
    Supports: headered .asc (first row year|month|...) and headerless .asc (29 fixed columns, last pad).
    """
    print(f"Reading and cleaning: {os.path.basename(file_path)} ...")

    try:
        df: pd.DataFrame = _read_asc_to_dataframe(file_path)
    except Exception as e:
        print(f"Failed to read {file_path}: {e}")
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    # --- Core filters ---
    df = df[df["Svc_Class"] == "F"]
    df = df[(df["departures_performed"] > 0) & (df["seats"] > 0)]
    df = df[df["origin"].str.match(r"^[A-Z]{3}$") & df["dest"].str.match(r"^[A-Z]{3}$")]
    df = df.dropna(subset=["origin", "dest", "passengers", "seats"])

    # --- Derived ---
    df["Cap"] = df["seats"] / df["departures_performed"]

    df = df.drop(columns=["Svc_Class"])
    df.columns = [col.lower() for col in df.columns]

    return df


def ingest_earnings_calendar_to_sql(conn: sqlite3.Connection, data_dir: str) -> None:
    """
    Load data/day.csv into SQLite table earnings_day (skip with message if missing).
    Columns: ticker, year, quarter, earnings_date (as in CSV).
    """
    csv_path: str = os.path.join(data_dir, DAY_CSV_FILENAME)
    if not os.path.isfile(csv_path):
        print(f"\n[WARNING] {csv_path} not found; skipping earnings calendar ingest.")
        return

    need_cols: List[str] = ["ticker", "year", "quarter", "earnings_date"]
    day_df: pd.DataFrame = pd.read_csv(csv_path)
    miss: List[str] = [c for c in need_cols if c not in day_df.columns]
    if miss:
        print(f"\n[WARNING] {csv_path} missing columns {miss}; skipping earnings calendar ingest.")
        return

    out: pd.DataFrame = day_df[need_cols].copy()
    out["earnings_date"] = pd.to_datetime(out["earnings_date"], errors="coerce")
    out["year"] = out["year"].astype(int)
    out["quarter"] = out["quarter"].astype(int)
    out = out.dropna(subset=["earnings_date"])

    out.to_sql(name=EARNINGS_TABLE_NAME, con=conn, if_exists="replace", index=False)
    print(f"\n[OK] Earnings calendar written to [{EARNINGS_TABLE_NAME}], rows: {len(out):,}")


def main() -> None:
    data_dir: str = "./data"

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created directory: {data_dir}")

    all_files: List[str] = sorted(glob.glob(os.path.join(data_dir, "*.asc")))

    if not all_files:
        print("No .asc files found; check the data directory.")
        return

    df_list: List[pd.DataFrame] = []
    for file in all_files:
        clean_df: pd.DataFrame = process_t100_segment_data(file)
        if not clean_df.empty:
            df_list.append(clean_df)

    if not df_list:
        print("No valid rows after cleaning.")
        return

    print("Concatenating all slices...")
    final_df: pd.DataFrame = pd.concat(df_list, ignore_index=True)
    final_df = final_df.drop_duplicates()

    print(f"\n[OK] Cleaning done. Final rows: {len(final_df):,}")

    db_path: str = os.path.join(data_dir, "quant_flights.db")
    table_name: str = "t100_segment"

    print(f"\nConnecting to SQLite [{db_path}] and writing table...")

    conn: sqlite3.Connection = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    final_df.to_sql(name=table_name, con=conn, if_exists="replace", index=False)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_t100_year_carrier "
        "ON t100_segment(year, carrier, origin, dest)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_t100_year_month "
        "ON t100_segment(year, month)"
    )
    conn.commit()

    ingest_earnings_calendar_to_sql(conn, data_dir)

    print("\nRunning validation SQL...")
    cursor: sqlite3.Cursor = conn.cursor()

    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    row_count: int = cursor.fetchone()[0]
    print(f"Row count in {table_name}: {row_count:,}")

    test_query: str = f"""
        SELECT origin, dest, SUM(passengers) as total_pax
        FROM {table_name}
        WHERE year = 2018
        GROUP BY origin, dest
        ORDER BY total_pax DESC
        LIMIT 3
    """
    try:
        print("\nTop 3 routes by 2018 passengers (SQL):")
        for row in cursor.execute(test_query):
            print(f"  {row[0]} -> {row[1]}: {int(row[2]):,} pax")
    except sqlite3.OperationalError:
        pass

    conn.close()
    print(f"\n[OK] Data written to SQLite: {os.path.abspath(db_path)}")


if __name__ == "__main__":
    main()
