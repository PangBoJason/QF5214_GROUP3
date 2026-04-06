"""
Module 5: Read four factors from quarterly_factors; MWU expert fusion; earnings / price logic.
Portfolio: each expert outputs Rank1=+0.5 / Rank6=-0.5; MWU updates after observing each quarter's realized outcome.

Fixes vs original:
- event timing fixed: portfolio forms at quarter-end + 5 bdays and earns return from formation date into the post-earnings window
- transaction cost: 50bps one-way slippage modeled via turnover
- bootstrap CI on annualized Sharpe
- factor correlation matrix + IC significance (t-test)
"""
from __future__ import annotations

import os
import sqlite3
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import scipy.stats as stats
import yfinance as yf

from clean_data import EARNINGS_TABLE_NAME
from factor_engineering import QUARTERLY_FACTORS_TABLE

DEFAULT_DB_PATH: str = os.path.join("data", "quant_flights.db")
EQUITY_PRICES_TABLE: str = "equity_adj_close_daily"

CARRIER_TO_TICKER: Dict[str, str] = {
    "DL": "DAL",
    "AA": "AAL",
    "UA": "UAL",
    "WN": "LUV",
    "AS": "ALK",
    "B6": "JBLU",
}

TICKERS: List[str] = list(CARRIER_TO_TICKER.values())
IATA_ORDER: List[str] = list(CARRIER_TO_TICKER.keys())
BENCHMARK: str = "SPY"
YF_START: str = "2017-01-01"

FACTOR_YEAR_START: int = 2017
FACTOR_YEAR_END: int = 2025

FACTOR_COLS: List[str] = ["factor_rpk", "factor_lf", "factor_ask", "factor_share"]
N_EXPERTS: int = len(FACTOR_COLS)
MWU_ETA: float = 2.0
TRANSACTION_COST_BPS: float = 50.0  # one-way slippage in basis points
YF_MAX_RETRIES: int = 3
YF_RETRY_SLEEP_SEC: float = 4.0
POST_EARNINGS_BDAYS: int = 5
MIN_VALID_NAMES_PER_QUARTER: int = 4


# ── Portfolio formation date ───────────────────────────────────────────────────


def _portfolio_formation_date(year: int, quarter: int) -> pd.Timestamp:
    """
    Portfolio is formed at quarter-end + 5 business days.
    Only earnings released on/before this date are usable (no look-ahead).
    """
    quarter_end_month: int = {1: 3, 2: 6, 3: 9, 4: 12}[quarter]
    last_day: pd.Timestamp = pd.Timestamp(year, quarter_end_month, 1) + pd.offsets.MonthEnd(0)
    return last_day + pd.offsets.BDay(5)


# ── DB helpers ─────────────────────────────────────────────────────────────────


def _load_earnings_calendar_from_db(
    db_path: str,
    year_start: int,
    year_end: int,
) -> pd.DataFrame:
    query: str = f"""
        SELECT ticker, year, quarter, earnings_date
        FROM {EARNINGS_TABLE_NAME}
        WHERE year >= ? AND year <= ?
    """
    conn: sqlite3.Connection = sqlite3.connect(db_path)
    try:
        cal: pd.DataFrame = pd.read_sql_query(query, conn, params=(year_start, year_end))
    except sqlite3.OperationalError as e:
        raise RuntimeError(
            f"Cannot read {EARNINGS_TABLE_NAME} (run clean_data.py to load day.csv): {e}"
        ) from e
    finally:
        conn.close()

    if cal.empty:
        return cal

    cal["earnings_date"] = pd.to_datetime(cal["earnings_date"], errors="coerce")
    cal["year"] = cal["year"].astype(int)
    cal["quarter"] = cal["quarter"].astype(int)
    cal = cal.dropna(subset=["earnings_date"])
    cal = cal.set_index(["ticker", "year", "quarter"]).sort_index()
    return cal


def _equity_table_exists(conn: sqlite3.Connection) -> bool:
    cur: sqlite3.Cursor = conn.cursor()
    cur.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (EQUITY_PRICES_TABLE,),
    )
    return cur.fetchone() is not None


def _normalize_close_frame(raw: pd.DataFrame, requested_syms: List[str]) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()

    clos = raw["Close"]
    if isinstance(clos, pd.Series):
        close: pd.DataFrame = clos.to_frame(name=requested_syms[0])
    else:
        close = clos.copy()

    close = close.sort_index()
    close.index = pd.to_datetime(close.index).normalize()
    return close


def _download_batch_once(all_syms: List[str], start: str, end: str, threads: bool) -> pd.DataFrame:
    raw: pd.DataFrame = yf.download(
        all_syms,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=threads,
    )
    return _normalize_close_frame(raw, all_syms)


def _download_single_symbol(sym: str, start: str, end: str) -> pd.DataFrame:
    raw: pd.DataFrame = yf.download(
        sym,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    out: pd.DataFrame = _normalize_close_frame(raw, [sym])
    if out.empty or sym not in out.columns:
        return pd.DataFrame()
    return out[[sym]]


def _wide_from_yfinance(all_syms: List[str], start: str, end: str) -> pd.DataFrame:
    last_err: str | None = None

    for attempt in range(1, YF_MAX_RETRIES + 1):
        for threads in (False, True):
            try:
                close: pd.DataFrame = _download_batch_once(all_syms, start, end, threads=threads)
                if not close.empty:
                    missing = [sym for sym in all_syms if sym not in close.columns or close[sym].dropna().empty]
                    if not missing:
                        return close
                    last_err = f"batch download missing symbols: {missing}"
                else:
                    last_err = "batch download returned empty frame"
            except Exception as exc:
                last_err = f"batch download failed: {exc}"

        if attempt < YF_MAX_RETRIES:
            time.sleep(YF_RETRY_SLEEP_SEC * attempt)

    single_frames: List[pd.DataFrame] = []
    missing_single: List[str] = []
    for sym in all_syms:
        ok = False
        for attempt in range(1, YF_MAX_RETRIES + 1):
            try:
                frame = _download_single_symbol(sym, start, end)
                if not frame.empty:
                    single_frames.append(frame)
                    ok = True
                    break
            except Exception as exc:
                last_err = f"single download failed for {sym}: {exc}"
            time.sleep(YF_RETRY_SLEEP_SEC * attempt)
        if not ok:
            missing_single.append(sym)

    if single_frames:
        close = pd.concat(single_frames, axis=1).sort_index()
        close.index = pd.to_datetime(close.index).normalize()
        available = [sym for sym in all_syms if sym in close.columns and not close[sym].dropna().empty]
        if len(available) == len(all_syms):
            return close[all_syms]
        missing_single = [sym for sym in all_syms if sym not in available]

    msg = "yfinance returned no usable price data"
    if missing_single:
        msg += f"; missing symbols after retries: {missing_single}"
    if last_err:
        msg += f"; last error: {last_err}"
    raise RuntimeError(msg)


def _save_adj_close_to_db(db_path: str, wide: pd.DataFrame) -> None:
    idx_name: str = wide.index.name if wide.index.name else "trade_date"
    temp: pd.DataFrame = wide.copy()
    temp.index.name = idx_name
    long_df: pd.DataFrame = temp.reset_index().melt(
        id_vars=idx_name,
        var_name="ticker",
        value_name="adj_close",
    ).rename(columns={idx_name: "trade_date"})
    long_df["trade_date"] = pd.to_datetime(long_df["trade_date"]).dt.strftime("%Y-%m-%d")
    long_df = long_df.dropna(subset=["adj_close"])

    conn: sqlite3.Connection = sqlite3.connect(db_path)
    try:
        long_df.to_sql(EQUITY_PRICES_TABLE, conn, if_exists="replace", index=False)
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_eq_price_ticker_date "
            f"ON {EQUITY_PRICES_TABLE}(ticker, trade_date)"
        )
        conn.commit()
    finally:
        conn.close()


def _db_prices_cover_range(
    wide: pd.DataFrame,
    all_syms: List[str],
    start: str,
    end: str,
) -> bool:
    req_lo: pd.Timestamp = pd.Timestamp(start).normalize()
    req_hi_excl: pd.Timestamp = pd.Timestamp(end).normalize()
    margin_lo: pd.Timedelta = pd.Timedelta(days=20)
    margin_hi: pd.Timedelta = pd.Timedelta(days=21)

    for sym in all_syms:
        if sym not in wide.columns:
            return False
        s: pd.Series = wide[sym].dropna()
        if s.empty:
            return False
        if s.index.min() > req_lo + margin_lo:
            return False
        if s.index.max() < req_hi_excl - margin_hi:
            return False
    return True


def _load_adj_close_from_db(
    db_path: str,
    all_syms: List[str],
    start: str,
    end: str,
) -> Optional[pd.DataFrame]:
    if not os.path.isfile(db_path):
        return None

    conn: sqlite3.Connection = sqlite3.connect(db_path)
    try:
        if not _equity_table_exists(conn):
            return None
        placeholders: str = ",".join("?" * len(all_syms))
        query: str = (
            f"SELECT trade_date, ticker, adj_close FROM {EQUITY_PRICES_TABLE} "
            f"WHERE ticker IN ({placeholders})"
        )
        raw: pd.DataFrame = pd.read_sql_query(query, conn, params=tuple(all_syms))
    finally:
        conn.close()

    if raw.empty:
        return None

    raw["trade_date"] = pd.to_datetime(raw["trade_date"], errors="coerce")
    raw = raw.dropna(subset=["trade_date", "adj_close"])
    wide: pd.DataFrame = raw.pivot(index="trade_date", columns="ticker", values="adj_close")
    wide = wide.sort_index()
    wide.index = pd.to_datetime(wide.index).normalize()

    if not _db_prices_cover_range(wide, all_syms, start, end):
        return None
    return wide


def load_or_fetch_adj_close(
    db_path: str,
    tickers: List[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    all_syms: List[str] = list(dict.fromkeys(list(tickers) + [BENCHMARK]))

    cached: Optional[pd.DataFrame] = _load_adj_close_from_db(db_path, all_syms, start, end)
    if cached is not None:
        print(f"========== Equity: loaded from DB table [{EQUITY_PRICES_TABLE}] ==========")
        return cached

    print(f"========== Equity: no DB cache; downloading via yfinance → [{EQUITY_PRICES_TABLE}] ... ==========")
    wide: pd.DataFrame = _wide_from_yfinance(all_syms, start, end)
    os.makedirs(os.path.dirname(os.path.abspath(db_path)) or ".", exist_ok=True)
    _save_adj_close_to_db(db_path, wide)
    print(f"========== Cached {len(wide):,} trading days × {len(wide.columns)} symbols to {db_path} ==========")
    return wide


# ── Event study ────────────────────────────────────────────────────────────────


def _loc_first_on_or_after(index: pd.DatetimeIndex, day: pd.Timestamp) -> int:
    ts: pd.Timestamp = pd.Timestamp(day).normalize()
    pos: int = int(index.searchsorted(ts, side="left"))
    if pos >= len(index):
        return len(index) - 1
    return pos


def _forward_excess_return(
    stock: pd.Series,
    spy: pd.Series,
    entry_date: pd.Timestamp,
    exit_date: pd.Timestamp,
) -> float:
    """
    Excess return (stock vs SPY) from entry_date to exit_date,
    using the first available trading day on/after each date.
    This is the actual tradable return for a portfolio formed at entry_date.
    """
    st: pd.Series = stock.dropna().sort_index()
    sp: pd.Series = spy.dropna().sort_index()

    entry_pos: int = _loc_first_on_or_after(st.index, entry_date)
    exit_pos: int = _loc_first_on_or_after(st.index, exit_date)
    if exit_pos >= len(st) or exit_pos <= entry_pos:
        return float("nan")

    sp_entry_pos: int = _loc_first_on_or_after(sp.index, entry_date)
    sp_exit_pos: int = _loc_first_on_or_after(sp.index, exit_date)
    if sp_exit_pos >= len(sp) or sp_exit_pos <= sp_entry_pos:
        return float("nan")

    p_entry: float = float(st.iloc[entry_pos])
    p_exit: float = float(st.iloc[exit_pos])
    if p_entry <= 0:
        return float("nan")

    m_entry: float = float(sp.iloc[sp_entry_pos])
    m_exit: float = float(sp.iloc[sp_exit_pos])
    if m_entry <= 0:
        return float("nan")

    return (p_exit / p_entry - 1.0) - (m_exit / m_entry - 1.0)


def _event_excess_return_after_entry(
    stock: pd.Series,
    spy: pd.Series,
    formation_date: pd.Timestamp,
    event_day: pd.Timestamp,
    post_event_bdays: int,
    max_exit_date: pd.Timestamp | None = None,
) -> float:
    """
    Tradable event return:
    enter on/after formation_date, exit on/after event_day + post_event_bdays.
    If the target exit exceeds the next formation date, cap at the next formation date.
    """
    if pd.Timestamp(event_day) < pd.Timestamp(formation_date):
        return float("nan")

    exit_day: pd.Timestamp = pd.Timestamp(event_day) + pd.offsets.BDay(post_event_bdays)
    if max_exit_date is not None:
        exit_day = min(exit_day, pd.Timestamp(max_exit_date))
    return _forward_excess_return(stock, spy, formation_date, exit_day)


# ── MWU helpers ────────────────────────────────────────────────────────────────


def _expert_rank_weights(values: np.ndarray, valid_mask: np.ndarray | None = None) -> np.ndarray:
    """Rank within the valid cross-section; top +0.5, bottom -0.5, rest 0."""
    w: np.ndarray = np.zeros(len(TICKERS), dtype=float)
    if values.shape[0] != len(TICKERS):
        return w

    if valid_mask is None:
        valid_mask = np.isfinite(values)
    else:
        valid_mask = valid_mask & np.isfinite(values)

    valid_idx: np.ndarray = np.flatnonzero(valid_mask)
    if valid_idx.size < 2:
        return w

    valid_vals: np.ndarray = values[valid_idx]
    order_local: np.ndarray = np.argsort(-valid_vals)
    w[int(valid_idx[order_local[0]])] = 0.5
    w[int(valid_idx[order_local[-1]])] = -0.5
    return w


def _normalize_l1_zero_sum(vec: np.ndarray) -> np.ndarray:
    """L1 normalize; preserves zero sum for long-short convex mixes."""
    s: float = float(np.sum(np.abs(vec)))
    if s <= 0.0:
        return vec
    return vec / s


def _load_quarterly_factors(db_path: str) -> pd.DataFrame:
    conn: sqlite3.Connection = sqlite3.connect(db_path)
    try:
        return pd.read_sql_query(f"SELECT * FROM {QUARTERLY_FACTORS_TABLE}", conn)
    finally:
        conn.close()


def _spearman_ic(x: np.ndarray, y: np.ndarray) -> float:
    sx: pd.Series = pd.Series(x)
    sy: pd.Series = pd.Series(y)
    if sx.nunique() <= 1 or sy.nunique() <= 1:
        return float("nan")
    return float(sx.corr(sy, method="spearman"))


# ── Bootstrap Sharpe ──────────────────────────────────────────────────────────


def _bootstrap_sharpe_ci(
    returns: np.ndarray,
    n_boot: int = 1000,
    ci: float = 0.95,
    random_seed: int = 42,
) -> tuple[float, float]:
    """Return (lo, hi) bootstrap CI for annualized Sharpe."""
    rng = np.random.default_rng(random_seed)
    sharpes = []
    n = len(returns)
    for _ in range(n_boot):
        samp = rng.choice(returns, size=n, replace=True)
        std = np.std(samp, ddof=0)
        if std > 0:
            sharpes.append((np.mean(samp) / std) * np.sqrt(4.0))
    sharpes_arr = np.array(sharpes)
    alpha = (1.0 - ci) / 2.0
    return float(np.quantile(sharpes_arr, alpha)), float(np.quantile(sharpes_arr, 1.0 - alpha))


# ── Main backtest ─────────────────────────────────────────────────────────────


def run_fundamental_backtest(
    db_path: str = DEFAULT_DB_PATH,
    yf_start: str = YF_START,
    yf_end: str | None = None,
    factor_year_start: int = FACTOR_YEAR_START,
    factor_year_end: int = FACTOR_YEAR_END,
    mwu_eta: float = MWU_ETA,
    transaction_cost_bps: float = TRANSACTION_COST_BPS,
    post_event_bdays: int = POST_EARNINGS_BDAYS,
    min_valid_names: int = MIN_VALID_NAMES_PER_QUARTER,
) -> Dict[str, Any]:
    if yf_end is None:
        yf_end = datetime.now().strftime("%Y-%m-%d")

    qfac: pd.DataFrame = _load_quarterly_factors(db_path)
    if qfac.empty:
        print("========== Fundamental backtest: quarterly_factors empty; run factor_engineering ==========")
        return {"n_quarters": 0}

    qfac = qfac[(qfac["year"] >= factor_year_start) & (qfac["year"] <= factor_year_end)].copy()
    qfac["carrier"] = qfac["carrier"].astype(str)

    earnings_cal: pd.DataFrame = _load_earnings_calendar_from_db(
        db_path,
        year_start=factor_year_start,
        year_end=factor_year_end + 1,  # +1 to include Q4 earnings released next year
    )
    if earnings_cal.empty:
        print(f"========== Fundamental backtest: no rows in {EARNINGS_TABLE_NAME} ==========")
        return {"n_quarters": 0}

    prices: pd.DataFrame = load_or_fetch_adj_close(db_path, TICKERS, yf_start, yf_end)
    spy: pd.Series = prices[BENCHMARK].astype(float)

    # Precompute next-quarter formation dates for forward return windows
    sorted_quarters: List[tuple] = sorted(qfac.groupby(["year", "quarter"], sort=True).groups.keys())
    next_formation_date_map: Dict[tuple, pd.Timestamp | None] = {}
    for i, (yr_i, qt_i) in enumerate(sorted_quarters):
        if i + 1 < len(sorted_quarters):
            ny, nq = sorted_quarters[i + 1]
            next_formation_date_map[(yr_i, qt_i)] = _portfolio_formation_date(int(ny), int(nq))
        else:
            next_formation_date_map[(yr_i, qt_i)] = None

    mu: np.ndarray = np.ones(N_EXPERTS, dtype=float) / float(N_EXPERTS)
    quarterly_returns_gross: List[float] = []
    quarterly_returns_net: List[float] = []
    meta_rows: List[Dict[str, Any]] = []
    ic_records: List[Dict[str, Any]] = []
    last_year_printed: int = -1
    prev_W: np.ndarray = np.zeros(len(TICKERS), dtype=float)
    tc_per_unit: float = transaction_cost_bps / 10000.0

    for (yr, qt), g in qfac.groupby(["year", "quarter"], sort=True):
        g = g.loc[g["carrier"].isin(IATA_ORDER)].copy()
        g = g.drop_duplicates(subset=["carrier"], keep="first")
        if len(g) != len(IATA_ORDER):
            continue
        if g[FACTOR_COLS].isna().any().any():
            continue

        # Portfolio forms at quarter-end + 5 bdays and realizes return
        # into the post-earnings window of that same quarter's earnings release.
        formation_date: pd.Timestamp = _portfolio_formation_date(int(yr), int(qt))
        next_fd: pd.Timestamp | None = next_formation_date_map.get((yr, qt))
        if next_fd is None:
            continue

        realized_returns: Dict[str, float] = {}
        event_dates: Dict[str, pd.Timestamp] = {}
        for tkr in TICKERS:
            if tkr not in prices.columns:
                realized_returns[tkr] = float("nan")
                continue
            try:
                row = earnings_cal.loc[(tkr, int(yr), int(qt))]
            except KeyError:
                realized_returns[tkr] = float("nan")
                continue
            event_day: pd.Timestamp = pd.Timestamp(row["earnings_date"])
            event_dates[tkr] = event_day
            if event_day < formation_date:
                realized_returns[tkr] = float("nan")
                continue
            realized_returns[tkr] = _event_excess_return_after_entry(
                prices[tkr].astype(float),
                spy,
                formation_date,
                event_day,
                post_event_bdays=post_event_bdays,
                max_exit_date=next_fd,
            )

        realized_vec: np.ndarray = np.array([realized_returns[tk] for tk in TICKERS], dtype=float)
        valid_mask: np.ndarray = np.isfinite(realized_vec)
        if int(valid_mask.sum()) < int(min_valid_names):
            continue

        g_idx: pd.DataFrame = g.set_index("carrier").reindex(IATA_ORDER)
        fac_mat: np.ndarray = g_idx[FACTOR_COLS].astype(float).to_numpy().T
        factor_valid_mask: np.ndarray = np.all(np.isfinite(fac_mat), axis=0)
        valid_mask = valid_mask & factor_valid_mask
        if int(valid_mask.sum()) < int(min_valid_names):
            continue

        W_exp: np.ndarray = np.zeros((N_EXPERTS, len(TICKERS)), dtype=float)
        for k in range(N_EXPERTS):
            W_exp[k] = _expert_rank_weights(fac_mat[k], valid_mask=valid_mask)

        # Use previous quarter's MWU weights to form this quarter's portfolio.
        W_raw: np.ndarray = np.sum(mu[:, np.newaxis] * W_exp, axis=0)
        W: np.ndarray = _normalize_l1_zero_sum(W_raw)
        if float(np.sum(np.abs(W))) <= 0.0:
            continue

        port_r_gross: float = float(np.dot(W, np.nan_to_num(realized_vec, nan=0.0)))

        # Transaction cost: turnover × one-way cost (L1 distance of weight change)
        turnover: float = float(np.sum(np.abs(W - prev_W)))
        port_r_net: float = port_r_gross - turnover * tc_per_unit
        prev_W = W.copy()

        pos_mask: np.ndarray = W > 0
        neg_mask: np.ndarray = W < 0
        ret_long: float = float(np.sum(W[pos_mask] * np.nan_to_num(realized_vec[pos_mask], nan=0.0)))
        ret_short: float = float(np.sum(W[neg_mask] * np.nan_to_num(realized_vec[neg_mask], nan=0.0)))

        # After observing this quarter's realized returns, update expert weights for the next quarter.
        payoffs: np.ndarray = np.zeros(N_EXPERTS, dtype=float)
        for k in range(N_EXPERTS):
            payoffs[k] = float(np.dot(W_exp[k], np.nan_to_num(realized_vec, nan=0.0)))

        for k in range(N_EXPERTS):
            mu[k] *= float(np.exp(mwu_eta * payoffs[k]))
        mu = mu / np.sum(mu) if np.sum(mu) > 0 else np.ones(N_EXPERTS) / N_EXPERTS

        # IC uses tradable post-formation event returns as the realized outcome.
        for k, col in enumerate(FACTOR_COLS):
            fac_vals = fac_mat[k]
            ic_records.append(
                {
                    "year": int(yr),
                    "quarter": int(qt),
                    "factor": col,
                    "ic": _spearman_ic(fac_vals[valid_mask], realized_vec[valid_mask]),
                }
            )

        quarterly_returns_gross.append(port_r_gross)
        quarterly_returns_net.append(port_r_net)
        meta_rows.append(
            {
                "year": int(yr),
                "quarter": int(qt),
                "portfolio_return_gross": port_r_gross,
                "portfolio_return_net": port_r_net,
                "turnover": turnover,
                "long_contribution": ret_long,
                "short_contribution": ret_short,
                "mu_rpk": float(mu[0]),
                "mu_lf": float(mu[1]),
                "mu_ask": float(mu[2]),
                "mu_share": float(mu[3]),
                "n_valid_names": int(valid_mask.sum()),
            }
        )

        if int(qt) == 4 and int(yr) != last_year_printed:
            last_year_printed = int(yr)
            print(
                f"[MWU year-end {yr}] expert weights (normalized): RPK={mu[0]:.3f}, LF={mu[1]:.3f}, "
                f"ASK={mu[2]:.3f}, Share={mu[3]:.3f}"
            )

    if not quarterly_returns_gross:
        print("========== Fundamental backtest: no complete quarters ==========")
        return {"n_quarters": 0}

    # ── Performance metrics (gross) ────────────────────────────────────────────
    rets_gross: np.ndarray = np.array(quarterly_returns_gross, dtype=float)
    rets_net: np.ndarray = np.array(quarterly_returns_net, dtype=float)

    def _summary(rets: np.ndarray) -> Dict[str, float]:
        mean_q = float(rets.mean())
        std_q = float(rets.std(ddof=0))
        sharpe = float((mean_q / std_q) * np.sqrt(4.0)) if std_q > 0 else float("nan")
        win_rate = float((rets > 0).mean())
        cum_ret = float((1.0 + rets).prod() - 1.0)
        return {"mean_q": mean_q, "std_q": std_q, "sharpe": sharpe, "win_rate": win_rate, "cum_ret": cum_ret}

    gross = _summary(rets_gross)
    net = _summary(rets_net)

    # Bootstrap CI on gross Sharpe
    sharpe_lo, sharpe_hi = _bootstrap_sharpe_ci(rets_gross)

    # t-test: is gross mean quarterly return significantly > 0?
    t_stat, p_value = stats.ttest_1samp(rets_gross, 0.0)

    # ── IC analysis ───────────────────────────────────────────────────────────
    ic_df: pd.DataFrame = pd.DataFrame(ic_records)
    mean_ic: pd.Series = ic_df.groupby("factor")["ic"].mean()

    # IC t-test per factor (H0: IC = 0)
    ic_pvalues: Dict[str, float] = {}
    for fname in FACTOR_COLS:
        ic_vals = ic_df[ic_df["factor"] == fname]["ic"].dropna().to_numpy()
        if len(ic_vals) > 1:
            _, pv = stats.ttest_1samp(ic_vals, 0.0)
            ic_pvalues[fname] = float(pv)
        else:
            ic_pvalues[fname] = float("nan")

    # ── Factor correlation matrix ──────────────────────────────────────────────
    qfac_valid = qfac[FACTOR_COLS].dropna()
    factor_corr: pd.DataFrame = qfac_valid.corr(method="spearman")

    # ── Regime breakdown ──────────────────────────────────────────────────────
    regime_map = {
        "pre_covid": (2018, 2019),
        "covid": (2020, 2020),
        "post_covid": (2021, 2025),
    }
    regime_stats: Dict[str, Dict[str, float]] = {}
    detail_df: pd.DataFrame = pd.DataFrame(meta_rows)
    for regime, (y_lo, y_hi) in regime_map.items():
        mask = (detail_df["year"] >= y_lo) & (detail_df["year"] <= y_hi)
        r_sub = rets_gross[mask.to_numpy()]
        if len(r_sub) > 0:
            regime_stats[regime] = _summary(r_sub)

    # ── Print report ──────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(" Fundamental MWU · quarterly returns")
    print("=" * 72)
    disp: pd.DataFrame = detail_df.copy()
    for col in ("portfolio_return_gross", "portfolio_return_net", "long_contribution", "short_contribution"):
        disp[col] = disp[col].map(lambda x: f"{float(x):.4%}")
    disp["turnover"] = disp["turnover"].map(lambda x: f"{float(x):.3f}")
    print(disp.to_string(index=False))
    print("-" * 72)
    print(" Mean IC by factor (Spearman vs tradable event return, 6-airline cross-section)")
    for fname in FACTOR_COLS:
        v: float = float(mean_ic.get(fname, float("nan")))
        pv: float = ic_pvalues.get(fname, float("nan"))
        print(f"  {fname}: IC={v:.4f}  p={pv:.3f}")
    print("-" * 72)
    print(" Factor correlation matrix (Spearman)")
    print(factor_corr.round(3).to_string())
    print("=" * 72)
    print(f"[GROSS] Cumulative: {gross['cum_ret']:.4%} | Mean Q: {gross['mean_q']:.4%} | "
          f"Win: {gross['win_rate']:.2%} | Ann.Sharpe: {gross['sharpe']:.4f} "
          f"[95% CI {sharpe_lo:.3f}, {sharpe_hi:.3f}]")
    print(f"        t-stat={t_stat:.3f}, p={p_value:.3f} (H0: mean return = 0)")
    print(f"[NET]   Cumulative: {net['cum_ret']:.4%} | Mean Q: {net['mean_q']:.4%} | "
          f"Win: {net['win_rate']:.2%} | Ann.Sharpe: {net['sharpe']:.4f}  "
          f"(TC={transaction_cost_bps:.0f}bps/side)")
    print("-" * 72)
    print(" Regime breakdown (gross):")
    for regime, rs in regime_stats.items():
        print(f"  {regime}: cum={rs['cum_ret']:.4%} sharpe={rs['sharpe']:.3f} win={rs['win_rate']:.2%}")
    print("=" * 72 + "\n")

    return {
        "n_quarters": int(len(rets_gross)),
        # Gross
        "cumulative_return": gross["cum_ret"],
        "mean_quarterly_return": gross["mean_q"],
        "win_rate": gross["win_rate"],
        "annualized_sharpe": gross["sharpe"],
        "sharpe_ci_lo": sharpe_lo,
        "sharpe_ci_hi": sharpe_hi,
        "t_stat": t_stat,
        "p_value": p_value,
        # Net
        "cumulative_return_net": net["cum_ret"],
        "mean_quarterly_return_net": net["mean_q"],
        "annualized_sharpe_net": net["sharpe"],
        # Detail
        "mean_quarterly_long": float(detail_df["long_contribution"].mean()),
        "mean_quarterly_short": float(detail_df["short_contribution"].mean()),
        "quarterly_detail": detail_df,
        "ic_by_factor": mean_ic.to_dict(),
        "ic_pvalues": ic_pvalues,
        "ic_detail": ic_df,
        "factor_corr": factor_corr,
        "regime_stats": regime_stats,
    }


if __name__ == "__main__":
    raise SystemExit("Call run_fundamental_backtest(db_path=...) from main.py")
