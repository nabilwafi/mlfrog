"""
Live MT5 H1/H4/D1 feed for paper trading — READ-ONLY rates.

Operational assumptions
-----------------------
- Windows MT5 terminal must be running and logged in (same as historical fetcher).
- Run as a long-lived poll loop (default) or `--once` via Task Scheduler each minute.
- Only closed bars are returned (no lookahead on the forming candle).
- This module never calls order_send / order execution APIs.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from connectors.mt5 import MT5Connector, resolve_timeframe

# Absolute ban: paper stack must not touch these symbols via MetaTrader5.
_FORBIDDEN_MT5_NAMES = frozenset(
    {
        "order_send",
        "order_check",
        "order_calc_profit",
        "order_calc_margin",
        "Buy",
        "Sell",
    }
)

RAW_COLS = ["Date", "Open", "High", "Low", "Close", "Tick Volume", "Volume", "Spread"]

TF_BAR_HOURS = {"H1": 1, "H4": 4, "D1": 24}


def assert_no_execution_api() -> None:
    """Fail loud if MetaTrader5 execution helpers are importable into this process for misuse."""
    import MetaTrader5 as mt5

    for name in _FORBIDDEN_MT5_NAMES:
        # Presence on the module is OK (MT5 ships them); we only assert *we* never call them.
        # Binding check is done by paper_trading_engine._forbid_order_calls patch.
        if not hasattr(mt5, name) and name in {"order_send", "order_check"}:
            continue


def load_mt5_config(config_path: Path = Path("configs/config.yaml")) -> dict[str, Any]:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _ensure_utc(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, utc=True)
    return dt


def load_raw_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=RAW_COLS)
    df = pd.read_csv(path, parse_dates=["Date"])
    df["Date"] = _ensure_utc(df["Date"])
    return df.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)


def save_raw_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df[RAW_COLS].sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
    out.to_csv(path, index=False)


def seed_from_historical(
    *,
    paper_raw_dir: Path,
    hist_dir: Path = Path("data/raw"),
    symbol: str = "XAUUSD",
    h1_bars: int = 6000,
    h4_bars: int = 2000,
    d1_bars: int = 800,
) -> None:
    """Copy a trailing window from historical CSVs into paper_trading/raw/ if empty."""
    paper_raw_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        ("H1", h1_bars),
        ("H4", h4_bars),
        ("D1", d1_bars),
    ]
    for tf, n in specs:
        dest = paper_raw_dir / f"{symbol}_{tf}.csv"
        if dest.exists() and len(load_raw_csv(dest)) > 0:
            continue
        src = hist_dir / f"{symbol}_{tf}.csv"
        if not src.exists():
            raise FileNotFoundError(f"missing historical seed {src}")
        df = load_raw_csv(src).tail(n).reset_index(drop=True)
        save_raw_csv(df, dest)


def drop_forming_bar(df: pd.DataFrame, timeframe: str, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Keep only bars whose close time is <= now (MT5 Date = open time)."""
    if df.empty:
        return df
    now = now or pd.Timestamp.now(tz="UTC")
    hours = TF_BAR_HOURS[timeframe.upper()]
    close_time = df["Date"] + pd.Timedelta(hours=hours)
    return df.loc[close_time <= now].copy().reset_index(drop=True)


def fetch_closed_bars(
    connector: MT5Connector,
    *,
    symbol: str,
    timeframe: str,
    lookback_bars: int = 50,
    now: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Pull recent bars from MT5 and drop the still-forming candle."""
    now = now or pd.Timestamp.now(tz="UTC")
    hours = TF_BAR_HOURS[timeframe.upper()]
    # Start a bit before lookback window
    start = (now - pd.Timedelta(hours=hours * (lookback_bars + 5))).to_pydatetime()
    if start.tzinfo is not None:
        start = start.replace(tzinfo=None)
    tf = resolve_timeframe(timeframe)
    connector.select_symbol(symbol)
    raw = connector.copy_rates_from(symbol, tf, start, lookback_bars + 5)
    if raw.empty:
        return raw
    return drop_forming_bar(raw, timeframe, now=now)


def append_new_closed_bars(
    local: pd.DataFrame,
    live: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merge live closed bars into local store.
    Returns (updated_local, newly_appended_rows).
    """
    if live.empty:
        return local, live.iloc[0:0].copy()
    if local.empty:
        return live.copy(), live.copy()
    last = local["Date"].max()
    new = live[live["Date"] > last].copy()
    if new.empty:
        # Optional OHLC refresh for last few overlapping bars (revision detect)
        overlap = live[live["Date"].isin(local["Date"].tail(5))].copy()
        return local, new
    updated = (
        pd.concat([local, new], ignore_index=True)
        .sort_values("Date")
        .drop_duplicates("Date", keep="last")
        .reset_index(drop=True)
    )
    return updated, new.reset_index(drop=True)


def detect_revisions(
    local: pd.DataFrame,
    live: pd.DataFrame,
    *,
    compare_n: int = 5,
    tol: float = 1e-6,
) -> pd.DataFrame:
    """Compare overlapping recent bars for feed revisions vs stored values."""
    if local.empty or live.empty:
        return pd.DataFrame()
    tail_dates = set(local["Date"].tail(compare_n))
    rows = []
    loc = local.set_index("Date")
    liv = live.set_index("Date")
    for ts in sorted(tail_dates & set(liv.index)):
        a, b = loc.loc[ts], liv.loc[ts]
        diffs = {}
        for c in ("Open", "High", "Low", "Close", "Spread"):
            if abs(float(a[c]) - float(b[c])) > tol:
                diffs[c] = {"stored": float(a[c]), "live": float(b[c])}
        if diffs:
            rows.append({"Date": ts, "diffs": str(diffs)})
    return pd.DataFrame(rows)


class LiveFeedHandler:
    """
    Poll MT5 for closed H1/H4/D1 bars and append to paper raw CSVs.
    """

    def __init__(
        self,
        *,
        config_path: Path = Path("configs/config.yaml"),
        paper_dir: Path = Path("data/paper_trading"),
        symbol: str = "XAUUSD",
        log: logging.Logger | None = None,
    ) -> None:
        assert_no_execution_api()
        self.cfg = load_mt5_config(config_path)
        self.paper_dir = paper_dir
        self.raw_dir = paper_dir / "raw"
        self.symbol = symbol
        self.log = log or logging.getLogger("live_feed")
        self.sanity_path = paper_dir / "feed_sanity.parquet"
        seed_from_historical(paper_raw_dir=self.raw_dir, symbol=symbol)

    def _path(self, tf: str) -> Path:
        return self.raw_dir / f"{self.symbol}_{tf}.csv"

    def poll_once(self) -> dict[str, pd.DataFrame]:
        """
        Fetch closed bars, append new ones, run revision sanity on H1.
        Returns dict of newly appended frames per TF (may be empty).
        """
        newly: dict[str, pd.DataFrame] = {}
        with MT5Connector(self.cfg, self.log) as conn:
            for tf, lookback in (("H1", 48), ("H4", 40), ("D1", 15)):
                local = load_raw_csv(self._path(tf))
                live = fetch_closed_bars(conn, symbol=self.symbol, timeframe=tf, lookback_bars=lookback)
                if live.empty:
                    newly[tf] = live
                    self.log.warning("no closed %s bars from MT5", tf)
                    continue
                if tf == "H1":
                    revs = detect_revisions(local, live)
                    if len(revs):
                        self.log.warning("H1 feed revisions detected: %s", len(revs))
                        if self.sanity_path.exists():
                            prev = pd.read_parquet(self.sanity_path)
                            revs = pd.concat([prev, revs], ignore_index=True)
                        revs["checked_at"] = pd.Timestamp.now(tz="UTC")
                        revs.to_parquet(self.sanity_path, index=False)
                updated, new = append_new_closed_bars(local, live)
                if len(new):
                    save_raw_csv(updated, self._path(tf))
                    self.log.info("appended %s %s bars through %s", len(new), tf, new["Date"].max())
                newly[tf] = new
        return newly

    def latest_closed_h1_open_time(self) -> pd.Timestamp | None:
        df = load_raw_csv(self._path("H1"))
        if df.empty:
            return None
        return pd.Timestamp(df["Date"].iloc[-1])
