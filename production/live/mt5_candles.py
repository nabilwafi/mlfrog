"""Fetch OHLCV from MetaTrader 5 into pandas (read-only)."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from data.connectors.mt5_constants import resolve_timeframe
from data.entities.candle import Candle
from data.entities.market_data import MarketData
from production.monitoring import mt5_session

logger = logging.getLogger(__name__)

_BAR_COLS = ["timestamp", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]
_IPC_LOST = -10004

_TF_SECONDS = {
    "M1": 60,
    "M5": 300,
    "M15": 900,
    "M30": 1800,
    "H1": 3600,
    "H4": 14400,
    "D1": 86400,
}


def _tf_seconds(timeframe: str) -> int:
    return int(_TF_SECONDS.get(str(timeframe).upper(), 3600))


def rates_to_frame(rates: Any, *, tz: str = "UTC") -> pd.DataFrame:
    if rates is None or len(rates) == 0:
        return pd.DataFrame(columns=_BAR_COLS)
    df = pd.DataFrame(rates)
    df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.rename(
        columns={
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "tick_volume": "tick_volume",
            "real_volume": "real_volume",
            "spread": "spread",
        }
    )
    return df[_BAR_COLS].sort_values("timestamp").reset_index(drop=True)


def frame_to_market(symbol: str, timeframe: str, frame: pd.DataFrame, *, tz: str = "UTC") -> MarketData:
    candles: list[Candle] = []
    for row in frame.itertuples(index=False):
        ts = pd.Timestamp(row.timestamp)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        candles.append(
            Candle(
                timestamp=ts.to_pydatetime(),
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                tick_volume=float(row.tick_volume or 0),
                spread=float(row.spread or 0),
                real_volume=float(row.real_volume or 0),
            )
        )
    return MarketData(symbol=symbol, timeframe=timeframe, timezone=tz, candles=tuple(candles))


class MT5CandleFeed:
    """Persistent MT5 session for live bar polling (shared IPC)."""

    def __init__(self, cfg: dict[str, Any], *, symbol: str, log: logging.Logger | None = None) -> None:
        self._cfg = cfg
        self._symbol = str(symbol)  # HF: XAUUSDc is case-sensitive — do NOT .upper()
        self._log = log or logger
        self._symbol_selected = False

    @property
    def symbol(self) -> str:
        return self._symbol

    def connect(self) -> None:
        if not mt5_session.ensure_connected(self._cfg):
            raise RuntimeError("MT5 unavailable — start terminal and enable algo trading")
        if not self._symbol_selected:
            mt5_session.select_symbol(self._symbol)
            self._symbol_selected = True

    def disconnect(self) -> None:
        # Shared session — shutdown only on app exit (run_paper_trading finally).
        self._symbol_selected = False

    def fetch(self, timeframe: str, *, count: int) -> pd.DataFrame:
        import MetaTrader5 as mt5

        if not mt5_session.ensure_connected(self._cfg):
            logger.warning("mt5_fetch_skipped reason=not_connected tf=%s", timeframe)
            return pd.DataFrame(columns=_BAR_COLS)
        if not self._symbol_selected:
            mt5_session.select_symbol(self._symbol)
            self._symbol_selected = True

        tf = resolve_timeframe(timeframe)
        for attempt in (1, 2):
            # ponytail: from_pos avoids datetime.now() local-vs-UTC footgun after weekend gaps
            rates = mt5.copy_rates_from_pos(self._symbol, tf, 0, int(count))
            if rates is not None:
                return rates_to_frame(rates)
            err = mt5.last_error()
            code = err[0] if isinstance(err, tuple) and err else None
            self._log.warning("copy_rates_from_pos_failed tf=%s attempt=%s err=%s", timeframe, attempt, err)
            if attempt == 1 and code == _IPC_LOST:
                mt5_session.ensure_connected(self._cfg)
                continue
            break
        return pd.DataFrame(columns=_BAR_COLS)

    def latest_closed_bar(self, timeframe: str = "H1") -> pd.DataFrame | None:
        """
        Return the most recently *closed* bar.
        MT5 timestamps are bar OPEN time. While a bar is still forming it is
        the last row — skip it. When the market is closed (weekend/gap), the
        last row is already closed, so use it (do not skip to [-2]).
        """
        frame = self.fetch(timeframe, count=5)
        if frame.empty:
            return None
        last = frame.iloc[-1]
        ts = pd.Timestamp(last["timestamp"])
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        bar_end = ts + pd.Timedelta(seconds=_tf_seconds(timeframe))
        now = pd.Timestamp.now(tz="UTC")
        if bar_end <= now:
            # last bar already fully closed (weekend / holiday / exact boundary)
            return frame.iloc[[-1]].reset_index(drop=True)
        if len(frame) < 2:
            return None
        return frame.iloc[[-2]].reset_index(drop=True)


def bar_age_seconds(ts: pd.Timestamp, timeframe: str, *, now: pd.Timestamp | None = None) -> float:
    """Seconds since bar close (open_ts + tf)."""
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    n = now if now is not None else pd.Timestamp.now(tz="UTC")
    bar_end = t + pd.Timedelta(seconds=_tf_seconds(timeframe))
    return float((n - bar_end).total_seconds())


def is_stale_closed_bar(ts: pd.Timestamp, timeframe: str, *, now: pd.Timestamp | None = None) -> bool:
    """True when bar closed long ago → market closed / weekend gap (not a live signal)."""
    tf = _tf_seconds(timeframe)
    # > 2 bars old = gap / closed session (H1 → stale after ~2h)
    return bar_age_seconds(ts, timeframe, now=now) > float(tf * 2)
