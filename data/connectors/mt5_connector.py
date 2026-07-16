"""MetaTrader 5 connection and bar data access."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import MetaTrader5 as mt5
import pandas as pd

_BAR_COLUMNS = [
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Tick Volume",
    "Volume",
    "Spread",
]


class MT5Connector:
    def __init__(self, cfg: dict[str, Any], log: logging.Logger) -> None:
        self._cfg = cfg.get("mt5", {})
        self._log = log
        self._connected = False

    def __enter__(self) -> MT5Connector:
        self.connect()
        return self

    def __exit__(self, *_: object) -> None:
        self.disconnect()

    def connect(self) -> None:
        if self._connected:
            return

        path = self._cfg.get("path") or None
        if not mt5.initialize(path=path):
            raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")

        login = int(self._cfg.get("login") or 0)
        if login:
            password = self._cfg.get("password", "")
            server = self._cfg.get("server", "")
            if not mt5.login(login, password=password, server=server):
                raise RuntimeError(f"MT5 login failed: {mt5.last_error()}")

        info = mt5.terminal_info()
        if info is None:
            mt5.shutdown()
            raise RuntimeError("MT5 terminal info unavailable")

        self._connected = True
        self._log.info("MT5 connected | build=%s | company=%s", info.build, info.company)

    def disconnect(self) -> None:
        if not self._connected:
            return
        mt5.shutdown()
        self._connected = False
        self._log.info("MT5 shutdown")

    def select_symbol(self, symbol: str) -> None:
        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"symbol_select({symbol!r}) failed: {mt5.last_error()}")

    def copy_rates_range(
        self,
        symbol: str,
        timeframe: int,
        date_from: datetime,
        date_to: datetime,
    ) -> pd.DataFrame:
        # MT5 Python bridge is picky about datetime params.
        # Ensure we pass naive datetimes (no tzinfo) and strip microseconds.
        if date_from.tzinfo is not None:
            date_from = date_from.replace(tzinfo=None)
        if date_to.tzinfo is not None:
            date_to = date_to.replace(tzinfo=None)
        date_from = date_from.replace(microsecond=0)
        date_to = date_to.replace(microsecond=0)

        # Defensive clamp: MT5 may reject future timestamps.
        now = datetime.now()
        if date_to > now:
            self._log.warning(
                "Clamping date_to to now to avoid MT5 invalid params | date_to=%s now=%s",
                date_to,
                now,
            )
            date_to = now

        if date_from >= date_to:
            self._log.warning(
                "Empty/invalid range for copy_rates_range | date_from=%s date_to=%s",
                date_from,
                date_to,
            )
            return pd.DataFrame()

        rates = mt5.copy_rates_range(symbol, timeframe, date_from, date_to)
        if rates is None:
            raise RuntimeError(f"copy_rates_range failed: {mt5.last_error()}")
        if len(rates) == 0:
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        # Use UTC explicitly so CSV always uses UTC timestamps.
        df["Date"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "tick_volume": "Tick Volume",
                "real_volume": "Volume",
                "spread": "Spread",
            }
        )
        return df[_BAR_COLUMNS]

    def copy_rates_from(
        self,
        symbol: str,
        timeframe: int,
        date_from: datetime,
        count: int,
    ) -> pd.DataFrame:
        """Fetch up to `count` bars starting at date_from (inclusive). Read-only."""
        if date_from.tzinfo is not None:
            date_from = date_from.replace(tzinfo=None)
        date_from = date_from.replace(microsecond=0)
        rates = mt5.copy_rates_from(symbol, timeframe, date_from, int(count))
        if rates is None:
            raise RuntimeError(f"copy_rates_from failed: {mt5.last_error()}")
        if len(rates) == 0:
            return pd.DataFrame()
        df = pd.DataFrame(rates)
        df["Date"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "tick_volume": "Tick Volume",
                "real_volume": "Volume",
                "spread": "Spread",
            }
        )
        return df[_BAR_COLUMNS]
