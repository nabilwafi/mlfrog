"""Indicator ABC — pluggable feature generators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class BaseIndicator(ABC):
    """One indicator (or indicator family) that reads OHLCV columns and emits Series/DataFrame."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params: dict[str, Any] = dict(params or {})

    @abstractmethod
    def name(self) -> str:
        """Stable registry key, e.g. 'ema', 'rsi'."""

    @abstractmethod
    def required_columns(self) -> list[str]:
        """OHLCV column names required on the input frame."""

    @abstractmethod
    def calculate(self, df: pd.DataFrame) -> pd.Series | pd.DataFrame:
        """Return aligned Series or multi-column DataFrame (same index as df)."""

    def feature_prefix(self) -> str:
        """Default output name prefix; override when params encode identity."""
        return self.name()

    def validate_columns(self, df: pd.DataFrame) -> None:
        missing = [c for c in self.required_columns() if c not in df.columns]
        if missing:
            raise ValueError(f"{self.name()}: missing columns {missing}")
