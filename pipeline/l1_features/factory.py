"""L1 Feature Factory — wraps LiveFeatureBuilder (same frozen feature vector)."""

from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd

from production.live.features import LiveFeatureBuilder


class FeatureFactory:
    """Thin wrap so paper/backtest/live share one L1 entrypoint."""

    def __init__(self, cfg: dict[str, Any], *, symbol: str, timezone: str = "UTC") -> None:
        self._builder = LiveFeatureBuilder(cfg, symbol=symbol, timezone=timezone)

    def build_panel(
        self,
        *,
        h1: pd.DataFrame,
        h4: pd.DataFrame,
        d1: pd.DataFrame,
        m5: pd.DataFrame,
    ) -> pd.DataFrame:
        return self._builder.build_panel(h1=h1, h4=h4, d1=d1, m5=m5)

    @staticmethod
    def features_hash(row: pd.Series | dict[str, Any], cols: list[str] | None = None) -> str:
        if isinstance(row, dict):
            keys = cols or [c for c in row.keys() if c != "timestamp"]
            parts = [f"{c}={row.get(c)}" for c in sorted(keys)]
        else:
            use = cols or [c for c in row.index if c != "timestamp"]
            parts = [f"{c}={row.get(c)}" for c in sorted(use)]
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]
