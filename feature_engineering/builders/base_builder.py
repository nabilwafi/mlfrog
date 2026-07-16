"""Builder ABC — each builder emits Features with metadata."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from feature_engineering.entities.feature import Feature


class BaseFeatureBuilder(ABC):
    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params: dict[str, Any] = dict(params or {})

    @abstractmethod
    def name(self) -> str:
        """Registry key."""

    @abstractmethod
    def build(self, frame: pd.DataFrame, *, context: dict[str, Any]) -> list[Feature]:
        """
        frame: OHLCV (+ optional precomputed intermediates in context['intermediates']).
        """
