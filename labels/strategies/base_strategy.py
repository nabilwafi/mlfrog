"""Label strategy ABC."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from data.entities.market_data import MarketData
from labels.entities.label import Label


class BaseLabelStrategy(ABC):
    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params: dict[str, Any] = dict(params or {})

    @abstractmethod
    def name(self) -> str:
        """Registry key, e.g. 'triple_barrier'."""

    @abstractmethod
    def generate(self, market_data: MarketData) -> tuple[Label, ...]:
        """Produce labels from MarketData (causal; no look-ahead beyond defined barriers)."""
