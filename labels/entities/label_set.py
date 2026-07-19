"""LabelSet aggregate — one side per set."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from labels.entities.label import Label, Side


@dataclass(frozen=True, slots=True)
class LabelSet:
    symbol: str
    timeframe: str
    strategy: str
    side: Side
    label_version: str
    created_at: datetime
    labels: tuple[Label, ...]

    def __post_init__(self) -> None:
        if not self.symbol.strip():
            raise ValueError("symbol must be non-empty")
        if not self.timeframe.strip():
            raise ValueError("timeframe must be non-empty")
        if not self.strategy.strip():
            raise ValueError("strategy must be non-empty")
        if self.side not in {"long", "short"}:
            raise ValueError(f"invalid side: {self.side!r}")
        object.__setattr__(self, "symbol", self.symbol.strip().upper())
        object.__setattr__(self, "timeframe", self.timeframe.strip().upper())
        object.__setattr__(self, "strategy", self.strategy.strip().lower())
        object.__setattr__(self, "side", self.side.strip().lower())  # type: ignore[arg-type]
        if not isinstance(self.labels, tuple):
            object.__setattr__(self, "labels", tuple(self.labels))
        for lab in self.labels:
            if lab.side != self.side:
                raise ValueError(
                    f"label side {lab.side!r} != LabelSet.side {self.side!r}"
                )

    @property
    def size(self) -> int:
        return len(self.labels)

    @property
    def class_counts(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for lab in self.labels:
            counts[lab.label] = counts.get(lab.label, 0) + 1
        return counts
