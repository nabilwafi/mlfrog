"""FeatureSet aggregate."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from features.entities.feature import Feature


@dataclass(frozen=True, slots=True)
class FeatureSet:
    symbol: str
    timeframe: str
    feature_version: str
    pipeline_version: str
    created_at: datetime
    features: tuple[Feature, ...]
    timestamps: tuple[datetime, ...]

    def __post_init__(self) -> None:
        if not self.symbol.strip():
            raise ValueError("symbol must be non-empty")
        if not self.timeframe.strip():
            raise ValueError("timeframe must be non-empty")
        object.__setattr__(self, "symbol", self.symbol.strip().upper())
        object.__setattr__(self, "timeframe", self.timeframe.strip().upper())
        if not isinstance(self.features, tuple):
            object.__setattr__(self, "features", tuple(self.features))
        if not isinstance(self.timestamps, tuple):
            object.__setattr__(self, "timestamps", tuple(self.timestamps))
        n = len(self.timestamps)
        for f in self.features:
            if len(f.values) != n:
                raise ValueError(
                    f"feature {f.name!r} length {len(f.values)} != timestamps {n}"
                )

    @property
    def feature_count(self) -> int:
        return len(self.features)

    @property
    def row_count(self) -> int:
        return len(self.timestamps)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.features)

    def to_frame_dict(self) -> dict[str, object]:
        out: dict[str, object] = {"timestamp": [t.isoformat() for t in self.timestamps]}
        for f in self.features:
            out[f.name] = list(f.values)
        return out
