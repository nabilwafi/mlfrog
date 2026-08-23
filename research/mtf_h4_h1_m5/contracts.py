"""Explicit parent-child signal contracts for H4 → H1 → M5 research."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

H4Trend = Literal["bullish", "bearish", "neutral"]
H4Structure = Literal["bullish", "bearish", "neutral"]
H4Volatility = Literal["low", "normal", "high"]
H4Strength = Literal["weak", "normal", "strong"]
H4Direction = Literal["long", "short", "neutral"]

M5Action = Literal["EXECUTE", "WAIT", "INVALIDATE"]


@dataclass(frozen=True)
class H4Signal:
    """Deterministic H4 context — no probability."""

    direction: H4Direction
    trend: H4Trend
    structure: H4Structure
    volatility: H4Volatility
    strength: H4Strength
    signal_timestamp: str  # ISO UTC — H4 bar open
    available_timestamp: str  # ISO UTC — when state is knowable (bar close)

    def to_features(self) -> dict[str, float]:
        """Numeric encoding for H1 ML (no raw H4 OHLC)."""
        dir_map = {"short": -1.0, "neutral": 0.0, "long": 1.0}
        tri_map = {"bearish": -1.0, "neutral": 0.0, "bullish": 1.0}
        vol_map = {"low": 0.0, "normal": 1.0, "high": 2.0}
        str_map = {"weak": 0.0, "normal": 1.0, "strong": 2.0}
        return {
            "h4_sig_direction": dir_map[self.direction],
            "h4_sig_trend": tri_map[self.trend],
            "h4_sig_structure": tri_map[self.structure],
            "h4_sig_vol": vol_map[self.volatility],
            "h4_sig_strength": str_map[self.strength],
        }

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


H4_SIGNAL_FEATURE_NAMES: tuple[str, ...] = (
    "h4_sig_direction",
    "h4_sig_trend",
    "h4_sig_structure",
    "h4_sig_vol",
    "h4_sig_strength",
)


@dataclass(frozen=True)
class D1Signal:
    """Deterministic D1 context — no probability."""

    direction: H4Direction
    trend: H4Trend
    structure: H4Structure
    volatility: H4Volatility
    strength: H4Strength
    signal_timestamp: str
    available_timestamp: str

    def to_features(self) -> dict[str, float]:
        dir_map = {"short": -1.0, "neutral": 0.0, "long": 1.0}
        tri_map = {"bearish": -1.0, "neutral": 0.0, "bullish": 1.0}
        vol_map = {"low": 0.0, "normal": 1.0, "high": 2.0}
        str_map = {"weak": 0.0, "normal": 1.0, "strong": 2.0}
        return {
            "d1_sig_direction": dir_map[self.direction],
            "d1_sig_trend": tri_map[self.trend],
            "d1_sig_structure": tri_map[self.structure],
            "d1_sig_vol": vol_map[self.volatility],
            "d1_sig_strength": str_map[self.strength],
        }

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


D1_SIGNAL_FEATURE_NAMES: tuple[str, ...] = (
    "d1_sig_direction",
    "d1_sig_trend",
    "d1_sig_structure",
    "d1_sig_vol",
    "d1_sig_strength",
)

@dataclass(frozen=True)
class H1Signal:
    """H1 ML trade signal."""

    direction: Literal["long", "short"]
    probability: float
    signal_timestamp: str
    available_timestamp: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class M5ExecutionSignal:
    """Deterministic M5 execution state — no probability."""

    action: M5Action
    reason: str
    signal_timestamp: str
    available_timestamp: str
    delay_m5_bars: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
