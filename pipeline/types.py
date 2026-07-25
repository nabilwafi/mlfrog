"""Shared layer I/O types for L1–L6."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class MarketState:
    """L2 — descriptive only; no trade decision."""

    trend: str = "unknown"  # bull | bear | sideways | ...
    volatility: str = "unknown"  # low | medium | high
    momentum: str = "unknown"  # weak | normal | strong
    session: str = "unknown"  # asia | london | overlap | newyork | other
    structure: str = "unknown"  # breakout | range | pullback | compression | ...
    regime_raw: str = "unknown"  # d1_regime passthrough

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PrimaryOut:
    side: str  # LONG | SHORT | NONE
    raw_score: float
    probability: float | None = None
    reason: str = ""


@dataclass
class MetaOut:
    trade: bool
    edge_score: float
    expected_r: float
    reason: str = ""


@dataclass
class RiskOut:
    entry: float
    stop: float
    target: float
    risk_pct: float
    rr: float
    lot: float
    atr: float
    side: str


@dataclass
class PortfolioDecision:
    accept: bool
    reason: str
    trade: dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceRecord:
    signal_id: str
    time: datetime
    features_hash: str
    market_state: dict[str, Any]
    primary: dict[str, Any]
    meta: dict[str, Any]
    risk: dict[str, Any] | None
    portfolio: dict[str, Any]
    entry: float | None = None
    exit: float | None = None
    pnl: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
