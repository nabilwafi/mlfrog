"""Step 4 — Market State (descriptive only; no trade gate)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class MarketState:
    trend: str = "unknown"
    volatility: str = "unknown"
    momentum: str = "unknown"
    session: str = "unknown"
    structure: str = "unknown"
    regime_raw: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _session_from_row(row: Any) -> str:
    get = row.get if hasattr(row, "get") else lambda k, d=None: row[k] if k in row.index else d
    if float(get("session_london_ny_overlap", 0) or 0) >= 0.5:
        return "overlap"
    if float(get("session_london", 0) or 0) >= 0.5:
        return "london"
    if float(get("session_newyork", 0) or 0) >= 0.5:
        return "newyork"
    if float(get("session_asia", 0) or 0) >= 0.5:
        return "asia"
    s = str(get("session", "") or "")
    return s if s and s != "unknown" else "other"


def _trend_from_regime(regime: str) -> str:
    r = (regime or "").lower()
    if "bull" in r:
        return "bull"
    if "bear" in r:
        return "bear"
    if "side" in r or "compress" in r:
        return "sideways"
    return "unknown"


def _structure_from_regime(regime: str) -> str:
    r = (regime or "").lower()
    if "compress" in r:
        return "compression"
    if "side" in r:
        return "range"
    if "bull" in r or "bear" in r:
        return "trend"
    return "unknown"


def _vol_bucket(row: Any) -> str:
    get = row.get if hasattr(row, "get") else lambda k, d=None: row[k] if k in row.index else d
    p = get("atr_percentile_252", None)
    if p is None:
        p = get("volatility_rank", None)
    if p is None or (isinstance(p, float) and p != p):
        return "unknown"
    pf = float(p)
    if pf >= 0.66:
        return "high"
    if pf <= 0.33:
        return "low"
    return "medium"


def _momentum(row: Any) -> str:
    get = row.get if hasattr(row, "get") else lambda k, d=None: row[k] if k in row.index else d
    slope = get("d1_ema_slope", None)
    if slope is None or (isinstance(slope, float) and slope != slope):
        return "unknown"
    s = abs(float(slope))
    if s < 1e-6:
        return "weak"
    if s > 1e-4:
        return "strong"
    return "normal"


def infer_market_state(row: Any) -> MarketState:
    get = row.get if hasattr(row, "get") else lambda k, d=None: row[k] if k in row.index else d
    regime = str(get("d1_regime", "unknown") or "unknown")
    return MarketState(
        trend=_trend_from_regime(regime),
        volatility=_vol_bucket(row),
        momentum=_momentum(row),
        session=_session_from_row(row),
        structure=_structure_from_regime(regime),
        regime_raw=regime,
    )
