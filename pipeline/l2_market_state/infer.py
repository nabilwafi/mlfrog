"""L2 Market State — describe conditions only; no trade decision / no new ML."""

from __future__ import annotations

from typing import Any

import pandas as pd

from pipeline.types import MarketState


def _session_from_row(row: pd.Series | dict[str, Any]) -> str:
    get = row.get if hasattr(row, "get") else lambda k, d=None: row[k] if k in row else d  # type: ignore[index]
    if float(get("session_london_ny_overlap", 0) or 0) >= 0.5:
        return "overlap"
    if float(get("session_london", 0) or 0) >= 0.5:
        return "london"
    if float(get("session_newyork", 0) or 0) >= 0.5:
        return "newyork"
    if float(get("session_asia", 0) or 0) >= 0.5:
        return "asia"
    # explicit label already set
    sess = str(get("session", "") or "")
    if sess and sess != "unknown":
        return sess
    return "other"


def _trend_from_regime(regime: str) -> str:
    r = (regime or "").lower()
    if "bull" in r:
        return "bull"
    if "bear" in r:
        return "bear"
    if "side" in r:
        return "sideways"
    if "compress" in r:
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


def _vol_bucket(row: pd.Series | dict[str, Any]) -> str:
    get = row.get if hasattr(row, "get") else lambda k, d=None: row[k] if k in row else d  # type: ignore[index]
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


def _momentum(row: pd.Series | dict[str, Any]) -> str:
    get = row.get if hasattr(row, "get") else lambda k, d=None: row[k] if k in row else d  # type: ignore[index]
    # ponytail: reuse d1_ema_slope magnitude when present
    slope = get("d1_ema_slope", None)
    if slope is None or (isinstance(slope, float) and slope != slope):
        return "unknown"
    s = abs(float(slope))
    if s < 1e-6:
        return "weak"
    if s > 1e-4:
        return "strong"
    return "normal"


def infer_market_state(
    row: pd.Series | dict[str, Any] | None = None,
    *,
    session: str | None = None,
    regime: str | None = None,
) -> MarketState:
    row = row if row is not None else {}
    get = row.get if hasattr(row, "get") else lambda k, d=None: (row[k] if k in row else d)  # type: ignore[index]
    regime_raw = str(regime if regime is not None else (get("d1_regime", "unknown") or "unknown"))
    sess = session if session is not None else _session_from_row(row)
    return MarketState(
        trend=_trend_from_regime(regime_raw),
        volatility=_vol_bucket(row),
        momentum=_momentum(row),
        session=sess,
        structure=_structure_from_regime(regime_raw),
        regime_raw=regime_raw,
    )
