"""Confidence-tier → size multiplier (no ML internals)."""

from __future__ import annotations

from settings.strategy import CONFIDENCE_SIZE_MULT


def confidence_size_multiplier(
    confidence_tier: str,
    mapping: dict[str, float] | None = None,
) -> float:
    """Unknown / low tiers default to 1.0 (no tilt) unless mapped."""
    m = mapping if mapping is not None else CONFIDENCE_SIZE_MULT
    return float(m.get(str(confidence_tier), 1.0))
