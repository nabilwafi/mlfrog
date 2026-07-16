"""Probability quality entities."""

from __future__ import annotations

from dataclasses import dataclass


# Fixed trading-relevant probability buckets (Sprint 14 brief).
PROB_BUCKETS: tuple[tuple[float, float | None, str], ...] = (
    (0.50, 0.55, "0.50-0.55"),
    (0.55, 0.60, "0.55-0.60"),
    (0.60, 0.65, "0.60-0.65"),
    (0.65, 0.70, "0.65-0.70"),
    (0.70, 0.75, "0.70-0.75"),
    (0.75, None, "0.75+"),
)

# Confidence tiers on P(success).
CONFIDENCE_TIERS: tuple[tuple[float, float | None, str], ...] = (
    (0.50, 0.60, "low"),
    (0.60, 0.70, "medium"),
    (0.70, None, "high"),
)


@dataclass(frozen=True)
class DistSummary:
    side: str
    n: int
    mean: float
    std: float
    p10: float
    p25: float
    p50: float
    p75: float
    p90: float
    p95: float
    p99: float
