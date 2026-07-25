"""L4 Meta Edge — gate on meta score; emit expected_r. Confidence unused."""

from __future__ import annotations

from pipeline import META_THRESHOLD, STRUCTURAL_RR
from pipeline.types import MetaOut


def expected_r_from_edge(edge_score: float, *, rr: float = STRUCTURAL_RR) -> float:
    """E[R] = p*RR - (1-p)*1 = p*(RR+1) - 1."""
    p = float(edge_score)
    return p * (float(rr) + 1.0) - 1.0


def decide_meta(edge_score: float, *, threshold: float = META_THRESHOLD) -> MetaOut:
    score = float(edge_score)
    er = expected_r_from_edge(score)
    if score < float(threshold):
        return MetaOut(
            trade=False,
            edge_score=score,
            expected_r=er,
            reason=f"meta_below_threshold ({score:.4f} < {threshold})",
        )
    return MetaOut(
        trade=True,
        edge_score=score,
        expected_r=er,
        reason="meta_ok",
    )
