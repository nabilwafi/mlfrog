"""Meta edge → risk_pct (option C). Meta is NOT a trade gate."""

from __future__ import annotations

from production import EXPECTED_R_REF, RISK_BASE, RISK_MAX, RISK_MIN
from settings.strategy import SL_ATR_MULT, TP_ATR_MULT

STRUCTURAL_RR: float = float(TP_ATR_MULT) / float(SL_ATR_MULT)


def expected_r_from_edge(edge_score: float, *, rr: float = STRUCTURAL_RR) -> float:
    """E[R] = p*RR - (1-p)*1 = p*(RR+1) - 1."""
    return float(edge_score) * (float(rr) + 1.0) - 1.0


def risk_pct_from_edge(edge_score: float) -> float:
    """
    Option C: risk_pct = clip(RISK_BASE * E[R] / EXPECTED_R_REF, RISK_MIN, RISK_MAX).
    Non-positive E[R] → RISK_MIN (still trade; Portfolio decides skip).
    """
    er = expected_r_from_edge(edge_score)
    if er <= 0:
        return float(RISK_MIN)
    raw = float(RISK_BASE) * er / float(EXPECTED_R_REF)
    return float(min(max(raw, float(RISK_MIN)), float(RISK_MAX)))
