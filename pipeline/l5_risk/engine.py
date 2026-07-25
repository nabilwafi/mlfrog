"""L5 Risk Engine — expected_r sizing (option C) + ATR barriers. No ML."""

from __future__ import annotations

from pipeline import EXPECTED_R_REF, RISK_BASE, RISK_MAX, RISK_MIN, STRUCTURAL_RR
from pipeline.types import RiskOut
from settings.strategy import CONTRACT_SIZE, SL_ATR_MULT, TP_ATR_MULT


def risk_pct_from_expected_r(
    expected_r: float,
    *,
    risk_base: float = RISK_BASE,
    expected_r_ref: float = EXPECTED_R_REF,
    risk_min: float = RISK_MIN,
    risk_max: float = RISK_MAX,
) -> float:
    """Option C: risk_pct = clip(base * E[R] / ref, min, max). Clamp <=0 to RISK_MIN."""
    er = float(expected_r)
    if er <= 0:
        return float(risk_min)
    raw = float(risk_base) * er / float(expected_r_ref)
    return float(min(max(raw, float(risk_min)), float(risk_max)))


def lots_from_risk(*, equity: float, atr: float, risk_pct: float) -> float:
    """Same convention as research `_lots_from_equity` (enforce_volume_min=False)."""
    if equity <= 0 or atr <= 0 or risk_pct <= 0:
        return 0.0
    sl_dist = float(SL_ATR_MULT) * float(atr)
    if sl_dist <= 0:
        return 0.0
    return max((float(equity) * float(risk_pct)) / (sl_dist * float(CONTRACT_SIZE)), 0.0)


def build_risk(
    *,
    side: str,
    entry: float,
    atr: float,
    equity: float,
    expected_r: float,
) -> RiskOut | None:
    side_l = str(side).lower()
    if side_l not in {"long", "short"}:
        return None
    if equity <= 0 or atr <= 0:
        return None
    risk_pct = risk_pct_from_expected_r(expected_r)
    lot = lots_from_risk(equity=equity, atr=atr, risk_pct=risk_pct)
    if lot <= 0:
        return None
    entry_f = float(entry)
    atr_f = float(atr)
    if side_l == "long":
        stop = entry_f - SL_ATR_MULT * atr_f
        target = entry_f + TP_ATR_MULT * atr_f
    else:
        stop = entry_f + SL_ATR_MULT * atr_f
        target = entry_f - TP_ATR_MULT * atr_f
    return RiskOut(
        entry=entry_f,
        stop=stop,
        target=target,
        risk_pct=risk_pct,
        rr=float(STRUCTURAL_RR),
        lot=lot,
        atr=atr_f,
        side=side_l,
    )
