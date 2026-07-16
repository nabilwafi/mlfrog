"""Fixed-fractional lots: equity * risk% * final_mult / (SL_dist * contract)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from risk.confidence_sizing.policy import confidence_size_multiplier
from settings.strategy import (
    CONTRACT_SIZE,
    RISK_PER_TRADE_PCT,
    SL_ATR_MULT,
    VOLUME_MIN,
    VOLUME_STEP,
)


@dataclass(frozen=True)
class SizeOrder:
    lots: float
    final_mult: float
    regime_mult: float
    confidence_mult: float
    risk_amount: float


def final_size_multiplier(regime_mult: float, confidence_mult: float) -> float:
    """Product rule: regime × confidence (never replace)."""
    return float(regime_mult) * float(confidence_mult)


def round_lots(lots: float, *, step: float = VOLUME_STEP, vmin: float = VOLUME_MIN) -> float:
    if lots < vmin:
        return 0.0
    steps = np.floor(lots / step + 1e-12)
    return float(max(vmin, steps * step))


def size_long(
    *,
    equity: float,
    atr: float,
    regime_mult: float = 1.0,
    confidence_tier: str = "n/a",
    confidence_map: dict[str, float] | None = None,
    enable_confidence_sizing: bool = False,
    risk_pct: float = RISK_PER_TRADE_PCT,
    sl_atr_mult: float = SL_ATR_MULT,
    contract_size: float = CONTRACT_SIZE,
) -> SizeOrder:
    conf_mult = (
        confidence_size_multiplier(confidence_tier, confidence_map)
        if enable_confidence_sizing
        else 1.0
    )
    final = final_size_multiplier(regime_mult, conf_mult)
    sl_dist = sl_atr_mult * float(atr)
    risk_amount = float(equity) * float(risk_pct) * final
    raw_lots = risk_amount / (sl_dist * contract_size) if sl_dist > 0 else 0.0
    lots = round_lots(raw_lots)
    return SizeOrder(
        lots=lots,
        final_mult=final,
        regime_mult=float(regime_mult),
        confidence_mult=float(conf_mult),
        risk_amount=risk_amount,
    )
