"""Decision layer — threshold + regime policy. No ML internals, no broker, no lots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Action = Literal["trade", "skip_threshold", "skip_guard", "skip_open", "skip_warmup"]


@dataclass(frozen=True)
class Decision:
    action: Action
    reason: str
    regime_tier: str
    regime_size_mult: float
    confidence_tier: str
    raw_p: float
    calibrated_p: float


def regime_size_multiplier(
    regime_tier: str,
    *,
    elevated_mult: float = 0.5,
) -> float:
    """Map regime tier → size mult (before confidence). Extreme is not sized — blocked upstream."""
    if regime_tier == "elevated":
        return float(elevated_mult)
    return 1.0


def decide_long_entry(
    *,
    raw_p: float,
    thr: float,
    regime_tier: str,
    position_open: bool,
    confidence_tier: str = "n/a",
    calibrated_p: float = float("nan"),
    elevated_mult: float = 0.5,
    enable_regime_guard: bool = True,
) -> Decision:
    """
    Pure entry policy (LONG). Same rules as backtest/paper:
      open position → skip; raw_p < thr → skip; extreme → skip_guard; else trade.
    """
    if position_open:
        return Decision(
            action="skip_open",
            reason="position already open (no stacking)",
            regime_tier=regime_tier,
            regime_size_mult=1.0,
            confidence_tier=confidence_tier,
            raw_p=raw_p,
            calibrated_p=calibrated_p,
        )
    if raw_p < thr:
        return Decision(
            action="skip_threshold",
            reason=f"raw_p {raw_p:.4f} < {thr}",
            regime_tier=regime_tier,
            regime_size_mult=1.0,
            confidence_tier=confidence_tier,
            raw_p=raw_p,
            calibrated_p=calibrated_p,
        )
    if enable_regime_guard and regime_tier == "extreme":
        return Decision(
            action="skip_guard",
            reason="blocked by regime guard (extreme)",
            regime_tier=regime_tier,
            regime_size_mult=0.0,
            confidence_tier=confidence_tier,
            raw_p=raw_p,
            calibrated_p=calibrated_p,
        )
    rmult = (
        regime_size_multiplier(regime_tier, elevated_mult=elevated_mult)
        if enable_regime_guard
        else 1.0
    )
    return Decision(
        action="trade",
        reason="enter",
        regime_tier=regime_tier,
        regime_size_mult=rmult,
        confidence_tier=confidence_tier,
        raw_p=raw_p,
        calibrated_p=calibrated_p,
    )
