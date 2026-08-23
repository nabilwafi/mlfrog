"""Deterministic position decision engine.

Produces decisions only — no broker I/O.
Default: no pyramid, no direct reverse (EXIT → FLAT → WAIT → optional ENTER).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from research.h1_reversal import (
    DEFAULT_ALLOW_DIRECT_REVERSE,
    DEFAULT_ALLOW_PYRAMID,
    DEFAULT_CONFIRM_BARS,
    DEFAULT_REENTRY_COOLDOWN_BARS,
    DecisionAction,
    H4Context,
    M5State,
    PositionSide,
    ReversalStage,
)


@dataclass
class MarketState:
    timestamp: Any = None
    price: float = 0.0
    atr: float = 0.0


@dataclass
class PositionState:
    side: PositionSide = "flat"
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    unrealized_r: float = 0.0
    mae_r: float = 0.0
    mfe_r: float = 0.0
    bars_held: int = 0  # manage-clock bars (e.g. M15) or H1 — caller defines
    bars_since_exit: int = 999
    last_exit_side: PositionSide | None = None
    reversal_stage: ReversalStage = "normal"
    weakening_bars: int = 0


@dataclass
class DecisionConfig:
    long_entry_threshold: float = 0.0  # unused when top-pct gate external; kept for API
    short_entry_threshold: float = 0.0
    require_long_gt_short: bool = True
    confirm_bars: int = DEFAULT_CONFIRM_BARS
    reentry_cooldown_bars: int = DEFAULT_REENTRY_COOLDOWN_BARS
    allow_pyramid: bool = DEFAULT_ALLOW_PYRAMID
    allow_direct_reverse: bool = DEFAULT_ALLOW_DIRECT_REVERSE
    # Baseline A-style: opposite calibrated/raw prob crosses
    weaken_opposite_prob: float = 0.55
    weaken_own_prob_max: float = 0.45
    confirm_opposite_prob: float = 0.60
    # Risk exits (optional overlays; trail still external)
    hard_exit_r: float | None = None  # e.g. -1.0
    use_h4_for_exit_boost: bool = True


@dataclass
class DecisionResult:
    action: DecisionAction
    reason: str
    reversal_stage: ReversalStage = "normal"
    detail: dict[str, Any] = field(default_factory=dict)


def evaluate(
    *,
    market: MarketState,
    p_long: float,
    p_short: float,
    reversal_prob: float | None,
    reversal_stage: ReversalStage,
    h4_context: H4Context,
    m5_state: M5State,
    position: PositionState,
    cfg: DecisionConfig | None = None,
    entry_gate_long: bool = False,
    entry_gate_short: bool = False,
) -> DecisionResult:
    """
    Core contract. Entry gates may be precomputed (top-pct); if False, WAIT when flat
    unless thresholds in cfg are used.
    """
    cfg = cfg or DecisionConfig()
    side = position.side
    pl, ps = float(p_long), float(p_short)
    rp = float(reversal_prob) if reversal_prob is not None else None

    if side == "flat":
        return _eval_flat(
            pl, ps, entry_gate_long, entry_gate_short, position, cfg,
        )

    if side == "long":
        return _eval_long(
            market, pl, ps, rp, reversal_stage, h4_context, m5_state, position, cfg,
        )
    return _eval_short(
        market, pl, ps, rp, reversal_stage, h4_context, m5_state, position, cfg,
    )


def _cooldown_blocks(position: PositionState, want: PositionSide, cfg: DecisionConfig) -> bool:
    if position.last_exit_side is None:
        return False
    if position.last_exit_side != want:
        return False
    return int(position.bars_since_exit) < int(cfg.reentry_cooldown_bars)


def _eval_flat(
    pl: float,
    ps: float,
    gate_long: bool,
    gate_short: bool,
    position: PositionState,
    cfg: DecisionConfig,
) -> DecisionResult:
    long_ok = gate_long or (pl >= cfg.long_entry_threshold > 0)
    short_ok = gate_short or (ps >= cfg.short_entry_threshold > 0)
    if cfg.require_long_gt_short:
        if long_ok and pl > ps and not _cooldown_blocks(position, "long", cfg):
            return DecisionResult(DecisionAction.ENTER_LONG, "flat_long_gate", detail={"p_long": pl, "p_short": ps})
        if short_ok and ps > pl and not _cooldown_blocks(position, "short", cfg):
            return DecisionResult(DecisionAction.ENTER_SHORT, "flat_short_gate", detail={"p_long": pl, "p_short": ps})
    else:
        if long_ok and not _cooldown_blocks(position, "long", cfg):
            return DecisionResult(DecisionAction.ENTER_LONG, "flat_long_gate", detail={"p_long": pl})
        if short_ok and not _cooldown_blocks(position, "short", cfg):
            return DecisionResult(DecisionAction.ENTER_SHORT, "flat_short_gate", detail={"p_short": ps})
    return DecisionResult(DecisionAction.WAIT, "flat_no_setup", detail={"p_long": pl, "p_short": ps})


def _update_stage_long(
    pl: float, ps: float, rp: float | None, stage: ReversalStage, weakening_bars: int, cfg: DecisionConfig,
) -> tuple[ReversalStage, int]:
    weaken = (ps >= cfg.weaken_opposite_prob and pl <= cfg.weaken_own_prob_max) or (
        rp is not None and rp >= cfg.weaken_opposite_prob
    )
    confirm = (ps >= cfg.confirm_opposite_prob and pl <= cfg.weaken_own_prob_max) or (
        rp is not None and rp >= cfg.confirm_opposite_prob
    )
    if stage == "normal":
        if weaken:
            return "weakening", 1
        return "normal", 0
    if stage == "weakening":
        wb = weakening_bars + 1
        if confirm and wb >= cfg.confirm_bars:
            return "confirmed", wb
        if not weaken:
            return "normal", 0
        return "weakening", wb
    # confirmed sticks until flat
    return "confirmed", weakening_bars


def _eval_long(
    market: MarketState,
    pl: float,
    ps: float,
    rp: float | None,
    stage: ReversalStage,
    h4: H4Context,
    m5: M5State,
    position: PositionState,
    cfg: DecisionConfig,
) -> DecisionResult:
    if not cfg.allow_pyramid and pl > ps:
        # same-direction signal while long → HOLD, never re-ENTER
        pass

    if cfg.hard_exit_r is not None and position.unrealized_r <= float(cfg.hard_exit_r):
        return DecisionResult(
            DecisionAction.EXIT_LONG,
            "hard_exit_r",
            reversal_stage=stage,
            detail={"unrealized_r": position.unrealized_r},
        )

    new_stage, wb = _update_stage_long(pl, ps, rp, stage, position.weakening_bars, cfg)

    if new_stage == "confirmed":
        if cfg.use_h4_for_exit_boost and h4 == "supportive" and m5 == "recovering":
            return DecisionResult(
                DecisionAction.WAIT_FOR_CONFIRMATION,
                "reversal_confirmed_but_h4_m5_supportive",
                reversal_stage=new_stage,
                detail={"p_long": pl, "p_short": ps, "p_reversal": rp, "weakening_bars": wb},
            )
        return DecisionResult(
            DecisionAction.EXIT_LONG,
            "bearish_reversal_confirmed",
            reversal_stage=new_stage,
            detail={"p_long": pl, "p_short": ps, "p_reversal": rp, "h4": h4, "m5": m5},
        )

    if new_stage == "weakening":
        return DecisionResult(
            DecisionAction.WAIT_FOR_CONFIRMATION,
            "long_thesis_weakening",
            reversal_stage=new_stage,
            detail={"p_long": pl, "p_short": ps, "weakening_bars": wb},
        )

    return DecisionResult(
        DecisionAction.HOLD_LONG,
        "thesis_valid",
        reversal_stage="normal",
        detail={"p_long": pl, "p_short": ps, "unrealized_r": position.unrealized_r, "mfe_r": position.mfe_r},
    )


def _update_stage_short(
    pl: float, ps: float, rp: float | None, stage: ReversalStage, weakening_bars: int, cfg: DecisionConfig,
) -> tuple[ReversalStage, int]:
    weaken = (pl >= cfg.weaken_opposite_prob and ps <= cfg.weaken_own_prob_max) or (
        rp is not None and rp >= cfg.weaken_opposite_prob
    )
    confirm = (pl >= cfg.confirm_opposite_prob and ps <= cfg.weaken_own_prob_max) or (
        rp is not None and rp >= cfg.confirm_opposite_prob
    )
    if stage == "normal":
        if weaken:
            return "weakening", 1
        return "normal", 0
    if stage == "weakening":
        wb = weakening_bars + 1
        if confirm and wb >= cfg.confirm_bars:
            return "confirmed", wb
        if not weaken:
            return "normal", 0
        return "weakening", wb
    return "confirmed", weakening_bars


def _eval_short(
    market: MarketState,
    pl: float,
    ps: float,
    rp: float | None,
    stage: ReversalStage,
    h4: H4Context,
    m5: M5State,
    position: PositionState,
    cfg: DecisionConfig,
) -> DecisionResult:
    if cfg.hard_exit_r is not None and position.unrealized_r <= float(cfg.hard_exit_r):
        return DecisionResult(
            DecisionAction.EXIT_SHORT,
            "hard_exit_r",
            reversal_stage=stage,
            detail={"unrealized_r": position.unrealized_r},
        )

    new_stage, wb = _update_stage_short(pl, ps, rp, stage, position.weakening_bars, cfg)

    if new_stage == "confirmed":
        if cfg.use_h4_for_exit_boost and h4 == "supportive" and m5 == "recovering":
            return DecisionResult(
                DecisionAction.WAIT_FOR_CONFIRMATION,
                "reversal_confirmed_but_h4_m5_supportive",
                reversal_stage=new_stage,
                detail={"p_long": pl, "p_short": ps, "p_reversal": rp, "weakening_bars": wb},
            )
        return DecisionResult(
            DecisionAction.EXIT_SHORT,
            "bullish_reversal_confirmed",
            reversal_stage=new_stage,
            detail={"p_long": pl, "p_short": ps, "p_reversal": rp, "h4": h4, "m5": m5},
        )

    if new_stage == "weakening":
        return DecisionResult(
            DecisionAction.WAIT_FOR_CONFIRMATION,
            "short_thesis_weakening",
            reversal_stage=new_stage,
            detail={"p_long": pl, "p_short": ps, "weakening_bars": wb},
        )

    return DecisionResult(
        DecisionAction.HOLD_SHORT,
        "thesis_valid",
        reversal_stage="normal",
        detail={"p_long": pl, "p_short": ps, "unrealized_r": position.unrealized_r, "mfe_r": position.mfe_r},
    )
