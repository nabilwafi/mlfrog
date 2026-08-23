"""H1 reversal research + position decision engine (research-first).

LONG/SHORT models stay separate. Reversal is position-aware.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

# Decision engine outputs (deterministic contract).
Action = Literal[
    "ENTER_LONG",
    "ENTER_SHORT",
    "HOLD_LONG",
    "HOLD_SHORT",
    "EXIT_LONG",
    "EXIT_SHORT",
    "WAIT",
    "WAIT_FOR_CONFIRMATION",
]

PositionSide = Literal["flat", "long", "short"]
ReversalStage = Literal["normal", "weakening", "confirmed"]
H4Context = Literal["supportive", "neutral", "conflicting", "unknown"]
M5State = Literal["recovering", "stagnating", "deteriorating", "unknown"]


class DecisionAction(str, Enum):
    ENTER_LONG = "ENTER_LONG"
    ENTER_SHORT = "ENTER_SHORT"
    HOLD_LONG = "HOLD_LONG"
    HOLD_SHORT = "HOLD_SHORT"
    EXIT_LONG = "EXIT_LONG"
    EXIT_SHORT = "EXIT_SHORT"
    WAIT = "WAIT"
    WAIT_FOR_CONFIRMATION = "WAIT_FOR_CONFIRMATION"


# Default anti-overtrading knobs (configurable).
DEFAULT_CONFIRM_BARS: int = 2  # H1 bars in WEAKENING before CONFIRMED
DEFAULT_REENTRY_COOLDOWN_BARS: int = 4  # H1 bars after EXIT before same-dir re-entry
DEFAULT_ALLOW_PYRAMID: bool = False
DEFAULT_ALLOW_DIRECT_REVERSE: bool = False  # EXIT then FLAT then wait; no flip
