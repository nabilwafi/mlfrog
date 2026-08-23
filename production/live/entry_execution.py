"""M15 pullback entry timing — wraps research execution rules for live/paper."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

import pandas as pd

from production import (
    ADV_INVALIDATE_R,
    MAX_WAIT_M15,
    PULLBACK_ATR,
    PULLBACK_RECOVERY_FRAC,
    SL_ATR_MULT,
)
from research.mtf_h4_h1_m5.m5_engine import M5ExecResult, build_m15_bars, decide_m15_execution

FillOutcome = Literal["pending", "execute", "invalidate"]


@dataclass(frozen=True)
class PendingFillResult:
    outcome: FillOutcome
    fill_price: float | None = None
    fill_timestamp: datetime | None = None
    reason: str = ""


def one_r_from_atr(atr: float) -> float:
    return float(SL_ATR_MULT) * float(atr)


def evaluate_m15_pullback(
    *,
    side: str,
    h1_signal_ts: datetime,
    h1_ref_price: float,
    atr: float,
    m15_ohlc: pd.DataFrame,
) -> PendingFillResult:
    """
    Scan closed M15 bars after H1 signal for pullback_recovery fill.
    Returns pending while still within max wait; execute/invalidate when resolved.
    """
    ts = h1_signal_ts if h1_signal_ts.tzinfo else h1_signal_ts.replace(tzinfo=timezone.utc)
    h1_ts = pd.Timestamp(ts).tz_convert("UTC")
    if m15_ohlc.empty:
        return PendingFillResult("pending", reason="no_m15_data")

    bars = build_m15_bars(m15_ohlc)
    res: M5ExecResult = decide_m15_execution(
        side=str(side).lower(),
        h1_signal_ts=h1_ts,
        h1_ref_price=float(h1_ref_price),
        one_r=one_r_from_atr(float(atr)),
        m15=bars,
        strategy="pullback_recovery",
        pullback_atr=float(PULLBACK_ATR),
        recovery_frac=float(PULLBACK_RECOVERY_FRAC),
    )
    if res.action == "EXECUTE" and res.fill_price is not None and res.available_timestamp is not None:
        fill_ts = res.available_timestamp.to_pydatetime()
        if fill_ts.tzinfo is None:
            fill_ts = fill_ts.replace(tzinfo=timezone.utc)
        return PendingFillResult(
            "execute",
            fill_price=float(res.fill_price),
            fill_timestamp=fill_ts,
            reason=str(res.reason),
        )
    if res.action == "INVALIDATE":
        return PendingFillResult("invalidate", reason=str(res.reason))

    # Still scanning — check horizon from bars seen after signal
    m15_ts = pd.to_datetime(bars["timestamp"], utc=True)
    start = int(m15_ts.searchsorted(h1_ts, side="right"))
    waited = max(0, len(bars) - start)
    if waited >= int(MAX_WAIT_M15):
        return PendingFillResult("invalidate", reason="max_wait_exceeded")
    return PendingFillResult("pending", reason="waiting_pullback")
