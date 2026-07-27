"""Paper broker — simulated fills; never MT5 order_send."""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from settings.strategy import ASSUMED_SLIPPAGE_POINTS, CONTRACT_SIZE, FALLBACK_SPREAD_POINTS, POINT

logger = logging.getLogger(__name__)


@dataclass
class FillResult:
    success: bool
    trade_id: str
    fill_price: float
    spread: float
    slippage: float
    latency_ms: float
    retry_count: int
    broker_response: str
    error_message: str | None = None
    broker_ticket: int | None = None


class PaperBroker:
    """
    Synchronous simulated broker. Idempotent trade IDs.
    Retries are local only (paper never blocks on DB/Telegram).
    """

    def __init__(self, *, max_retries: int = 2) -> None:
        self._max_retries = max_retries
        self._open: dict[str, dict[str, Any]] = {}

    def open_order(
        self,
        *,
        side: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        lot: float,
        trade_id: str | None = None,
    ) -> FillResult:
        tid = trade_id or uuid.uuid4().hex
        if tid in self._open:
            # idempotent — return existing
            pos = self._open[tid]
            return FillResult(
                success=True,
                trade_id=tid,
                fill_price=float(pos["fill_price"]),
                spread=float(pos["spread"]),
                slippage=float(pos["slippage"]),
                latency_ms=0.0,
                retry_count=0,
                broker_response="IDEMPOTENT_OK",
            )

        last_err = None
        for attempt in range(self._max_retries + 1):
            t0 = time.perf_counter()
            try:
                spread = FALLBACK_SPREAD_POINTS * POINT
                slip = ASSUMED_SLIPPAGE_POINTS * POINT
                fill = entry_price + slip if side == "long" else entry_price - slip
                latency = (time.perf_counter() - t0) * 1000
                self._open[tid] = {
                    "side": side,
                    "fill_price": fill,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "lot": lot,
                    "spread": spread,
                    "slippage": slip,
                }
                return FillResult(
                    success=True,
                    trade_id=tid,
                    fill_price=fill,
                    spread=spread,
                    slippage=slip,
                    latency_ms=latency,
                    retry_count=attempt,
                    broker_response="FILLED",
                )
            except Exception as exc:  # pragma: no cover
                last_err = str(exc)
                time.sleep(0.01 * (attempt + 1))
        return FillResult(
            success=False,
            trade_id=tid,
            fill_price=entry_price,
            spread=0.0,
            slippage=0.0,
            latency_ms=0.0,
            retry_count=self._max_retries,
            broker_response="ERROR",
            error_message=last_err or "unknown",
        )

    def close_order(self, trade_id: str, exit_price: float) -> FillResult:
        pos = self._open.pop(trade_id, None)
        if pos is None:
            return FillResult(
                success=False,
                trade_id=trade_id,
                fill_price=exit_price,
                spread=0.0,
                slippage=0.0,
                latency_ms=0.0,
                retry_count=0,
                broker_response="NOT_FOUND",
                error_message="trade not open",
            )
        return FillResult(
            success=True,
            trade_id=trade_id,
            fill_price=exit_price,
            spread=float(pos["spread"]),
            slippage=0.0,
            latency_ms=0.0,
            retry_count=0,
            broker_response="CLOSED",
        )

    @staticmethod
    def pnl(side: str, entry: float, exit_px: float, lot: float) -> float:
        if side == "long":
            return lot * CONTRACT_SIZE * (exit_px - entry)
        return lot * CONTRACT_SIZE * (entry - exit_px)

    @staticmethod
    def mark_excursions(side: str, entry: float, high: float, low: float) -> tuple[float, float]:
        """Return (mae, mfe) as price fractions."""
        if side == "long":
            mae = max(0.0, (entry - low) / entry)
            mfe = max(0.0, (high - entry) / entry)
        else:
            mae = max(0.0, (high - entry) / entry)
            mfe = max(0.0, (entry - low) / entry)
        return mae, mfe
