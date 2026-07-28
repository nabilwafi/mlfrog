"""Paper broker — simulated fills; never MT5 order_send.

Open book keyed by ticket_id (synthetic). Idempotent via signal/trade_id map.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any

from settings.strategy import ASSUMED_SLIPPAGE_POINTS, CONTRACT_SIZE, FALLBACK_SPREAD_POINTS, POINT

logger = logging.getLogger(__name__)


@dataclass
class FillResult:
    success: bool
    trade_id: str  # str(ticket_id) after fill — state/broker identity
    fill_price: float
    spread: float
    slippage: float
    latency_ms: float
    retry_count: int
    broker_response: str
    error_message: str | None = None
    broker_ticket: int | None = None


class PaperBroker:
    def __init__(self, *, max_retries: int = 2) -> None:
        self._max_retries = max_retries
        self._open: dict[int, dict[str, Any]] = {}  # ticket_id → pos
        self._by_client_id: dict[str, int] = {}  # client trade_id → ticket
        # ponytail: paper tickets are synthetic ints; live uses MT5 position tickets
        self._next_ticket = int(time.time()) % 1_000_000_000 + 1

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
        client_id = trade_id or uuid.uuid4().hex
        if client_id in self._by_client_id:
            ticket = self._by_client_id[client_id]
            pos = self._open[ticket]
            return FillResult(
                success=True,
                trade_id=str(ticket),
                fill_price=float(pos["fill_price"]),
                spread=float(pos["spread"]),
                slippage=float(pos["slippage"]),
                latency_ms=0.0,
                retry_count=0,
                broker_response="IDEMPOTENT_OK",
                broker_ticket=ticket,
            )

        last_err = None
        for attempt in range(self._max_retries + 1):
            t0 = time.perf_counter()
            try:
                spread = FALLBACK_SPREAD_POINTS * POINT
                slip = ASSUMED_SLIPPAGE_POINTS * POINT
                fill = entry_price + slip if side == "long" else entry_price - slip
                latency = (time.perf_counter() - t0) * 1000
                ticket = self._next_ticket
                self._next_ticket += 1
                self._open[ticket] = {
                    "client_id": client_id,
                    "side": side,
                    "fill_price": fill,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "lot": lot,
                    "spread": spread,
                    "slippage": slip,
                }
                self._by_client_id[client_id] = ticket
                return FillResult(
                    success=True,
                    trade_id=str(ticket),
                    fill_price=fill,
                    spread=spread,
                    slippage=slip,
                    latency_ms=latency,
                    retry_count=attempt,
                    broker_response="FILLED",
                    broker_ticket=ticket,
                )
            except Exception as exc:  # pragma: no cover
                last_err = str(exc)
                time.sleep(0.01 * (attempt + 1))
        return FillResult(
            success=False,
            trade_id=client_id,
            fill_price=entry_price,
            spread=0.0,
            slippage=0.0,
            latency_ms=0.0,
            retry_count=self._max_retries,
            broker_response="ERROR",
            error_message=last_err or "unknown",
        )

    def close_order(self, trade_id: str, exit_price: float) -> FillResult:
        """trade_id is str(ticket_id)."""
        try:
            ticket = int(trade_id)
        except (TypeError, ValueError):
            ticket = self._by_client_id.get(str(trade_id))
            if ticket is None:
                return FillResult(
                    success=False,
                    trade_id=str(trade_id),
                    fill_price=exit_price,
                    spread=0.0,
                    slippage=0.0,
                    latency_ms=0.0,
                    retry_count=0,
                    broker_response="NOT_FOUND",
                    error_message="trade not open",
                )
        pos = self._open.pop(ticket, None)
        if pos is None:
            return FillResult(
                success=False,
                trade_id=str(ticket),
                fill_price=exit_price,
                spread=0.0,
                slippage=0.0,
                latency_ms=0.0,
                retry_count=0,
                broker_response="NOT_FOUND",
                error_message="trade not open",
            )
        client_id = str(pos.get("client_id") or "")
        if client_id:
            self._by_client_id.pop(client_id, None)
        return FillResult(
            success=True,
            trade_id=str(ticket),
            fill_price=exit_price,
            spread=float(pos["spread"]),
            slippage=0.0,
            latency_ms=0.0,
            retry_count=0,
            broker_response="CLOSED",
            broker_ticket=ticket,
        )

    @staticmethod
    def pnl(side: str, entry: float, exit_px: float, lot: float) -> float:
        if side == "long":
            return lot * CONTRACT_SIZE * (exit_px - entry)
        return lot * CONTRACT_SIZE * (entry - exit_px)

    @staticmethod
    def mark_excursions(side: str, entry: float, high: float, low: float) -> tuple[float, float]:
        if side == "long":
            mae = max(0.0, (entry - low) / entry)
            mfe = max(0.0, (high - entry) / entry)
        else:
            mae = max(0.0, (high - entry) / entry)
            mfe = max(0.0, (entry - low) / entry)
        return mae, mfe
