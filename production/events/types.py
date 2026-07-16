"""Typed production events (hot path publishes; workers consume)."""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def new_correlation_id() -> str:
    return uuid.uuid4().hex


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class EventType(str, Enum):
    SIGNAL = "signal"
    TRADE_OPENED = "trade_opened"
    TRADE_CLOSED = "trade_closed"
    TRADE_SKIPPED = "trade_skipped"
    EXECUTION_ERROR = "execution_error"
    HEAT_TRIGGERED = "heat_triggered"
    METRIC = "metric"
    AUDIT = "audit"
    DAILY_SUMMARY = "daily_summary"
    HEALTH = "health"
    SHUTDOWN = "shutdown"


@dataclass
class ProductionEvent:
    event_type: EventType
    correlation_id: str
    timestamp: datetime
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["event_type"] = self.event_type.value
        d["timestamp"] = self.timestamp.isoformat()
        return d


def make_event(event_type: EventType, payload: dict[str, Any], *, correlation_id: str | None = None) -> ProductionEvent:
    return ProductionEvent(
        event_type=event_type,
        correlation_id=correlation_id or new_correlation_id(),
        timestamp=utcnow(),
        payload=dict(payload),
    )
