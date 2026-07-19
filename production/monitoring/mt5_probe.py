"""MT5 liveness probe for health / Telegram (read-only)."""

from __future__ import annotations

import logging
from typing import Any, Callable

from production.monitoring.mt5_session import status as mt5_status

logger = logging.getLogger(__name__)


def probe_mt5(cfg: dict[str, Any]) -> dict[str, Any]:
    """Soft-check MT5 — uses shared session; never shuts down IPC."""
    return mt5_status(cfg)


def make_mt5_probe(cfg: dict[str, Any]) -> Callable[[], dict[str, Any]]:
    return lambda: probe_mt5(cfg)
