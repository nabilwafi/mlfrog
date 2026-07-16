"""Publish HEALTH events to EventBus (Telegram via worker)."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

from production.events.bus import EventBus
from production.events.types import EventType, make_event, utcnow
from production.paper.state import PortfolioState

logger = logging.getLogger(__name__)


class HealthReporter:
    """ponytail: in-process heartbeat; upgrade to external probe if multi-host."""

    def __init__(
        self,
        bus: EventBus,
        state: PortfolioState,
        *,
        interval_seconds: float = 300.0,
        probe_throttle_seconds: float = 60.0,
        mt5_probe: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        self.bus = bus
        self.state = state
        self.interval_seconds = max(0.0, float(interval_seconds))
        self.probe_throttle_seconds = max(0.0, float(probe_throttle_seconds))
        self._mt5_probe = mt5_probe
        self._last_mt5_ok: bool | None = None
        self._last_payload: dict[str, Any] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_probe_tele = 0.0
        self._lock = threading.Lock()

    def snapshot(self, *, reason: str, status: str = "ok", extra: dict[str, Any] | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": status,
            "reason": reason,
            "equity": self.state.equity,
            "open_positions": len(self.state.open_positions),
            "trades_today": self.state.trades_today,
            "skips": self.state.skips,
            "heat_triggered_today": self.state.heat_triggered_today,
            "drawdown": self.state.drawdown(),
            "bus_pending": self.bus.pending,
            "bus_dropped": self.bus.dropped,
            "timestamp": utcnow().isoformat(),
        }
        if self._mt5_probe is not None:
            try:
                payload.update(self._mt5_probe())
            except Exception as exc:
                payload["mt5"] = "down"
                payload["mt5_error"] = str(exc)
            if payload.get("mt5") != "connected":
                payload["status"] = "degraded"
            mt5_ok = payload.get("mt5") == "connected"
            if self._last_mt5_ok is True and not mt5_ok:
                payload["reason"] = f"{reason}|mt5_down"
                payload["status"] = "degraded"
            elif self._last_mt5_ok is False and mt5_ok:
                payload["reason"] = f"{reason}|mt5_up"
            self._last_mt5_ok = mt5_ok
        if extra:
            payload.update(extra)
        self._last_payload = dict(payload)
        return payload

    def last_snapshot(self) -> dict[str, Any]:
        return dict(self._last_payload)

    def publish(self, *, reason: str, status: str = "ok", extra: dict[str, Any] | None = None) -> None:
        self.bus.publish(make_event(EventType.HEALTH, self.snapshot(reason=reason, status=status, extra=extra)))

    def on_http_probe(self) -> None:
        """Called from /health — throttled so k8s probes do not spam Telegram."""
        now = time.monotonic()
        with self._lock:
            throttled = now - self._last_probe_tele < self.probe_throttle_seconds
            if not throttled:
                self._last_probe_tele = now
        if throttled:
            if self._mt5_probe is None:
                return
            prev = self._last_mt5_ok
            payload = self.snapshot(reason="http_probe_silent")
            # always alert immediately when MT5 drops, even under throttle
            if prev is True and self._last_mt5_ok is False:
                self.bus.publish(make_event(EventType.HEALTH, payload))
            return
        self.publish(reason="http_probe")

    def start(self) -> None:
        self.publish(reason="start")
        if self.interval_seconds <= 0:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="health-heartbeat", daemon=True)
        self._thread.start()
        logger.info("health_heartbeat_started interval_s=%s", self.interval_seconds)

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self.publish(reason="stop")

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                self.publish(reason="heartbeat")
            except Exception:
                logger.exception("health_heartbeat_failed")
