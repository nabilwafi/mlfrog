"""Paper runtime loop — poll bars / replay signals; graceful shutdown."""

from __future__ import annotations

import logging
import signal
import threading
import time
from datetime import datetime, timezone
from typing import Any

from production.events.bus import EventBus
from production.events.types import EventType, make_event
from production.paper.pipeline import IncomingSignal, ProductionPipeline, SignalSource

logger = logging.getLogger(__name__)


class PaperRuntime:
    def __init__(
        self,
        *,
        pipeline: ProductionPipeline,
        bus: EventBus,
        source: SignalSource,
        poll_seconds: float = 5.0,
    ) -> None:
        self.pipeline = pipeline
        self.bus = bus
        self.source = source
        self.poll_seconds = poll_seconds
        self._stop = threading.Event()

    def request_stop(self, *_args: Any) -> None:
        logger.info("graceful_shutdown_requested")
        self._stop.set()

    def install_signal_handlers(self) -> None:
        signal.signal(signal.SIGINT, self.request_stop)
        signal.signal(signal.SIGTERM, self.request_stop)

    def run_forever(self) -> None:
        self.install_signal_handlers()
        self.bus.publish(
            make_event(EventType.AUDIT, {"component": "runtime", "action": "start", "detail": {}})
        )
        last_day = None
        while not self._stop.is_set():
            try:
                signals = self.source.next_signals()
                for sig in signals:
                    self.pipeline.process_signal(sig)
                # daily summary once per UTC day rollover
                today = datetime.now(timezone.utc).date()
                if last_day is not None and today != last_day:
                    self.pipeline.emit_daily_summary()
                last_day = today
            except Exception:
                logger.exception("runtime_tick_failed")
                self.bus.publish(
                    make_event(
                        EventType.EXECUTION_ERROR,
                        {"trade_id": None, "error_message": "runtime_tick_failed", "retry_count": 0},
                    )
                )
            self._stop.wait(self.poll_seconds)
        self.pipeline.emit_daily_summary()
        self.bus.publish(
            make_event(EventType.AUDIT, {"component": "runtime", "action": "stop", "detail": {}})
        )
        self.bus.publish(make_event(EventType.SHUTDOWN, {}))


class ReplaySignalSource:
    """Replay frozen research signals for infrastructure validation (no retrain)."""

    def __init__(self, signals: list[IncomingSignal]) -> None:
        self._signals = list(signals)
        self._i = 0

    def next_signals(self) -> list[IncomingSignal]:
        if self._i >= len(self._signals):
            return []
        # emit one per tick for deterministic tests
        s = self._signals[self._i]
        self._i += 1
        return [s]

    @property
    def exhausted(self) -> bool:
        return self._i >= len(self._signals)


class IdleSignalSource:
    def next_signals(self) -> list[IncomingSignal]:
        return []
