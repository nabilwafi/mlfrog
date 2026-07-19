"""In-process async event bus — execution never waits for consumers."""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable
from typing import Any

from production.events.types import EventType, ProductionEvent

logger = logging.getLogger(__name__)

Handler = Callable[[ProductionEvent], None]


class EventBus:
    """
    Fire-and-forget publish. Workers drain in background threads.
    Broker/execution path must only call publish(); never join on workers.
    """

    def __init__(self, *, maxsize: int = 10_000) -> None:
        self._q: queue.Queue[ProductionEvent | None] = queue.Queue(maxsize=maxsize)
        self._handlers: dict[EventType | None, list[Handler]] = {}
        self._workers: list[threading.Thread] = []
        self._stopped = threading.Event()
        self._dropped = 0

    def subscribe(self, event_type: EventType | None, handler: Handler) -> None:
        """None = all events."""
        self._handlers.setdefault(event_type, []).append(handler)

    def publish(self, event: ProductionEvent) -> bool:
        """Non-blocking for callers: put with timeout 0; drop if full (count it)."""
        if self._stopped.is_set():
            return False
        try:
            self._q.put_nowait(event)
            return True
        except queue.Full:
            self._dropped += 1
            logger.error(
                "event_bus_full dropped=%s type=%s corr=%s",
                self._dropped,
                event.event_type.value,
                event.correlation_id,
            )
            return False

    def start(self, *, n_workers: int = 2) -> None:
        self._stopped.clear()
        for i in range(max(1, n_workers)):
            t = threading.Thread(target=self._loop, name=f"event-bus-{i}", daemon=True)
            t.start()
            self._workers.append(t)
        logger.info("event_bus_started workers=%s", n_workers)

    def stop(self, *, drain_timeout_s: float = 5.0) -> None:
        self._stopped.set()
        for _ in self._workers:
            try:
                self._q.put(None, timeout=0.5)
            except queue.Full:
                pass
        for t in self._workers:
            t.join(timeout=drain_timeout_s)
        self._workers.clear()
        logger.info("event_bus_stopped dropped=%s", self._dropped)

    @property
    def dropped(self) -> int:
        return self._dropped

    @property
    def pending(self) -> int:
        return self._q.qsize()

    def _loop(self) -> None:
        while True:
            try:
                item = self._q.get(timeout=0.5)
            except queue.Empty:
                if self._stopped.is_set():
                    break
                continue
            if item is None:
                break
            self._dispatch(item)

    def _dispatch(self, event: ProductionEvent) -> None:
        handlers = list(self._handlers.get(event.event_type, []))
        handlers.extend(self._handlers.get(None, []))
        for h in handlers:
            try:
                h(event)
            except Exception:
                # Worker failure must NEVER crash the bus / execution path.
                logger.exception(
                    "worker_handler_failed type=%s corr=%s handler=%s",
                    event.event_type.value,
                    event.correlation_id,
                    getattr(h, "__name__", repr(h)),
                )


class MetricsCollector:
    """In-memory counters/latencies for monitoring endpoints."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.counters: dict[str, int] = {}
        self.latencies_ms: dict[str, list[float]] = {}

    def incr(self, name: str, n: int = 1) -> None:
        with self._lock:
            self.counters[name] = self.counters.get(name, 0) + n

    def observe_ms(self, name: str, value: float) -> None:
        with self._lock:
            self.latencies_ms.setdefault(name, []).append(float(value))
            # ponytail: cap series
            if len(self.latencies_ms[name]) > 5_000:
                self.latencies_ms[name] = self.latencies_ms[name][-2_500:]

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            lat = {}
            for k, xs in self.latencies_ms.items():
                if not xs:
                    continue
                s = sorted(xs)
                lat[k] = {
                    "count": len(s),
                    "p50": s[len(s) // 2],
                    "p95": s[int(len(s) * 0.95)],
                    "max": s[-1],
                    "mean": sum(s) / len(s),
                }
            return {"counters": dict(self.counters), "latencies_ms": lat}
