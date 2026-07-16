"""Lightweight monitoring HTTP endpoints (stdlib)."""

from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from production.events.bus import EventBus, MetricsCollector
from production.monitoring.health import HealthReporter
from production.paper.state import PortfolioState

logger = logging.getLogger(__name__)


def start_monitoring_server(
    *,
    host: str,
    port: int,
    metrics: MetricsCollector,
    state: PortfolioState,
    bus: EventBus,
    health: HealthReporter | None = None,
) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:  # quieter
            logger.debug("http " + fmt, *args)

        def _json(self, code: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, default=str).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path in ("/health", "/healthz"):
                if health is not None:
                    health.on_http_probe()
                    snap = health.last_snapshot()
                    mt5 = snap.get("mt5")
                    code = 503 if mt5 == "down" else 200
                    self._json(
                        code,
                        {
                            "status": snap.get("status", "ok"),
                            "open_positions": len(state.open_positions),
                            "mt5": mt5,
                            "mt5_error": snap.get("mt5_error"),
                            "mt5_login": snap.get("mt5_login"),
                            "mt5_server": snap.get("mt5_server"),
                            "mt5_connected": snap.get("mt5_connected"),
                            "mt5_trade_allowed": snap.get("mt5_trade_allowed"),
                        },
                    )
                    return
                self._json(200, {"status": "ok", "open_positions": len(state.open_positions)})
                return
            if self.path == "/metrics":
                snap = metrics.snapshot()
                snap["equity"] = state.equity
                snap["drawdown"] = state.drawdown()
                snap["event_bus_pending"] = bus.pending
                snap["event_bus_dropped"] = bus.dropped
                self._json(200, snap)
                return
            if self.path == "/portfolio":
                self._json(
                    200,
                    {
                        "equity": state.equity,
                        "peak_equity": state.peak_equity,
                        "drawdown": state.drawdown(),
                        "day_pnl": state.day_pnl,
                        "open": list(state.open_positions.keys()),
                        "trades_today": state.trades_today,
                        "skips": state.skips,
                        "heat_triggered_today": state.heat_triggered_today,
                    },
                )
                return
            self._json(404, {"error": "not_found"})

    server = ThreadingHTTPServer((host, port), Handler)
    t = threading.Thread(target=server.serve_forever, name="monitoring-http", daemon=True)
    t.start()
    logger.info("monitoring_started host=%s port=%s", host, port)
    return server
