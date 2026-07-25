"""L6 Portfolio Execution — heat, duplicate, paper broker, trade journal."""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Any

from pipeline import (
    CONFIDENCE_ENABLED,
    CONFIDENCE_SKIP,
    FEATURE_VERSION,
    LABEL_VERSION,
    META_THRESHOLD,
    META_VERSION,
    MODEL_VERSION,
    PIPELINE_VERSION,
    RISK_BASE,
)
from pipeline.types import MarketState, MetaOut, PortfolioDecision, PrimaryOut, RiskOut
from production.events.bus import EventBus
from production.events.types import EventType, make_event
from production.paper.broker import PaperBroker
from production.paper.state import OpenPosition, PortfolioState


class PortfolioExecution:
    def __init__(
        self,
        *,
        bus: EventBus | None = None,
        state: PortfolioState,
        broker: PaperBroker | None = None,
        symbol: str = "XAUUSD",
        environment: str = "paper",
    ) -> None:
        self.bus = bus
        self.state = state
        self.broker = broker or PaperBroker()
        self.symbol = symbol
        self.environment = str(environment or "paper")

    def _publish(self, et: EventType, payload: dict[str, Any], correlation_id: str | None = None) -> None:
        if self.bus is None:
            return
        self.bus.publish(make_event(et, payload, correlation_id=correlation_id))

    def decide_and_execute(
        self,
        *,
        signal_id: str,
        timestamp: datetime,
        symbol: str,
        primary: PrimaryOut,
        meta: MetaOut,
        risk: RiskOut | None,
        market_state: MarketState,
        probability: float,
        confidence: float = 0.0,
        correlation_id: str | None = None,
        entry_price: float = 0.0,
    ) -> PortfolioDecision:
        now = timestamp if timestamp.tzinfo else timestamp.replace(tzinfo=timezone.utc)
        self.state.roll_day(now)
        corr = correlation_id or uuid.uuid4().hex
        entry_px = float(entry_price or (risk.entry if risk else 0.0))

        if not self.state.register_signal_key(signal_id):
            self._skip(corr, signal_id, symbol, primary, meta, "duplicate", None, None, now, entry_px)
            return PortfolioDecision(accept=False, reason="duplicate")

        base_signal = {
            "signal_id": signal_id,
            "symbol": symbol,
            "side": primary.side.lower() if primary.side != "NONE" else "",
            "probability": probability,
            "meta_probability": meta.edge_score,
            "confidence": confidence,
            "expected_r": meta.expected_r,
            "threshold_meta": META_THRESHOLD,
            "threshold_confidence": CONFIDENCE_SKIP,
            "confidence_enabled": CONFIDENCE_ENABLED,
            "model_version": MODEL_VERSION,
            "meta_version": META_VERSION,
            "feature_version": FEATURE_VERSION,
            "label_version": LABEL_VERSION,
            "pipeline_version": PIPELINE_VERSION,
            "accepted": False,
            "session": market_state.session,
            "regime": market_state.regime_raw,
            "market_state": market_state.to_dict(),
        }

        if not meta.trade:
            self.state.meta_rejects += 1
            self.state.skips += 1
            self._publish(EventType.SIGNAL, {**base_signal, "accepted": False}, corr)
            self._skip(corr, signal_id, symbol, primary, meta, "meta", META_THRESHOLD, meta.edge_score, now, entry_px)
            return PortfolioDecision(accept=False, reason="meta")

        # Confidence intentionally unused when CONFIDENCE_ENABLED is False

        if self.state.daily_heat_blocked():
            self.state.heat_triggered_today += 1
            self.state.skips += 1
            self._publish(
                EventType.HEAT_TRIGGERED,
                {"day_pnl": self.state.day_pnl, "equity": self.state.equity},
                corr,
            )
            self._publish(EventType.SIGNAL, {**base_signal, "accepted": False}, corr)
            self._skip(
                corr,
                signal_id,
                symbol,
                primary,
                meta,
                "heat",
                -1.0,
                self.state.day_pnl / max(self.state.r_unit(), 1e-9),
                now,
                entry_px,
            )
            return PortfolioDecision(accept=False, reason="heat")

        if risk is None:
            self.state.skips += 1
            self._publish(EventType.SIGNAL, {**base_signal, "accepted": False}, corr)
            self._skip(corr, signal_id, symbol, primary, meta, "risk", RISK_BASE, self.state.equity, now, entry_px)
            return PortfolioDecision(accept=False, reason="risk")

        t_br0 = time.perf_counter()
        fill = self.broker.open_order(
            side=risk.side,
            entry_price=risk.entry,
            stop_loss=risk.stop,
            take_profit=risk.target,
            lot=risk.lot,
            trade_id=signal_id,
        )
        broker_ms = (time.perf_counter() - t_br0) * 1000
        self._publish(EventType.METRIC, {"name": "broker_latency_ms", "value": broker_ms}, corr)

        if not fill.success:
            self._publish(
                EventType.EXECUTION_ERROR,
                {
                    "trade_id": fill.trade_id,
                    "error_message": fill.error_message,
                    "retry_count": fill.retry_count,
                    "latency_ms": fill.latency_ms,
                    "broker_response": fill.broker_response,
                },
                corr,
            )
            return PortfolioDecision(accept=False, reason="broker_error", trade={"error": fill.error_message})

        self._publish(EventType.SIGNAL, {**base_signal, "accepted": True}, corr)

        trade_json = {
            "signal_time": now.isoformat(),
            "entry_time": now.isoformat(),
            "side": risk.side,
            "entry": fill.fill_price,
            "stop": risk.stop,
            "target": risk.target,
            "lot": risk.lot,
            "rr": risk.rr,
            "risk_pct": risk.risk_pct,
            "market_state": market_state.to_dict(),
            "expected_r": meta.expected_r,
            "edge_score": meta.edge_score,
            "decision": "accepted",
            "reason": meta.reason,
        }

        pos = OpenPosition(
            trade_id=fill.trade_id,
            signal_id=signal_id,
            side=risk.side,
            entry_time=now,
            entry_price=fill.fill_price,
            stop_loss=risk.stop,
            take_profit=risk.target,
            lot=risk.lot,
            risk_pct=risk.risk_pct,
            atr=risk.atr,
            correlation_id=corr,
            meta={
                "probability": probability,
                "meta_probability": meta.edge_score,
                "confidence": confidence,
                "expected_r": meta.expected_r,
                "session": market_state.session,
                "regime": market_state.regime_raw,
                "market_state": market_state.to_dict(),
            },
        )
        self.state.open_positions[fill.trade_id] = pos
        self.state.trades_today += 1

        self._publish(
            EventType.TRADE_OPENED,
            {
                "trade_id": fill.trade_id,
                "signal_id": signal_id,
                "symbol": symbol,
                "side": risk.side,
                "entry_time": now,
                "entry_price": fill.fill_price,
                "stop_loss": risk.stop,
                "take_profit": risk.target,
                "lot": risk.lot,
                "risk_pct": risk.risk_pct,
                "rr": risk.rr,
                "expected_r": meta.expected_r,
                "edge_score": meta.edge_score,
                "latency_ms": fill.latency_ms,
                "broker_response": fill.broker_response,
                "spread": fill.spread,
                "slippage": fill.slippage,
                "retry_count": fill.retry_count,
                "session": market_state.session,
                "regime": market_state.regime_raw,
                "probability": probability,
                "meta_probability": meta.edge_score,
                "confidence": confidence,
                "environment": self.environment,
            },
            corr,
        )
        self._publish(
            EventType.AUDIT,
            {"component": "pipeline", "action": "trade_opened", "detail": {"trade_id": fill.trade_id}},
            corr,
        )
        return PortfolioDecision(accept=True, reason="opened", trade=trade_json)

    def on_bar(self, *, high: float, low: float, close: float, timestamp: datetime) -> list[dict[str, Any]]:
        closed = []
        for tid, pos in list(self.state.open_positions.items()):
            mae, mfe = PaperBroker.mark_excursions(pos.side, pos.entry_price, high, low)
            pos.mae = max(pos.mae, mae)
            pos.mfe = max(pos.mfe, mfe)
            hit_sl = (low <= pos.stop_loss) if pos.side == "long" else (high >= pos.stop_loss)
            hit_tp = (high >= pos.take_profit) if pos.side == "long" else (low <= pos.take_profit)
            reason = None
            exit_px = close
            if hit_sl and hit_tp:
                reason, exit_px = "SL", pos.stop_loss
            elif hit_sl:
                reason, exit_px = "SL", pos.stop_loss
            elif hit_tp:
                reason, exit_px = "TP", pos.take_profit
            if reason is None:
                continue
            fill = self.broker.close_order(tid, exit_px)
            pnl = PaperBroker.pnl(pos.side, pos.entry_price, exit_px, pos.lot)
            ru = max(self.state.r_unit(), 1e-9)
            pnl_r = pnl / ru
            self.state.apply_pnl(pnl)
            if pnl > 0:
                self.state.wins_today += 1
            del self.state.open_positions[tid]
            duration = int((timestamp - pos.entry_time).total_seconds())
            payload = {
                "trade_id": tid,
                "signal_id": pos.signal_id,
                "symbol": self.symbol,
                "side": pos.side,
                "entry_time": pos.entry_time,
                "exit_time": timestamp,
                "entry_price": pos.entry_price,
                "exit_price": exit_px,
                "stop_loss": pos.stop_loss,
                "take_profit": pos.take_profit,
                "lot": pos.lot,
                "risk_pct": pos.risk_pct,
                "pnl": pnl,
                "pnl_r": pnl_r,
                "duration_seconds": duration,
                "mae": pos.mae,
                "mfe": pos.mfe,
                "exit_reason": reason,
                "status": "closed",
                "session": pos.meta.get("session"),
                "regime": pos.meta.get("regime"),
                "probability": pos.meta.get("probability"),
                "meta_probability": pos.meta.get("meta_probability"),
                "confidence": pos.meta.get("confidence"),
                "expected_r": pos.meta.get("expected_r"),
                "equity": self.state.equity,
                "broker_response": fill.broker_response,
                "environment": self.environment,
            }
            self._publish(EventType.TRADE_CLOSED, payload, pos.correlation_id)
            closed.append(payload)
        return closed

    def emit_daily_summary(self, day: datetime | None = None, *, extra: dict[str, Any] | None = None) -> None:
        now = day or datetime.now(timezone.utc)
        trades = int(self.state.trades_today)
        wins = int(self.state.wins_today)
        losses = max(0, trades - wins)
        wr = (wins / trades) if trades else 0.0
        start_eq = float(self.state.day_start_equity or self.state.equity)
        pnl = float(self.state.pnl_today)
        pnl_pct = (pnl / start_eq * 100.0) if start_eq > 0 else 0.0
        status = "ok"
        if self.state.heat_triggered_today or self.state.equity <= 0:
            status = "degraded"
        report_date = self.state.day.isoformat() if self.state.day else now.date().isoformat()
        payload: dict[str, Any] = {
            "date": report_date,
            "status": status,
            "equity": self.state.equity,
            "daily_r": self.state.day_pnl / max(self.state.r_unit(), 1e-9),
            "drawdown": self.state.drawdown(),
            "heat_triggered": self.state.heat_triggered_today,
            "trades": trades,
            "wins": wins,
            "losses": losses,
            "winrate": wr,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "skipped": self.state.skips,
            "meta_rejects": self.state.meta_rejects,
            "confidence_rejects": self.state.confidence_rejects,
            "uptime_seconds": time.monotonic() - float(self.state.started_mono),
        }
        if extra:
            payload.update(extra)
        self._publish(EventType.DAILY_SUMMARY, payload)
        self.state.roll_day(now)

    def _skip(
        self,
        corr: str,
        signal_id: str,
        symbol: str,
        primary: PrimaryOut,
        meta: MetaOut,
        reason: str,
        threshold: float | None,
        current: float | None,
        timestamp: datetime,
        entry_price: float = 0.0,
    ) -> None:
        self._publish(
            EventType.TRADE_SKIPPED,
            {
                "signal_id": signal_id,
                "symbol": symbol,
                "reason": reason,
                "threshold": threshold,
                "current_value": current,
                "entry_price": entry_price,
                "timestamp": timestamp,
                "environment": self.environment,
                "detail": {
                    "meta_probability": meta.edge_score,
                    "expected_r": meta.expected_r,
                    "probability": primary.probability,
                    "side": primary.side,
                },
            },
            corr,
        )
