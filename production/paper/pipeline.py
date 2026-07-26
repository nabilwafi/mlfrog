"""Production signal filter + paper execution (frozen research policy)."""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol

from production import (
    BLOCK_OPPOSITE_SIDE,
    CONFIDENCE_ENABLED,
    CONFIDENCE_SKIP,
    EXIT_HORIZON_BARS,
    EXIT_MODE,
    FEATURE_VERSION,
    FIXED_LOT,
    LABEL_VERSION,
    MAX_OPEN_POSITIONS,
    META_AS_GATE,
    META_THRESHOLD,
    META_VERSION,
    MODEL_VERSION,
    PIPELINE_VERSION,
    SIZE_FROM_PRIMARY,
    SIZE_MODE,
    TRAIL_ACTIVATE_R,
    TRAIL_ATR_MULT,
)
from production.events.bus import EventBus
from production.events.types import EventType, make_event
from production.paper.broker import PaperBroker
from production.paper.edge_sizing import expected_r_from_edge, risk_pct_from_edge
from production.paper.state import OpenPosition, PortfolioState
from research.portfolio_backtest.services.engine import _lots_from_equity
from settings.strategy import SL_ATR_MULT, TP_ATR_MULT

logger = logging.getLogger(__name__)


@dataclass
class IncomingSignal:
    """Pre-computed frozen-model outputs for one bar (paper / replay)."""

    timestamp: datetime
    symbol: str
    side: str
    probability: float
    meta_probability: float
    confidence: float
    entry_price: float
    atr: float
    session: str = "unknown"
    regime: str = "unknown"
    trend: str = "unknown"
    volatility: str = "unknown"
    momentum: str = "unknown"
    structure: str = "unknown"
    bar_key: str = ""  # for idempotency


class SignalSource(Protocol):
    def next_signals(self) -> list[IncomingSignal]: ...


class ProductionPipeline:
    """
    Hot path: filter → broker execute → publish events → return.
    Never waits on DB / Telegram.
    """

    def __init__(
        self,
        *,
        bus: EventBus,
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

    def process_signal(self, sig: IncomingSignal) -> dict[str, Any]:
        t_inf0 = time.perf_counter()
        now = sig.timestamp if sig.timestamp.tzinfo else sig.timestamp.replace(tzinfo=timezone.utc)
        self.state.roll_day(now)
        corr = uuid.uuid4().hex
        signal_id = uuid.uuid5(uuid.NAMESPACE_URL, f"{sig.symbol}:{sig.side}:{sig.bar_key or sig.timestamp.isoformat()}").hex

        # Duplicate prevention
        dup_key = f"{signal_id}"
        if not self.state.register_signal_key(dup_key):
            self._skip(corr, signal_id, sig, "duplicate", None, None)
            return {"status": "skipped", "reason": "duplicate", "correlation_id": corr}

        # Multi-entry cap + optional no-opposite
        n_open = len(self.state.open_positions)
        if n_open >= int(MAX_OPEN_POSITIONS):
            self.state.skips += 1
            self._skip(corr, signal_id, sig, "max_open", float(MAX_OPEN_POSITIONS), float(n_open))
            return {"status": "skipped", "reason": "max_open", "correlation_id": corr}

        if BLOCK_OPPOSITE_SIDE and n_open > 0:
            open_sides = {p.side for p in self.state.open_positions.values()}
            sig_side = str(sig.side).lower()
            if open_sides and sig_side not in open_sides:
                self.state.skips += 1
                self._skip(corr, signal_id, sig, "opposite_blocked", 1.0, float(n_open))
                return {"status": "skipped", "reason": "opposite_blocked", "correlation_id": corr}

        # Publish signal row (accepted TBD)
        base_signal = {
            "signal_id": signal_id,
            "symbol": sig.symbol,
            "side": sig.side,
            "probability": sig.probability,
            "meta_probability": sig.meta_probability,
            "confidence": sig.confidence,
            "expected_r": expected_r_from_edge(float(sig.meta_probability)),
            "threshold_meta": META_THRESHOLD,
            "threshold_confidence": CONFIDENCE_SKIP,
            "meta_as_gate": META_AS_GATE,
            "confidence_enabled": CONFIDENCE_ENABLED,
            "model_version": MODEL_VERSION,
            "meta_version": META_VERSION,
            "feature_version": FEATURE_VERSION,
            "label_version": LABEL_VERSION,
            "pipeline_version": PIPELINE_VERSION,
            "accepted": False,
            "session": sig.session,
            "regime": sig.regime,
        }

        inference_ms = (time.perf_counter() - t_inf0) * 1000
        self.bus.publish(
            make_event(EventType.METRIC, {"name": "inference_latency_ms", "value": inference_ms}, correlation_id=corr)
        )

        # Meta as gate only if restored; step 3 = edge for sizing only
        if META_AS_GATE and float(sig.meta_probability) < META_THRESHOLD:
            self.state.meta_rejects += 1
            self.state.skips += 1
            self.bus.publish(make_event(EventType.SIGNAL, {**base_signal, "accepted": False}, correlation_id=corr))
            self._skip(corr, signal_id, sig, "meta", META_THRESHOLD, sig.meta_probability)
            return {"status": "skipped", "reason": "meta", "correlation_id": corr}

        if CONFIDENCE_ENABLED and float(sig.confidence) < CONFIDENCE_SKIP:
            self.state.confidence_rejects += 1
            self.state.skips += 1
            self.bus.publish(make_event(EventType.SIGNAL, {**base_signal, "accepted": False}, correlation_id=corr))
            self._skip(corr, signal_id, sig, "confidence", CONFIDENCE_SKIP, sig.confidence)
            return {"status": "skipped", "reason": "confidence", "correlation_id": corr}

        # Portfolio heat
        if self.state.daily_heat_blocked():
            self.state.heat_triggered_today += 1
            self.state.skips += 1
            self.bus.publish(make_event(EventType.HEAT_TRIGGERED, {"day_pnl": self.state.day_pnl, "equity": self.state.equity}, correlation_id=corr))
            self.bus.publish(make_event(EventType.SIGNAL, {**base_signal, "accepted": False}, correlation_id=corr))
            self._skip(corr, signal_id, sig, "heat", -1.0, self.state.day_pnl / max(self.state.r_unit(), 1e-9))
            return {"status": "skipped", "reason": "heat", "correlation_id": corr}

        # Risk / sizing
        edge = float(sig.probability if SIZE_FROM_PRIMARY else sig.meta_probability)
        expected_r = expected_r_from_edge(edge)
        risk_pct = risk_pct_from_edge(edge)

        if self.state.equity <= 0 or sig.atr <= 0:
            self.state.skips += 1
            self.bus.publish(make_event(EventType.SIGNAL, {**base_signal, "accepted": False}, correlation_id=corr))
            self._skip(corr, signal_id, sig, "risk", risk_pct, self.state.equity)
            return {"status": "skipped", "reason": "risk", "correlation_id": corr}

        if str(SIZE_MODE).lower() == "fixed":
            lots = float(FIXED_LOT)
        else:
            lots = _lots_from_equity(
                self.state.equity,
                float(sig.atr),
                mode="risk",
                fixed_lots=None,
                risk_pct=risk_pct,
                enforce_volume_min=True,
            )
        if lots <= 0:
            self.state.skips += 1
            self._skip(corr, signal_id, sig, "risk", risk_pct, lots)
            return {"status": "skipped", "reason": "risk", "correlation_id": corr}

        side = str(sig.side).lower()
        entry = float(sig.entry_price)
        atr = float(sig.atr)
        if side == "long":
            sl = entry - SL_ATR_MULT * atr
            tp = entry + TP_ATR_MULT * atr
        else:
            sl = entry + SL_ATR_MULT * atr
            tp = entry - TP_ATR_MULT * atr

        # --- SYNC broker only ---
        t_br0 = time.perf_counter()
        fill = self.broker.open_order(
            side=side,
            entry_price=entry,
            stop_loss=sl,
            take_profit=tp,
            lot=lots,
            trade_id=signal_id,  # idempotent: trade_id == signal_id
        )
        broker_ms = (time.perf_counter() - t_br0) * 1000
        self.bus.publish(
            make_event(EventType.METRIC, {"name": "broker_latency_ms", "value": broker_ms}, correlation_id=corr)
        )

        if not fill.success:
            self.bus.publish(
                make_event(
                    EventType.EXECUTION_ERROR,
                    {
                        "trade_id": fill.trade_id,
                        "error_message": fill.error_message,
                        "retry_count": fill.retry_count,
                        "latency_ms": fill.latency_ms,
                        "broker_response": fill.broker_response,
                    },
                    correlation_id=corr,
                )
            )
            return {"status": "error", "correlation_id": corr, "error": fill.error_message}

        self.bus.publish(make_event(EventType.SIGNAL, {**base_signal, "accepted": True}, correlation_id=corr))

        pos = OpenPosition(
            trade_id=fill.trade_id,
            signal_id=signal_id,
            side=side,
            entry_time=now,
            entry_price=fill.fill_price,
            stop_loss=sl,
            take_profit=tp,
            lot=lots,
            risk_pct=risk_pct,
            atr=atr,
            correlation_id=corr,
            extreme_fav=fill.fill_price,
            bars_held=0,
            meta={
                "probability": sig.probability,
                "meta_probability": sig.meta_probability,
                "confidence": sig.confidence,
                "expected_r": expected_r,
                "session": sig.session,
                "regime": sig.regime,
                "trend": sig.trend,
                "volatility": sig.volatility,
                "momentum": sig.momentum,
                "structure": sig.structure,
                "exit_mode": EXIT_MODE,
                "trail_atr_mult": TRAIL_ATR_MULT,
                "trail_activate_r": TRAIL_ACTIVATE_R,
                "edge_score": edge,
                "initial_sl": sl,
            },
        )
        self.state.open_positions[fill.trade_id] = pos
        self.state.trades_today += 1

        self.bus.publish(
            make_event(
                EventType.TRADE_OPENED,
                {
                    "trade_id": fill.trade_id,
                    "signal_id": signal_id,
                    "symbol": sig.symbol,
                    "side": side,
                    "entry_time": now,
                    "entry_price": fill.fill_price,
                    "stop_loss": sl,
                    "take_profit": tp,
                    "lot": lots,
                    "risk_pct": risk_pct,
                    "expected_r": expected_r,
                    "edge_score": edge,
                    "latency_ms": fill.latency_ms,
                    "broker_response": fill.broker_response,
                    "spread": fill.spread,
                    "slippage": fill.slippage,
                    "retry_count": fill.retry_count,
                    "session": sig.session,
                    "regime": sig.regime,
                    "trend": sig.trend,
                    "volatility": sig.volatility,
                    "momentum": sig.momentum,
                    "structure": sig.structure,
                    "probability": sig.probability,
                    "meta_probability": sig.meta_probability,
                    "confidence": sig.confidence,
                    "exit_mode": EXIT_MODE,
                    "trail_atr_mult": TRAIL_ATR_MULT,
                    "trail_activate_r": TRAIL_ACTIVATE_R,
                    "environment": self.environment,
                },
                correlation_id=corr,
            )
        )
        self.bus.publish(
            make_event(
                EventType.AUDIT,
                {"component": "pipeline", "action": "trade_opened", "detail": {"trade_id": fill.trade_id}},
                correlation_id=corr,
            )
        )
        return {"status": "opened", "trade_id": fill.trade_id, "correlation_id": corr}

    def _update_trail(self, pos: OpenPosition, *, high: float, low: float) -> bool:
        """Ratchet SL after +TRAIL_ACTIVATE_R using TRAIL_ATR_MULT. Returns True if SL moved."""
        if str(EXIT_MODE).lower() != "atr_trail":
            return False
        atr = float(pos.atr)
        if atr <= 0:
            return False
        prev = float(pos.stop_loss)
        one_r = float(SL_ATR_MULT) * atr
        if pos.side == "long":
            pos.extreme_fav = max(float(pos.extreme_fav or pos.entry_price), float(high))
            fav = pos.extreme_fav - pos.entry_price
            if fav >= float(TRAIL_ACTIVATE_R) * one_r:
                trail_sl = pos.extreme_fav - float(TRAIL_ATR_MULT) * atr
                pos.stop_loss = max(pos.stop_loss, trail_sl)
        else:
            pos.extreme_fav = min(float(pos.extreme_fav or pos.entry_price), float(low))
            fav = pos.entry_price - pos.extreme_fav
            if fav >= float(TRAIL_ACTIVATE_R) * one_r:
                trail_sl = pos.extreme_fav + float(TRAIL_ATR_MULT) * atr
                pos.stop_loss = min(pos.stop_loss, trail_sl)
        return abs(float(pos.stop_loss) - prev) > 1e-9

    def on_bar(self, *, high: float, low: float, close: float, timestamp: datetime) -> list[dict[str, Any]]:
        """Mark open positions; ATR-trail ratchet then SL/TP/TIMEOUT (SL first same-bar)."""
        closed = []
        for tid, pos in list(self.state.open_positions.items()):
            pos.bars_held = int(pos.bars_held) + 1
            mae, mfe = PaperBroker.mark_excursions(pos.side, pos.entry_price, high, low)
            pos.mae = max(pos.mae, mae)
            pos.mfe = max(pos.mfe, mfe)
            moved = self._update_trail(pos, high=high, low=low)
            if moved:
                u_pnl = PaperBroker.pnl(pos.side, pos.entry_price, float(close), pos.lot)
                self.bus.publish(
                    make_event(
                        EventType.TRAIL_UPDATE,
                        {
                            "trade_id": tid,
                            "signal_id": pos.signal_id,
                            "symbol": self.symbol,
                            "side": pos.side,
                            "entry_price": pos.entry_price,
                            "mark_price": float(close),
                            "stop_loss": pos.stop_loss,
                            "lot": pos.lot,
                            "unrealized_pnl": u_pnl,
                            "trail_atr_mult": pos.meta.get("trail_atr_mult", TRAIL_ATR_MULT),
                            "trail_activate_r": pos.meta.get("trail_activate_r", TRAIL_ACTIVATE_R),
                            "session": pos.meta.get("session"),
                            "regime": pos.meta.get("regime"),
                            "trend": pos.meta.get("trend"),
                            "volatility": pos.meta.get("volatility"),
                            "momentum": pos.meta.get("momentum"),
                            "structure": pos.meta.get("structure"),
                            "timestamp": timestamp,
                            "environment": self.environment,
                        },
                        correlation_id=pos.correlation_id,
                    )
                )

            hit_sl = (low <= pos.stop_loss) if pos.side == "long" else (high >= pos.stop_loss)
            hit_tp = (high >= pos.take_profit) if pos.side == "long" else (low <= pos.take_profit)
            reason = None
            exit_px = close
            if hit_sl and hit_tp:
                reason, exit_px = "SL", pos.stop_loss
            elif hit_sl:
                init_sl = float(pos.meta.get("initial_sl", pos.stop_loss))
                reason = "TRAIL" if abs(pos.stop_loss - init_sl) > 1e-9 else "SL"
                exit_px = pos.stop_loss
            elif hit_tp:
                reason, exit_px = "TP", pos.take_profit
            elif pos.bars_held >= int(EXIT_HORIZON_BARS):
                reason, exit_px = "TIMEOUT", close
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
                "trend": pos.meta.get("trend"),
                "volatility": pos.meta.get("volatility"),
                "momentum": pos.meta.get("momentum"),
                "structure": pos.meta.get("structure"),
                "probability": pos.meta.get("probability"),
                "meta_probability": pos.meta.get("meta_probability"),
                "confidence": pos.meta.get("confidence"),
                "trail_atr_mult": pos.meta.get("trail_atr_mult", TRAIL_ATR_MULT),
                "equity": self.state.equity,
                "broker_response": fill.broker_response,
                "environment": self.environment,
            }
            self.bus.publish(make_event(EventType.TRADE_CLOSED, payload, correlation_id=pos.correlation_id))
            closed.append(payload)
        return closed

    def emit_daily_summary(
        self,
        day: datetime | None = None,
        *,
        extra: dict[str, Any] | None = None,
        roll: bool = True,
    ) -> None:
        """Publish stats for the current trading day. Optionally roll calendar after."""
        now = day or datetime.now(timezone.utc)
        if self.state.day is None:
            self.state.day = now.date()
            if self.state.day_start_equity <= 0:
                self.state.day_start_equity = float(self.state.equity)
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
        # Prefer broker balance when synced; else day-start equity (paper)
        bal = self.state.balance
        payload: dict[str, Any] = {
            "date": report_date,
            "symbol": self.symbol,
            "status": status,
            "balance": float(bal) if bal is not None else start_eq,
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
            "best_trade_pnl": self.state.best_trade_pnl_today,
            "worst_trade_pnl": self.state.worst_trade_pnl_today,
            "skipped": self.state.skips,
            "meta_rejects": self.state.meta_rejects,
            "confidence_rejects": self.state.confidence_rejects,
            "uptime_seconds": time.monotonic() - float(self.state.started_mono),
            "environment": self.environment,
        }
        if extra:
            payload.update(extra)
        self.bus.publish(make_event(EventType.DAILY_SUMMARY, payload))
        # roll after publish so midnight summary still has yesterday's counters
        if roll:
            self.state.roll_day(now)

    def _skip(
        self,
        corr: str,
        signal_id: str,
        sig: IncomingSignal,
        reason: str,
        threshold: float | None,
        current: float | None,
    ) -> None:
        self.bus.publish(
            make_event(
                EventType.TRADE_SKIPPED,
                {
                    "signal_id": signal_id,
                    "symbol": sig.symbol,
                    "reason": reason,
                    "threshold": threshold,
                    "current_value": current,
                    "entry_price": float(sig.entry_price),
                    "timestamp": sig.timestamp,
                    "environment": self.environment,
                    "detail": {
                        "meta_probability": sig.meta_probability,
                        "confidence": sig.confidence,
                        "probability": sig.probability,
                    },
                },
                correlation_id=corr,
            )
        )
