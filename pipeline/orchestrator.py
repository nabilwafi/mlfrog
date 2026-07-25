"""TradingPipeline orchestrator — L1–L6 wiring for paper + scored replay."""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from pipeline.l1_features.factory import FeatureFactory
from pipeline.l2_market_state.infer import infer_market_state
from pipeline.l3_primary.alpha import primary_from_signal
from pipeline.l4_meta_edge.edge import decide_meta
from pipeline.l5_risk.engine import build_risk
from pipeline.l6_portfolio.execution import PortfolioExecution
from pipeline.logging.trace import TraceLogger
from pipeline.types import TraceRecord
from production.events.types import EventType, make_event


class TradingPipeline:
    """
    Shared decision path:
      PrimaryOut + Meta edge → Risk (expected_r) → PortfolioExecution
    L1–L3 live scoring stays in adapters; replay feeds precomputed scores here.
    """

    def __init__(
        self,
        *,
        portfolio: PortfolioExecution,
        trace: TraceLogger | None = None,
    ) -> None:
        self.portfolio = portfolio
        self.trace = trace or TraceLogger()

    def run_scored_signal(
        self,
        *,
        timestamp: datetime,
        symbol: str,
        side: str,
        probability: float,
        meta_probability: float,
        entry_price: float,
        atr: float,
        session: str = "unknown",
        regime: str = "unknown",
        bar_key: str = "",
        confidence: float = 0.0,
        feature_row: dict[str, Any] | None = None,
        features_hash: str = "",
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        now = timestamp if timestamp.tzinfo else timestamp.replace(tzinfo=timezone.utc)
        corr = uuid.uuid4().hex
        signal_id = uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"{symbol}:{side}:{bar_key or now.isoformat()}",
        ).hex

        market_state = infer_market_state(feature_row, session=session, regime=regime)
        primary = primary_from_signal(side=side, probability=probability)
        meta = decide_meta(meta_probability)
        risk = None
        if meta.trade and primary.side in {"LONG", "SHORT"}:
            risk = build_risk(
                side=primary.side.lower(),
                entry=float(entry_price),
                atr=float(atr),
                equity=self.portfolio.state.equity,
                expected_r=meta.expected_r,
            )

        if self.portfolio.bus is not None:
            ms = (time.perf_counter() - t0) * 1000
            self.portfolio.bus.publish(
                make_event(EventType.METRIC, {"name": "inference_latency_ms", "value": ms}, correlation_id=corr)
            )

        decision = self.portfolio.decide_and_execute(
            signal_id=signal_id,
            timestamp=now,
            symbol=symbol,
            primary=primary,
            meta=meta,
            risk=risk,
            market_state=market_state,
            probability=float(probability),
            confidence=float(confidence),
            correlation_id=corr,
            entry_price=float(entry_price),
        )

        fhash = features_hash
        if not fhash and feature_row:
            fhash = FeatureFactory.features_hash(feature_row)  # type: ignore[arg-type]
        self.trace.log(
            TraceRecord(
                signal_id=signal_id,
                time=now,
                features_hash=fhash,
                market_state=market_state.to_dict(),
                primary=asdict(primary),
                meta=asdict(meta),
                risk=asdict(risk) if risk else None,
                portfolio={"accept": decision.accept, "reason": decision.reason, "trade": decision.trade},
                entry=float(entry_price),
            )
        )

        if decision.accept:
            return {"status": "opened", "trade_id": signal_id, "correlation_id": corr, "trade": decision.trade}
        return {"status": "skipped", "reason": decision.reason, "correlation_id": corr}

    def on_bar(self, *, high: float, low: float, close: float, timestamp: datetime) -> list[dict[str, Any]]:
        return self.portfolio.on_bar(high=high, low=low, close=close, timestamp=timestamp)

    def emit_daily_summary(self, day: datetime | None = None, *, extra: dict[str, Any] | None = None) -> None:
        self.portfolio.emit_daily_summary(day, extra=extra)
