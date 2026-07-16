"""LabelPipeline — MarketData → strategy(side) → validate → LabelSet."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

from data.entities.market_data import MarketData
from labels.entities.label_set import LabelSet
from labels.exceptions import StrategyError
from labels.registry.strategy_registry import StrategyRegistry
from labels.validators.label_validator import LabelValidator

logger = logging.getLogger(__name__)


class LabelPipeline:
    def __init__(self, config: dict[str, Any], validator: LabelValidator | None = None) -> None:
        self._cfg = config
        StrategyRegistry.discover()
        horizon = None
        params = config.get("params") or {}
        if "horizon" in params:
            horizon = int(params["horizon"])
        self._validator = validator or LabelValidator(expected_horizon=horizon)

    def _resolve_params(self, strategy: str, side: str) -> dict[str, Any]:
        params = dict(self._cfg.get("params") or {})
        overrides = (self._cfg.get("strategies") or {}).get(strategy)
        if isinstance(overrides, dict):
            params = {**params, **dict(overrides.get("params") or overrides)}
        if "tp_atr_mult" not in params and "take_profit_atr_mult" in params:
            params["tp_atr_mult"] = params["take_profit_atr_mult"]
        if "sl_atr_mult" not in params and "stop_loss_atr_mult" in params:
            params["sl_atr_mult"] = params["stop_loss_atr_mult"]
        params["side"] = side
        return params

    def run(
        self,
        market_data: MarketData,
        *,
        strategy: str | None = None,
        side: str = "long",
    ) -> tuple[LabelSet, dict]:
        key = (strategy or str(self._cfg.get("strategy", "triple_barrier"))).lower()
        side_l = side.lower()
        if side_l not in {"long", "short"}:
            raise StrategyError(f"invalid side: {side!r}")
        params = self._resolve_params(key, side_l)

        t0 = time.perf_counter()
        try:
            strat = StrategyRegistry.create(key, params)
            labels = strat.generate(market_data)
        except Exception as exc:
            raise StrategyError(f"strategy {key!r} side={side_l} failed: {exc}") from exc

        label_set = LabelSet(
            symbol=market_data.symbol,
            timeframe=market_data.timeframe,
            strategy=key,
            side=side_l,  # type: ignore[arg-type]
            label_version=str(self._cfg.get("label_version", "v1")),
            created_at=datetime.now(tz=timezone.utc),
            labels=labels,
        )
        report = self._validator.validate(label_set)
        elapsed = time.perf_counter() - t0
        logger.info(
            "LabelPipeline complete | strategy=%s side=%s n=%s duration=%.3fs",
            key,
            side_l,
            label_set.size,
            elapsed,
        )
        return label_set, report
