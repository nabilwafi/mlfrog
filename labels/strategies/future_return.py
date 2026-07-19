"""Future-return labeling (functional stub — forward return sign)."""

from __future__ import annotations

import logging

from data.entities.market_data import MarketData
from labels.entities.label import Label
from labels.exceptions import StrategyError
from labels.registry.strategy_registry import StrategyRegistry
from labels.strategies.base_strategy import BaseLabelStrategy

logger = logging.getLogger(__name__)


@StrategyRegistry.register
class FutureReturnStrategy(BaseLabelStrategy):
    """
    params: horizon (int), threshold (float, default 0)
    """

    def name(self) -> str:
        return "future_return"

    def generate(self, market_data: MarketData) -> tuple[Label, ...]:
        horizon = int(self.params.get("horizon", 8))
        threshold = float(self.params.get("threshold", 0.0))
        side = str(self.params.get("side", "long")).lower()
        if side not in {"long", "short"}:
            raise StrategyError(f"unsupported side: {side!r}")
        if horizon < 1:
            raise StrategyError("horizon must be >= 1")
        candles = market_data.candles
        n = len(candles)
        out: list[Label] = []
        for i in range(0, n - horizon):
            entry = candles[i].close
            future = candles[i + horizon].close
            if side == "long":
                ret = (future - entry) / entry
            else:
                ret = (entry - future) / entry
            if ret > threshold:
                y, reason = 1, "TP"
            elif ret < -threshold:
                y, reason = -1, "SL"
            else:
                y, reason = 0, "TIMEOUT"
            if side == "long":
                tp, sl = entry * 1.01, entry * 0.99
            else:
                tp, sl = entry * 0.99, entry * 1.01
            out.append(
                Label(
                    timestamp=candles[i].timestamp,
                    entry_price=entry,
                    tp_price=float(tp),
                    sl_price=float(sl),
                    expire_timestamp=candles[i + horizon].timestamp,
                    holding_bars=horizon,
                    realized_return=float(ret),
                    exit_reason=reason,  # type: ignore[arg-type]
                    side=side,  # type: ignore[arg-type]
                    label=y,
                    metadata={"strategy": "future_return", "threshold": threshold},
                )
            )
        logger.info("FutureReturn stub generated | labels=%s", len(out))
        return tuple(out)
