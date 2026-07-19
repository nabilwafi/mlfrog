"""Triple-barrier labeling (López de Prado style) — LONG by default."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from data.entities.market_data import MarketData
from labels.entities.label import Label
from labels.exceptions import StrategyError
from labels.registry.strategy_registry import StrategyRegistry
from labels.strategies.base_strategy import BaseLabelStrategy

logger = logging.getLogger(__name__)


def _wilder_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    n = len(close)
    tr = np.empty(n, dtype=float)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    atr = np.full(n, np.nan, dtype=float)
    if n < period:
        return atr
    atr[period - 1] = float(np.mean(tr[:period]))
    alpha = 1.0 / period
    for i in range(period, n):
        atr[i] = atr[i - 1] * (1.0 - alpha) + tr[i] * alpha
    return atr


@StrategyRegistry.register
class TripleBarrierStrategy(BaseLabelStrategy):
    """
    Config params (all required unless noted):
      horizon: int
      tp_atr_mult: float
      sl_atr_mult: float
      atr_period: int (default 14)
      side: 'long' | 'short' (default long)
      min_atr: float (default 0) — skip bars with ATR below this
      risk_reward_ratio: optional; if set, tp_atr_mult = sl_atr_mult * ratio (overrides tp)
    """

    def name(self) -> str:
        return "triple_barrier"

    def generate(self, market_data: MarketData) -> tuple[Label, ...]:
        if market_data.total_candles == 0:
            raise StrategyError("empty MarketData")

        horizon = int(self.params["horizon"])
        sl_mult = float(self.params["sl_atr_mult"])
        if "risk_reward_ratio" in self.params and self.params["risk_reward_ratio"] is not None:
            rr = float(self.params["risk_reward_ratio"])
            if rr <= 0:
                raise StrategyError("risk_reward_ratio must be > 0")
            tp_mult = sl_mult * rr
        else:
            tp_mult = float(self.params["tp_atr_mult"])
        atr_period = int(self.params.get("atr_period", 14))
        side = str(self.params.get("side", "long")).lower()
        min_atr = float(self.params.get("min_atr", 0.0))
        if side not in {"long", "short"}:
            raise StrategyError(f"unsupported side: {side!r}")
        if horizon < 1:
            raise StrategyError("horizon must be >= 1")
        if tp_mult <= 0 or sl_mult <= 0:
            raise StrategyError("tp/sl ATR multipliers must be > 0")

        candles = market_data.candles
        n = len(candles)
        high = np.array([c.high for c in candles], dtype=float)
        low = np.array([c.low for c in candles], dtype=float)
        close = np.array([c.close for c in candles], dtype=float)
        atr = _wilder_atr(high, low, close, atr_period)

        out: list[Label] = []
        # Last usable entry index: need room for at least 1 forward bar path up to horizon
        last_i = n - 2
        for i in range(atr_period - 1, last_i + 1):
            a = atr[i]
            if not np.isfinite(a) or a <= min_atr:
                continue
            entry = float(close[i])
            ts = candles[i].timestamp
            if side == "long":
                tp = entry + tp_mult * a
                sl = entry - sl_mult * a
            else:
                tp = entry - tp_mult * a
                sl = entry + sl_mult * a

            end_j = min(i + horizon, n - 1)
            expire_ts = candles[end_j].timestamp
            exit_reason = "TIMEOUT"
            exit_price = float(close[end_j])
            holding = end_j - i
            hit_j = end_j

            for j in range(i + 1, end_j + 1):
                # Intra-bar: if both hit in same bar, conservative = SL first (worse case)
                if side == "long":
                    hit_sl = low[j] <= sl
                    hit_tp = high[j] >= tp
                    if hit_sl and hit_tp:
                        exit_reason, exit_price, hit_j, holding = "SL", sl, j, j - i
                        break
                    if hit_sl:
                        exit_reason, exit_price, hit_j, holding = "SL", sl, j, j - i
                        break
                    if hit_tp:
                        exit_reason, exit_price, hit_j, holding = "TP", tp, j, j - i
                        break
                else:
                    hit_sl = high[j] >= sl
                    hit_tp = low[j] <= tp
                    if hit_sl and hit_tp:
                        exit_reason, exit_price, hit_j, holding = "SL", sl, j, j - i
                        break
                    if hit_sl:
                        exit_reason, exit_price, hit_j, holding = "SL", sl, j, j - i
                        break
                    if hit_tp:
                        exit_reason, exit_price, hit_j, holding = "TP", tp, j, j - i
                        break

            if side == "long":
                realized = (exit_price - entry) / entry
            else:
                realized = (entry - exit_price) / entry

            if exit_reason == "TP":
                y = 1
            elif exit_reason == "SL":
                y = -1
            else:
                y = 0

            out.append(
                Label(
                    timestamp=ts,
                    entry_price=entry,
                    tp_price=float(tp),
                    sl_price=float(sl),
                    expire_timestamp=expire_ts,
                    holding_bars=int(holding),
                    realized_return=float(realized),
                    exit_reason=exit_reason,  # type: ignore[arg-type]
                    side=side,  # type: ignore[arg-type]
                    label=y,
                    metadata={
                        "atr": float(a),
                        "tp_atr_mult": tp_mult,
                        "sl_atr_mult": sl_mult,
                        "horizon": horizon,
                        "exit_index": int(hit_j),
                        "entry_index": int(i),
                    },
                )
            )

        logger.info(
            "TripleBarrier generated | symbol=%s n=%s labels=%s",
            market_data.symbol,
            n,
            len(out),
        )
        return tuple(out)
