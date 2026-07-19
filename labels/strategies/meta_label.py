"""Meta-labeling stub — secondary labels from primary event outcomes.

Full meta-labeling (primary model → bet size) lands in a later sprint.
This stub emits labels aligned to a simple primary triple-barrier-like path
using the same MarketData so the pipeline/registry stay exercisable.
"""

from __future__ import annotations

import logging

from data.entities.market_data import MarketData
from labels.entities.label import Label
from labels.exceptions import StrategyError
from labels.registry.strategy_registry import StrategyRegistry
from labels.strategies.base_strategy import BaseLabelStrategy
from labels.strategies.triple_barrier import TripleBarrierStrategy

logger = logging.getLogger(__name__)


@StrategyRegistry.register
class MetaLabelStrategy(BaseLabelStrategy):
    """
    Functional stub: runs an inner triple-barrier pass, then maps outcomes to
    meta-labels {1: take trade (TP), 0: skip (SL/TIMEOUT)}.

    params: same barrier params as triple_barrier (horizon, tp_atr_mult, sl_atr_mult, ...)
    """

    def name(self) -> str:
        return "meta_label"

    def generate(self, market_data: MarketData) -> tuple[Label, ...]:
        required = ("horizon", "tp_atr_mult", "sl_atr_mult")
        for key in required:
            if key not in self.params:
                raise StrategyError(f"meta_label requires param {key!r}")
        primary = TripleBarrierStrategy(self.params).generate(market_data)
        out: list[Label] = []
        for lab in primary:
            meta_y = 1 if lab.label == 1 else 0
            out.append(
                Label(
                    timestamp=lab.timestamp,
                    entry_price=lab.entry_price,
                    tp_price=lab.tp_price,
                    sl_price=lab.sl_price,
                    expire_timestamp=lab.expire_timestamp,
                    holding_bars=lab.holding_bars,
                    realized_return=lab.realized_return,
                    exit_reason=lab.exit_reason,
                    side=lab.side,
                    label=meta_y,
                    metadata={
                        **dict(lab.metadata),
                        "strategy": "meta_label",
                        "primary_label": lab.label,
                    },
                )
            )
        logger.info("MetaLabel stub generated | labels=%s", len(out))
        return tuple(out)
