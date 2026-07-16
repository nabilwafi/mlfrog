"""Causal asof-join of higher-TF context onto base-TF timestamps."""

from __future__ import annotations

import logging

import pandas as pd

from market_context.exceptions import ContextJoinError
from market_context.services.timeframe_utils import available_at

logger = logging.getLogger(__name__)


class ContextJoinService:
    """Attach latest *completed* context bar to each base (e.g. H1) candle."""

    def join(
        self,
        base_timestamps: pd.Series,
        context: pd.DataFrame,
        *,
        context_timeframe: str,
        context_cols: list[str],
    ) -> pd.DataFrame:
        if context.empty:
            raise ContextJoinError("context frame is empty")
        if "timestamp" not in context.columns:
            raise ContextJoinError("context missing timestamp")

        base = pd.DataFrame(
            {"timestamp": pd.to_datetime(base_timestamps, utc=True)}
        ).sort_values("timestamp")
        ctx = context.copy()
        ctx["timestamp"] = pd.to_datetime(ctx["timestamp"], utc=True)
        ctx = ctx.sort_values("timestamp").reset_index(drop=True)

        missing = [c for c in context_cols if c not in ctx.columns]
        if missing:
            raise ContextJoinError(f"context missing columns: {missing}")

        right = ctx[["timestamp", *context_cols]].copy()
        right = right.rename(columns={"timestamp": "context_bar_timestamp"})
        right["available_at"] = available_at(ctx["timestamp"], context_timeframe)
        right = right.sort_values("available_at").reset_index(drop=True)

        merged = pd.merge_asof(
            base,
            right,
            left_on="timestamp",
            right_on="available_at",
            direction="backward",
        )
        # available_at is join key only
        if "available_at" in merged.columns:
            merged = merged.drop(columns=["available_at"])

        attached = int(merged[context_cols[0]].notna().sum()) if context_cols else 0
        logger.info(
            "Context asof-join | base=%s attached=%s context_tf=%s",
            len(base),
            attached,
            context_timeframe,
        )
        if attached == 0:
            raise ContextJoinError("asof-join attached zero context rows")
        return merged.reset_index(drop=True)
