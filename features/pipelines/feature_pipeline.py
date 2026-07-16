"""FeaturePipeline — MarketData → indicators → interactions → FeatureSet."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from data.entities.market_data import MarketData
from features.entities.feature import Feature
from features.entities.feature_set import FeatureSet
from features.exceptions import FeatureError, IndicatorError
from features.registry.indicator_registry import IndicatorRegistry
from features.validators.feature_validator import FeatureValidator

logger = logging.getLogger(__name__)


def market_data_to_frame(market_data: MarketData) -> pd.DataFrame:
    rows = [
        {
            "timestamp": c.timestamp,
            "open": c.open,
            "high": c.high,
            "low": c.low,
            "close": c.close,
            "tick_volume": c.tick_volume,
            "spread": c.spread,
            "real_volume": c.real_volume,
        }
        for c in market_data.candles
    ]
    return pd.DataFrame(rows)


class FeaturePipeline:
    """
    Config-driven pipeline. Indicators/interactions are resolved via IndicatorRegistry only.
    Future SMC/ICT modules register the same way — no pipeline edits required.
    """

    def __init__(
        self,
        config: dict[str, Any],
        validator: FeatureValidator | None = None,
    ) -> None:
        self._cfg = config
        self._validator = validator or FeatureValidator(allow_nan=True, max_nan_ratio=1.0)
        IndicatorRegistry.discover()

    def run(self, market_data: MarketData) -> tuple[FeatureSet, dict[str, int]]:
        if market_data.total_candles == 0:
            raise FeatureError("cannot build features from empty MarketData")

        t0 = time.perf_counter()
        df = market_data_to_frame(market_data)
        work = df.copy()

        indicator_specs = list(self._cfg.get("indicators", []))
        interaction_specs = list(self._cfg.get("interactions", []))

        indicator_cols_before = set(work.columns)
        for spec in indicator_specs:
            work = self._apply_spec(work, spec, stage="indicator")
        n_indicator_features = len(set(work.columns) - indicator_cols_before - {"timestamp", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"})

        after_ind = set(work.columns)
        for spec in interaction_specs:
            work = self._apply_spec(work, spec, stage="interaction")
        n_interaction_features = len(set(work.columns) - after_ind)

        feature_cols = [
            c
            for c in work.columns
            if c
            not in {
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "tick_volume",
                "spread",
                "real_volume",
            }
        ]
        if not feature_cols:
            raise FeatureError("pipeline produced zero feature columns")

        drop_na = bool(self._cfg.get("drop_na", True))
        nan_before = int(work[feature_cols].isna().any(axis=1).sum())
        if drop_na:
            work = work.dropna(subset=feature_cols).reset_index(drop=True)
        nan_removed = nan_before if drop_na else 0

        features: list[Feature] = []
        for col in feature_cols:
            series = work[col].astype(float)
            features.append(
                Feature(
                    name=col,
                    dtype=str(series.dtype),
                    values=tuple(float(x) for x in series.tolist()),
                    metadata={"source": "pipeline"},
                )
            )

        timestamps = tuple(pd.to_datetime(work["timestamp"], utc=True).tolist())
        # ensure datetime objects
        timestamps = tuple(
            t.to_pydatetime() if hasattr(t, "to_pydatetime") else t for t in timestamps
        )

        feature_set = FeatureSet(
            symbol=market_data.symbol,
            timeframe=market_data.timeframe,
            feature_version=str(self._cfg.get("feature_version", "v1")),
            pipeline_version=str(self._cfg.get("pipeline_version", "1.0")),
            created_at=datetime.now(tz=timezone.utc),
            features=tuple(features),
            timestamps=timestamps,
        )

        # After dropna, disallow remaining NaN/Inf
        FeatureValidator(allow_nan=False, min_rows=1).validate(feature_set)

        elapsed = time.perf_counter() - t0
        logger.info(
            "Pipeline complete | duration=%.3fs indicators=%s interactions=%s total=%s nan_removed=%s",
            elapsed,
            n_indicator_features,
            n_interaction_features,
            feature_set.feature_count,
            nan_removed,
        )
        stats = {
            "raw_candles": market_data.total_candles,
            "indicators": n_indicator_features,
            "interactions": n_interaction_features,
            "total_features": feature_set.feature_count,
            "nan_removed": nan_removed,
            "row_count": feature_set.row_count,
        }
        return feature_set, stats

    def _apply_spec(self, work: pd.DataFrame, spec: dict[str, Any], *, stage: str) -> pd.DataFrame:
        key = str(spec["name"])
        params = dict(spec.get("params") or {})
        t0 = time.perf_counter()
        try:
            indicator = IndicatorRegistry.create(key, params)
            result = indicator.calculate(work)
        except Exception as exc:
            raise IndicatorError(f"{stage} {key!r} failed: {exc}") from exc
        elapsed = time.perf_counter() - t0
        logger.info("Executed %s | name=%s duration=%.4fs", stage, key, elapsed)

        if isinstance(result, pd.Series):
            col = result.name or indicator.feature_prefix()
            if col in work.columns:
                raise FeatureError(f"duplicate feature column {col!r} from {key}")
            work = work.copy()
            work[col] = result.values
            return work
        if isinstance(result, pd.DataFrame):
            work = work.copy()
            for col in result.columns:
                if col in work.columns:
                    raise FeatureError(f"duplicate feature column {col!r} from {key}")
                work[col] = result[col].values
            return work
        raise IndicatorError(f"{key} returned unsupported type {type(result)}")
