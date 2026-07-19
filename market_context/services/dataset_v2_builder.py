"""Build dataset v2 = H1 engineered features + H4 context + labels (no retrain)."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from datasets.builders.dataset_builder import DatasetBuilder
from datasets.entities.dataset import Dataset
from market_context.exceptions import ContextInputError

logger = logging.getLogger(__name__)

_META = {
    "timestamp",
    "symbol",
    "timeframe",
    "feature_version",
    "label_version",
    "split",
    "side",
    "strategy",
    "label",
    "expire_timestamp",
    "holding_bars",
    "exit_reason",
    "context_bar_timestamp",
}


class DatasetV2Builder:
    """Enrich Sprint-6 H1 feature matrix with context columns, then time-split."""

    def __init__(
        self,
        *,
        dataset_config: dict[str, Any],
        datasets_root: str | Path,
        feature_version: str = "v2",
    ) -> None:
        self._builder = DatasetBuilder(dataset_config)
        self._datasets_root = Path(datasets_root)
        self._feature_version = feature_version

    def build_and_save(
        self,
        *,
        h1_features: pd.DataFrame,
        context_on_h1: pd.DataFrame,
        labels: pd.DataFrame,
        symbol: str,
        timeframe: str,
        side: str,
        strategy: str,
        label_version: str,
        context_cols: list[str],
    ) -> dict[str, Dataset]:
        if "timestamp" not in h1_features.columns:
            raise ContextInputError("H1 feature matrix missing timestamp")
        feats = h1_features.copy()
        feats["timestamp"] = pd.to_datetime(feats["timestamp"], utc=True)

        ctx = context_on_h1[["timestamp", *context_cols]].copy()
        ctx["timestamp"] = pd.to_datetime(ctx["timestamp"], utc=True)
        ctx = ctx.dropna(subset=context_cols)

        enriched = feats.merge(ctx, on="timestamp", how="inner")
        if enriched.empty:
            raise ContextInputError("no overlap between H1 features and context")

        lab = labels.copy()
        lab["timestamp"] = pd.to_datetime(lab["timestamp"], utc=True)
        keep_lab = ["timestamp", "label"]
        for optional in ("expire_timestamp", "holding_bars", "exit_reason", "side"):
            if optional in lab.columns:
                keep_lab.append(optional)
        merged = enriched.merge(lab[keep_lab], on="timestamp", how="inner")
        if merged.empty:
            raise ContextInputError("no overlap between enriched features and labels")

        merged = merged.sort_values("timestamp").reset_index(drop=True)
        merged["symbol"] = symbol.upper()
        merged["timeframe"] = timeframe.upper()
        merged["feature_version"] = self._feature_version
        merged["label_version"] = label_version
        merged["strategy"] = strategy
        merged["side"] = side.lower()

        splits = self._builder.split_time(
            merged,
            symbol=symbol,
            timeframe=timeframe,
            side=side,
            strategy=strategy,
            feature_version=self._feature_version,
            label_version=label_version,
        )
        for ds in splits.values():
            self._save_v2(ds)
        logger.info(
            "Dataset v2 saved | side=%s splits=%s version=%s",
            side,
            list(splits),
            self._feature_version,
        )
        return splits

    def _save_v2(self, dataset: Dataset) -> None:
        root = (
            self._datasets_root
            / dataset.symbol.upper()
            / dataset.timeframe.upper()
            / dataset.side.lower()
            / "v2"
        )
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{dataset.split.lower()}.parquet"
        dataset.frame.to_parquet(path, index=False)
        meta = {
            "symbol": dataset.symbol,
            "timeframe": dataset.timeframe,
            "side": dataset.side,
            "strategy": dataset.strategy,
            "feature_version": dataset.feature_version,
            "label_version": dataset.label_version,
            "split": dataset.split,
            "created_at": datetime.now(tz=timezone.utc).isoformat(),
            "rows": dataset.size,
            "feature_names": [c for c in dataset.frame.columns if c not in _META],
            "metadata": {**dataset.metadata, "dataset_layout": "v2"},
        }
        (root / f"{dataset.split.lower()}.meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
