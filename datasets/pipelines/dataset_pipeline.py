"""DatasetPipeline — FeatureSet + LabelSet → aligned splits → Dataset map."""

from __future__ import annotations

import logging
import time
from typing import Any

from datasets.builders.dataset_builder import DatasetBuilder
from datasets.entities.dataset import Dataset
from datasets.validators.dataset_validator import DatasetValidator
from features.entities.feature_set import FeatureSet
from labels.entities.label_set import LabelSet

logger = logging.getLogger(__name__)


class DatasetPipeline:
    def __init__(self, config: dict[str, Any]) -> None:
        self._cfg = config
        self._builder = DatasetBuilder(config)
        self._validator = DatasetValidator()

    def run(
        self, feature_set: FeatureSet, label_set: LabelSet
    ) -> dict[str, Dataset]:
        t0 = time.perf_counter()
        frame = self._builder.build_aligned_frame(feature_set, label_set)
        splits = self._builder.split_time(
            frame,
            symbol=feature_set.symbol,
            timeframe=feature_set.timeframe,
            side=label_set.side,
            strategy=label_set.strategy,
            feature_version=feature_set.feature_version,
            label_version=label_set.label_version,
        )
        for name, ds in splits.items():
            report = self._validator.validate(ds)
            logger.info(
                "Validated split | split=%s rows=%s classes=%s",
                name,
                report["rows"],
                report["class_counts"],
            )
        elapsed = time.perf_counter() - t0
        logger.info(
            "DatasetPipeline complete | side=%s splits=%s duration=%.3fs",
            label_set.side,
            list(splits),
            elapsed,
        )
        return splits
