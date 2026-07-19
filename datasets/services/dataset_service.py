"""DatasetService — load features+labels → build → validate → save."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from datasets.entities.dataset import Dataset
from datasets.pipelines.dataset_pipeline import DatasetPipeline
from datasets.repositories.dataset_repository import DatasetRepository
from features.repositories.feature_repository import FeatureRepository
from labels.repositories.label_repository import LabelRepository

logger = logging.getLogger(__name__)


class DatasetService:
    def __init__(
        self,
        feature_repository: FeatureRepository,
        label_repository: LabelRepository,
        dataset_repository: DatasetRepository,
        dataset_config: dict[str, Any],
        *,
        timezone: str = "UTC",
    ) -> None:
        self._feature_repo = feature_repository
        self._label_repo = label_repository
        self._dataset_repo = dataset_repository
        self._cfg = dataset_config
        self._timezone = timezone
        self._pipeline = DatasetPipeline(dataset_config)

    def run(
        self,
        symbol: str,
        timeframe: str,
        *,
        sides: list[str] | None = None,
    ) -> dict[str, dict[str, tuple[Dataset, Path]]]:
        feature_version = str(self._cfg.get("feature_version", "v1"))
        label_version = str(self._cfg.get("label_version", "v1"))
        strategy = str(self._cfg.get("strategy", "triple_barrier"))
        side_list = sides or list(self._cfg.get("sides") or ["long"])

        features = self._feature_repo.load_parquet(
            symbol, timeframe, feature_version, timezone=self._timezone
        )
        logger.info(
            "Loaded FeatureSet | rows=%s features=%s",
            features.row_count,
            features.feature_count,
        )

        all_results: dict[str, dict[str, tuple[Dataset, Path]]] = {}
        for side in side_list:
            labels = self._label_repo.load_parquet(
                symbol,
                timeframe,
                side,
                strategy,
                label_version,
                timezone=self._timezone,
            )
            logger.info(
                "Loaded LabelSet | side=%s n=%s classes=%s",
                side,
                labels.size,
                labels.class_counts,
            )
            splits = self._pipeline.run(features, labels)
            saved: dict[str, tuple[Dataset, Path]] = {}
            for split_name, ds in splits.items():
                path = self._dataset_repo.save_parquet(ds)
                saved[split_name] = (ds, path)
                logger.info(
                    "Dataset saved | side=%s split=%s rows=%s path=%s",
                    side,
                    split_name,
                    ds.size,
                    path,
                )
            all_results[side] = saved
        return all_results
