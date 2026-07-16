"""ResearchService — load datasets/labels/config and run ResearchPipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from datasets.repositories.dataset_repository import DatasetRepository
from research.analyzers.walk_forward_analyzer import WalkForwardWindow
from research.pipelines.research_pipeline import ResearchPipeline
from research.repositories.research_repository import ResearchRepository

logger = logging.getLogger(__name__)


class ResearchService:
    def __init__(
        self,
        dataset_repository: DatasetRepository,
        research_repository: ResearchRepository,
        *,
        config: dict[str, Any] | None = None,
        label_loader: Callable[..., pd.DataFrame] | None = None,
    ) -> None:
        self._dataset_repo = dataset_repository
        self._research_repo = research_repository
        self._cfg = dict(config or {})
        self._label_loader = label_loader
        self._pipeline = ResearchPipeline(research_repository)

    def run(
        self,
        symbol: str,
        timeframe: str,
        side: str,
    ) -> tuple[Any, Path]:
        split_names = list(
            self._cfg.get("splits") or ["train", "validation", "test", "sealed"]
        )
        splits = {}
        for name in split_names:
            try:
                splits[name] = self._dataset_repo.load_parquet(
                    symbol, timeframe, side, name
                )
            except Exception as exc:
                if name in {"train", "validation"}:
                    raise
                logger.info("Optional split %s skipped: %s", name, exc)

        algorithm = str(self._cfg.get("algorithm", "lightgbm"))
        trainer_params = dict(self._cfg.get("trainer_params") or {})
        windows = self._parse_windows(self._cfg.get("windows"))
        label_frame = None
        if self._label_loader is not None:
            try:
                label_frame = self._label_loader(symbol, timeframe, side)
            except Exception as exc:
                logger.warning("Label frame unavailable: %s", exc)

        logger.info(
            "ResearchService.run | %s %s %s algorithm=%s",
            symbol,
            timeframe,
            side,
            algorithm,
        )
        return self._pipeline.run(
            splits=splits,
            algorithm=algorithm,
            trainer_params=trainer_params,
            target_mode=str(self._cfg.get("target_mode", "exclude_timeout")),
            threshold=float(self._cfg.get("threshold", 0.5)),
            random_seed=int(self._cfg.get("random_seed", 42)),
            dataset_version=str(self._cfg.get("dataset_version", "v1")),
            windows=windows,
            label_frame=label_frame,
        )

    @staticmethod
    def _parse_windows(raw: Any) -> list[WalkForwardWindow] | None:
        if not raw:
            return None
        out: list[WalkForwardWindow] = []
        for item in raw:
            out.append(
                WalkForwardWindow(
                    train_start_year=int(item["train_start_year"]),
                    train_end_year=int(item["train_end_year"]),
                    valid_year=int(item["valid_year"]),
                )
            )
        return out
