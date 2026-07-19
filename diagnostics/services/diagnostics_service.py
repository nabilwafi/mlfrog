"""DiagnosticsService — load train artifacts and run DiagnosticsPipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from datasets.repositories.dataset_repository import DatasetRepository
from diagnostics.exceptions import DiagnosticsInputError
from diagnostics.pipelines.diagnostics_pipeline import DiagnosticsPipeline
from diagnostics.repositories.diagnostics_repository import DiagnosticsRepository
from models.entities.model import Model
from models.registries.trainer_registry import TrainerRegistry
from models.repositories.model_repository import ModelRepository

logger = logging.getLogger(__name__)


class DiagnosticsService:
    def __init__(
        self,
        dataset_repository: DatasetRepository,
        model_repository: ModelRepository,
        diagnostics_repository: DiagnosticsRepository,
        *,
        label_loader: Any | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self._dataset_repo = dataset_repository
        self._model_repo = model_repository
        self._diag_repo = diagnostics_repository
        self._label_loader = label_loader
        self._cfg = dict(config or {})
        self._pipeline = DiagnosticsPipeline(diagnostics_repository)
        TrainerRegistry.discover()

    def run(
        self,
        symbol: str,
        timeframe: str,
        side: str,
    ) -> tuple[Any, Path]:
        model = self._model_repo.load_model(symbol, timeframe, side)
        target_mode = str(
            self._cfg.get("target_mode")
            or model.metadata.get("target_mode")
            or "exclude_timeout"
        )
        threshold = float(self._cfg.get("threshold", 0.5))

        split_names = list(
            self._cfg.get("splits") or ["train", "validation", "test", "sealed"]
        )
        splits: dict[str, Any] = {}
        for name in split_names:
            try:
                splits[name] = self._dataset_repo.load_parquet(
                    symbol, timeframe, side, name
                )
            except Exception as exc:
                if name in {"train", "validation"}:
                    raise DiagnosticsInputError(
                        f"required split {name!r} missing: {exc}"
                    ) from exc
                logger.info("Optional split %s unavailable: %s", name, exc)

        label_frame = self._load_labels(symbol, timeframe, side, model)
        trainer = TrainerRegistry.create(model.algorithm, {})
        model_path = self._model_repo.model_path(symbol, timeframe, side)

        logger.info(
            "DiagnosticsService.run | %s %s %s algorithm=%s",
            symbol,
            timeframe,
            side,
            model.algorithm,
        )
        return self._pipeline.run(
            splits=splits,
            model=model,
            trainer=trainer,
            model_path=model_path,
            label_frame=label_frame,
            threshold=threshold,
            target_mode=target_mode,
        )

    def _load_labels(
        self,
        symbol: str,
        timeframe: str,
        side: str,
        model: Model,
    ) -> pd.DataFrame | None:
        if self._label_loader is None:
            return None
        try:
            return self._label_loader(symbol, timeframe, side, model)
        except Exception as exc:
            logger.warning("Label frame unavailable for diagnostics: %s", exc)
            return None
