"""ModelService — load datasets, resolve trainer, run TrainingPipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from datasets.repositories.dataset_repository import DatasetRepository
from models.entities.model import Model
from models.evaluators.evaluator import Evaluator
from models.pipelines.training_pipeline import TrainingPipeline
from models.registries.trainer_registry import TrainerRegistry
from models.repositories.model_repository import ModelRepository
from models.validators.training_validator import TrainingValidator

logger = logging.getLogger(__name__)


class ModelService:
    def __init__(
        self,
        dataset_repository: DatasetRepository,
        model_repository: ModelRepository,
        models_config: dict[str, Any],
    ) -> None:
        self._dataset_repo = dataset_repository
        self._model_repo = model_repository
        self._cfg = models_config
        self._pipeline = TrainingPipeline(
            TrainingValidator(),
            Evaluator(),
            model_repository,
        )
        TrainerRegistry.discover()

    def run(
        self,
        symbol: str,
        timeframe: str,
        side: str,
        algorithm: str,
    ) -> tuple[Model, Path]:
        algos = self._cfg.get("algorithms") or {}
        algo_cfg = dict(algos.get(algorithm) or {})
        params = dict(algo_cfg.get("params") or algo_cfg)

        target_mode = str(
            self._cfg.get("target_mode", params.pop("target_mode", "exclude_timeout"))
        )
        random_seed = int(self._cfg.get("random_seed", params.pop("random_seed", 42)))
        threshold = float(self._cfg.get("threshold", 0.5))
        dataset_version = str(self._cfg.get("dataset_version", "v1"))

        train = self._dataset_repo.load_parquet(symbol, timeframe, side, "train")
        validation = self._dataset_repo.load_parquet(
            symbol, timeframe, side, "validation"
        )

        trainer = TrainerRegistry.create(algorithm, params)
        context = {
            "target_mode": target_mode,
            "random_seed": random_seed,
            "dataset_version": dataset_version,
        }
        logger.info(
            "ModelService.run | %s %s %s algorithm=%s target_mode=%s",
            symbol,
            timeframe,
            side,
            algorithm,
            target_mode,
        )
        return self._pipeline.run(
            train,
            validation,
            trainer,
            context=context,
            threshold=threshold,
        )
