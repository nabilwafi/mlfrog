"""CatBoost binary classifier trainer."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from datasets.entities.dataset import Dataset
from models.entities.model import Model
from models.exceptions import TrainerError
from models.registries.trainer_registry import TrainerRegistry
from models.trainers.base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


@TrainerRegistry.register
class CatBoostTrainer(BaseTrainer):
    def name(self) -> str:
        return "catboost"

    def fit(self, train: Dataset, validation: Dataset, *, context: dict[str, Any]) -> Model:
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:
            raise TrainerError(
                "catboost is not installed; pip install catboost"
            ) from exc

        target_mode = str(context.get("target_mode", "exclude_timeout"))
        seed = int(context.get("random_seed", self.params.get("random_state", 42)))
        early_stopping = int(self.params.get("early_stopping_rounds", 50))
        use_class_weight = bool(self.params.get("class_weight", True))

        x_train, y_train, feature_names = self.prepare_xy(train, target_mode=target_mode)
        x_val, y_val, _ = self.prepare_xy(validation, target_mode=target_mode)
        if len(y_train) == 0 or len(y_val) == 0:
            raise TrainerError("empty train/validation after target mapping")

        class_weights = None
        if use_class_weight:
            n_pos = max(int((y_train == 1).sum()), 1)
            n_neg = max(int((y_train == 0).sum()), 1)
            # CatBoost: weights for classes [0, 1]
            class_weights = [1.0, n_neg / n_pos]

        params = {
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "iterations": int(self.params.get("iterations", self.params.get("num_boost_round", 500))),
            "learning_rate": float(self.params.get("learning_rate", 0.05)),
            "depth": int(self.params.get("depth", self.params.get("max_depth", 6))),
            "l2_leaf_reg": float(self.params.get("l2_leaf_reg", 3.0)),
            "random_seed": seed,
            "verbose": bool(self.params.get("verbose", False)),
            "allow_writing_files": False,
        }
        if class_weights is not None:
            params["class_weights"] = class_weights

        model = CatBoostClassifier(**params)
        fit_kwargs: dict[str, Any] = {
            "eval_set": (x_val, y_val),
            "use_best_model": True,
        }
        if early_stopping > 0:
            fit_kwargs["early_stopping_rounds"] = early_stopping

        logger.info(
            "Training CatBoost | train=%s val=%s pos_rate=%.3f",
            len(y_train),
            len(y_val),
            float(y_train.mean()),
        )
        try:
            model.fit(x_train, y_train, **fit_kwargs)
        except Exception as exc:
            raise TrainerError(f"CatBoost train failed: {exc}") from exc

        importances = model.get_feature_importance()
        importance = {
            name: float(val) for name, val in zip(feature_names, importances, strict=True)
        }

        return Model(
            symbol=train.symbol,
            timeframe=train.timeframe,
            side=train.side,
            algorithm=self.name(),
            feature_version=train.feature_version,
            label_version=train.label_version,
            dataset_version=str(context.get("dataset_version", "v1")),
            trained_at=datetime.now(tz=timezone.utc),
            hyperparameters={**params, "early_stopping_rounds": early_stopping},
            feature_names=tuple(feature_names),
            train_rows=int(len(y_train)),
            validation_rows=int(len(y_val)),
            feature_importance=importance,
            metadata={
                "best_iteration": int(getattr(model, "best_iteration_", 0) or 0),
                "target_mode": target_mode,
                "strategy": train.strategy,
            },
            estimator=model,
        )

    def predict_proba(self, model: Model, frame: pd.DataFrame) -> np.ndarray:
        clf = model.estimator
        x = frame.loc[:, list(model.feature_names)].astype(float)
        proba = clf.predict_proba(x)[:, 1]
        return np.asarray(proba, dtype=float)
