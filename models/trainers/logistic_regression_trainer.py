"""Logistic Regression binary trainer."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from datasets.entities.dataset import Dataset
from models.entities.model import Model
from models.exceptions import TrainerError
from models.registries.trainer_registry import TrainerRegistry
from models.trainers.base_trainer import BaseTrainer


@TrainerRegistry.register
class LogisticRegressionTrainer(BaseTrainer):
    def name(self) -> str:
        return "logistic_regression"

    def fit(self, train: Dataset, validation: Dataset, *, context: dict[str, Any]) -> Model:
        target_mode = str(context.get("target_mode", "exclude_timeout"))
        seed = int(context.get("random_seed", self.params.get("random_state", 42)))
        x_train, y_train, feature_names = self.prepare_xy(train, target_mode=target_mode)
        x_val, y_val, _ = self.prepare_xy(validation, target_mode=target_mode)
        if len(y_train) == 0 or len(y_val) == 0:
            raise TrainerError("empty train/validation after target mapping")

        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=int(self.params.get("max_iter", 1000)),
                        C=float(self.params.get("C", 1.0)),
                        class_weight="balanced" if self.params.get("class_weight", True) else None,
                        random_state=seed,
                        solver=str(self.params.get("solver", "lbfgs")),
                    ),
                ),
            ]
        )
        try:
            pipe.fit(x_train, y_train)
        except Exception as exc:
            raise TrainerError(f"logistic regression failed: {exc}") from exc

        coef = pipe.named_steps["clf"].coef_.ravel()
        importance = {n: float(abs(v)) for n, v in zip(feature_names, coef, strict=True)}
        return Model(
            symbol=train.symbol,
            timeframe=train.timeframe,
            side=train.side,
            algorithm=self.name(),
            feature_version=train.feature_version,
            label_version=train.label_version,
            dataset_version=str(context.get("dataset_version", "v1")),
            trained_at=datetime.now(tz=timezone.utc),
            hyperparameters=dict(self.params),
            feature_names=tuple(feature_names),
            train_rows=int(len(y_train)),
            validation_rows=int(len(y_val)),
            feature_importance=importance,
            metadata={"best_iteration": 0, "target_mode": target_mode},
            estimator=pipe,
        )

    def predict_proba(self, model: Model, frame: pd.DataFrame) -> np.ndarray:
        x = frame.loc[:, list(model.feature_names)].astype(float)
        return np.asarray(model.estimator.predict_proba(x)[:, 1], dtype=float)
