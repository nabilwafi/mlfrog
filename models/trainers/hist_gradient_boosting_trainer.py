"""HistGradientBoosting binary trainer."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from datasets.entities.dataset import Dataset
from models.entities.model import Model
from models.exceptions import TrainerError
from models.registries.trainer_registry import TrainerRegistry
from models.trainers.base_trainer import BaseTrainer


@TrainerRegistry.register
class HistGradientBoostingTrainer(BaseTrainer):
    def name(self) -> str:
        return "hist_gradient_boosting"

    def fit(self, train: Dataset, validation: Dataset, *, context: dict[str, Any]) -> Model:
        target_mode = str(context.get("target_mode", "exclude_timeout"))
        seed = int(context.get("random_seed", self.params.get("random_state", 42)))
        x_train, y_train, feature_names = self.prepare_xy(train, target_mode=target_mode)
        x_val, y_val, _ = self.prepare_xy(validation, target_mode=target_mode)
        if len(y_train) == 0 or len(y_val) == 0:
            raise TrainerError("empty train/validation after target mapping")

        # class weight via sample_weight
        sample_weight = None
        if bool(self.params.get("class_weight", True)):
            n_pos = max(int((y_train == 1).sum()), 1)
            n_neg = max(int((y_train == 0).sum()), 1)
            w_pos = n_neg / n_pos
            sample_weight = np.where(y_train == 1, w_pos, 1.0)

        clf = HistGradientBoostingClassifier(
            max_iter=int(self.params.get("max_iter", 200)),
            learning_rate=float(self.params.get("learning_rate", 0.05)),
            max_depth=int(self.params.get("max_depth", 6)),
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=int(self.params.get("n_iter_no_change", 20)),
            random_state=seed,
        )
        try:
            clf.fit(x_train, y_train, sample_weight=sample_weight)
        except Exception as exc:
            raise TrainerError(f"hist gradient boosting failed: {exc}") from exc

        # HGB has no native feature_importances_; leave empty
        best_iter = int(getattr(clf, "n_iter_", 0) or 0)
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
            feature_importance={},
            metadata={"best_iteration": best_iter, "target_mode": target_mode},
            estimator=clf,
        )

    def predict_proba(self, model: Model, frame: pd.DataFrame) -> np.ndarray:
        x = frame.loc[:, list(model.feature_names)].astype(float)
        return np.asarray(model.estimator.predict_proba(x)[:, 1], dtype=float)
