"""LightGBM binary classifier trainer."""

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
class LightGBMTrainer(BaseTrainer):
    def name(self) -> str:
        return "lightgbm"

    def fit(self, train: Dataset, validation: Dataset, *, context: dict[str, Any]) -> Model:
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise TrainerError("lightgbm is not installed; pip install lightgbm") from exc

        target_mode = str(context.get("target_mode", "exclude_timeout"))
        seed = int(context.get("random_seed", self.params.get("random_state", 42)))
        early_stopping = int(self.params.get("early_stopping_rounds", 50))
        num_boost_round = int(self.params.get("num_boost_round", 500))
        use_class_weight = bool(self.params.get("class_weight", True))

        x_train, y_train, feature_names = self.prepare_xy(train, target_mode=target_mode)
        x_val, y_val, _ = self.prepare_xy(validation, target_mode=target_mode)
        if len(y_train) == 0 or len(y_val) == 0:
            raise TrainerError("empty train/validation after target mapping")

        scale_pos_weight = 1.0
        if use_class_weight:
            n_pos = max(int((y_train == 1).sum()), 1)
            n_neg = max(int((y_train == 0).sum()), 1)
            scale_pos_weight = n_neg / n_pos

        lgb_params = {
            "objective": "binary",
            "metric": ["binary_logloss", "auc"],
            "boosting_type": str(self.params.get("boosting_type", "gbdt")),
            "learning_rate": float(self.params.get("learning_rate", 0.05)),
            "num_leaves": int(self.params.get("num_leaves", 63)),
            "max_depth": int(self.params.get("max_depth", -1)),
            "min_child_samples": int(self.params.get("min_child_samples", 40)),
            "subsample": float(self.params.get("subsample", 0.8)),
            "colsample_bytree": float(self.params.get("colsample_bytree", 0.8)),
            "reg_alpha": float(self.params.get("reg_alpha", 0.0)),
            "reg_lambda": float(self.params.get("reg_lambda", 0.0)),
            "scale_pos_weight": float(self.params.get("scale_pos_weight", scale_pos_weight)),
            "verbosity": int(self.params.get("verbosity", -1)),
            "seed": seed,
            "feature_fraction_seed": seed,
            "bagging_seed": seed,
        }

        dtrain = lgb.Dataset(x_train, label=y_train, feature_name=feature_names, free_raw_data=False)
        dval = lgb.Dataset(
            x_val, label=y_val, reference=dtrain, feature_name=feature_names, free_raw_data=False
        )

        callbacks = [lgb.log_evaluation(period=0)]
        if early_stopping > 0:
            callbacks.append(lgb.early_stopping(early_stopping, verbose=False))

        logger.info(
            "Training LightGBM | train=%s val=%s pos_rate=%.3f scale_pos_weight=%.3f",
            len(y_train),
            len(y_val),
            float(y_train.mean()),
            lgb_params["scale_pos_weight"],
        )
        try:
            booster = lgb.train(
                lgb_params,
                dtrain,
                num_boost_round=num_boost_round,
                valid_sets=[dtrain, dval],
                valid_names=["train", "valid"],
                callbacks=callbacks,
            )
        except Exception as exc:
            raise TrainerError(f"LightGBM train failed: {exc}") from exc

        gain = booster.feature_importance(importance_type="gain")
        importance = {
            name: float(val) for name, val in zip(feature_names, gain, strict=True)
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
            hyperparameters={**lgb_params, "num_boost_round": num_boost_round, "early_stopping_rounds": early_stopping},
            feature_names=tuple(feature_names),
            train_rows=int(len(y_train)),
            validation_rows=int(len(y_val)),
            feature_importance=importance,
            metadata={
                "best_iteration": int(getattr(booster, "best_iteration", 0) or 0),
                "target_mode": target_mode,
                "strategy": train.strategy,
            },
            estimator=booster,
        )

    def predict_proba(self, model: Model, frame: pd.DataFrame) -> np.ndarray:
        x = frame.loc[:, list(model.feature_names)].astype(float)
        booster = model.estimator
        raw = booster.predict(x, num_iteration=getattr(booster, "best_iteration", None) or None)
        return np.asarray(raw, dtype=float)
