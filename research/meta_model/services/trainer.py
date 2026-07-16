"""Production Meta LightGBM trainer (fixed params; no HPO)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from research.meta_model import FINAL_FEATURES

logger = logging.getLogger(__name__)

DEFAULT_LGBM_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.0,
    "reg_lambda": 0.0,
    "verbosity": -1,
}


class MetaModelTrainer:
    def __init__(
        self,
        *,
        features: tuple[str, ...] = FINAL_FEATURES,
        lgbm_params: dict[str, Any] | None = None,
        num_boost_round: int = 200,
        early_stopping_rounds: int = 30,
        random_seed: int = 42,
    ) -> None:
        self.features = list(features)
        self.params = {**DEFAULT_LGBM_PARAMS, **(lgbm_params or {})}
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.random_seed = random_seed

    def _xy(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        x = frame[self.features].astype(float)
        y = frame["meta_label"].astype(int).to_numpy()
        return x, y

    def _fit(
        self, x_tr: pd.DataFrame, y_tr: np.ndarray, x_va: pd.DataFrame, y_va: np.ndarray
    ) -> tuple[lgb.Booster, pd.Series]:
        med = x_tr.median(numeric_only=True)
        x_tr = x_tr.fillna(med)
        x_va = x_va.fillna(med)
        n_pos = max(int((y_tr == 1).sum()), 1)
        n_neg = max(int((y_tr == 0).sum()), 1)
        params = {
            **self.params,
            "seed": self.random_seed,
            "feature_fraction_seed": self.random_seed,
            "bagging_seed": self.random_seed,
            "scale_pos_weight": n_neg / n_pos,
        }
        dtrain = lgb.Dataset(x_tr, label=y_tr, feature_name=self.features, free_raw_data=False)
        dval = lgb.Dataset(
            x_va, label=y_va, reference=dtrain, feature_name=self.features, free_raw_data=False
        )
        callbacks = [lgb.log_evaluation(0)]
        if self.early_stopping_rounds > 0:
            callbacks.append(lgb.early_stopping(self.early_stopping_rounds, verbose=False))
        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=self.num_boost_round,
            valid_sets=[dval],
            valid_names=["val"],
            callbacks=callbacks,
        )
        return booster, med

    def leave_one_year_out(self, panel: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, lgb.Booster]]:
        """Train LOO; return OOF predictions + per-year boosters."""
        years = sorted(int(y) for y in panel["valid_year"].dropna().unique())
        parts: list[pd.DataFrame] = []
        models: dict[int, lgb.Booster] = {}
        for year in years:
            val = panel.loc[panel["valid_year"] == year].copy()
            train = panel.loc[panel["valid_year"] != year].copy()
            if train.empty or val.empty:
                continue
            if train["meta_label"].nunique() < 2:
                logger.warning("Skip year=%s — train labels degenerate", year)
                continue
            x_tr, y_tr = self._xy(train)
            x_va, y_va = self._xy(val)
            med = x_tr.median(numeric_only=True)
            booster, _ = self._fit(x_tr, y_tr, x_va.fillna(med), y_va)
            x_va = x_va.fillna(med)
            proba = booster.predict(x_va, num_iteration=booster.best_iteration)
            out = val.copy()
            out["meta_proba"] = proba
            parts.append(out)
            models[year] = booster
            logger.info(
                "Meta LOO year=%s n_train=%s n_val=%s best_iter=%s",
                year,
                len(train),
                len(val),
                booster.best_iteration,
            )
        oof = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        return oof, models

    def fit_full(self, panel: pd.DataFrame) -> tuple[lgb.Booster, pd.Series]:
        """Train on all years for deployment artifact (early-stop on last 20% by time)."""
        frame = panel.sort_values("timestamp").reset_index(drop=True)
        n = len(frame)
        cut = max(int(n * 0.8), 1)
        train, hold = frame.iloc[:cut], frame.iloc[cut:]
        if hold.empty or hold["meta_label"].nunique() < 2:
            train, hold = frame, frame
        x_tr, y_tr = self._xy(train)
        x_h, y_h = self._xy(hold)
        booster, med = self._fit(x_tr, y_tr, x_h, y_h)
        return booster, med

    @staticmethod
    def save_booster(booster: lgb.Booster, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        booster.save_model(str(path))
