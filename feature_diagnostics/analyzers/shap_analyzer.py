"""Walk-forward TreeSHAP importance via LightGBM pred_contrib."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from research.analyzers._common import filter_years

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ShapWindow:
    train_start_year: int
    train_end_year: int
    valid_year: int

    @property
    def name(self) -> str:
        return f"train_{self.train_start_year}_{self.train_end_year}_val_{self.valid_year}"


class ShapImportanceAnalyzer:
    def __init__(
        self,
        *,
        windows: list[ShapWindow] | None = None,
        random_state: int = 42,
        num_boost_round: int = 80,
        early_stopping_rounds: int = 20,
    ) -> None:
        self.windows = windows or [
            ShapWindow(2020, 2022, 2023),
            ShapWindow(2021, 2023, 2024),
            ShapWindow(2022, 2024, 2025),
            ShapWindow(2023, 2025, 2026),
        ]
        self.random_state = random_state
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds

    def analyze(
        self,
        frame: pd.DataFrame,
        feature_names: list[str],
        *,
        y_col: str = "_y",
        metadata: dict[str, dict[str, Any]] | None = None,
    ) -> tuple[dict[str, Any], pd.DataFrame]:
        """
        frame must contain timestamp, feature columns, and binary y_col.
        """
        meta = metadata or {}
        per_window: list[pd.DataFrame] = []
        for window in self.windows:
            train = filter_years(frame, window.train_start_year, window.train_end_year)
            valid = filter_years(frame, window.valid_year, window.valid_year)
            if train.empty or valid.empty:
                logger.warning("SHAP skip empty window %s", window.name)
                continue
            x_train = train[feature_names].astype(float)
            y_train = train[y_col].astype(int).to_numpy()
            x_val = valid[feature_names].astype(float)
            y_val = valid[y_col].astype(int).to_numpy()
            if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
                continue

            dtrain = lgb.Dataset(x_train, label=y_train, feature_name=feature_names)
            dval = lgb.Dataset(x_val, label=y_val, reference=dtrain, feature_name=feature_names)
            booster = lgb.train(
                {
                    "objective": "binary",
                    "metric": "auc",
                    "learning_rate": 0.05,
                    "num_leaves": 31,
                    "verbosity": -1,
                    "seed": self.random_state,
                },
                dtrain,
                num_boost_round=self.num_boost_round,
                valid_sets=[dval],
                callbacks=[
                    lgb.early_stopping(self.early_stopping_rounds, verbose=False),
                    lgb.log_evaluation(0),
                ],
            )
            # TreeSHAP: last column is bias
            contrib = booster.predict(x_val, pred_contrib=True)
            shap_abs = np.abs(contrib[:, :-1]).mean(axis=0)
            per_window.append(
                pd.DataFrame(
                    {
                        "feature": feature_names,
                        "window": window.name,
                        "valid_year": window.valid_year,
                        "mean_abs_shap": shap_abs,
                        "best_iteration": int(getattr(booster, "best_iteration", 0) or 0),
                    }
                )
            )
            logger.info("SHAP window %s complete | best_iter=%s", window.name, booster.best_iteration)

        if not per_window:
            empty = pd.DataFrame(
                columns=["feature", "mean_abs_shap", "shap_rank_mean", "n_windows"]
            )
            return {"n_windows": 0}, empty

        long = pd.concat(per_window, ignore_index=True)
        agg = (
            long.groupby("feature", as_index=False)
            .agg(
                mean_abs_shap=("mean_abs_shap", "mean"),
                std_abs_shap=("mean_abs_shap", "std"),
                n_windows=("window", "nunique"),
            )
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )
        agg["shap_rank"] = np.arange(1, len(agg) + 1)
        # consistency: average rank across windows
        long["rank_in_window"] = long.groupby("window")["mean_abs_shap"].rank(
            ascending=False, method="average"
        )
        rank_mean = long.groupby("feature")["rank_in_window"].mean().rename("shap_rank_mean")
        agg = agg.merge(rank_mean, on="feature", how="left")
        agg["category"] = agg["feature"].map(lambda f: (meta.get(f) or {}).get("category", ""))

        summary = {
            "n_windows": int(long["window"].nunique()),
            "top_features": agg.head(15).to_dict(orient="records"),
            "per_window_top": (
                long.sort_values("mean_abs_shap", ascending=False)
                .groupby("window")
                .head(5)
                .to_dict(orient="records")
            ),
        }
        return summary, agg
