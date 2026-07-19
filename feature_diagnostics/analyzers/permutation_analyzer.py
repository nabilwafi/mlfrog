"""Permutation importance via sklearn on a LightGBM fit."""

from __future__ import annotations

import logging
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class PermutationImportanceAnalyzer:
    def analyze(
        self,
        x: pd.DataFrame,
        y: np.ndarray,
        *,
        random_state: int = 42,
        n_repeats: int = 5,
        max_rows: int = 20_000,
        metadata: dict[str, dict[str, Any]] | None = None,
    ) -> tuple[dict[str, Any], pd.DataFrame]:
        meta = metadata or {}
        x_work = x
        y_work = y
        if len(x) > max_rows:
            rng = np.random.default_rng(random_state)
            idx = rng.choice(len(x), size=max_rows, replace=False)
            x_work = x.iloc[idx].reset_index(drop=True)
            y_work = y[idx]

        x_train, x_val, y_train, y_val = train_test_split(
            x_work,
            y_work,
            test_size=0.25,
            random_state=random_state,
            stratify=y_work if len(np.unique(y_work)) > 1 else None,
        )
        model = lgb.LGBMClassifier(
            n_estimators=80,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            verbosity=-1,
        )
        model.fit(x_train, y_train)
        logger.info("Permutation importance | train=%s val=%s", len(x_train), len(x_val))
        result = permutation_importance(
            model,
            x_val,
            y_val,
            n_repeats=n_repeats,
            random_state=random_state,
            scoring="roc_auc",
        )
        rows = []
        for i, name in enumerate(x.columns):
            rows.append(
                {
                    "feature": name,
                    "permutation_importance_mean": float(result.importances_mean[i]),
                    "permutation_importance_std": float(result.importances_std[i]),
                    "category": (meta.get(name) or {}).get("category", ""),
                }
            )
        frame = (
            pd.DataFrame(rows)
            .sort_values("permutation_importance_mean", ascending=False)
            .reset_index(drop=True)
        )
        summary = {
            "top_features": frame.head(15).to_dict(orient="records"),
            "n_rows_used": int(len(x_work)),
        }
        return summary, frame
