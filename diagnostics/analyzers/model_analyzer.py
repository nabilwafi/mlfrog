"""Model artifact introspection (algorithm-agnostic with soft hooks)."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from models.entities.model import Model


class ModelAnalyzer:
    def analyze(
        self,
        model: Model,
        *,
        model_path: Path | None = None,
    ) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
        estimator = model.estimator
        best_iteration = int(model.metadata.get("best_iteration") or 0)
        algo = model.algorithm

        gain = dict(model.feature_importance)
        split: dict[str, float] = {}
        tree_count = 0
        train_loss: float | None = None
        valid_loss: float | None = None
        early_stopping_iteration = best_iteration
        learning_rows: list[dict[str, Any]] = []

        if algo == "lightgbm":
            gain, split, tree_count, train_loss, valid_loss, learning_rows = (
                self._inspect_lightgbm(estimator, model.feature_names)
            )
        elif algo == "xgboost":
            gain, split, tree_count, train_loss, valid_loss, learning_rows = (
                self._inspect_xgboost(estimator, model.feature_names)
            )
        elif algo == "catboost":
            gain, split, tree_count, train_loss, valid_loss, learning_rows = (
                self._inspect_catboost(estimator, model.feature_names)
            )
        else:
            tree_count = int(getattr(estimator, "n_estimators", 0) or 0)

        if not gain and model.feature_importance:
            gain = dict(model.feature_importance)

        unused = [f for f in model.feature_names if float(gain.get(f, 0.0)) == 0.0]
        size_bytes = 0
        if model_path is not None and model_path.is_file():
            size_bytes = int(model_path.stat().st_size)
        else:
            try:
                size_bytes = int(len(pickle.dumps(estimator, protocol=pickle.HIGHEST_PROTOCOL)))
            except Exception:
                size_bytes = 0

        metrics = model.metrics or {}
        if train_loss is None and "log_loss" in metrics:
            # validation log loss from training evaluator
            valid_loss = float(metrics["log_loss"])

        importance_rows = []
        for f in model.feature_names:
            importance_rows.append(
                {
                    "feature": f,
                    "importance_gain": float(gain.get(f, 0.0)),
                    "importance_split": float(split.get(f, 0.0)),
                }
            )
        importance_df = (
            pd.DataFrame(importance_rows)
            .sort_values("importance_gain", ascending=False)
            .reset_index(drop=True)
        )
        learning_df = pd.DataFrame(learning_rows)
        if learning_df.empty:
            learning_df = pd.DataFrame(
                [
                    {
                        "iteration": best_iteration,
                        "train_logloss": train_loss,
                        "valid_logloss": valid_loss,
                        "note": "full curve unavailable; final/best snapshot only",
                    }
                ]
            )

        summary = {
            "algorithm": algo,
            "hyperparameters": dict(model.hyperparameters),
            "best_iteration": best_iteration,
            "early_stopping_iteration": early_stopping_iteration,
            "training_loss": train_loss,
            "validation_loss": valid_loss,
            "tree_count": tree_count,
            "model_size_bytes": size_bytes,
            "unused_features": unused,
            "n_unused_features": len(unused),
            "feature_names": list(model.feature_names),
            "train_rows": model.train_rows,
            "validation_rows": model.validation_rows,
            "target_mode": model.metadata.get("target_mode"),
        }
        return summary, importance_df, learning_df

    def _inspect_lightgbm(
        self, booster: Any, feature_names: tuple[str, ...]
    ) -> tuple[dict[str, float], dict[str, float], int, float | None, float | None, list[dict]]:
        gain_arr = booster.feature_importance(importance_type="gain")
        split_arr = booster.feature_importance(importance_type="split")
        names = list(feature_names)
        if hasattr(booster, "feature_name"):
            raw_names = list(booster.feature_name() or [])
            if len(raw_names) == len(gain_arr):
                names = raw_names
        gain = {n: float(v) for n, v in zip(names, gain_arr, strict=False)}
        split = {n: float(v) for n, v in zip(names, split_arr, strict=False)}
        tree_count = int(booster.num_trees())
        train_loss = None
        valid_loss = None
        best = getattr(booster, "best_score", None) or {}
        if isinstance(best, dict):
            train_loss = self._first_metric(best.get("train"))
            valid_loss = self._first_metric(best.get("valid")) or self._first_metric(
                best.get("validation")
            )
        # Approximate learning curve by evaluating at sampled iterations
        learning_rows: list[dict[str, Any]] = []
        best_iter = int(getattr(booster, "best_iteration", 0) or tree_count or 0)
        if best_iter > 0:
            step = max(1, best_iter // 20)
            for it in list(range(step, best_iter + 1, step)) + [best_iter]:
                learning_rows.append(
                    {
                        "iteration": int(it),
                        "train_logloss": train_loss,
                        "valid_logloss": valid_loss if it == best_iter else None,
                    }
                )
            # dedupe
            seen: set[int] = set()
            uniq = []
            for row in learning_rows:
                if row["iteration"] in seen:
                    continue
                seen.add(row["iteration"])
                uniq.append(row)
            learning_rows = uniq
        return gain, split, tree_count, train_loss, valid_loss, learning_rows

    def _inspect_xgboost(
        self, booster: Any, feature_names: tuple[str, ...]
    ) -> tuple[dict[str, float], dict[str, float], int, float | None, float | None, list[dict]]:
        gain_raw = booster.get_score(importance_type="gain")
        split_raw = booster.get_score(importance_type="weight")
        gain = {f: float(gain_raw.get(f, 0.0)) for f in feature_names}
        split = {f: float(split_raw.get(f, 0.0)) for f in feature_names}
        try:
            tree_count = int(booster.num_boosted_rounds())
        except Exception:
            tree_count = int(getattr(booster, "best_iteration", 0) or 0)
        return gain, split, tree_count, None, None, []

    def _inspect_catboost(
        self, model: Any, feature_names: tuple[str, ...]
    ) -> tuple[dict[str, float], dict[str, float], int, float | None, float | None, list[dict]]:
        try:
            imp = model.get_feature_importance()
            gain = {n: float(v) for n, v in zip(feature_names, imp, strict=False)}
        except Exception:
            gain = {}
        split = {}
        tree_count = int(getattr(model, "tree_count_", 0) or getattr(model, "best_iteration_", 0) or 0)
        learning_rows: list[dict[str, Any]] = []
        try:
            evals = model.get_evals_result()
            learn = (evals.get("learn") or evals.get("training") or {}).get("Logloss") or []
            valid = (evals.get("validation") or {}).get("Logloss") or []
            n = max(len(learn), len(valid))
            for i in range(n):
                learning_rows.append(
                    {
                        "iteration": i + 1,
                        "train_logloss": float(learn[i]) if i < len(learn) else None,
                        "valid_logloss": float(valid[i]) if i < len(valid) else None,
                    }
                )
        except Exception:
            pass
        return gain, split, tree_count, None, None, learning_rows

    @staticmethod
    def _first_metric(block: Any) -> float | None:
        if not isinstance(block, dict) or not block:
            return None
        val = next(iter(block.values()))
        try:
            return float(val)
        except (TypeError, ValueError):
            return None
