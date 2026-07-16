"""Unified model catalog for the benchmark."""

from __future__ import annotations

from typing import Any


BENCHMARK_MODELS: tuple[str, ...] = (
    "logistic_regression",
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
    "xgboost",
    "lightgbm",
    "catboost",
)


def default_params(algorithm: str) -> dict[str, Any]:
    """Sane defaults — not tuned; identical evaluation protocol."""
    common_seed = 42
    table: dict[str, dict[str, Any]] = {
        "logistic_regression": {
            "max_iter": 1000,
            "C": 1.0,
            "class_weight": True,
            "random_state": common_seed,
        },
        "random_forest": {
            "n_estimators": 200,
            "min_samples_leaf": 5,
            "class_weight": True,
            "n_jobs": -1,
            "random_state": common_seed,
        },
        "extra_trees": {
            "n_estimators": 200,
            "min_samples_leaf": 5,
            "class_weight": True,
            "n_jobs": -1,
            "random_state": common_seed,
        },
        "hist_gradient_boosting": {
            "max_iter": 200,
            "learning_rate": 0.05,
            "max_depth": 6,
            "n_iter_no_change": 20,
            "class_weight": True,
            "random_state": common_seed,
        },
        "xgboost": {
            "learning_rate": 0.05,
            "max_depth": 6,
            "num_boost_round": 300,
            "early_stopping_rounds": 40,
            "class_weight": True,
            "verbosity": 0,
        },
        "lightgbm": {
            "learning_rate": 0.05,
            "num_leaves": 63,
            "num_boost_round": 300,
            "early_stopping_rounds": 40,
            "class_weight": True,
            "verbosity": -1,
        },
        "catboost": {
            "learning_rate": 0.05,
            "depth": 6,
            "iterations": 300,
            "early_stopping_rounds": 40,
            "class_weight": True,
            "verbose": False,
        },
    }
    return dict(table.get(algorithm, {}))
