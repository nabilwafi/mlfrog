"""Unit tests for model training layer."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from datasets.entities.dataset import Dataset
from datasets.repositories.dataset_repository import DatasetRepository
from models.evaluators.evaluator import Evaluator
from models.exceptions import RegistryError, TrainingValidationError
from models.registries.trainer_registry import TrainerRegistry
from models.repositories.model_repository import ModelRepository
from models.services.model_service import ModelService
from models.validators.training_validator import TrainingValidator


def _frame(n: int, *, split: str, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    # Separable signal: feature_a correlated with positive class
    label_raw = rng.choice([-1, 0, 1], size=n, p=[0.35, 0.3, 0.35])
    feature_a = label_raw.astype(float) + rng.normal(0, 0.3, size=n)
    feature_b = rng.normal(0, 1, size=n)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "symbol": "XAUUSD",
            "timeframe": "H1",
            "feature_version": "v1",
            "label_version": "v1",
            "split": split,
            "side": "long",
            "label": label_raw,
            "feature_a": feature_a,
            "feature_b": feature_b,
        }
    )


def _dataset(split: str, n: int, seed: int) -> Dataset:
    return Dataset(
        symbol="XAUUSD",
        timeframe="H1",
        side="long",
        strategy="triple_barrier",
        feature_version="v1",
        label_version="v1",
        split=split,
        created_at=datetime.now(tz=timezone.utc),
        frame=_frame(n, split=split, seed=seed),
    )


class TrainerRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        TrainerRegistry.clear()
        TrainerRegistry.discover()

    def test_discovers_lightgbm(self) -> None:
        self.assertIn("lightgbm", TrainerRegistry.keys())
        self.assertIn("xgboost", TrainerRegistry.keys())
        self.assertIn("catboost", TrainerRegistry.keys())
        self.assertIn("logistic_regression", TrainerRegistry.keys())
        self.assertIn("random_forest", TrainerRegistry.keys())
        self.assertIn("extra_trees", TrainerRegistry.keys())
        self.assertIn("hist_gradient_boosting", TrainerRegistry.keys())

    def test_unknown_raises(self) -> None:
        with self.assertRaises(RegistryError):
            TrainerRegistry.create("no_such_algo")


class EvaluatorTests(unittest.TestCase):
    def test_metrics(self) -> None:
        y = np.array([0, 0, 1, 1])
        p = np.array([0.1, 0.4, 0.6, 0.9])
        result = Evaluator().evaluate(y, p, feature_importance={"a": 1.0})
        self.assertIn("roc_auc", result.metrics)
        self.assertIn("pr_auc", result.metrics)
        self.assertIn("confusion_matrix", result.metrics)
        self.assertGreater(result.metrics["roc_auc"], 0.9)


class TrainingValidatorTests(unittest.TestCase):
    def test_ok(self) -> None:
        TrainingValidator().validate(_dataset("train", 40, 1), _dataset("validation", 20, 2))

    def test_split_mismatch(self) -> None:
        with self.assertRaises(TrainingValidationError):
            TrainingValidator().validate(
                _dataset("validation", 40, 1), _dataset("validation", 20, 2)
            )


class LightGBMTrainTests(unittest.TestCase):
    def setUp(self) -> None:
        TrainerRegistry.clear()
        TrainerRegistry.discover()

    def test_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ds_repo = DatasetRepository(root / "datasets")
            model_repo = ModelRepository(root / "models")
            train = _dataset("train", 120, 1)
            val = _dataset("validation", 40, 2)
            ds_repo.save_parquet(train)
            ds_repo.save_parquet(val)

            service = ModelService(
                ds_repo,
                model_repo,
                {
                    "dataset_version": "v1",
                    "target_mode": "exclude_timeout",
                    "random_seed": 42,
                    "threshold": 0.5,
                    "algorithms": {
                        "lightgbm": {
                            "params": {
                                "num_boost_round": 40,
                                "early_stopping_rounds": 10,
                                "learning_rate": 0.1,
                                "num_leaves": 15,
                                "verbosity": -1,
                                "class_weight": True,
                            }
                        }
                    },
                },
            )
            model, out_dir = service.run("XAUUSD", "H1", "long", "lightgbm")
            self.assertEqual(model.algorithm, "lightgbm")
            self.assertTrue((out_dir / "model.pkl").is_file())
            self.assertTrue((out_dir / "metadata.json").is_file())
            self.assertTrue((out_dir / "feature_importance.parquet").is_file())
            self.assertTrue((out_dir / "training_report.md").is_file())
            self.assertIn("roc_auc", model.metrics)

            loaded = model_repo.load_model("XAUUSD", "H1", "long")
            self.assertEqual(loaded.algorithm, "lightgbm")
            self.assertEqual(len(loaded.feature_names), 2)


if __name__ == "__main__":
    unittest.main()
