"""Unit tests for diagnostics engine."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from datasets.entities.dataset import Dataset
from datasets.repositories.dataset_repository import DatasetRepository
from diagnostics.analyzers.probability_analyzer import ProbabilityAnalyzer
from diagnostics.repositories.diagnostics_repository import DiagnosticsRepository
from diagnostics.services.diagnostics_service import DiagnosticsService
from models.registries.trainer_registry import TrainerRegistry
from models.repositories.model_repository import ModelRepository
from models.services.model_service import ModelService


def _frame(n: int, *, split: str, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    label_raw = rng.choice([-1, 0, 1], size=n, p=[0.35, 0.3, 0.35])
    feature_a = label_raw.astype(float) + rng.normal(0, 0.25, size=n)
    feature_b = rng.normal(0, 1, size=n)
    feature_const = np.ones(n)
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
            "feature_const": feature_const,
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


class ProbabilityAnalyzerTests(unittest.TestCase):
    def test_threshold_sweep(self) -> None:
        y = np.array([0, 0, 1, 1, 0, 1])
        p = np.array([0.1, 0.2, 0.6, 0.8, 0.4, 0.7])
        summary, hist = ProbabilityAnalyzer().analyze(y, p)
        self.assertIn("flags", summary)
        self.assertEqual(len(hist), 20)
        thr_sum, thr = ProbabilityAnalyzer().threshold_sweep(y, p)
        self.assertGreater(len(thr), 10)
        self.assertIn("best_f1_threshold", thr_sum)


class DiagnosticsEndToEndTests(unittest.TestCase):
    def setUp(self) -> None:
        TrainerRegistry.clear()
        TrainerRegistry.discover()

    def test_pipeline_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ds_repo = DatasetRepository(root / "datasets")
            model_repo = ModelRepository(root / "models")
            diag_repo = DiagnosticsRepository(root / "diagnostics")

            train = _dataset("train", 160, 1)
            val = _dataset("validation", 60, 2)
            test = _dataset("test", 40, 3)
            ds_repo.save_parquet(train)
            ds_repo.save_parquet(val)
            ds_repo.save_parquet(test)

            ModelService(
                ds_repo,
                model_repo,
                {
                    "dataset_version": "v1",
                    "target_mode": "exclude_timeout",
                    "random_seed": 7,
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
            ).run("XAUUSD", "H1", "long", "lightgbm")

            report, out = DiagnosticsService(
                ds_repo,
                model_repo,
                diag_repo,
                config={"target_mode": "exclude_timeout", "threshold": 0.5},
            ).run("XAUUSD", "H1", "long")

            self.assertTrue((out / "diagnostics_report.md").is_file())
            self.assertTrue((out / "metadata.json").is_file())
            for name in (
                "probability_distribution.csv",
                "threshold_analysis.csv",
                "feature_statistics.csv",
                "feature_importance.csv",
                "prediction_distribution.csv",
                "learning_curve.csv",
                "drift_report.csv",
            ):
                self.assertTrue((out / name).is_file(), name)
            self.assertIn("feature_const", report.features.get("constant_features") or [])


if __name__ == "__main__":
    unittest.main()
