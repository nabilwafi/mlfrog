"""Unit tests for model benchmark framework."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from models.registries.trainer_registry import TrainerRegistry
from research.model_benchmark.repositories.benchmark_repository import BenchmarkRepository
from research.model_benchmark.services.model_catalog import BENCHMARK_MODELS, default_params
from research.model_benchmark.usecases.run_model_benchmark import RunModelBenchmarkUseCase

UTC = ZoneInfo("UTC")


def _write_mini_artifacts(root: Path) -> tuple[Path, Path, pd.DataFrame]:
    feat_dir = root / "features" / "XAUUSD" / "H1"
    feat_dir.mkdir(parents=True)

    rng = np.random.default_rng(7)
    rows = []
    labels = []
    for year in range(2020, 2027):
        for i in range(60):
            t = datetime(year, 1, 1, tzinfo=UTC) + timedelta(hours=i)
            a = rng.normal(0, 1)
            b = rng.normal(0, 1)
            c = rng.normal(0, 1)
            y = 1 if (0.8 * a + 0.2 * c + rng.normal(0, 0.35)) > 0 else -1
            if rng.random() < 0.15:
                y = 0
            rows.append(
                {
                    "timestamp": t,
                    "ema20_distance_atr": a,
                    "rsi_percentile": (b + 3) / 6,
                    "atr_percent": abs(c),
                    "hour_sin": np.sin(2 * np.pi * (i % 24) / 24),
                }
            )
            labels.append({"timestamp": t, "label": y})

    feats = pd.DataFrame(rows)
    labs = pd.DataFrame(labels)
    feats.to_parquet(feat_dir / "feature_matrix.parquet", index=False)
    meta = {
        "features": [
            {"name": "ema20_distance_atr", "category": "trend"},
            {"name": "rsi_percentile", "category": "momentum"},
            {"name": "atr_percent", "category": "volatility"},
            {"name": "hour_sin", "category": "session"},
        ]
    }
    (feat_dir / "feature_metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    return feat_dir, root / "benchmark", labs


class CatalogTests(unittest.TestCase):
    def test_seven_models(self) -> None:
        self.assertEqual(len(BENCHMARK_MODELS), 7)
        self.assertIn("logistic_regression", BENCHMARK_MODELS)
        self.assertIn("catboost", BENCHMARK_MODELS)

    def test_default_params_keys(self) -> None:
        for algo in BENCHMARK_MODELS:
            self.assertIsInstance(default_params(algo), dict)


class RegistryCoverageTests(unittest.TestCase):
    def setUp(self) -> None:
        TrainerRegistry.clear()
        TrainerRegistry.discover()

    def test_all_benchmark_trainers_registered(self) -> None:
        keys = set(TrainerRegistry.keys())
        for algo in BENCHMARK_MODELS:
            self.assertIn(algo, keys)


class BenchmarkE2ETests(unittest.TestCase):
    def test_runs_and_writes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            feat_dir, out_dir, labs = _write_mini_artifacts(root)
            # Keep runtime small: sklearn + lightgbm only
            results, out = RunModelBenchmarkUseCase(
                BenchmarkRepository(out_dir),
                config={
                    "algorithms": [
                        "logistic_regression",
                        "random_forest",
                        "lightgbm",
                    ],
                    "target_mode": "exclude_timeout",
                    "threshold": 0.5,
                    "random_seed": 42,
                    "model_params": {
                        "random_forest": {"n_estimators": 30},
                        "lightgbm": {"num_boost_round": 40, "early_stopping_rounds": 10},
                    },
                    "windows": [
                        {
                            "train_start_year": 2020,
                            "train_end_year": 2022,
                            "valid_year": 2023,
                        },
                        {
                            "train_start_year": 2021,
                            "train_end_year": 2023,
                            "valid_year": 2024,
                        },
                    ],
                },
            ).execute(
                symbol="XAUUSD",
                timeframe="H1",
                side="long",
                feature_matrix_path=feat_dir / "feature_matrix.parquet",
                feature_metadata_path=feat_dir / "feature_metadata.json",
                labels=labs,
            )
            self.assertEqual(len(results), 3)
            self.assertTrue((out / "model_benchmark.csv").is_file())
            self.assertTrue((out / "model_rankings.csv").is_file())
            self.assertTrue((out / "model_benchmark.md").is_file())
            self.assertTrue((out / "charts" / "roc_comparison.png").is_file())
            ok = [r for r in results if any(w.status == "ok" for w in r.windows)]
            self.assertGreaterEqual(len(ok), 1)


if __name__ == "__main__":
    unittest.main()
