"""Unit tests for research engine."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from datasets.entities.dataset import Dataset
from datasets.repositories.dataset_repository import DatasetRepository
from research.analyzers.feature_drift_analyzer import FeatureDriftAnalyzer
from research.repositories.research_repository import ResearchRepository
from research.services.research_service import ResearchService


def _frame(n: int, *, year: int, split: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range(f"{year}-01-01", periods=n, freq="h", tz="UTC")
    label_raw = rng.choice([-1, 0, 1], size=n, p=[0.35, 0.3, 0.35])
    # mild year drift in feature_a mean
    shift = 0.15 * (year - 2020)
    feature_a = label_raw.astype(float) + rng.normal(shift, 0.3, size=n)
    feature_b = rng.normal(0, 1, size=n)
    atr = np.abs(rng.normal(10 + shift, 2, size=n))
    adx = np.abs(rng.normal(20, 5, size=n))
    returns = rng.normal(0.001 * (year - 2022), 0.01, size=n)
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
            "atr_14": atr,
            "adx_14": adx,
            "returns_1": returns,
            "ema_distance_ema_20_close": returns * 10,
        }
    )


def _ds(split: str, year: int, n: int, seed: int) -> Dataset:
    return Dataset(
        symbol="XAUUSD",
        timeframe="H1",
        side="long",
        strategy="triple_barrier",
        feature_version="v1",
        label_version="v1",
        split=split,
        created_at=datetime.now(tz=timezone.utc),
        frame=_frame(n, year=year, split=split, seed=seed),
    )


class FeatureDriftTests(unittest.TestCase):
    def test_sorted_by_psi(self) -> None:
        train = _ds("train", 2020, 200, 1)
        val = _ds("validation", 2024, 80, 2)
        summary, drift = FeatureDriftAnalyzer().analyze(train, val)
        self.assertIn("drift_severity", drift.columns)
        self.assertGreaterEqual(summary["n_features"], 2)
        if len(drift) >= 2 and drift["psi"].notna().all():
            self.assertGreaterEqual(drift.iloc[0]["psi"], drift.iloc[1]["psi"])


class ResearchEndToEndTests(unittest.TestCase):
    def test_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ds_repo = DatasetRepository(root / "datasets")
            # Build chronological multi-year coverage via split labels only
            train = Dataset(
                symbol="XAUUSD",
                timeframe="H1",
                side="long",
                strategy="triple_barrier",
                feature_version="v1",
                label_version="v1",
                split="train",
                created_at=datetime.now(tz=timezone.utc),
                frame=pd.concat(
                    [
                        _frame(120, year=2020, split="train", seed=1),
                        _frame(120, year=2021, split="train", seed=2),
                        _frame(120, year=2022, split="train", seed=3),
                    ],
                    ignore_index=True,
                ),
            )
            validation = Dataset(
                symbol="XAUUSD",
                timeframe="H1",
                side="long",
                strategy="triple_barrier",
                feature_version="v1",
                label_version="v1",
                split="validation",
                created_at=datetime.now(tz=timezone.utc),
                frame=_frame(100, year=2023, split="validation", seed=4),
            )
            test = Dataset(
                symbol="XAUUSD",
                timeframe="H1",
                side="long",
                strategy="triple_barrier",
                feature_version="v1",
                label_version="v1",
                split="test",
                created_at=datetime.now(tz=timezone.utc),
                frame=_frame(80, year=2024, split="test", seed=5),
            )
            ds_repo.save_parquet(train)
            ds_repo.save_parquet(validation)
            ds_repo.save_parquet(test)

            report, out = ResearchService(
                ds_repo,
                ResearchRepository(root / "research"),
                config={
                    "algorithm": "lightgbm",
                    "trainer_params": {
                        "num_boost_round": 30,
                        "early_stopping_rounds": 8,
                        "learning_rate": 0.1,
                        "num_leaves": 15,
                        "verbosity": -1,
                        "class_weight": True,
                    },
                    "target_mode": "exclude_timeout",
                    "threshold": 0.5,
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
                    "splits": ["train", "validation", "test"],
                },
            ).run("XAUUSD", "H1", "long")

            for name in (
                "walk_forward_report.md",
                "walk_forward_metrics.csv",
                "feature_stability.csv",
                "feature_drift.csv",
                "regime_report.md",
                "regime_statistics.csv",
                "label_stability.csv",
                "research_summary.md",
            ):
                self.assertTrue((out / name).is_file(), name)
            self.assertIn("failure_driver", report.conclusions)
            self.assertIn("model_strategy_recommendation", report.conclusions)


if __name__ == "__main__":
    unittest.main()
