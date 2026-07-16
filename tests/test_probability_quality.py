"""Unit tests for probability quality (Sprint 14)."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from research.model_v2.services.experiment_catalog import LONG_STRUCTURE_CONTEXT
from research.probability_quality.repositories import ProbabilityQualityRepository
from research.probability_quality.services.analyzers import (
    build_bucket_analysis,
    build_confidence_analysis,
    monotonicity_metrics,
)
from research.probability_quality.usecases import RunProbabilityQualityUseCase

UTC = ZoneInfo("UTC")


def _mini(root: Path) -> tuple[Path, Path, dict[str, pd.DataFrame]]:
    feat_dir = root / "features" / "XAUUSD" / "H1"
    struct_dir = root / "h4_structure"
    feat_dir.mkdir(parents=True)
    struct_dir.mkdir(parents=True)
    rng = np.random.default_rng(13)
    rows_f, rows_s, labs_l, labs_sh = [], [], [], []
    for year in range(2020, 2027):
        for i in range(40):
            t = datetime(year, 1, 1, tzinfo=UTC) + timedelta(hours=i)
            a = rng.normal(0, 1)
            sq = float(rng.random())
            score = 0.55 * a + 0.25 * sq + rng.normal(0, 0.35)
            y = 1 if score > 0 else -1
            if rng.random() < 0.12:
                y = 0
            ret = float(0.01 if y == 1 else (-0.01 if y == -1 else 0.0))
            rows_f.append({"timestamp": t, "ema20_distance_atr": a, "rsi_percentile": float(rng.random())})
            row = {"timestamp": t, "ctx_h4_swing_quality": sq}
            for f in LONG_STRUCTURE_CONTEXT:
                row.setdefault(f, float(rng.normal(0, 1) if "distance" in f else rng.random()))
            rows_s.append(row)
            labs_l.append({"timestamp": t, "label": y, "realized_return": ret})
            labs_sh.append(
                {
                    "timestamp": t,
                    "label": -y if y != 0 else 0,
                    "realized_return": -ret if y != 0 else 0.0,
                }
            )
    pd.DataFrame(rows_f).to_parquet(feat_dir / "feature_matrix.parquet", index=False)
    pd.DataFrame(rows_s).to_parquet(struct_dir / "structure_features.parquet", index=False)
    return (
        feat_dir / "feature_matrix.parquet",
        struct_dir / "structure_features.parquet",
        {"long": pd.DataFrame(labs_l), "short": pd.DataFrame(labs_sh)},
    )


class AnalyzerTests(unittest.TestCase):
    def test_buckets_and_mono(self) -> None:
        rng = np.random.default_rng(1)
        n = 500
        p = rng.uniform(0.5, 0.9, size=n)
        y = (rng.random(n) < p).astype(int)
        ret = np.where(y == 1, 0.01, -0.01)
        df = pd.DataFrame(
            {
                "side": "long",
                "y_true": y,
                "y_prob": p,
                "realized_return": ret,
                "valid_year": 2023,
            }
        )
        buckets = build_bucket_analysis(df)
        self.assertFalse(buckets.empty)
        conf = build_confidence_analysis(df)
        self.assertEqual(set(conf["confidence"]), {"low", "medium", "high"})
        mono = monotonicity_metrics(buckets, "long")
        self.assertIn("win_rate_spearman", mono)


class E2ETests(unittest.TestCase):
    def test_writes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            feat, struct, labels = _mini(root)
            preds, out = RunProbabilityQualityUseCase(
                ProbabilityQualityRepository(root / "research" / "probability_quality"),
                config={
                    "algorithm": "lightgbm",
                    "trainer_params": {
                        "num_boost_round": 25,
                        "early_stopping_rounds": 8,
                        "verbosity": -1,
                    },
                    "target_mode": "exclude_timeout",
                    "random_seed": 42,
                    "windows": [
                        {
                            "train_start_year": 2020,
                            "train_end_year": 2022,
                            "valid_year": 2023,
                        }
                    ],
                },
            ).execute(
                symbol="XAUUSD",
                timeframe="H1",
                feature_matrix_path=feat,
                structure_features_path=struct,
                labels_by_side=labels,
                sides=["long", "short"],
            )
            self.assertGreater(len(preds), 0)
            self.assertTrue((out / "probability_quality_report.md").is_file())
            self.assertTrue((out / "probability_bucket_analysis.csv").is_file())
            self.assertTrue((out / "confidence_analysis.csv").is_file())
            self.assertTrue((out / "predictions.parquet").is_file())
            self.assertTrue((out / "charts" / "probability_histogram.png").is_file())
            self.assertTrue((out / "charts" / "bucket_performance.png").is_file())
            self.assertTrue((out / "charts" / "calibration_curve.png").is_file())
            self.assertTrue((out / "charts" / "confidence_distribution.png").is_file())


if __name__ == "__main__":
    unittest.main()
