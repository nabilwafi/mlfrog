"""Unit tests for probability collapse diagnosis (Sprint 15)."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from research.model_v2.services.experiment_catalog import LONG_STRUCTURE_CONTEXT
from research.probability_diagnosis.repositories import ProbabilityDiagnosisRepository
from research.probability_diagnosis.services.analyzers import (
    build_baseline_vs_v2,
    build_label_distribution,
    build_prediction_distribution,
)
from research.probability_diagnosis.usecases import RunProbabilityDiagnosisUseCase

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
            rows_f.append(
                {"timestamp": t, "ema20_distance_atr": a, "rsi_percentile": float(rng.random())}
            )
            row = {"timestamp": t, "ctx_h4_swing_quality": sq}
            for f in LONG_STRUCTURE_CONTEXT:
                row.setdefault(
                    f, float(rng.normal(0, 1) if "distance" in f else rng.random())
                )
            rows_s.append(row)
            labs_l.append({"timestamp": t, "label": y})
            labs_sh.append({"timestamp": t, "label": -y if y != 0 else 0})
    pd.DataFrame(rows_f).to_parquet(feat_dir / "feature_matrix.parquet", index=False)
    pd.DataFrame(rows_s).to_parquet(struct_dir / "structure_features.parquet", index=False)
    return (
        feat_dir / "feature_matrix.parquet",
        struct_dir / "structure_features.parquet",
        {"long": pd.DataFrame(labs_l), "short": pd.DataFrame(labs_sh)},
    )


class AnalyzerTests(unittest.TestCase):
    def test_tables(self) -> None:
        rng = np.random.default_rng(0)
        rows = []
        for side in ("long", "short"):
            for exp in ("A_baseline", "B_v2_context"):
                for split in ("train", "validation"):
                    n = 200
                    p = rng.normal(0.41, 0.02, size=n).clip(0.2, 0.6)
                    y = (rng.random(n) < 0.4).astype(int)
                    rows.append(
                        pd.DataFrame(
                            {
                                "side": side,
                                "experiment_id": exp,
                                "split": split,
                                "window": "w1",
                                "valid_year": 2023,
                                "y_true": y,
                                "y_prob": p,
                            }
                        )
                    )
        preds = pd.concat(rows, ignore_index=True)
        labels = build_label_distribution(preds)
        self.assertEqual(len(labels), 2)
        dist = build_prediction_distribution(preds)
        self.assertFalse(dist.empty)
        cmp_ = build_baseline_vs_v2(dist)
        self.assertEqual(len(cmp_), 2)


class E2ETests(unittest.TestCase):
    def test_writes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            feat, struct, labels = _mini(root)
            preds, out = RunProbabilityDiagnosisUseCase(
                ProbabilityDiagnosisRepository(root / "research" / "probability_diagnosis"),
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
            self.assertTrue((out / "probability_diagnosis_report.md").is_file())
            self.assertTrue((out / "prediction_distribution.csv").is_file())
            self.assertTrue((out / "wf_probability_drift.csv").is_file())
            self.assertTrue((out / "charts" / "prediction_histogram.png").is_file())
            self.assertTrue((out / "charts" / "wf_probability_shift.png").is_file())
            self.assertTrue((out / "charts" / "baseline_vs_v2_probability.png").is_file())
            # train + val for A and B
            self.assertIn("train", set(preds["split"]))
            self.assertIn("validation", set(preds["split"]))
            self.assertIn("A_baseline", set(preds["experiment_id"]))
            self.assertIn("B_v2_context", set(preds["experiment_id"]))


if __name__ == "__main__":
    unittest.main()
