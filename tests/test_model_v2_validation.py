"""Unit tests for model v2 validation (Sprint 13)."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from research.model_v2.repositories.model_v2_repository import ModelV2Repository
from research.model_v2.services.experiment_catalog import (
    LONG_STRUCTURE_CONTEXT,
    SHORT_STRUCTURE_CONTEXT,
    resolve_side_experiments,
)
from research.model_v2.usecases.run_model_v2_validation import RunModelV2ValidationUseCase

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
            y = 1 if (0.55 * a + 0.25 * sq + rng.normal(0, 0.35)) > 0 else -1
            if rng.random() < 0.12:
                y = 0
            rows_f.append({"timestamp": t, "ema20_distance_atr": a, "rsi_percentile": float(rng.random())})
            row = {"timestamp": t, "ctx_h4_swing_quality": sq}
            for f in LONG_STRUCTURE_CONTEXT:
                row.setdefault(f, float(rng.normal(0, 1) if "distance" in f else rng.random()))
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


class CatalogTests(unittest.TestCase):
    def test_ab_per_side(self) -> None:
        exps = resolve_side_experiments(
            ["a", "b"], side="long", context_features=LONG_STRUCTURE_CONTEXT
        )
        self.assertEqual(len(exps), 2)
        self.assertEqual(exps[0].experiment_id, "A_baseline")
        self.assertEqual(exps[1].experiment_id, "B_v2_context")
        self.assertEqual(len(SHORT_STRUCTURE_CONTEXT), 1)


class E2ETests(unittest.TestCase):
    def test_writes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            feat, struct, labels = _mini(root)
            results, out = RunModelV2ValidationUseCase(
                ModelV2Repository(root / "models" / "v2"),
                config={
                    "algorithm": "lightgbm",
                    "trainer_params": {
                        "num_boost_round": 25,
                        "early_stopping_rounds": 8,
                        "verbosity": -1,
                    },
                    "target_mode": "exclude_timeout",
                    "threshold": 0.5,
                    "random_seed": 42,
                    "compute_shap": True,
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
            self.assertEqual(len(results), 4)
            self.assertTrue((out / "comparison.csv").is_file())
            self.assertTrue((out / "feature_importance.csv").is_file())
            self.assertTrue((out / "long_model_v2_report.md").is_file())
            self.assertTrue((out / "short_model_v2_report.md").is_file())
            self.assertTrue((out / "charts" / "comparison_delta.png").is_file())


if __name__ == "__main__":
    unittest.main()
