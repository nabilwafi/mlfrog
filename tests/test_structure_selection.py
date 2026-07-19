"""Unit tests for structure selection research (Sprint 12)."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from research.structure_selection.repositories.structure_selection_repository import (
    StructureSelectionRepository,
)
from research.structure_selection.services.experiment_catalog import (
    DEFAULT_TOP_STRUCTURE,
    resolve_experiments,
)
from research.structure_selection.usecases.run_structure_selection import (
    RunStructureSelectionUseCase,
)

UTC = ZoneInfo("UTC")


def _mini(root: Path) -> tuple[Path, Path, Path, dict[str, pd.DataFrame]]:
    feat_dir = root / "features" / "XAUUSD" / "H1"
    struct_dir = root / "h4_structure"
    ctx_dir = root / "context" / "XAUUSD" / "H1"
    feat_dir.mkdir(parents=True)
    struct_dir.mkdir(parents=True)
    ctx_dir.mkdir(parents=True)

    rng = np.random.default_rng(9)
    rows_f, rows_s, rows_c, labs_l, labs_sh = [], [], [], [], []
    for year in range(2020, 2027):
        for i in range(45):
            t = datetime(year, 1, 1, tzinfo=UTC) + timedelta(hours=i)
            a = rng.normal(0, 1)
            sq = float(rng.random())
            deq = float(a)
            y = 1 if (0.5 * a + 0.2 * sq + rng.normal(0, 0.4)) > 0 else -1
            if rng.random() < 0.15:
                y = 0
            rows_f.append(
                {
                    "timestamp": t,
                    "ema20_distance_atr": a,
                    "rsi_percentile": float(rng.random()),
                }
            )
            row_s = {
                "timestamp": t,
                "ctx_h4_swing_quality": sq,
                "ctx_h4_distance_from_equilibrium": deq,
                "ctx_h4_swing_strength": float(rng.random()),
                "ctx_h4_swing_high_distance_atr": float(rng.normal(-1, 0.5)),
                "ctx_h4_rejection_strength": float(rng.random()),
            }
            for f in DEFAULT_TOP_STRUCTURE:
                row_s.setdefault(f, float(rng.random()))
            rows_s.append(row_s)
            rows_c.append(
                {
                    "timestamp": t,
                    "ctx_h4_volatility_regime": float(rng.random()),
                    "ctx_h4_market_regime": float(rng.uniform(-1, 1)),
                    "ctx_h4_trend_direction": float(rng.uniform(-1, 1)),
                }
            )
            labs_l.append({"timestamp": t, "label": y})
            labs_sh.append({"timestamp": t, "label": -y if y != 0 else 0})

    pd.DataFrame(rows_f).to_parquet(feat_dir / "feature_matrix.parquet", index=False)
    pd.DataFrame(rows_s).to_parquet(struct_dir / "structure_features.parquet", index=False)
    pd.DataFrame(
        {
            "feature": list(DEFAULT_TOP_STRUCTURE),
            "importance_long": [5, 4, 3, 2, 1],
            "importance_short": [5, 4, 3, 2, 1],
        }
    ).to_csv(struct_dir / "feature_statistics.csv", index=False)
    pd.DataFrame(rows_c).to_parquet(ctx_dir / "context.parquet", index=False)
    return (
        feat_dir / "feature_matrix.parquet",
        struct_dir / "structure_features.parquet",
        struct_dir / "feature_statistics.csv",
        {"long": pd.DataFrame(labs_l), "short": pd.DataFrame(labs_sh)},
    )


class CatalogTests(unittest.TestCase):
    def test_five_experiments(self) -> None:
        exps = resolve_experiments(["a"], top_structure=DEFAULT_TOP_STRUCTURE)
        self.assertEqual(len(exps), 5)
        self.assertEqual(exps[0].experiment_id, "A_baseline")
        self.assertEqual(exps[-1].experiment_id, "E_top_structure")


class E2ETests(unittest.TestCase):
    def test_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            feat, struct, stats, labels = _mini(root)
            results, out = RunStructureSelectionUseCase(
                StructureSelectionRepository(root / "structure_selection"),
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
                    "top_k": 5,
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
                sides=["long"],
                feature_matrix_path=feat,
                structure_features_path=struct,
                labels_by_side={"long": labels["long"]},
                feature_statistics_path=stats,
                regime_context_path=root / "context" / "XAUUSD" / "H1" / "context.parquet",
            )
            self.assertEqual(len(results), 5)
            self.assertTrue((out / "experiment_results.csv").is_file())
            self.assertTrue((out / "feature_stability.csv").is_file())
            self.assertTrue((out / "selection_report.md").is_file())
            self.assertTrue((out / "charts" / "experiment_roc.png").is_file())


if __name__ == "__main__":
    unittest.main()
