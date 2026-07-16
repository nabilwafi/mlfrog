"""Unit tests for feature ablation framework."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from research.feature_ablation.repositories.ablation_repository import AblationRepository
from research.feature_ablation.services.experiment_catalog import (
    ExperimentResolver,
    default_experiment_catalog,
)
from research.feature_ablation.usecases.run_feature_ablation import RunFeatureAblationUseCase

UTC = ZoneInfo("UTC")


def _write_mini_artifacts(root: Path) -> tuple[Path, Path, Path, pd.DataFrame]:
    feat_dir = root / "features" / "XAUUSD" / "H1"
    diag_dir = root / "research" / "XAUUSD" / "H1" / "long"
    feat_dir.mkdir(parents=True)
    diag_dir.mkdir(parents=True)

    rng = np.random.default_rng(1)
    rows = []
    labels = []
    for year in range(2020, 2027):
        for i in range(80):
            t = datetime(year, 1, 1, tzinfo=UTC) + timedelta(hours=i)
            a = rng.normal(0, 1)
            b = rng.normal(0, 1)
            c = rng.normal(0, 1)
            d = rng.normal(0, 1)
            e = rng.normal(0, 1)
            f = rng.normal(0, 1)
            y = 1 if (0.7 * a + 0.3 * d + rng.normal(0, 0.4)) > 0 else -1
            if rng.random() < 0.2:
                y = 0
            rows.append(
                {
                    "timestamp": t,
                    "ema20_distance_atr": a,
                    "ema50_slope": b,
                    "macd_normalized": c,
                    "rsi_percentile": (d + 3) / 6,
                    "atr_percent": abs(e),
                    "body_percent": abs(rng.normal(0.3, 0.1)),
                    "hour_sin": np.sin(2 * np.pi * (i % 24) / 24),
                    "rolling_zscore": f,
                }
            )
            labels.append({"timestamp": t, "label": y})

    feats = pd.DataFrame(rows)
    labs = pd.DataFrame(labels)
    feats.to_parquet(feat_dir / "feature_matrix.parquet", index=False)
    meta = {
        "features": [
            {"name": "ema20_distance_atr", "category": "trend"},
            {"name": "ema50_slope", "category": "trend"},
            {"name": "macd_normalized", "category": "momentum"},
            {"name": "rsi_percentile", "category": "momentum"},
            {"name": "atr_percent", "category": "volatility"},
            {"name": "body_percent", "category": "candle"},
            {"name": "hour_sin", "category": "session"},
            {"name": "rolling_zscore", "category": "statistical"},
        ]
    }
    (feat_dir / "feature_metadata.json").write_text(json.dumps(meta), encoding="utf-8")

    names = [c for c in feats.columns if c != "timestamp"]
    pd.DataFrame(
        {
            "feature": names,
            "decision": ["keep"] * len(names),
            "category": [meta["features"][i]["category"] for i in range(len(names))],
        }
    ).to_csv(diag_dir / "feature_selection_candidates.csv", index=False)
    pd.DataFrame(
        {"feature": names, "stable_flag": [True, False, False, True, False, False, True, False]}
    ).to_csv(diag_dir / "feature_stability.csv", index=False)
    pd.DataFrame(
        {"feature": names, "mean_abs_shap": list(range(len(names), 0, -1))}
    ).to_csv(diag_dir / "feature_shap_importance.csv", index=False)
    pd.DataFrame(
        {"feature": names, "mutual_information": list(np.linspace(0.2, 0.01, len(names)))}
    ).to_csv(diag_dir / "feature_mutual_information.csv", index=False)

    return feat_dir, diag_dir, root / "ablation", labs


class CatalogTests(unittest.TestCase):
    def test_thirteen_experiments(self) -> None:
        self.assertEqual(len(default_experiment_catalog()), 13)

    def test_resolver_remove_category(self) -> None:
        resolver = ExperimentResolver(
            all_features=["a", "b", "c"],
            category_by_feature={"a": "trend", "b": "momentum", "c": "session"},
            keep_features=["a", "c"],
            stable_features=["c"],
            shap_ranked=["a", "b", "c"],
            mi_ranked=["b", "a", "c"],
        )
        exp = [e for e in default_experiment_catalog() if e.experiment_id == "remove_trend"][0]
        resolved = resolver.resolve(exp)
        self.assertNotIn("a", resolved.feature_names)
        self.assertIn("b", resolved.feature_names)


class AblationE2ETests(unittest.TestCase):
    def test_runs_and_writes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            feat_dir, diag_dir, out_dir, labs = _write_mini_artifacts(root)
            results, out = RunFeatureAblationUseCase(
                AblationRepository(out_dir),
                config={
                    "algorithm": "lightgbm",
                    "trainer_params": {
                        "num_boost_round": 25,
                        "early_stopping_rounds": 5,
                        "learning_rate": 0.1,
                        "num_leaves": 15,
                        "verbosity": -1,
                        "class_weight": True,
                    },
                    "target_mode": "exclude_timeout",
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
                diagnostics_dir=diag_dir,
                labels=labs,
            )
            self.assertEqual(len(results), 13)
            for name in (
                "feature_ablation_summary.csv",
                "feature_ablation_metrics.csv",
                "feature_ablation_rankings.csv",
                "feature_ablation_report.md",
            ):
                self.assertTrue((out / name).is_file(), name)
            self.assertTrue((out / "charts" / "roc_comparison.png").is_file())


if __name__ == "__main__":
    unittest.main()
