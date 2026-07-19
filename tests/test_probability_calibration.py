"""Unit tests for probability calibration (Sprint 16)."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from research.model_v2.services.experiment_catalog import LONG_STRUCTURE_CONTEXT
from research.probability_calibration.repositories import ProbabilityCalibrationRepository
from research.probability_calibration.services.calibration_fitter import (
    build_calibration_comparison,
    calibrate_walk_forward,
)
from research.probability_calibration.services.threshold_analyzer import (
    analyze_percentile_thresholds,
)
from research.probability_calibration.usecases import RunProbabilityCalibrationUseCase

UTC = ZoneInfo("UTC")


def _synthetic_preds() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    for side in ("long", "short"):
        for year in (2023, 2024):
            window = f"train_{year-3}_{year-1}_val_{year}"
            # train
            for i in range(120):
                t = datetime(year - 2, 1, 1, tzinfo=UTC) + timedelta(hours=i)
                p = float(np.clip(rng.normal(0.41, 0.03), 0.2, 0.7))
                y = int(rng.random() < (p + 0.05))
                rows.append(
                    {
                        "side": side,
                        "experiment_id": "B_v2_context",
                        "window": window,
                        "valid_year": year,
                        "split": "train",
                        "timestamp": t,
                        "y_true": y,
                        "y_prob": p,
                        "realized_return": 0.01 if y else -0.01,
                    }
                )
            # validation — slightly shifted
            for i in range(80):
                t = datetime(year, 1, 1, tzinfo=UTC) + timedelta(hours=i)
                p = float(np.clip(rng.normal(0.42, 0.03), 0.2, 0.7))
                y = int(rng.random() < p)
                rows.append(
                    {
                        "side": side,
                        "experiment_id": "B_v2_context",
                        "window": window,
                        "valid_year": year,
                        "split": "validation",
                        "timestamp": t,
                        "y_true": y,
                        "y_prob": p,
                        "realized_return": 0.01 if y else -0.01,
                    }
                )
    return pd.DataFrame(rows)


class FitterTests(unittest.TestCase):
    def test_calibrate_and_thresholds(self) -> None:
        preds = _synthetic_preds()
        cal = calibrate_walk_forward(preds)
        self.assertFalse(cal.empty)
        self.assertIn("y_prob_platt", cal.columns)
        cmp_ = build_calibration_comparison(cal)
        self.assertEqual(set(cmp_["method"]), {"raw", "platt", "isotonic"})
        thr = analyze_percentile_thresholds(cal, prob_col="y_prob_raw", cost=0.0001)
        self.assertFalse(thr.empty)
        self.assertIn("profit_factor", thr.columns)


class E2ETests(unittest.TestCase):
    def test_writes_from_diagnosis_parquet(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            diag = root / "diagnosis"
            diag.mkdir()
            preds = _synthetic_preds()
            # add structure join targets
            struct_rows = []
            for ts in preds["timestamp"].unique():
                struct_rows.append(
                    {
                        "timestamp": ts,
                        "ctx_h4_swing_quality": 0.5,
                        "ctx_h4_distance_from_equilibrium": 0.1,
                        "ctx_h4_rejection_strength": 0.2,
                        **{f: 0.0 for f in LONG_STRUCTURE_CONTEXT if f not in {
                            "ctx_h4_swing_quality",
                            "ctx_h4_distance_from_equilibrium",
                            "ctx_h4_rejection_strength",
                        }},
                    }
                )
            struct_path = root / "structure_features.parquet"
            pd.DataFrame(struct_rows).to_parquet(struct_path, index=False)
            preds_path = diag / "predictions.parquet"
            preds.drop(columns=["realized_return"]).to_parquet(preds_path, index=False)

            # labels with returns
            labels = {}
            for side in ("long", "short"):
                sub = preds.loc[preds["side"] == side, ["timestamp", "y_true", "realized_return"]]
                lab = sub.rename(columns={"y_true": "label"})
                lab["label"] = lab["label"].map({1: 1, 0: -1})
                labels[side] = lab

            # dummy feature matrix (unused when diagnosis parquet present)
            feat = root / "features.parquet"
            pd.DataFrame({"timestamp": preds["timestamp"].unique()[:5], "a": 1.0}).to_parquet(
                feat, index=False
            )

            cal, out = RunProbabilityCalibrationUseCase(
                ProbabilityCalibrationRepository(root / "out"),
                config={"transaction_cost": 0.0001},
            ).execute(
                symbol="XAUUSD",
                timeframe="H1",
                feature_matrix_path=feat,
                structure_features_path=struct_path,
                labels_by_side=labels,
                sides=["long", "short"],
                diagnosis_predictions_path=preds_path,
            )
            self.assertGreater(len(cal), 0)
            self.assertTrue((out / "probability_calibration_report.md").is_file())
            self.assertTrue((out / "threshold_analysis.csv").is_file())
            self.assertTrue((out / "context_filter_analysis.csv").is_file())
            self.assertTrue((out / "charts" / "reliability_curve.png").is_file())
            self.assertTrue((out / "charts" / "threshold_expectancy.png").is_file())
            self.assertTrue((out / "charts" / "context_filter.png").is_file())


if __name__ == "__main__":
    unittest.main()
