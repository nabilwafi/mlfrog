"""Unit tests for temporal stability research (synthetic panel)."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from research.temporal_stability.reports.report import build_answers
from research.temporal_stability.services import class_metrics, ece
from research.temporal_stability.services.data import recency_weights
from research.temporal_stability.services.experiments import rolling_performance
from research.temporal_stability.services.drift import year_pair_drift


def _synth(n_years: int = 8, n_per: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for yi, year in enumerate(range(2016, 2016 + n_years)):
        for i in range(n_per):
            # mild aging: signal weaker in later years
            strength = 0.8 - 0.05 * yi
            x0 = rng.normal()
            label = 1 if (x0 + rng.normal() * 0.5) > 0 else -1
            rows.append(
                {
                    "timestamp": pd.Timestamp(f"{year}-01-01", tz="UTC") + pd.Timedelta(hours=i),
                    "year": year,
                    "side": "long",
                    "label": label,
                    "feat_a": x0 * strength + rng.normal() * 0.1,
                    "feat_b": rng.normal(),
                    "feat_c": rng.normal() + 0.1 * yi,
                }
            )
    return pd.DataFrame(rows)


class MetricsTests(unittest.TestCase):
    def test_ece_and_class_metrics(self) -> None:
        y = np.array([0, 0, 1, 1, 1, 0, 1, 0])
        p = np.array([0.1, 0.2, 0.8, 0.7, 0.9, 0.3, 0.6, 0.4])
        self.assertTrue(ece(y, p) >= 0)
        m = class_metrics(y, p)
        self.assertIn("roc_auc", m)
        self.assertTrue(m["roc_auc"] > 0.5)


class WeightTests(unittest.TestCase):
    def test_recency_modes(self) -> None:
        y = np.array([2018, 2019, 2020, 2021])
        for mode in ("uniform", "linear", "exponential", "half_life"):
            w = recency_weights(y, mode=mode)
            self.assertEqual(len(w), 4)
            self.assertTrue(np.all(w > 0))


class DriftTests(unittest.TestCase):
    def test_year_pair_drift(self) -> None:
        panel = _synth(n_years=4, n_per=40)
        summary, detail = year_pair_drift(panel, features=["feat_a", "feat_b", "feat_c"])
        self.assertFalse(summary.empty)
        self.assertFalse(detail.empty)
        self.assertIn("feature_psi_mean", summary.columns)


class RollingSmokeTests(unittest.TestCase):
    def test_rolling_runs(self) -> None:
        panel = _synth(n_years=7, n_per=60)
        out = rolling_performance(panel, min_train_years=3)
        self.assertFalse(out.empty)
        self.assertIn("roc_auc", out.columns)

    def test_answers(self) -> None:
        panel = _synth(n_years=7, n_per=60)
        rolling = rolling_performance(panel, min_train_years=3)
        answers = build_answers({"rolling": rolling, "aging": pd.DataFrame(), "recency": pd.DataFrame()})
        self.assertIn("recommendation", answers)
        self.assertIn("q1_model_ages", answers)


if __name__ == "__main__":
    unittest.main()
