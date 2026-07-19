"""Unit tests for feature diagnostics."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from feature_diagnostics.analyzers.correlation_analyzer import CorrelationAnalyzer
from feature_diagnostics.analyzers.redundancy_analyzer import RedundancyAnalyzer
from feature_diagnostics.repositories.feature_diagnostics_repository import (
    FeatureDiagnosticsRepository,
)
from feature_diagnostics.services.feature_diagnostics_service import FeatureDiagnosticsService

UTC = ZoneInfo("UTC")


def _synth(n: int = 800) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rng = np.random.default_rng(0)
    ts = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    # years spanning walk-forward windows
    # rebuild with multi-year coverage
    parts = []
    labels = []
    t0 = datetime(2020, 1, 1, tzinfo=UTC)
    idx = 0
    for year in range(2020, 2027):
        for i in range(120):
            t = datetime(year, 1, 1, tzinfo=UTC) + timedelta(hours=i)
            a = rng.normal(0, 1)
            b = a + rng.normal(0, 0.05)  # redundant with a
            c = rng.normal(year - 2023, 1)  # drifting
            d = rng.normal(0, 1)
            y = 1 if (0.8 * a + 0.2 * d + rng.normal(0, 0.5)) > 0 else -1
            if rng.random() < 0.25:
                y = 0
            parts.append(
                {
                    "timestamp": t,
                    "ema20_distance_atr": a,
                    "ema50_distance_atr": b,
                    "atr_percent": abs(c),
                    "rsi_percentile": (d + 3) / 6,
                    "hour_sin": np.sin(2 * np.pi * (i % 24) / 24),
                    "volatility_regime_score": rng.uniform(-1, 1),
                    "ema_alignment_score": rng.uniform(-1, 1),
                }
            )
            labels.append({"timestamp": t, "label": y})
            idx += 1
    feats = pd.DataFrame(parts)
    labs = pd.DataFrame(labels)
    meta = {
        "symbol": "XAUUSD",
        "timeframe": "H1",
        "features": [
            {"name": c, "category": "trend" if "ema" in c else "volatility" if "atr" in c or "vol" in c else "session" if "hour" in c else "momentum",
             "stationary": True, "normalized": True, "drift_sensitive": False, "depends_on": [], "description": c}
            for c in feats.columns if c != "timestamp"
        ],
    }
    return feats, labs, meta


class CorrelationTests(unittest.TestCase):
    def test_redundant_pair(self) -> None:
        feats, _, _ = _synth(200)
        x = feats.drop(columns=["timestamp"])
        _, corr = CorrelationAnalyzer().analyze(x)
        summary, red = RedundancyAnalyzer().analyze(corr, threshold=0.9)
        self.assertGreaterEqual(summary["n_groups"], 1)


class FeatureDiagnosticsE2ETests(unittest.TestCase):
    def test_writes_artifacts(self) -> None:
        feats, labs, meta = _synth()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            feat_dir = root / "features" / "XAUUSD" / "H1"
            feat_dir.mkdir(parents=True)
            feats.to_parquet(feat_dir / "feature_matrix.parquet", index=False)
            (feat_dir / "feature_metadata.json").write_text(json.dumps(meta), encoding="utf-8")

            report, out = FeatureDiagnosticsService(
                FeatureDiagnosticsRepository(root / "research"),
                features_root=root / "features",
                config={
                    "target_mode": "exclude_timeout",
                    "corr_threshold": 0.9,
                    "random_seed": 0,
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
                label_loader=lambda *_: labs,
            ).run("XAUUSD", "H1", "long")

            for name in (
                "feature_correlation.csv",
                "feature_mutual_information.csv",
                "feature_permutation_importance.csv",
                "feature_shap_importance.csv",
                "feature_stability.csv",
                "feature_redundancy.csv",
                "feature_selection_candidates.csv",
                "feature_diagnostics_report.md",
            ):
                self.assertTrue((out / name).is_file(), name)
            self.assertTrue(len(report.keep) + len(report.remove) >= 1)
            self.assertIsInstance(report.candidate_library_v3, list)


if __name__ == "__main__":
    unittest.main()
