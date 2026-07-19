"""Unit tests for meta feature ablation (fixed LGBM; not final meta)."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from research.meta_feature_ablation.services.correlation import (
    pearson_spearman,
    removal_recommendations,
    variance_inflation_factors,
)
from research.meta_feature_ablation.services.feature_groups import ABLATION_STAGES, available_features
from research.meta_feature_ablation.usecases import RunMetaFeatureAblationUseCase
from research.meta_feature_ablation.repositories import MetaFeatureAblationRepository

UTC = ZoneInfo("UTC")


class CorrelationTests(unittest.TestCase):
    def test_vif_and_removal(self) -> None:
        rng = np.random.default_rng(0)
        n = 200
        a = rng.normal(size=n)
        b = a + rng.normal(scale=0.01, size=n)  # near-duplicate
        c = rng.normal(size=n)
        df = pd.DataFrame({"a": a, "b": b, "c": c})
        pearson, _ = pearson_spearman(df, ["a", "b", "c"])
        vif = variance_inflation_factors(df, ["a", "b", "c"])
        rem = removal_recommendations(pearson, vif, pearson_thr=0.95, vif_thr=10.0)
        self.assertTrue(set(rem["feature"]).intersection({"a", "b"}))


class AblationE2ETests(unittest.TestCase):
    def test_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rng = np.random.default_rng(2)
            rows = []
            t0 = datetime(2023, 1, 1, tzinfo=UTC)
            for year in (2023, 2024):
                for i in range(60):
                    t = t0.replace(year=year) + timedelta(hours=i * 3)
                    y = int(rng.random() < 0.45)
                    net = 0.005 if y else -0.004
                    p = 0.4 + 0.1 * y + rng.normal(0, 0.02)
                    rows.append(
                        {
                            "side": "long",
                            "window": f"val_{year}",
                            "valid_year": year,
                            "timestamp": t,
                            "meta_label": y,
                            "net_return": net,
                            "holding_bars": float(rng.integers(2, 20)),
                            "raw_probability": p,
                            "probability_rank": float(i),
                            "probability_percentile": float(i) / 60.0,
                            "probability_margin": float(rng.random()),
                            "ctx_h4_rejection_strength": float(rng.random() + 0.2 * y),
                            "ctx_h4_swing_quality": float(rng.random()),
                            "ctx_h4_swing_strength": float(rng.random()),
                            "ctx_h4_distance_from_equilibrium": float(rng.normal()),
                            "ctx_h4_premium_discount_zone": float(rng.random()),
                            "ctx_h4_h4_range_position": float(rng.random()),
                            "ctx_h4_swing_high_distance_atr": float(rng.random()),
                            "ctx_h4_swing_low_distance_atr": float(rng.random()),
                            "rolling_quantile": float(rng.random()),
                            "rolling_std": float(rng.random()),
                            "rolling_volatility": float(rng.random()),
                            "ema_trend_duration": float(rng.integers(0, 10)),
                            "volatility_rank": float(rng.random()),
                            "ema_alignment_score": float(rng.normal()),
                            "rsi_percentile": float(rng.random()),
                            "atr_percent": float(rng.random()),
                            "momentum_rank": float(rng.random()),
                            "hour_of_day": float(rng.integers(0, 24)),
                            "day_of_week": float(rng.integers(0, 5)),
                            "month": float(rng.integers(1, 12)),
                            "session_london": float(rng.integers(0, 2)),
                            "session_newyork": float(rng.integers(0, 2)),
                            "session_london_ny_overlap": float(rng.integers(0, 2)),
                            "session_asia": float(rng.integers(0, 2)),
                            "distance_to_sl": float(rng.random()),
                            "distance_to_tp": float(rng.random()),
                            "atr_entry": float(rng.random()),
                            "atr_stop_size": float(rng.random()),
                            "initial_rr": float(rng.uniform(0.5, 2.0)),
                        }
                    )
            panel = pd.DataFrame(rows)
            panel_path = root / "panel.parquet"
            cand_path = root / "candidates.parquet"
            panel.to_parquet(panel_path, index=False)
            panel[
                ["side", "window", "valid_year", "timestamp", "holding_bars", "net_return", "meta_label"]
            ].assign(percentile=0.03).to_parquet(cand_path, index=False)

            # Smoke: stages resolve features
            for sid, wanted in ABLATION_STAGES:
                feats = available_features(list(panel.columns), wanted)
                self.assertGreaterEqual(len(feats), 3, msg=sid)

            out = RunMetaFeatureAblationUseCase(
                MetaFeatureAblationRepository(root / "out"),
                config={
                    "random_seed": 42,
                    "threshold": 0.5,
                    "num_boost_round": 30,
                    "early_stopping_rounds": 5,
                    "transaction_cost": 0.00015,
                    "compute_shap": False,
                },
            ).execute(
                symbol="XAUUSD",
                timeframe="H1",
                panel_path=panel_path,
                candidates_path=cand_path,
            )
            self.assertTrue((out / "meta_feature_ablation_report.md").is_file())
            self.assertTrue((out / "vif_table.csv").is_file())
            self.assertTrue((out / "stage_summary.csv").is_file())
            self.assertTrue((out / "charts" / "ablation_curve.png").is_file())


if __name__ == "__main__":
    unittest.main()
