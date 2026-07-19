"""Unit tests for context impact research (Sprint 10)."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from research.context_impact.services.experiment_catalog import (
    ALL_CONTEXT,
    resolve_experiments,
)
from research.context_impact.repositories.context_impact_repository import ContextImpactRepository
from research.context_impact.usecases.run_context_impact import RunContextImpactUseCase

UTC = ZoneInfo("UTC")


def _write_mini(root: Path) -> tuple[Path, Path, dict[str, pd.DataFrame]]:
    feat_dir = root / "features" / "XAUUSD" / "H1"
    ctx_dir = root / "context" / "XAUUSD" / "H1"
    feat_dir.mkdir(parents=True)
    ctx_dir.mkdir(parents=True)

    rng = np.random.default_rng(3)
    rows_f, rows_c, labs_long, labs_short = [], [], [], []
    for year in range(2020, 2027):
        for i in range(50):
            t = datetime(year, 1, 1, tzinfo=UTC) + timedelta(hours=i)
            a = rng.normal(0, 1)
            b = rng.normal(0, 1)
            # Context weakly informative for long labels
            td = float(np.tanh(a))
            y_long = 1 if (0.6 * a + 0.3 * td + rng.normal(0, 0.4)) > 0 else -1
            y_short = 1 if (0.5 * b + rng.normal(0, 0.5)) > 0 else -1
            if rng.random() < 0.15:
                y_long = 0
                y_short = 0
            rows_f.append(
                {
                    "timestamp": t,
                    "ema20_distance_atr": a,
                    "rsi_percentile": (b + 3) / 6,
                    "atr_percent": abs(rng.normal(0.2, 0.05)),
                }
            )
            rows_c.append(
                {
                    "timestamp": t,
                    "ctx_h4_trend_direction": td,
                    "ctx_h4_trend_strength": abs(td),
                    "ctx_h4_market_regime": td * 0.5,
                    "ctx_h4_volatility_regime": float(rng.random()),
                    "ctx_h4_compression": float(rng.random()),
                    "ctx_h4_expansion": float(rng.random()),
                    "ctx_h4_swing_quality": float(rng.random()),
                }
            )
            labs_long.append({"timestamp": t, "label": y_long})
            labs_short.append({"timestamp": t, "label": y_short})

    pd.DataFrame(rows_f).to_parquet(feat_dir / "feature_matrix.parquet", index=False)
    pd.DataFrame(rows_c).to_parquet(ctx_dir / "context.parquet", index=False)
    return (
        feat_dir / "feature_matrix.parquet",
        ctx_dir / "context.parquet",
        {"long": pd.DataFrame(labs_long), "short": pd.DataFrame(labs_short)},
    )


class CatalogTests(unittest.TestCase):
    def test_five_experiments(self) -> None:
        exps = resolve_experiments(["a", "b"])
        self.assertEqual(len(exps), 5)
        ids = [e.experiment_id for e in exps]
        self.assertEqual(ids[0], "baseline")
        self.assertEqual(ids[-1], "all_context")
        all_exp = exps[-1]
        self.assertEqual(len(all_exp.feature_names), 2 + len(ALL_CONTEXT))


class ContextImpactE2ETests(unittest.TestCase):
    def test_runs_long_and_short(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            feat_path, ctx_path, labels = _write_mini(root)
            results, out = RunContextImpactUseCase(
                ContextImpactRepository(root / "research" / "context_impact"),
                config={
                    "algorithm": "lightgbm",
                    "trainer_params": {
                        "num_boost_round": 30,
                        "early_stopping_rounds": 8,
                        "verbosity": -1,
                    },
                    "target_mode": "exclude_timeout",
                    "threshold": 0.5,
                    "random_seed": 42,
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
                sides=["long", "short"],
                feature_matrix_path=feat_path,
                context_path=ctx_path,
                labels_by_side=labels,
            )
            self.assertEqual(len(results), 10)  # 5 × 2
            self.assertTrue((out / "comparison.csv").is_file())
            self.assertTrue((out / "ablation_results.csv").is_file())
            self.assertTrue((out / "context_impact_report.md").is_file())
            self.assertTrue((out / "charts" / "roc_comparison.png").is_file())
            sides = {r.side for r in results}
            self.assertEqual(sides, {"long", "short"})


if __name__ == "__main__":
    unittest.main()
