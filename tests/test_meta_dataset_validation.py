"""Unit tests for meta dataset validation (research only)."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from research.meta_dataset_validation.repositories import MetaDatasetValidationRepository
from research.meta_dataset_validation.services.analyzers import (
    attach_meta_label,
    build_candidate_trades,
    build_percentile_summary,
    build_stability,
)
from research.meta_dataset_validation.usecases import RunMetaDatasetValidationUseCase

UTC = ZoneInfo("UTC")


def _mini_preds() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    rows = []
    for side in ("long", "short"):
        for year in (2023, 2024, 2025):
            window = f"train_{year-3}_{year-1}_val_{year}"
            for i in range(100):
                t = datetime(year, 1, 1, tzinfo=UTC) + timedelta(hours=i)
                p = float(np.clip(rng.normal(0.42, 0.03), 0.25, 0.65))
                y = int(rng.random() < p)
                ret = 0.01 if y else -0.008
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
                        "y_prob_raw": p,
                        "realized_return": ret,
                    }
                )
    return pd.DataFrame(rows)


class AnalyzerTests(unittest.TestCase):
    def test_candidates(self) -> None:
        preds = _mini_preds()
        c = attach_meta_label(build_candidate_trades(preds), cost=0.0001)
        self.assertFalse(c.empty)
        self.assertIn("meta_label", c.columns)
        summary = build_percentile_summary(c, cost=0.0001)
        self.assertTrue((summary["n_trades"] > 0).any())
        stab = build_stability(c)
        self.assertIn("meta_trainable_heuristic", stab.columns)


class E2ETests(unittest.TestCase):
    def test_writes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            preds = _mini_preds()
            labels = {}
            candle_rows = []
            for side in ("long", "short"):
                sub = preds.loc[preds["side"] == side]
                labels[side] = pd.DataFrame(
                    {
                        "timestamp": sub["timestamp"],
                        "holding_bars": 4,
                        "realized_return": sub["realized_return"],
                        "entry_price": 2000.0,
                        "exit_reason": "TP",
                        "metadata_json": "{}",
                    }
                )
            # Minimal candles covering timestamps
            ts = sorted(preds["timestamp"].unique())
            for i, t in enumerate(ts):
                px = 2000 + i * 0.1
                candle_rows.append(
                    {"timestamp": t, "open": px, "high": px + 1, "low": px - 1, "close": px}
                )
            candles = pd.DataFrame(candle_rows)

            cands, out = RunMetaDatasetValidationUseCase(
                MetaDatasetValidationRepository(root / "out"),
                config={"transaction_cost": 0.0001},
            ).execute(
                symbol="XAUUSD",
                timeframe="H1",
                predictions=preds,
                labels_by_side=labels,
                candles=candles,
                sides=["long", "short"],
            )
            self.assertGreater(len(cands), 0)
            self.assertTrue((out / "meta_dataset_validation_report.md").is_file())
            self.assertTrue((out / "percentile_quality.csv").is_file())
            self.assertTrue((out / "charts" / "return_hist.png").is_file())
            self.assertTrue((out / "charts" / "wf_expectancy.png").is_file())


if __name__ == "__main__":
    unittest.main()
