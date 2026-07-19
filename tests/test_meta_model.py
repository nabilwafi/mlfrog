"""Unit tests for production meta model."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from research.meta_model import FINAL_FEATURES
from research.meta_model.repositories import MetaModelRepository
from research.meta_model.usecases import RunMetaModelUseCase

UTC = ZoneInfo("UTC")


class MetaModelE2ETests(unittest.TestCase):
    def test_writes_models_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rng = np.random.default_rng(3)
            rows = []
            t0 = datetime(2023, 1, 1, tzinfo=UTC)
            for year in (2023, 2024):
                for i in range(80):
                    t = t0.replace(year=year) + timedelta(hours=i * 2)
                    y = int(rng.random() < 0.48)
                    # Signal: raw_probability weakly predictive
                    p = 0.35 + 0.2 * y + rng.normal(0, 0.05)
                    net = 0.004 if y else -0.003
                    row = {
                        "side": "long",
                        "window": f"val_{year}",
                        "valid_year": year,
                        "timestamp": t,
                        "meta_label": y,
                        "net_return": net,
                        "holding_bars": float(rng.integers(2, 15)),
                    }
                    for f in FINAL_FEATURES:
                        if f == "raw_probability":
                            row[f] = p
                        elif f.startswith("probability"):
                            row[f] = float(rng.random())
                        elif f in ("hour_of_day", "day_of_week", "month"):
                            row[f] = float(rng.integers(0, 12))
                        else:
                            row[f] = float(rng.normal() + 0.3 * y)
                    rows.append(row)
            panel = pd.DataFrame(rows)
            panel_path = root / "panel.parquet"
            cand_path = root / "cand.parquet"
            panel.to_parquet(panel_path, index=False)
            panel[
                ["side", "window", "valid_year", "timestamp", "net_return", "holding_bars", "meta_label"]
            ].assign(percentile=0.03).to_parquet(cand_path, index=False)

            out = RunMetaModelUseCase(
                MetaModelRepository(root / "out"),
                config={
                    "random_seed": 42,
                    "num_boost_round": 40,
                    "early_stopping_rounds": 10,
                    "reference_threshold": 0.5,
                    "starting_equity": 80.0,
                    "transaction_cost": 0.00015,
                    "percentile": 0.03,
                },
            ).execute(
                symbol="XAUUSD",
                timeframe="H1",
                panel_path=panel_path,
                candidates_path=cand_path,
            )
            self.assertTrue((out / "meta_model_report.md").is_file())
            self.assertTrue((out / "models" / "meta_full.txt").is_file())
            self.assertTrue((out / "summary_metrics.csv").is_file())
            self.assertTrue((out / "charts" / "equity_curve.png").is_file())
            self.assertTrue((out / "answers.json").is_file())


if __name__ == "__main__":
    unittest.main()
