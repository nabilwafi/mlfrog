"""Unit tests for meta feature research (no meta training)."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from research.meta_feature_research.repositories import MetaFeatureResearchRepository
from research.meta_feature_research.services.univariate import score_all_features
from research.meta_feature_research.usecases import RunMetaFeatureResearchUseCase

UTC = ZoneInfo("UTC")


class UnivariateTests(unittest.TestCase):
    def test_scores(self) -> None:
        rng = np.random.default_rng(0)
        n = 200
        y = (rng.random(n) > 0.5).astype(int)
        x = y.astype(float) + rng.normal(0, 0.3, size=n)
        df = pd.DataFrame({"meta_label": y, "signal": x, "noise": rng.normal(size=n)})
        scored = score_all_features(df, ["signal", "noise"])
        sig = float(scored.loc[scored["feature"] == "signal", "mutual_info"].iloc[0])
        noi = float(scored.loc[scored["feature"] == "noise", "mutual_info"].iloc[0])
        self.assertGreater(sig, noi)


class E2ETests(unittest.TestCase):
    def test_writes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rng = np.random.default_rng(1)
            rows = []
            h1_rows = []
            h4_rows = []
            m15_rows = []
            t0 = datetime(2023, 1, 1, tzinfo=UTC)
            for side in ("long", "short"):
                for year in (2023, 2024):
                    window = f"train_{year-3}_{year-1}_val_{year}"
                    for i in range(40):
                        t = t0.replace(year=year) + timedelta(hours=i * 4)
                        p = float(rng.uniform(0.4, 0.5))
                        y = int(rng.random() < 0.45)
                        net = 0.01 if y else -0.01
                        rows.append(
                            {
                                "side": side,
                                "window": window,
                                "valid_year": year,
                                "timestamp": t,
                                "y_prob_raw": p,
                                "percentile": 0.03,
                                "meta_label": y,
                                "net_return": net,
                                "entry_price": 2000.0,
                                "metadata_json": '{"atr": 10.0, "tp_atr_mult": 2.0, "sl_atr_mult": 1.5}',
                            }
                        )
                        h1_rows.append(
                            {
                                "timestamp": t,
                                "rsi_percentile": float(rng.random()),
                                "atr_percent": float(rng.random()),
                                "ema_alignment_score": float(rng.normal()),
                            }
                        )
                        h4_rows.append(
                            {
                                "timestamp": t,
                                "ctx_h4_rejection_strength": float(rng.random()),
                                "ctx_h4_swing_quality": float(rng.random()),
                                "ctx_h4_distance_from_equilibrium": float(rng.normal()),
                            }
                        )
            # M15 denser bars
            for i in range(500):
                t = t0 + timedelta(minutes=15 * i)
                px = 2000 + i * 0.01
                m15_rows.append(
                    {
                        "timestamp": t,
                        "open": px,
                        "high": px + 0.5,
                        "low": px - 0.5,
                        "close": px + 0.1,
                        "tick_volume": float(100 + rng.integers(0, 50)),
                    }
                )

            cand = root / "candidates.parquet"
            h1 = root / "h1.parquet"
            h4 = root / "h4.parquet"
            m15 = root / "m15.parquet"
            pd.DataFrame(rows).to_parquet(cand, index=False)
            pd.DataFrame(h1_rows).drop_duplicates("timestamp").to_parquet(h1, index=False)
            pd.DataFrame(h4_rows).drop_duplicates("timestamp").to_parquet(h4, index=False)
            pd.DataFrame(m15_rows).to_parquet(m15, index=False)

            panel, out = RunMetaFeatureResearchUseCase(
                MetaFeatureResearchRepository(root / "out"),
                config={"percentile": 0.03},
            ).execute(
                symbol="XAUUSD",
                timeframe="H1",
                candidates_path=cand,
                h1_features_path=h1,
                h4_structure_path=h4,
                m15_candles_path=m15,
            )
            self.assertGreater(len(panel), 0)
            self.assertTrue((out / "meta_feature_research_report.md").is_file())
            self.assertTrue((out / "univariate_scores.csv").is_file())
            self.assertTrue((out / "charts" / "feature_importance.png").is_file())
            self.assertTrue((out / "charts" / "m15_vs_h4_gain.png").is_file())
            # no leakage cols as features in scores
            scores = pd.read_csv(out / "univariate_scores.csv")
            for bad in ("mae", "mfe", "holding_bars", "realized_return", "meta_label"):
                self.assertNotIn(bad, set(scores["feature"]))


if __name__ == "__main__":
    unittest.main()
