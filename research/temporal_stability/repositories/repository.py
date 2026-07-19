"""Persist temporal stability artifacts."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class TemporalStabilityRepository:
    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    @property
    def root(self) -> Path:
        return self._root

    def save(self, artifacts: dict[str, Any], *, report_md: str, answers: dict[str, Any]) -> Path:
        out = self._root
        out.mkdir(parents=True, exist_ok=True)
        (out / "charts").mkdir(parents=True, exist_ok=True)

        frames = {
            "rolling_performance.csv": artifacts.get("rolling"),
            "rolling_metrics.csv": artifacts.get("rolling"),
            "year_summary_ml.csv": artifacts.get("year_ml"),
            "year_summary.csv": artifacts.get("year_frozen"),
            "aging_report.csv": artifacts.get("aging"),
            "recency_weights.csv": artifacts.get("recency"),
            "retrain_frequency.csv": artifacts.get("retrain"),
            "drift_report.csv": artifacts.get("drift"),
            "feature_importance_over_time.csv": artifacts.get("feature_importance"),
            "feature_stability.csv": artifacts.get("feature_stability"),
            "threshold_report.csv": artifacts.get("threshold"),
            "threshold_by_year.csv": artifacts.get("threshold"),
            "probability_stability.csv": artifacts.get("probability"),
            "monte_carlo_by_year.csv": artifacts.get("monte_carlo"),
        }
        for filename, df in frames.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                df.to_csv(out / filename, index=False)

        year_md = artifacts.get("year_summary_md")
        if year_md:
            (out / "year_summary.md").write_text(str(year_md), encoding="utf-8")

        (out / "answers.json").write_text(json.dumps(answers, indent=2, default=str), encoding="utf-8")
        (out / "temporal_stability_report.md").write_text(report_md, encoding="utf-8")
        logger.info("Saved temporal stability artifacts | dir=%s", out)
        return out
