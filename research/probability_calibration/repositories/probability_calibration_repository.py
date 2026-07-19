"""Persist Sprint 16 artifacts."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class ProbabilityCalibrationRepository:
    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    @property
    def root(self) -> Path:
        return self._root

    def save(
        self,
        *,
        calibrated: pd.DataFrame,
        calibration_comparison: pd.DataFrame,
        thresholds: pd.DataFrame,
        context: pd.DataFrame,
        report_md: str,
    ) -> Path:
        out = self._root
        out.mkdir(parents=True, exist_ok=True)
        calibrated.to_parquet(out / "calibrated_predictions.parquet", index=False)
        calibration_comparison.to_csv(out / "calibration_comparison.csv", index=False)
        thresholds.to_csv(out / "threshold_analysis.csv", index=False)
        context.to_csv(out / "context_filter_analysis.csv", index=False)
        (out / "probability_calibration_report.md").write_text(report_md, encoding="utf-8")
        logger.info("Saved probability calibration artifacts | dir=%s", out)
        return out
