"""Persist probability diagnosis artifacts."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class ProbabilityDiagnosisRepository:
    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    @property
    def root(self) -> Path:
        return self._root

    def save(
        self,
        *,
        predictions: pd.DataFrame,
        labels: pd.DataFrame,
        pred_dist: pd.DataFrame,
        wf_drift: pd.DataFrame,
        compare: pd.DataFrame,
        report_md: str,
    ) -> Path:
        out = self._root
        out.mkdir(parents=True, exist_ok=True)
        predictions.to_parquet(out / "predictions.parquet", index=False)
        labels.to_csv(out / "label_distribution.csv", index=False)
        pred_dist.to_csv(out / "prediction_distribution.csv", index=False)
        wf_drift.to_csv(out / "wf_probability_drift.csv", index=False)
        compare.to_csv(out / "baseline_vs_v2.csv", index=False)
        (out / "probability_diagnosis_report.md").write_text(report_md, encoding="utf-8")
        logger.info("Saved probability diagnosis artifacts | dir=%s", out)
        return out
