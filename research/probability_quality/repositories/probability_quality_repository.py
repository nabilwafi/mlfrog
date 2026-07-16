"""Persist probability quality artifacts."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class ProbabilityQualityRepository:
    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    @property
    def root(self) -> Path:
        return self._root

    def save(
        self,
        *,
        predictions: pd.DataFrame,
        buckets: pd.DataFrame,
        deciles: pd.DataFrame,
        confidence: pd.DataFrame,
        relative_confidence: pd.DataFrame,
        distribution: pd.DataFrame,
        monotonicity: pd.DataFrame,
        report_md: str,
    ) -> Path:
        out = self._root
        out.mkdir(parents=True, exist_ok=True)
        predictions.to_parquet(out / "predictions.parquet", index=False)
        buckets.to_csv(out / "probability_bucket_analysis.csv", index=False)
        deciles.to_csv(out / "probability_decile_analysis.csv", index=False)
        confidence.to_csv(out / "confidence_analysis.csv", index=False)
        relative_confidence.to_csv(out / "confidence_relative_analysis.csv", index=False)
        distribution.to_csv(out / "probability_distribution.csv", index=False)
        monotonicity.to_csv(out / "monotonicity.csv", index=False)
        (out / "probability_quality_report.md").write_text(report_md, encoding="utf-8")
        logger.info("Saved probability quality artifacts | dir=%s", out)
        return out
