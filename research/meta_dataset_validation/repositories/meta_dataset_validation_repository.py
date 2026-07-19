"""Persist meta-dataset validation artifacts."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class MetaDatasetValidationRepository:
    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    @property
    def root(self) -> Path:
        return self._root

    def save(
        self,
        *,
        candidates: pd.DataFrame,
        summary: pd.DataFrame,
        wf: pd.DataFrame,
        stability: pd.DataFrame,
        report_md: str,
    ) -> Path:
        out = self._root
        out.mkdir(parents=True, exist_ok=True)
        candidates.to_parquet(out / "candidate_trades.parquet", index=False)
        summary.to_csv(out / "percentile_quality.csv", index=False)
        wf.to_csv(out / "wf_percentile_metrics.csv", index=False)
        stability.to_csv(out / "dataset_stability.csv", index=False)
        (out / "meta_dataset_validation_report.md").write_text(report_md, encoding="utf-8")
        logger.info("Saved meta dataset validation artifacts | dir=%s", out)
        return out
