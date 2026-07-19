"""Persist ablation CSVs, markdown, and charts."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from research.feature_ablation.entities.experiment_result import ExperimentResult

logger = logging.getLogger(__name__)


class AblationRepository:
    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    @property
    def root(self) -> Path:
        return self._root

    def save(
        self,
        *,
        summary: pd.DataFrame,
        metrics: pd.DataFrame,
        rankings: pd.DataFrame,
        report_md: str,
        charts: dict[str, Path] | None = None,
    ) -> Path:
        out = self._root
        out.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out / "feature_ablation_summary.csv", index=False)
        metrics.to_csv(out / "feature_ablation_metrics.csv", index=False)
        rankings.to_csv(out / "feature_ablation_rankings.csv", index=False)
        (out / "feature_ablation_report.md").write_text(report_md, encoding="utf-8")
        logger.info("Saved ablation artifacts | dir=%s", out)
        return out
