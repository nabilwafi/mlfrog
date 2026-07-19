"""Persist structure selection research outputs."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class StructureSelectionRepository:
    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    @property
    def root(self) -> Path:
        return self._root

    def save(
        self,
        *,
        experiment_results: pd.DataFrame,
        feature_stability: pd.DataFrame,
        regime_results: pd.DataFrame,
        window_metrics: pd.DataFrame,
        report_md: str,
    ) -> Path:
        out = self._root
        out.mkdir(parents=True, exist_ok=True)
        experiment_results.to_csv(out / "experiment_results.csv", index=False)
        feature_stability.to_csv(out / "feature_stability.csv", index=False)
        if not regime_results.empty:
            regime_results.to_csv(out / "regime_results.csv", index=False)
        window_metrics.to_csv(out / "window_metrics.csv", index=False)
        (out / "selection_report.md").write_text(report_md, encoding="utf-8")
        logger.info("Saved structure selection artifacts | dir=%s", out)
        return out
