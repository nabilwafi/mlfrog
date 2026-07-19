"""Persist context impact research artifacts."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class ContextImpactRepository:
    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    @property
    def root(self) -> Path:
        return self._root

    def save(
        self,
        *,
        comparison: pd.DataFrame,
        ablation: pd.DataFrame,
        windows: pd.DataFrame,
        report_md: str,
    ) -> Path:
        out = self._root
        out.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(out / "comparison.csv", index=False)
        ablation.to_csv(out / "ablation_results.csv", index=False)
        windows.to_csv(out / "window_metrics.csv", index=False)
        (out / "context_impact_report.md").write_text(report_md, encoding="utf-8")
        logger.info("Saved context impact artifacts | dir=%s", out)
        return out
