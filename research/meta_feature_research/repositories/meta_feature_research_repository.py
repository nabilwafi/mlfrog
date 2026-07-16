"""Persist meta feature research artifacts."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class MetaFeatureResearchRepository:
    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    @property
    def root(self) -> Path:
        return self._root

    def save(
        self,
        *,
        panel: pd.DataFrame,
        univariate: pd.DataFrame,
        group_importance: pd.DataFrame,
        stability: pd.DataFrame,
        interactions: pd.DataFrame,
        session_table: pd.DataFrame,
        report_md: str,
    ) -> Path:
        out = self._root
        out.mkdir(parents=True, exist_ok=True)
        # Drop bulky metadata before parquet
        drop = [c for c in ("metadata_json",) if c in panel.columns]
        panel.drop(columns=drop, errors="ignore").to_parquet(out / "meta_feature_panel.parquet", index=False)
        univariate.to_csv(out / "univariate_scores.csv", index=False)
        group_importance.to_csv(out / "group_importance.csv", index=False)
        stability.to_csv(out / "feature_stability.csv", index=False)
        interactions.to_csv(out / "interactions.csv", index=False)
        session_table.to_csv(out / "session_analysis.csv", index=False)
        (out / "meta_feature_research_report.md").write_text(report_md, encoding="utf-8")
        logger.info("Saved meta feature research artifacts | dir=%s", out)
        return out
