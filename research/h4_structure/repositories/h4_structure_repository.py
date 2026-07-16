"""Persist H4 structure research artifacts."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from market_context.entities.context_feature import ContextFeatureSpec

logger = logging.getLogger(__name__)


class H4StructureRepository:
    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    @property
    def root(self) -> Path:
        return self._root

    def save(
        self,
        *,
        structure_features: pd.DataFrame,
        specs: list[ContextFeatureSpec],
        ablation: pd.DataFrame,
        feature_statistics: pd.DataFrame,
        report_md: str,
    ) -> Path:
        out = self._root
        out.mkdir(parents=True, exist_ok=True)
        structure_features.to_parquet(out / "structure_features.parquet", index=False)
        ablation.to_csv(out / "ablation_results.csv", index=False)
        feature_statistics.to_csv(out / "feature_statistics.csv", index=False)
        pd.DataFrame([s.to_dict() for s in specs]).to_csv(
            out / "structure_feature_defs.csv", index=False
        )
        (out / "structure_report.md").write_text(report_md, encoding="utf-8")
        logger.info("Saved H4 structure artifacts | dir=%s", out)
        return out
