"""Persist meta feature ablation artifacts."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class MetaFeatureAblationRepository:
    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    @property
    def root(self) -> Path:
        return self._root

    def save(
        self,
        *,
        window_metrics: pd.DataFrame,
        stage_summary: pd.DataFrame,
        oof: pd.DataFrame,
        importance: pd.DataFrame,
        stability: pd.DataFrame,
        pearson: pd.DataFrame,
        spearman: pd.DataFrame,
        vif: pd.DataFrame,
        clusters: pd.DataFrame,
        removal: pd.DataFrame,
        interactions: pd.DataFrame,
        recommended: pd.DataFrame,
        report_md: str,
    ) -> Path:
        out = self._root
        out.mkdir(parents=True, exist_ok=True)
        window_metrics.to_csv(out / "window_metrics.csv", index=False)
        stage_summary.to_csv(out / "stage_summary.csv", index=False)
        if not oof.empty:
            oof.to_parquet(out / "oof_predictions.parquet", index=False)
        importance.to_csv(out / "feature_importance.csv", index=False)
        stability.to_csv(out / "importance_stability.csv", index=False)
        pearson.to_csv(out / "pearson_correlation.csv")
        spearman.to_csv(out / "spearman_correlation.csv")
        vif.to_csv(out / "vif_table.csv", index=False)
        clusters.to_csv(out / "correlation_clusters.csv", index=False)
        removal.to_csv(out / "removal_recommendations.csv", index=False)
        interactions.to_csv(out / "interactions.csv", index=False)
        recommended.to_csv(out / "recommended_features.csv", index=False)
        (out / "meta_feature_ablation_report.md").write_text(report_md, encoding="utf-8")
        logger.info("Saved meta feature ablation artifacts | dir=%s", out)
        return out
