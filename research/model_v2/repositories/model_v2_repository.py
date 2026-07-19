"""Persist model v2 validation artifacts under artifacts/models/v2/."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class ModelV2Repository:
    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    @property
    def root(self) -> Path:
        return self._root

    def save(
        self,
        *,
        comparison: pd.DataFrame,
        feature_importance: pd.DataFrame,
        feature_stability: pd.DataFrame,
        window_metrics: pd.DataFrame,
        long_report_md: str,
        short_report_md: str,
    ) -> Path:
        out = self._root
        out.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(out / "comparison.csv", index=False)
        feature_importance.to_csv(out / "feature_importance.csv", index=False)
        feature_stability.to_csv(out / "feature_stability.csv", index=False)
        window_metrics.to_csv(out / "window_metrics.csv", index=False)
        (out / "long_model_v2_report.md").write_text(long_report_md, encoding="utf-8")
        (out / "short_model_v2_report.md").write_text(short_report_md, encoding="utf-8")
        logger.info("Saved model v2 validation artifacts | dir=%s", out)
        return out
