"""Persist confidence-layer artifacts."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class ConfidenceLayerRepository:
    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    @property
    def root(self) -> Path:
        return self._root

    def save(
        self,
        *,
        panel: pd.DataFrame,
        weights: pd.DataFrame,
        buckets: pd.DataFrame,
        regimes: pd.DataFrame,
        risk_search: pd.DataFrame,
        tp_research: pd.DataFrame,
        trail_research: pd.DataFrame,
        calibration_bins: pd.DataFrame,
        calibration_summary: dict[str, Any],
        shap_importance: pd.DataFrame,
        shap_interactions: pd.DataFrame,
        m5_research: dict[str, Any],
        answers: dict[str, Any],
        report_md: str,
    ) -> Path:
        out = self._root
        out.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(out / "confidence_panel.parquet", index=False)
        weights.to_csv(out / "confidence_weights.csv", index=False)
        buckets.to_csv(out / "confidence_buckets.csv", index=False)
        regimes.to_csv(out / "regime_performance.csv", index=False)
        risk_search.to_csv(out / "dynamic_risk_search.csv", index=False)
        tp_research.to_csv(out / "dynamic_tp_research.csv", index=False)
        trail_research.to_csv(out / "dynamic_trail_research.csv", index=False)
        calibration_bins.to_csv(out / "calibration_bins.csv", index=False)
        shap_importance.to_csv(out / "shap_importance.csv", index=False)
        shap_interactions.to_csv(out / "shap_interactions.csv", index=False)
        (out / "calibration_summary.json").write_text(
            json.dumps(calibration_summary, indent=2), encoding="utf-8"
        )
        (out / "m5_research.json").write_text(json.dumps(m5_research, indent=2), encoding="utf-8")
        (out / "answers.json").write_text(json.dumps(answers, indent=2, default=str), encoding="utf-8")
        (out / "confidence_layer_report.md").write_text(report_md, encoding="utf-8")
        logger.info("Saved confidence layer artifacts | dir=%s", out)
        return out
