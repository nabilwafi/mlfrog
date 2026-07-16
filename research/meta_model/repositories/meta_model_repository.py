"""Persist production meta model artifacts."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import lightgbm as lgb
import pandas as pd

logger = logging.getLogger(__name__)


class MetaModelRepository:
    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    @property
    def root(self) -> Path:
        return self._root

    def save(
        self,
        *,
        oof: pd.DataFrame,
        by_year: pd.DataFrame,
        summary: pd.DataFrame,
        equity: pd.DataFrame,
        answers: dict[str, Any],
        report_md: str,
        loo_models: dict[int, lgb.Booster],
        full_model: lgb.Booster,
        feature_medians: pd.Series,
        features: list[str],
    ) -> Path:
        out = self._root
        out.mkdir(parents=True, exist_ok=True)
        models = out / "models"
        models.mkdir(parents=True, exist_ok=True)

        oof.to_parquet(out / "oof_predictions.parquet", index=False)
        by_year.to_csv(out / "wf_metrics.csv", index=False)
        summary.to_csv(out / "summary_metrics.csv", index=False)
        equity.to_csv(out / "equity_curve.csv", index=False)
        (out / "answers.json").write_text(json.dumps(answers, indent=2, default=str), encoding="utf-8")
        (out / "meta_model_report.md").write_text(report_md, encoding="utf-8")
        pd.DataFrame({"feature": features}).to_csv(out / "features.csv", index=False)
        feature_medians.to_csv(out / "feature_medians.csv", header=["median"])

        for year, booster in loo_models.items():
            booster.save_model(str(models / f"loo_val_{year}.txt"))
        full_model.save_model(str(models / "meta_full.txt"))

        logger.info("Saved production meta model artifacts | dir=%s", out)
        return out
