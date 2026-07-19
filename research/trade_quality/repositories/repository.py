"""Persist trade quality artifacts."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class TradeQualityRepository:
    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    def save(
        self,
        *,
        panel: pd.DataFrame,
        stability: pd.DataFrame,
        tq_buckets: pd.DataFrame,
        conf_buckets: pd.DataFrame,
        risk_search: pd.DataFrame,
        interactions: pd.DataFrame,
        capital: pd.DataFrame,
        yearly: pd.DataFrame,
        shap_df: pd.DataFrame,
        perm_df: pd.DataFrame,
        answers: dict[str, Any],
        report_md: str,
    ) -> Path:
        out = self._root
        out.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(out / "trade_quality_panel.parquet", index=False)
        stability.to_csv(out / "method_stability.csv", index=False)
        tq_buckets.to_csv(out / "tq_buckets.csv", index=False)
        conf_buckets.to_csv(out / "confidence_buckets.csv", index=False)
        risk_search.to_csv(out / "dynamic_risk_search.csv", index=False)
        interactions.to_csv(out / "interactions.csv", index=False)
        capital.to_csv(out / "capital_allocation.csv", index=False)
        yearly.to_csv(out / "yearly_stability.csv", index=False)
        shap_df.to_csv(out / "shap_importance.csv", index=False)
        perm_df.to_csv(out / "permutation_importance.csv", index=False)
        (out / "answers.json").write_text(json.dumps(answers, indent=2, default=str), encoding="utf-8")
        (out / "trade_quality_report.md").write_text(report_md, encoding="utf-8")
        logger.info("Saved trade quality artifacts | dir=%s", out)
        return out
