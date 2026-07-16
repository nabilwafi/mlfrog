"""RunMetaDatasetValidationUseCase — research only, no meta training."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from research.meta_dataset_validation.reports import MetaDatasetChartBuilder, build_report
from research.meta_dataset_validation.repositories import MetaDatasetValidationRepository
from research.meta_dataset_validation.services import (
    answer_research,
    attach_meta_label,
    build_candidate_trades,
    build_percentile_summary,
    build_stability,
    build_wf_summary,
    enrich_predictions,
)

logger = logging.getLogger(__name__)


class RunMetaDatasetValidationUseCase:
    def __init__(
        self,
        repository: MetaDatasetValidationRepository,
        *,
        config: dict[str, Any] | None = None,
    ) -> None:
        self._repo = repository
        self._cfg = dict(config or {})
        self._charts = MetaDatasetChartBuilder()

    def execute(
        self,
        *,
        symbol: str,
        timeframe: str,
        predictions: pd.DataFrame,
        labels_by_side: dict[str, pd.DataFrame],
        candles: pd.DataFrame | None = None,
        sides: list[str] | None = None,
    ) -> tuple[pd.DataFrame, Path]:
        cost = float(self._cfg.get("transaction_cost", 0.00015))
        wanted = {s.lower() for s in (sides or ["long", "short"])}
        pred = predictions.copy()
        pred["timestamp"] = pd.to_datetime(pred["timestamp"], utc=True)
        if "experiment_id" in pred.columns:
            pred = pred.loc[pred["experiment_id"] == "B_v2_context"]
        if "split" in pred.columns:
            pred = pred.loc[pred["split"] == "validation"]
        pred = pred.loc[pred["side"].isin(wanted)].copy()
        if pred.empty:
            raise RuntimeError("no validation predictions for v2 models")

        # Prefer raw probs (Sprint 16 conclusion)
        if "y_prob_raw" not in pred.columns and "y_prob" in pred.columns:
            pred["y_prob_raw"] = pred["y_prob"]

        logger.info("Enriching predictions with holding/MAE/MFE | n=%s", len(pred))
        enriched = enrich_predictions(pred, labels_by_side, candles)

        candidates = build_candidate_trades(enriched, prob_col="y_prob_raw")
        candidates = attach_meta_label(candidates, cost=cost)
        if candidates.empty:
            raise RuntimeError("no candidate trades generated")

        summary = build_percentile_summary(candidates, cost=cost)
        wf = build_wf_summary(candidates, cost=cost)
        stability = build_stability(candidates)
        answers = answer_research(summary=summary, wf=wf, stability=stability)

        report = build_report(
            symbol=symbol,
            timeframe=timeframe,
            cost=cost,
            summary=summary,
            wf=wf,
            stability=stability,
            answers=answers,
        )

        out = self._repo.save(
            candidates=candidates,
            summary=summary,
            wf=wf,
            stability=stability,
            report_md=report,
        )

        # Chart focus: long recommended quality pct if available else 0.10
        focus = answers.get("long", {}).get("best_quality", {}).get("percentile", 0.10)
        self._charts.write_all(
            out_dir=out,
            candidates=candidates,
            summary=summary,
            wf=wf,
            focus_percentile=float(focus) if focus == focus else 0.10,
        )
        return candidates, out
