"""RunContextImpactUseCase — measure H4 context contribution (no engine changes)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from research.analyzers.walk_forward_analyzer import WalkForwardWindow
from research.context_impact.entities.experiment_result import ExperimentResult
from research.context_impact.reports.context_impact_charts import ContextImpactChartBuilder
from research.context_impact.reports.context_impact_report import (
    ContextImpactReportBuilder,
    build_comparison,
)
from research.context_impact.repositories.context_impact_repository import ContextImpactRepository
from research.context_impact.services.experiment_catalog import resolve_experiments
from research.context_impact.services.panel_builder import ContextImpactPanelBuilder
from research.context_impact.services.walk_forward_runner import ContextImpactWalkForwardRunner

logger = logging.getLogger(__name__)


class RunContextImpactUseCase:
    def __init__(
        self,
        repository: ContextImpactRepository,
        *,
        config: dict[str, Any] | None = None,
    ) -> None:
        self._repo = repository
        self._cfg = dict(config or {})
        self._panel_builder = ContextImpactPanelBuilder()
        self._report_builder = ContextImpactReportBuilder()
        self._chart_builder = ContextImpactChartBuilder()

    def execute(
        self,
        *,
        symbol: str,
        timeframe: str,
        sides: list[str],
        feature_matrix_path: Path,
        context_path: Path,
        labels_by_side: dict[str, pd.DataFrame],
    ) -> tuple[list[ExperimentResult], Path]:
        windows = self._parse_windows(self._cfg.get("windows"))
        runner = ContextImpactWalkForwardRunner(
            windows=windows,
            trainer_params=dict(self._cfg.get("trainer_params") or {}),
            target_mode=str(self._cfg.get("target_mode", "exclude_timeout")),
            threshold=float(self._cfg.get("threshold", 0.5)),
            random_seed=int(self._cfg.get("random_seed", 42)),
            algorithm=str(self._cfg.get("algorithm", "lightgbm")),
        )

        results: list[ExperimentResult] = []
        for side in sides:
            labels = labels_by_side.get(side)
            if labels is None or labels.empty:
                logger.warning("Skipping side=%s — missing labels", side)
                continue
            panel, h1_features, _ = self._panel_builder.build(
                feature_matrix_path=feature_matrix_path,
                context_path=context_path,
                labels=labels,
                symbol=symbol,
                timeframe=timeframe,
                side=side,
            )
            experiments = resolve_experiments(h1_features)
            for exp in experiments:
                logger.info(
                    "Running context impact | side=%s exp=%s feats=%s",
                    side,
                    exp.experiment_id,
                    len(exp.feature_names),
                )
                results.append(runner.run(panel, exp, side=side))

        ablation = pd.DataFrame([r.summary_row() for r in results])
        comparison = build_comparison(results)
        windows_df = pd.DataFrame([w.to_dict() for r in results for w in r.windows])
        report_md = self._report_builder.build(
            results=results,
            comparison=comparison,
            ablation=ablation,
            symbol=symbol,
            timeframe=timeframe,
        )
        out = self._repo.save(
            comparison=comparison,
            ablation=ablation,
            windows=windows_df,
            report_md=report_md,
        )
        self._chart_builder.write_all(
            out_dir=out, ablation=ablation, comparison=comparison, results=results
        )
        return results, out

    @staticmethod
    def _parse_windows(raw: Any) -> list[WalkForwardWindow] | None:
        if not raw:
            return None
        return [
            WalkForwardWindow(
                int(w["train_start_year"]),
                int(w["train_end_year"]),
                int(w["valid_year"]),
            )
            for w in raw
        ]
