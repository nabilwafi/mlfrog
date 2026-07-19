"""RunModelV2ValidationUseCase — Long/Short H1+context A/B validation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from research.analyzers.walk_forward_analyzer import WalkForwardWindow
from research.model_v2.reports.model_v2_charts import ModelV2ChartBuilder
from research.model_v2.reports.model_v2_report import (
    ModelV2ReportBuilder,
    build_comparison,
    build_feature_importance,
)
from research.model_v2.repositories.model_v2_repository import ModelV2Repository
from research.model_v2.services.experiment_catalog import SIDE_SPECS, resolve_side_experiments
from research.structure_selection.entities.selection_result import ExperimentResult
from research.structure_selection.services.panel_builder import SelectionPanelBuilder
from research.structure_selection.services.stability_analyzer import build_feature_stability
from research.structure_selection.services.walk_forward_runner import SelectionWalkForwardRunner

logger = logging.getLogger(__name__)


class RunModelV2ValidationUseCase:
    def __init__(
        self,
        repository: ModelV2Repository,
        *,
        config: dict[str, Any] | None = None,
    ) -> None:
        self._repo = repository
        self._cfg = dict(config or {})
        self._panel = SelectionPanelBuilder()
        self._report = ModelV2ReportBuilder()
        self._charts = ModelV2ChartBuilder()

    def execute(
        self,
        *,
        symbol: str,
        timeframe: str,
        feature_matrix_path: Path,
        structure_features_path: Path,
        labels_by_side: dict[str, pd.DataFrame],
        sides: list[str] | None = None,
    ) -> tuple[list[ExperimentResult], Path]:
        wanted = {s.lower() for s in (sides or ["long", "short"])}
        specs = [s for s in SIDE_SPECS if s.side in wanted]

        windows = self._parse_windows(self._cfg.get("windows"))
        runner = SelectionWalkForwardRunner(
            windows=windows,
            trainer_params=dict(self._cfg.get("trainer_params") or {}),
            target_mode=str(self._cfg.get("target_mode", "exclude_timeout")),
            threshold=float(self._cfg.get("threshold", 0.5)),
            random_seed=int(self._cfg.get("random_seed", 42)),
            algorithm=str(self._cfg.get("algorithm", "lightgbm")),
            compute_shap=bool(self._cfg.get("compute_shap", True)),
        )

        results: list[ExperimentResult] = []
        for spec in specs:
            labels = labels_by_side.get(spec.side)
            if labels is None or labels.empty:
                logger.warning("Skipping side=%s — missing labels", spec.side)
                continue
            panel, h1_features, structure_cols = self._panel.build(
                feature_matrix_path=feature_matrix_path,
                structure_features_path=structure_features_path,
                labels=labels,
                symbol=symbol,
                timeframe=timeframe,
                side=spec.side,
            )
            missing = [f for f in spec.context_features if f not in structure_cols]
            if missing:
                raise ValueError(f"structure features missing for {spec.side}: {missing}")

            for exp in resolve_side_experiments(
                h1_features,
                side=spec.side,
                context_features=spec.context_features,
            ):
                logger.info(
                    "Model v2 validation | side=%s exp=%s feats=%s",
                    spec.side,
                    exp.experiment_id,
                    len(exp.feature_names),
                )
                results.append(
                    runner.run(
                        panel,
                        exp,
                        side=spec.side,
                        structure_feature_names=list(spec.context_features),
                    )
                )

        comparison = build_comparison(results)
        feature_importance = build_feature_importance(results)
        feature_stability = build_feature_stability(results)
        window_metrics = pd.DataFrame([w.to_dict() for r in results for w in r.windows])

        reports: dict[str, str] = {}
        for spec in specs:
            reports[spec.side] = self._report.build_side_report(
                side=spec.side,
                results=results,
                comparison=comparison,
                feature_importance=feature_importance,
                feature_stability=feature_stability,
                context_features=spec.context_features,
                symbol=symbol,
                timeframe=timeframe,
            )

        out = self._repo.save(
            comparison=comparison,
            feature_importance=feature_importance,
            feature_stability=feature_stability,
            window_metrics=window_metrics,
            long_report_md=reports.get("long", "# Long Model v2\n\n_(not run)_\n"),
            short_report_md=reports.get("short", "# Short Model v2\n\n_(not run)_\n"),
        )
        self._charts.write_all(
            out_dir=out,
            results=results,
            comparison=comparison,
            feature_importance=feature_importance,
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
