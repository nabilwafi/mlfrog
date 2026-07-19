"""RunStructureSelectionUseCase — select robust H4 structure features."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from research.analyzers.walk_forward_analyzer import WalkForwardWindow
from research.structure_selection.entities.selection_result import ExperimentResult
from research.structure_selection.reports.selection_charts import SelectionChartBuilder
from research.structure_selection.reports.selection_report import SelectionReportBuilder
from research.structure_selection.repositories.structure_selection_repository import (
    StructureSelectionRepository,
)
from research.structure_selection.services.experiment_catalog import (
    load_top_structure_features,
    resolve_experiments,
)
from research.structure_selection.services.panel_builder import SelectionPanelBuilder
from research.structure_selection.services.regime_analyzer import RegimeAnalyzer
from research.structure_selection.services.stability_analyzer import build_feature_stability
from research.structure_selection.services.walk_forward_runner import SelectionWalkForwardRunner

logger = logging.getLogger(__name__)


class RunStructureSelectionUseCase:
    def __init__(
        self,
        repository: StructureSelectionRepository,
        *,
        config: dict[str, Any] | None = None,
    ) -> None:
        self._repo = repository
        self._cfg = dict(config or {})
        self._panel = SelectionPanelBuilder()
        self._report = SelectionReportBuilder()
        self._charts = SelectionChartBuilder()

    def execute(
        self,
        *,
        symbol: str,
        timeframe: str,
        sides: list[str],
        feature_matrix_path: Path,
        structure_features_path: Path,
        labels_by_side: dict[str, pd.DataFrame],
        feature_statistics_path: Path | None = None,
        regime_context_path: Path | None = None,
    ) -> tuple[list[ExperimentResult], Path]:
        top_k = int(self._cfg.get("top_k", 5))
        top_structure = load_top_structure_features(feature_statistics_path, top_k=top_k)
        if self._cfg.get("top_structure_features"):
            top_structure = tuple(self._cfg["top_structure_features"])

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
        regime_analyzer = RegimeAnalyzer(
            windows=windows or runner.windows,
            trainer_params=dict(self._cfg.get("trainer_params") or {}),
            target_mode=str(self._cfg.get("target_mode", "exclude_timeout")),
            random_seed=int(self._cfg.get("random_seed", 42)),
            algorithm=str(self._cfg.get("algorithm", "lightgbm")),
        )

        results: list[ExperimentResult] = []
        regime_frames: list[pd.DataFrame] = []
        structure_cols: list[str] = []

        for side in sides:
            labels = labels_by_side.get(side)
            if labels is None or labels.empty:
                logger.warning("Skipping side=%s — missing labels", side)
                continue
            panel, h1_features, structure_cols = self._panel.build(
                feature_matrix_path=feature_matrix_path,
                structure_features_path=structure_features_path,
                labels=labels,
                symbol=symbol,
                timeframe=timeframe,
                side=side,
                regime_context_path=regime_context_path,
            )
            experiments = resolve_experiments(h1_features, top_structure=top_structure)
            for exp in experiments:
                logger.info(
                    "Structure selection | side=%s exp=%s feats=%s",
                    side,
                    exp.experiment_id,
                    len(exp.feature_names),
                )
                results.append(
                    runner.run(
                        panel,
                        exp,
                        side=side,
                        structure_feature_names=structure_cols,
                    )
                )
            # Regime analysis on D (minimal informative pair)
            d_exp = next(e for e in experiments if e.experiment_id == "D_swing_plus_eq")
            regime_frames.append(regime_analyzer.analyze(panel, d_exp, side=side))

        experiment_results = pd.DataFrame([r.summary_row() for r in results])
        feature_stability = build_feature_stability(results)
        regime_results = (
            pd.concat(regime_frames, ignore_index=True)
            if any(not f.empty for f in regime_frames)
            else pd.DataFrame()
        )
        window_metrics = pd.DataFrame([w.to_dict() for r in results for w in r.windows])
        report_md = self._report.build(
            results=results,
            experiment_results=experiment_results,
            feature_stability=feature_stability,
            regime_results=regime_results,
            top_structure=top_structure,
            symbol=symbol,
            timeframe=timeframe,
        )
        out = self._repo.save(
            experiment_results=experiment_results,
            feature_stability=feature_stability,
            regime_results=regime_results,
            window_metrics=window_metrics,
            report_md=report_md,
        )
        self._charts.write_all(
            out_dir=out,
            experiment_results=experiment_results,
            feature_stability=feature_stability,
            regime_results=regime_results,
            results=results,
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
