"""RunFeatureAblationUseCase — orchestrate all ablation experiments."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from research.analyzers.walk_forward_analyzer import WalkForwardWindow
from research.feature_ablation.entities.experiment_result import ExperimentResult
from research.feature_ablation.reports.ablation_charts import AblationChartBuilder
from research.feature_ablation.reports.ablation_report import AblationReportBuilder, build_rankings
from research.feature_ablation.repositories.ablation_repository import AblationRepository
from research.feature_ablation.services.experiment_catalog import (
    ExperimentResolver,
    default_experiment_catalog,
    load_diagnostics_lists,
)
from research.feature_ablation.services.panel_builder import AblationPanelBuilder
from research.feature_ablation.services.walk_forward_runner import AblationWalkForwardRunner

logger = logging.getLogger(__name__)


class RunFeatureAblationUseCase:
    def __init__(
        self,
        repository: AblationRepository,
        *,
        config: dict[str, Any] | None = None,
    ) -> None:
        self._repo = repository
        self._cfg = dict(config or {})
        self._panel_builder = AblationPanelBuilder()
        self._report_builder = AblationReportBuilder()
        self._chart_builder = AblationChartBuilder()

    def execute(
        self,
        *,
        symbol: str,
        timeframe: str,
        side: str,
        feature_matrix_path: Path,
        feature_metadata_path: Path,
        diagnostics_dir: Path,
        labels: pd.DataFrame,
    ) -> tuple[list[ExperimentResult], Path]:
        panel, all_features, category_by_feature = self._panel_builder.build(
            feature_matrix_path=feature_matrix_path,
            feature_metadata_path=feature_metadata_path,
            labels=labels,
            symbol=symbol,
            timeframe=timeframe,
            side=side,
        )
        diag = load_diagnostics_lists(diagnostics_dir)
        resolver = ExperimentResolver(
            all_features=all_features,
            category_by_feature=category_by_feature,
            keep_features=list(diag["keep"]),
            stable_features=list(diag["stable"]),
            shap_ranked=list(diag["shap_ranked"]),
            mi_ranked=list(diag["mi_ranked"]),
        )

        windows = self._parse_windows(self._cfg.get("windows"))
        runner = AblationWalkForwardRunner(
            windows=windows,
            trainer_params=dict(self._cfg.get("trainer_params") or {}),
            target_mode=str(self._cfg.get("target_mode", "exclude_timeout")),
            threshold=float(self._cfg.get("threshold", 0.5)),
            random_seed=int(self._cfg.get("random_seed", 42)),
            algorithm=str(self._cfg.get("algorithm", "lightgbm")),
        )

        catalog = default_experiment_catalog()
        results: list[ExperimentResult] = []
        for exp in catalog:
            resolved = resolver.resolve(exp)
            logger.info(
                "Running ablation experiment %s | n_features=%s",
                resolved.experiment_id,
                len(resolved.feature_names),
            )
            result = runner.run(
                panel,
                resolved,
                category_counts=resolver.category_counts(resolved.feature_names),
            )
            results.append(result)

        summary = pd.DataFrame([r.summary_row() for r in results])
        metrics = pd.DataFrame(
            [w.to_dict() for r in results for w in r.windows]
        )
        rankings = build_rankings(summary)
        report_md = self._report_builder.build(
            results=results,
            summary=summary,
            rankings=rankings,
            symbol=symbol,
            timeframe=timeframe,
            side=side,
        )
        out = self._repo.save(
            summary=summary,
            metrics=metrics,
            rankings=rankings,
            report_md=report_md,
        )
        self._chart_builder.write_all(out_dir=out, summary=summary, results=results)
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
