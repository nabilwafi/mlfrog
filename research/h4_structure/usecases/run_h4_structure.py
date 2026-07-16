"""RunH4StructureUseCase — build structure features + ablation research."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from data.repositories.market_repository import MarketRepository
from market_context.builders.h4_structure_builder import ALL_STRUCTURE_FEATURES
from research.analyzers.walk_forward_analyzer import WalkForwardWindow
from research.h4_structure.entities.experiment_result import ExperimentResult
from research.h4_structure.reports.structure_charts import StructureChartBuilder
from research.h4_structure.reports.structure_report import (
    StructureReportBuilder,
    build_feature_statistics,
)
from research.h4_structure.repositories.h4_structure_repository import H4StructureRepository
from research.h4_structure.services.experiment_catalog import resolve_experiments
from research.h4_structure.services.panel_builder import StructurePanelBuilder
from research.h4_structure.services.walk_forward_runner import StructureWalkForwardRunner

logger = logging.getLogger(__name__)


class RunH4StructureUseCase:
    def __init__(
        self,
        repository: H4StructureRepository,
        market_repo: MarketRepository,
        *,
        config: dict[str, Any] | None = None,
    ) -> None:
        self._repo = repository
        self._market_repo = market_repo
        self._cfg = dict(config or {})
        self._panel = StructurePanelBuilder(dict(self._cfg.get("builder_params") or {}))
        self._report = StructureReportBuilder()
        self._charts = StructureChartBuilder()

    def execute(
        self,
        *,
        symbol: str,
        base_timeframe: str,
        context_timeframe: str,
        sides: list[str],
        feature_matrix_path: Path,
        labels_by_side: dict[str, pd.DataFrame],
        timezone: str = "UTC",
    ) -> tuple[list[ExperimentResult], Path]:
        structure_h4, specs = self._panel.build_structure_on_h4(
            market_repo=self._market_repo,
            symbol=symbol,
            context_timeframe=context_timeframe,
            timezone=timezone,
            drop_na=bool(self._cfg.get("drop_na", True)),
        )
        structure_cols = [s.name for s in specs]
        structure_on_h1 = self._panel.join_to_h1(
            market_repo=self._market_repo,
            structure_h4=structure_h4,
            structure_cols=structure_cols,
            symbol=symbol,
            base_timeframe=base_timeframe,
            context_timeframe=context_timeframe,
            timezone=timezone,
        )

        windows = self._parse_windows(self._cfg.get("windows"))
        runner = StructureWalkForwardRunner(
            windows=windows,
            trainer_params=dict(self._cfg.get("trainer_params") or {}),
            target_mode=str(self._cfg.get("target_mode", "exclude_timeout")),
            threshold=float(self._cfg.get("threshold", 0.5)),
            random_seed=int(self._cfg.get("random_seed", 42)),
            algorithm=str(self._cfg.get("algorithm", "lightgbm")),
        )

        results: list[ExperimentResult] = []
        importance_by_side: dict[str, dict[str, float]] = {}
        for side in sides:
            labels = labels_by_side.get(side)
            if labels is None or labels.empty:
                logger.warning("Skipping side=%s — missing labels", side)
                continue
            panel, h1_features = self._panel.build_ml_panel(
                feature_matrix_path=feature_matrix_path,
                structure_on_h1=structure_on_h1,
                structure_cols=structure_cols,
                labels=labels,
                symbol=symbol,
                timeframe=base_timeframe,
                side=side,
            )
            for exp in resolve_experiments(h1_features):
                logger.info(
                    "H4 structure ablation | side=%s exp=%s feats=%s",
                    side,
                    exp.experiment_id,
                    len(exp.feature_names),
                )
                result = runner.run(
                    panel,
                    exp,
                    side=side,
                    structure_feature_names=list(ALL_STRUCTURE_FEATURES),
                )
                results.append(result)
                if exp.experiment_id == "all_structure":
                    importance_by_side[side] = dict(result.mean_importance)

        ablation = pd.DataFrame([r.summary_row() for r in results])
        feature_statistics = build_feature_statistics(
            structure_on_h1, structure_cols, importance_by_side
        )
        report_md = self._report.build(
            results=results,
            ablation=ablation,
            feature_statistics=feature_statistics,
            symbol=symbol,
            timeframe=base_timeframe,
        )
        out = self._repo.save(
            structure_features=structure_on_h1,
            specs=specs,
            ablation=ablation,
            feature_statistics=feature_statistics,
            report_md=report_md,
        )
        self._charts.write_all(
            out_dir=out,
            ablation=ablation,
            feature_statistics=feature_statistics,
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
