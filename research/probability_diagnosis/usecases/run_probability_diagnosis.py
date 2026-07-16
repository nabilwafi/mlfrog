"""RunProbabilityDiagnosisUseCase — Sprint 15."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from research.analyzers.walk_forward_analyzer import WalkForwardWindow
from research.model_v2.services.experiment_catalog import SIDE_SPECS, resolve_side_experiments
from research.probability_diagnosis.reports import DiagnosisChartBuilder, build_report
from research.probability_diagnosis.repositories import ProbabilityDiagnosisRepository
from research.probability_diagnosis.services import (
    DiagnosisWalkForwardRunner,
    build_baseline_vs_v2,
    build_label_distribution,
    build_prediction_distribution,
    build_wf_drift,
    diagnose_answers,
)
from research.structure_selection.services.panel_builder import SelectionPanelBuilder

logger = logging.getLogger(__name__)


class RunProbabilityDiagnosisUseCase:
    def __init__(
        self,
        repository: ProbabilityDiagnosisRepository,
        *,
        config: dict[str, Any] | None = None,
    ) -> None:
        self._repo = repository
        self._cfg = dict(config or {})
        self._panel = SelectionPanelBuilder()
        self._charts = DiagnosisChartBuilder()

    def execute(
        self,
        *,
        symbol: str,
        timeframe: str,
        feature_matrix_path: Path,
        structure_features_path: Path,
        labels_by_side: dict[str, pd.DataFrame],
        sides: list[str] | None = None,
    ) -> tuple[pd.DataFrame, Path]:
        wanted = {s.lower() for s in (sides or ["long", "short"])}
        specs = [s for s in SIDE_SPECS if s.side in wanted]
        trainer_params = dict(self._cfg.get("trainer_params") or {})

        windows = self._parse_windows(self._cfg.get("windows"))
        runner = DiagnosisWalkForwardRunner(
            windows=windows,
            trainer_params=trainer_params,
            target_mode=str(self._cfg.get("target_mode", "exclude_timeout")),
            random_seed=int(self._cfg.get("random_seed", 42)),
            algorithm=str(self._cfg.get("algorithm", "lightgbm")),
        )

        parts: list[pd.DataFrame] = []
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
                    "Probability diagnosis | side=%s exp=%s feats=%s",
                    spec.side,
                    exp.experiment_id,
                    len(exp.feature_names),
                )
                parts.append(runner.collect(panel, exp, side=spec.side))

        predictions = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        labels_df = build_label_distribution(predictions) if not predictions.empty else pd.DataFrame()
        pred_dist = (
            build_prediction_distribution(predictions) if not predictions.empty else pd.DataFrame()
        )
        wf_drift = build_wf_drift(predictions) if not predictions.empty else pd.DataFrame()
        compare = build_baseline_vs_v2(pred_dist) if not pred_dist.empty else pd.DataFrame()
        answers = diagnose_answers(
            labels=labels_df,
            pred_dist=pred_dist,
            compare=compare,
            trainer_params=trainer_params,
        )
        report = build_report(
            symbol=symbol,
            timeframe=timeframe,
            labels=labels_df,
            pred_dist=pred_dist,
            wf_drift=wf_drift,
            compare=compare,
            answers=answers,
        )

        out = self._repo.save(
            predictions=predictions,
            labels=labels_df,
            pred_dist=pred_dist,
            wf_drift=wf_drift,
            compare=compare,
            report_md=report,
        )
        self._charts.write_all(
            out_dir=out,
            predictions=predictions,
            wf_drift=wf_drift,
            compare=compare,
        )
        return predictions, out

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
