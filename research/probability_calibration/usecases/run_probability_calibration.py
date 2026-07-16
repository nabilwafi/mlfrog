"""RunProbabilityCalibrationUseCase — Sprint 16."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from research.analyzers.walk_forward_analyzer import WalkForwardWindow
from research.model_v2.services.experiment_catalog import SIDE_SPECS, resolve_side_experiments
from research.probability_calibration.reports import (
    CalibrationChartBuilder,
    build_answers,
    build_report,
)
from research.probability_calibration.repositories import ProbabilityCalibrationRepository
from research.probability_calibration.services import (
    CONTEXT_FILTER_FEATURES,
    analyze_context_filters,
    analyze_percentile_thresholds,
    build_calibration_comparison,
    calibrate_walk_forward,
    pick_best_method,
    pick_optimal_threshold,
)
from research.probability_diagnosis.services.diagnosis_runner import DiagnosisWalkForwardRunner
from research.structure_selection.services.panel_builder import SelectionPanelBuilder

logger = logging.getLogger(__name__)


class RunProbabilityCalibrationUseCase:
    def __init__(
        self,
        repository: ProbabilityCalibrationRepository,
        *,
        config: dict[str, Any] | None = None,
    ) -> None:
        self._repo = repository
        self._cfg = dict(config or {})
        self._panel = SelectionPanelBuilder()
        self._charts = CalibrationChartBuilder()

    def execute(
        self,
        *,
        symbol: str,
        timeframe: str,
        feature_matrix_path: Path,
        structure_features_path: Path,
        labels_by_side: dict[str, pd.DataFrame],
        sides: list[str] | None = None,
        diagnosis_predictions_path: Path | None = None,
    ) -> tuple[pd.DataFrame, Path]:
        cost = float(self._cfg.get("transaction_cost", 0.00015))
        wanted = {s.lower() for s in (sides or ["long", "short"])}

        predictions = self._load_or_collect(
            symbol=symbol,
            timeframe=timeframe,
            feature_matrix_path=feature_matrix_path,
            structure_features_path=structure_features_path,
            labels_by_side=labels_by_side,
            sides=sorted(wanted),
            diagnosis_predictions_path=diagnosis_predictions_path,
        )
        predictions = predictions.loc[
            (predictions["experiment_id"] == "B_v2_context")
            & (predictions["side"].isin(wanted))
        ].copy()

        predictions = self._attach_returns(predictions, labels_by_side)
        predictions = self._attach_structure(predictions, structure_features_path)

        calibrated = calibrate_walk_forward(predictions)
        if calibrated.empty:
            raise RuntimeError("no calibrated validation rows — check diagnosis predictions")

        comparison = build_calibration_comparison(calibrated)
        best_methods = {
            side: pick_best_method(comparison, side)
            for side in sorted(calibrated["side"].unique())
        }

        # Thresholds on raw + each calibrator
        thr_parts = []
        for col in ("y_prob_raw", "y_prob_platt", "y_prob_isotonic"):
            thr_parts.append(
                analyze_percentile_thresholds(calibrated, prob_col=col, cost=cost)
            )
        thresholds = pd.concat(thr_parts, ignore_index=True) if thr_parts else pd.DataFrame()

        # Context filters use best method per side (apply column selection)
        ctx_parts = []
        for side, method in best_methods.items():
            col = {
                "raw": "y_prob_raw",
                "platt": "y_prob_platt",
                "isotonic": "y_prob_isotonic",
            }.get(method, "y_prob_raw")
            side_df = calibrated.loc[calibrated["side"] == side]
            ctx_parts.append(
                analyze_context_filters(side_df, prob_col=col, cost=cost)
            )
        context = pd.concat(ctx_parts, ignore_index=True) if ctx_parts else pd.DataFrame()

        # Optimal from best-method thresholds only
        optimal: dict[str, dict] = {}
        for side, method in best_methods.items():
            col = {
                "raw": "y_prob_raw",
                "platt": "y_prob_platt",
                "isotonic": "y_prob_isotonic",
            }.get(method, "y_prob_raw")
            sub = thresholds.loc[
                (thresholds["side"] == side) & (thresholds["prob_col"] == col)
            ]
            optimal[side] = pick_optimal_threshold(sub, side)

        answers = build_answers(
            calibration=comparison,
            thresholds=thresholds,
            context=context,
            optimal=optimal,
            best_methods=best_methods,
        )
        report = build_report(
            symbol=symbol,
            timeframe=timeframe,
            cost=cost,
            calibration=comparison,
            thresholds=thresholds,
            context=context,
            optimal=optimal,
            answers=answers,
        )

        out = self._repo.save(
            calibrated=calibrated,
            calibration_comparison=comparison,
            thresholds=thresholds,
            context=context,
            report_md=report,
        )
        self._charts.write_all(
            out_dir=out,
            calibrated=calibrated,
            thresholds=thresholds,
            context=context,
        )
        return calibrated, out

    def _load_or_collect(
        self,
        *,
        symbol: str,
        timeframe: str,
        feature_matrix_path: Path,
        structure_features_path: Path,
        labels_by_side: dict[str, pd.DataFrame],
        sides: list[str],
        diagnosis_predictions_path: Path | None,
    ) -> pd.DataFrame:
        if diagnosis_predictions_path is not None and diagnosis_predictions_path.is_file():
            logger.info("Loading diagnosis predictions | %s", diagnosis_predictions_path)
            preds = pd.read_parquet(diagnosis_predictions_path)
            preds["timestamp"] = pd.to_datetime(preds["timestamp"], utc=True)
            return preds

        # Fallback: re-collect B_v2 train+val
        logger.info("Diagnosis predictions missing — collecting via DiagnosisWalkForwardRunner")
        windows = self._parse_windows(self._cfg.get("windows"))
        runner = DiagnosisWalkForwardRunner(
            windows=windows,
            trainer_params=dict(self._cfg.get("trainer_params") or {}),
            target_mode=str(self._cfg.get("target_mode", "exclude_timeout")),
            random_seed=int(self._cfg.get("random_seed", 42)),
            algorithm=str(self._cfg.get("algorithm", "lightgbm")),
        )
        specs = [s for s in SIDE_SPECS if s.side in set(sides)]
        parts = []
        for spec in specs:
            labels = labels_by_side.get(spec.side)
            if labels is None or labels.empty:
                continue
            panel, h1, structure_cols = self._panel.build(
                feature_matrix_path=feature_matrix_path,
                structure_features_path=structure_features_path,
                labels=labels,
                symbol=symbol,
                timeframe=timeframe,
                side=spec.side,
            )
            missing = [f for f in spec.context_features if f not in structure_cols]
            if missing:
                raise ValueError(f"missing structure features: {missing}")
            exps = resolve_side_experiments(
                h1, side=spec.side, context_features=spec.context_features
            )
            v2 = next(e for e in exps if e.experiment_id == "B_v2_context")
            parts.append(runner.collect(panel, v2, side=spec.side))
        if not parts:
            return pd.DataFrame()
        return pd.concat(parts, ignore_index=True)

    @staticmethod
    def _attach_returns(
        predictions: pd.DataFrame, labels_by_side: dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        if "realized_return" in predictions.columns and predictions["realized_return"].notna().any():
            return predictions
        parts = []
        for side, g in predictions.groupby("side"):
            lab = labels_by_side.get(side)
            if lab is None or lab.empty or "realized_return" not in lab.columns:
                gg = g.copy()
                gg["realized_return"] = float("nan")
                parts.append(gg)
                continue
            lab = lab.copy()
            lab["timestamp"] = pd.to_datetime(lab["timestamp"], utc=True)
            extra = lab[["timestamp", "realized_return"]].drop_duplicates("timestamp")
            parts.append(g.merge(extra, on="timestamp", how="left"))
        return pd.concat(parts, ignore_index=True)

    @staticmethod
    def _attach_structure(predictions: pd.DataFrame, structure_path: Path) -> pd.DataFrame:
        if not structure_path.is_file():
            return predictions
        struct = pd.read_parquet(structure_path)
        struct["timestamp"] = pd.to_datetime(struct["timestamp"], utc=True)
        cols = ["timestamp", *[c for c in CONTEXT_FILTER_FEATURES if c in struct.columns]]
        if len(cols) == 1:
            return predictions
        return predictions.merge(struct[cols].drop_duplicates("timestamp"), on="timestamp", how="left")

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
