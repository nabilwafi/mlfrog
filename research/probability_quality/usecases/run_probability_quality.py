"""RunProbabilityQualityUseCase — Sprint 14."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from research.analyzers.walk_forward_analyzer import WalkForwardWindow
from research.model_v2.services.experiment_catalog import SIDE_SPECS, resolve_side_experiments
from research.probability_quality.reports import (
    ProbabilityQualityChartBuilder,
    build_report,
    compare_sides,
)
from research.probability_quality.repositories import ProbabilityQualityRepository
from research.probability_quality.services import (
    PredictionWalkForwardRunner,
    build_bucket_analysis,
    build_confidence_analysis,
    build_decile_analysis,
    build_relative_confidence,
    distribution_summary,
    monotonicity_metrics,
)
from research.structure_selection.services.panel_builder import SelectionPanelBuilder

logger = logging.getLogger(__name__)


class RunProbabilityQualityUseCase:
    def __init__(
        self,
        repository: ProbabilityQualityRepository,
        *,
        config: dict[str, Any] | None = None,
    ) -> None:
        self._repo = repository
        self._cfg = dict(config or {})
        self._panel = SelectionPanelBuilder()
        self._charts = ProbabilityQualityChartBuilder()

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

        windows = self._parse_windows(self._cfg.get("windows"))
        runner = PredictionWalkForwardRunner(
            windows=windows,
            trainer_params=dict(self._cfg.get("trainer_params") or {}),
            target_mode=str(self._cfg.get("target_mode", "exclude_timeout")),
            random_seed=int(self._cfg.get("random_seed", 42)),
            algorithm=str(self._cfg.get("algorithm", "lightgbm")),
        )

        pred_parts: list[pd.DataFrame] = []
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

            panel = self._attach_returns(panel, labels)

            # v2 context experiment only (B)
            exps = resolve_side_experiments(
                h1_features,
                side=spec.side,
                context_features=spec.context_features,
            )
            v2 = next(e for e in exps if e.experiment_id == "B_v2_context")
            logger.info(
                "Probability quality | side=%s feats=%s",
                spec.side,
                len(v2.feature_names),
            )
            pred_parts.append(runner.collect(panel, v2, side=spec.side))

        predictions = (
            pd.concat(pred_parts, ignore_index=True)
            if pred_parts
            else pd.DataFrame()
        )
        buckets = build_bucket_analysis(predictions) if not predictions.empty else pd.DataFrame()
        deciles = build_decile_analysis(predictions) if not predictions.empty else pd.DataFrame()
        confidence = (
            build_confidence_analysis(predictions) if not predictions.empty else pd.DataFrame()
        )
        relative_confidence = (
            build_relative_confidence(predictions) if not predictions.empty else pd.DataFrame()
        )

        dist_map = {
            side: distribution_summary(predictions, side)
            for side in sorted(predictions["side"].unique())
        } if not predictions.empty else {}
        dist_df = pd.DataFrame(
            [
                {
                    "side": d.side,
                    "n": d.n,
                    "mean": d.mean,
                    "std": d.std,
                    "p10": d.p10,
                    "p25": d.p25,
                    "p50": d.p50,
                    "p75": d.p75,
                    "p90": d.p90,
                    "p95": d.p95,
                    "p99": d.p99,
                }
                for d in dist_map.values()
            ]
        )

        mono = {
            side: monotonicity_metrics(buckets, side)
            for side in sorted(buckets["side"].unique())
        } if not buckets.empty else {}
        mono_decile = {
            side: monotonicity_metrics(deciles, side)
            for side in sorted(deciles["side"].unique())
        } if not deciles.empty else {}
        mono_df = pd.DataFrame(
            [
                {**mono.get(s, {}), "scheme": "absolute_bucket"}
                for s in mono
            ]
            + [
                {**mono_decile.get(s, {}), "scheme": "decile"}
                for s in mono_decile
            ]
        )

        side_compare = compare_sides(
            buckets=buckets,
            deciles=deciles,
            confidence=confidence,
            relative_confidence=relative_confidence,
            mono=mono,
            mono_decile=mono_decile,
            dist=dist_map,
        )
        report = build_report(
            symbol=symbol,
            timeframe=timeframe,
            dist=dist_map,
            buckets=buckets,
            deciles=deciles,
            confidence=confidence,
            relative_confidence=relative_confidence,
            mono=mono,
            mono_decile=mono_decile,
            side_compare=side_compare,
        )

        out = self._repo.save(
            predictions=predictions,
            buckets=buckets,
            deciles=deciles,
            confidence=confidence,
            relative_confidence=relative_confidence,
            distribution=dist_df,
            monotonicity=mono_df,
            report_md=report,
        )
        self._charts.write_all(
            out_dir=out,
            predictions=predictions,
            buckets=buckets,
            deciles=deciles,
            confidence=relative_confidence if not relative_confidence.empty else confidence,
        )
        return predictions, out

    @staticmethod
    def _attach_returns(panel: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
        lab = labels.copy()
        lab["timestamp"] = pd.to_datetime(lab["timestamp"], utc=True)
        cols = ["timestamp"]
        if "realized_return" in lab.columns:
            cols.append("realized_return")
        else:
            lab["realized_return"] = float("nan")
            cols.append("realized_return")
        if "exit_reason" in lab.columns:
            cols.append("exit_reason")
        extra = lab[cols].drop_duplicates(subset=["timestamp"])
        out = panel.merge(extra, on="timestamp", how="left")
        return out

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
