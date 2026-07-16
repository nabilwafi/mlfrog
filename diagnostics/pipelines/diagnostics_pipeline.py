"""Orchestrate analyzers -> warnings -> markdown/CSV artifacts."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from datasets.entities.dataset import Dataset
from diagnostics.analyzers.dataset_analyzer import DatasetAnalyzer
from diagnostics.analyzers.drift_analyzer import DriftAnalyzer
from diagnostics.analyzers.feature_analyzer import FeatureAnalyzer
from diagnostics.analyzers.label_analyzer import LabelAnalyzer
from diagnostics.analyzers.model_analyzer import ModelAnalyzer
from diagnostics.analyzers.prediction_analyzer import PredictionAnalyzer
from diagnostics.analyzers.probability_analyzer import ProbabilityAnalyzer
from diagnostics.analyzers._common import map_binary_target
from diagnostics.entities.diagnostics_report import DiagnosticsReport
from diagnostics.repositories.diagnostics_repository import DiagnosticsRepository
from models.entities.model import Model
from models.trainers.base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if value != value:
            return "n/a"
        return f"{value:.{digits}f}"
    return str(value)


class DiagnosticsPipeline:
    def __init__(self, repository: DiagnosticsRepository) -> None:
        self._repo = repository
        self._dataset = DatasetAnalyzer()
        self._features = FeatureAnalyzer()
        self._labels = LabelAnalyzer()
        self._model = ModelAnalyzer()
        self._predictions = PredictionAnalyzer()
        self._probabilities = ProbabilityAnalyzer()
        self._drift = DriftAnalyzer()

    def run(
        self,
        *,
        splits: dict[str, Dataset],
        model: Model,
        trainer: BaseTrainer,
        model_path: Path | None = None,
        label_frame: pd.DataFrame | None = None,
        threshold: float = 0.5,
        target_mode: str | None = None,
    ) -> tuple[DiagnosticsReport, Path]:
        if "train" not in splits or "validation" not in splits:
            raise ValueError("splits must include train and validation")

        train = splits["train"]
        validation = splits["validation"]
        mode = target_mode or str(model.metadata.get("target_mode") or "exclude_timeout")

        x_train, y_train, _ = map_binary_target(train, target_mode=mode)
        x_val, y_val, ts_val = map_binary_target(validation, target_mode=mode)
        p_val = np.asarray(trainer.predict_proba(model, x_val), dtype=float)
        p_train = np.asarray(trainer.predict_proba(model, x_train), dtype=float)

        dataset_summary = self._dataset.analyze(splits)
        feature_summary, feature_stats = self._features.analyze(train)
        label_summary = self._labels.analyze(splits, label_frame=label_frame)
        model_summary, importance_df, learning_df = self._model.analyze(
            model, model_path=model_path
        )
        pred_summary, pred_dist = self._predictions.analyze(
            y_val, p_val, timestamps=ts_val, threshold=threshold
        )
        prob_summary, prob_hist = self._probabilities.analyze(y_val, p_val)
        thr_summary, thr_frame = self._probabilities.threshold_sweep(y_val, p_val)
        drift_summary, drift_df = self._drift.analyze(
            train,
            validation,
            y_train=y_train,
            y_val=y_val,
            p_train=p_train,
            p_val=p_val,
            target_mode=mode,
        )

        report = DiagnosticsReport(
            symbol=model.symbol,
            timeframe=model.timeframe,
            side=model.side,
            algorithm=model.algorithm,
            created_at=datetime.now(tz=timezone.utc),
            dataset=dataset_summary,
            features=feature_summary,
            labels=label_summary,
            model=model_summary,
            predictions=pred_summary,
            probabilities={**prob_summary, "threshold_sweep": thr_summary},
            thresholds=thr_summary,
            drift=drift_summary,
            metadata={
                "target_mode": mode,
                "feature_version": model.feature_version,
                "label_version": model.label_version,
                "dataset_version": model.dataset_version,
                "threshold": threshold,
            },
        )
        self._emit_warnings(report)

        tables = {
            "probability_distribution.csv": prob_hist,
            "threshold_analysis.csv": thr_frame,
            "feature_statistics.csv": feature_stats,
            "feature_importance.csv": importance_df,
            "prediction_distribution.csv": pred_dist,
            "learning_curve.csv": learning_df,
            "drift_report.csv": drift_df,
        }
        markdown = self._build_markdown(report)
        out = self._repo.save(report, markdown=markdown, tables=tables)
        logger.info(
            "Diagnostics complete | warnings=%s path=%s",
            len(report.warnings),
            out,
        )
        return report, out

    def _emit_warnings(self, report: DiagnosticsReport) -> None:
        best_iter = int(report.model.get("best_iteration") or 0)
        if best_iter < 10:
            report.add_warning(
                "early_stop_too_soon",
                f"Best iteration < 10 (best_iteration={best_iter})",
                "The booster stopped almost immediately. This usually means the "
                "validation signal is noise-like, features have little predictive "
                "power, or early stopping is too aggressive relative to learning rate.",
                severity="critical",
            )

        roc = report.predictions.get("roc_auc")
        if isinstance(roc, float) and roc == roc and roc < 0.52:
            report.add_warning(
                "roc_near_random",
                f"ROC-AUC below 0.52 (roc_auc={_fmt(roc)})",
                "Discrimination is near chance on validation. Investigate label "
                "definition, target_mode, feature leakage/staleness, and regime shift "
                "before calibration or backtesting.",
                severity="critical",
            )

        pred_pos = int(report.predictions.get("predicted_positives") or 0)
        pred_neg = int(report.predictions.get("predicted_negatives") or 0)
        if pred_pos == 0 or pred_neg == 0:
            report.add_warning(
                "single_class_predictions",
                "Model predicts only one class at the configured threshold",
                "All validation rows map to one side of the decision threshold. "
                "Check probability collapse and run threshold_analysis.csv for "
                "operating points that actually fire trades.",
                severity="critical",
            )

        const = list(report.features.get("constant_features") or [])
        if const:
            report.add_warning(
                "constant_features",
                f"Constant features detected: {const[:8]}",
                "Constant columns add no information and can inflate dimensionality. "
                "Drop them from the feature pipeline.",
            )

        near = list(report.features.get("near_constant_features") or [])
        if near:
            report.add_warning(
                "near_constant_features",
                f"Near-constant features detected: {near[:8]}",
                "Near-zero variance features rarely help tree models and should be reviewed.",
            )

        max_psi = report.drift.get("max_psi")
        if isinstance(max_psi, float) and max_psi == max_psi and max_psi >= 0.25:
            report.add_warning(
                "severe_feature_drift",
                f"Severe train-validation drift (max PSI={_fmt(max_psi)})",
                "Feature distributions shifted between train and validation. "
                "A weak validation score may reflect non-stationarity rather than "
                "a broken training loop. Inspect drift_report.csv.",
                severity="critical",
            )

        flags = dict(report.probabilities.get("flags") or {})
        if flags.get("collapsed_probability_distribution"):
            report.add_warning(
                "probability_collapsed",
                "Probability collapsed",
                "Predicted probabilities occupy a narrow band. The model is not "
                "expressing confidence variation — typically from under-training "
                "(tiny best_iteration) or uninformative features.",
                severity="critical",
            )
        if flags.get("collapsed_predictions"):
            report.add_warning(
                "predictions_collapsed",
                "Collapsed predictions",
                "Probability std is extremely low; ranking and thresholding become unreliable.",
                severity="critical",
            )
        if flags.get("overconfident_model"):
            report.add_warning(
                "overconfident",
                "Overconfident model",
                "A large share of probabilities sit near 0 or 1. Verify label noise "
                "and whether validation metrics support that confidence.",
            )
        if flags.get("underconfident_model"):
            report.add_warning(
                "underconfident",
                "Underconfident model",
                "Probabilities cluster around 0.5 with low dispersion. Ranking may "
                "still exist but threshold 0.5 will not separate classes cleanly.",
            )

        thr = float(report.metadata.get("threshold") or 0.5)
        if abs(thr - 0.5) < 1e-9 and pred_pos == 0:
            report.add_warning(
                "threshold_zero_positives",
                "Threshold 0.5 predicts zero positives",
                "No validation row exceeds 0.5. Either the score scale is compressed "
                "below 0.5 or the decision threshold is misaligned with the score "
                "distribution — use threshold_analysis.csv.",
                severity="critical",
            )

        imb = dict(report.dataset.get("dataset_imbalance") or {})
        ratio = imb.get("tp_sl_imbalance_ratio")
        if isinstance(ratio, float) and ratio == ratio and ratio >= 2.0:
            report.add_warning(
                "class_imbalance",
                f"Class imbalance (TP/SL ratio={_fmt(ratio)})",
                "Barrier outcomes are unbalanced. Confirm class_weight / scale_pos_weight "
                "and prefer PR-AUC over accuracy when interpreting results.",
            )

        pairs = list(report.features.get("high_correlation_pairs") or [])
        extreme = [p for p in pairs if abs(float(p.get("correlation", 0))) >= 0.98]
        if extreme:
            sample = extreme[0]
            report.add_warning(
                "extreme_correlation",
                f"Correlation above 0.98 ({sample.get('feature_a')} vs {sample.get('feature_b')})",
                "Near-duplicate features waste splits and can destabilize importance. "
                "Keep one representative from each highly correlated group.",
            )

        unused = list(report.model.get("unused_features") or [])
        if len(unused) >= max(3, int(0.3 * len(report.model.get("feature_names") or unused or [1]))):
            report.add_warning(
                "many_unused_features",
                f"Many unused features ({len(unused)})",
                "A large share of features received zero gain. The model may have "
                "stopped before exploring the feature space, or features are redundant.",
            )

    def _build_markdown(self, report: DiagnosticsReport) -> str:
        lines: list[str] = [
            f"# Diagnostics Report — {report.symbol} {report.timeframe} {report.side}",
            "",
            f"- Algorithm: `{report.algorithm}`",
            f"- Created (UTC): `{report.created_at.isoformat()}`",
            f"- Target mode: `{report.metadata.get('target_mode')}`",
            f"- Feature / label / dataset versions: "
            f"`{report.metadata.get('feature_version')}` / "
            f"`{report.metadata.get('label_version')}` / "
            f"`{report.metadata.get('dataset_version')}`",
            "",
            "## Warnings",
            "",
        ]
        if not report.warnings:
            lines.append("No automatic warnings fired.")
        else:
            for w in report.warnings:
                mark = "CRITICAL" if w.severity == "critical" else "WARN"
                lines.append(f"- **[!] [{mark}]** {w.message}")
                lines.append(f"  - {w.explanation}")
        lines.extend(["", "## Dataset", ""])
        d = report.dataset
        lines.extend(
            [
                f"- Total rows: `{d.get('total_rows')}`",
                f"- Train / validation / test: "
                f"`{d.get('train_rows')}` / `{d.get('validation_rows')}` / `{d.get('test_rows')}`",
                f"- Missing values (train): `{ (d.get('missing_values') or {}).get('train') }`",
                f"- Duplicate timestamps (train): `{ (d.get('duplicate_timestamps') or {}).get('train') }`",
                f"- Chronological train: `{(d.get('chronological_ordering') or {}).get('train')}`",
                f"- Imbalance: `{d.get('dataset_imbalance')}`",
                f"- Class distribution: `{d.get('class_distribution')}`",
                "",
                "## Features",
                "",
                f"- Feature count: `{report.features.get('n_features')}`",
                f"- Constant: `{report.features.get('constant_features')}`",
                f"- Near-constant: `{report.features.get('near_constant_features')}`",
                f"- High-correlation pairs (>{report.features.get('corr_threshold')}): "
                f"`{len(report.features.get('high_correlation_pairs') or [])}`",
                "",
                "## Labels",
                "",
                f"- TP / SL / TIMEOUT %: "
                f"`{_fmt(report.labels.get('tp_pct'))}` / "
                f"`{_fmt(report.labels.get('sl_pct'))}` / "
                f"`{_fmt(report.labels.get('timeout_pct'))}`",
                f"- Positive / negative rate: "
                f"`{_fmt(report.labels.get('positive_rate'))}` / "
                f"`{_fmt(report.labels.get('negative_rate'))}`",
                f"- Holding period: `{report.labels.get('holding_period')}`",
                f"- Exit reasons: `{report.labels.get('exit_reason_distribution')}`",
                "",
                "## Model",
                "",
                f"- Best iteration: `{report.model.get('best_iteration')}`",
                f"- Tree count: `{report.model.get('tree_count')}`",
                f"- Train / valid loss: "
                f"`{_fmt(report.model.get('training_loss'))}` / "
                f"`{_fmt(report.model.get('validation_loss'))}`",
                f"- Model size (bytes): `{report.model.get('model_size_bytes')}`",
                f"- Unused features: `{report.model.get('unused_features')}`",
                "",
                "## Predictions (validation)",
                "",
                f"- ROC-AUC / PR-AUC: "
                f"`{_fmt(report.predictions.get('roc_auc'))}` / "
                f"`{_fmt(report.predictions.get('pr_auc'))}`",
                f"- Precision / Recall / F1 / Accuracy: "
                f"`{_fmt(report.predictions.get('precision'))}` / "
                f"`{_fmt(report.predictions.get('recall'))}` / "
                f"`{_fmt(report.predictions.get('f1'))}` / "
                f"`{_fmt(report.predictions.get('accuracy'))}`",
                f"- Predicted + / -: "
                f"`{report.predictions.get('predicted_positives')}` / "
                f"`{report.predictions.get('predicted_negatives')}`",
                f"- Confusion matrix: `{report.predictions.get('confusion_matrix')}`",
                "",
                "## Probabilities",
                "",
                f"- Min / Max / Mean / Median / Std: "
                f"`{_fmt(report.probabilities.get('min'))}` / "
                f"`{_fmt(report.probabilities.get('max'))}` / "
                f"`{_fmt(report.probabilities.get('mean'))}` / "
                f"`{_fmt(report.probabilities.get('median'))}` / "
                f"`{_fmt(report.probabilities.get('std'))}`",
                f"- Percentiles: `{report.probabilities.get('percentiles')}`",
                f"- By class: `{report.probabilities.get('probability_by_class')}`",
                f"- Overlap: `{_fmt(report.probabilities.get('probability_overlap'))}`",
                f"- Flags: `{report.probabilities.get('flags')}`",
                f"- Best F1 threshold sweep: `{report.thresholds.get('best_f1_threshold')}`",
                "",
                "## Drift (train vs validation)",
                "",
                f"- Max PSI / Max KS: "
                f"`{_fmt(report.drift.get('max_psi'))}` / `{_fmt(report.drift.get('max_ks'))}`",
                f"- Severe PSI / KS feature counts: "
                f"`{report.drift.get('n_severe_psi_features')}` / "
                f"`{report.drift.get('n_severe_ks_features')}`",
                f"- Target drift: `{report.drift.get('target_drift')}`",
                f"- Probability drift: `{report.drift.get('probability_drift')}`",
                f"- Top PSI features: `{report.drift.get('top_psi_features')}`",
                "",
                "## Artifacts",
                "",
                "- `probability_distribution.csv`",
                "- `threshold_analysis.csv`",
                "- `feature_statistics.csv`",
                "- `feature_importance.csv`",
                "- `prediction_distribution.csv`",
                "- `learning_curve.csv`",
                "- `drift_report.csv`",
                "- `metadata.json`",
                "",
            ]
        )
        return "\n".join(lines)
