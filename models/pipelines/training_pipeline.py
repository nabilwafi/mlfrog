"""Training pipeline: validate -> train -> evaluate -> persist."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from datasets.entities.dataset import Dataset
from models.entities.model import Model
from models.evaluators.evaluator import EvaluationResult, Evaluator
from models.repositories.model_repository import ModelRepository
from models.trainers.base_trainer import BaseTrainer
from models.validators.training_validator import TrainingValidator

logger = logging.getLogger(__name__)


def _fmt_metric(value: Any) -> str:
    if isinstance(value, float):
        if value != value:  # NaN
            return "n/a"
        return f"{value:.6f}"
    return str(value)


def build_training_report(
    model: Model,
    evaluation: EvaluationResult,
    *,
    train: Dataset,
    validation: Dataset,
) -> str:
    m = evaluation.metrics
    cm = m.get("confusion_matrix") or {}
    lines = [
        f"# Training Report — {model.symbol} {model.timeframe} {model.side}",
        "",
        "## Model",
        "",
        f"- Algorithm: `{model.algorithm}`",
        f"- Dataset version: `{model.dataset_version}`",
        f"- Feature version: `{model.feature_version}`",
        f"- Label version: `{model.label_version}`",
        f"- Trained at (UTC): `{model.trained_at.isoformat()}`",
        f"- Train rows (post target map): `{model.train_rows}`",
        f"- Validation rows (post target map): `{model.validation_rows}`",
        f"- Raw train rows: `{train.size}`",
        f"- Raw validation rows: `{validation.size}`",
        f"- Target mode: `{model.metadata.get('target_mode', '')}`",
        "",
        "## Validation Metrics",
        "",
        f"| Metric | Value |",
        f"| --- | --- |",
        f"| ROC-AUC | {_fmt_metric(m.get('roc_auc'))} |",
        f"| PR-AUC | {_fmt_metric(m.get('pr_auc'))} |",
        f"| Precision | {_fmt_metric(m.get('precision'))} |",
        f"| Recall | {_fmt_metric(m.get('recall'))} |",
        f"| F1 | {_fmt_metric(m.get('f1'))} |",
        f"| Accuracy | {_fmt_metric(m.get('accuracy'))} |",
        f"| Log Loss | {_fmt_metric(m.get('log_loss'))} |",
        f"| Brier Score | {_fmt_metric(m.get('brier_score'))} |",
        f"| Positive rate | {_fmt_metric(m.get('positive_rate'))} |",
        f"| Threshold | {_fmt_metric(m.get('threshold'))} |",
        "",
        "## Confusion Matrix",
        "",
        f"|  | Pred 0 | Pred 1 |",
        f"| --- | --- | --- |",
        f"| Actual 0 | {cm.get('tn', 0)} | {cm.get('fp', 0)} |",
        f"| Actual 1 | {cm.get('fn', 0)} | {cm.get('tp', 0)} |",
        "",
        "## Top Feature Importance",
        "",
        "| Feature | Importance |",
        "| --- | --- |",
    ]
    ranked = sorted(
        evaluation.feature_importance.items(), key=lambda kv: kv[1], reverse=True
    )
    for name, score in ranked[:30]:
        lines.append(f"| `{name}` | {_fmt_metric(score)} |")
    if not ranked:
        lines.append("| _(none)_ |  |")
    lines.append("")
    lines.append("## Hyperparameters")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(model.hyperparameters, indent=2, default=str))
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


class TrainingPipeline:
    """Algorithm-agnostic orchestration. Trainers come from TrainerRegistry."""

    def __init__(
        self,
        validator: TrainingValidator,
        evaluator: Evaluator,
        repository: ModelRepository,
    ) -> None:
        self._validator = validator
        self._evaluator = evaluator
        self._repository = repository

    def run(
        self,
        train: Dataset,
        validation: Dataset,
        trainer: BaseTrainer,
        *,
        context: dict[str, Any] | None = None,
        threshold: float = 0.5,
    ) -> tuple[Model, Path]:
        ctx = dict(context or {})
        self._validator.validate(train, validation)

        logger.info(
            "Fitting trainer=%s symbol=%s tf=%s side=%s",
            trainer.name(),
            train.symbol,
            train.timeframe,
            train.side,
        )
        model = trainer.fit(train, validation, context=ctx)

        target_mode = str(ctx.get("target_mode", "exclude_timeout"))
        x_val, y_val, _ = trainer.prepare_xy(validation, target_mode=target_mode)
        y_proba = trainer.predict_proba(model, x_val)
        evaluation = self._evaluator.evaluate(
            y_val,
            y_proba,
            feature_importance=model.feature_importance,
            threshold=threshold,
        )
        model.metrics = evaluation.metrics

        report = build_training_report(
            model, evaluation, train=train, validation=validation
        )
        importance_frame = self._evaluator.importance_frame(evaluation.feature_importance)
        out_dir = self._repository.save(
            model,
            importance_frame=importance_frame,
            report_markdown=report,
        )
        logger.info(
            "Training complete | algorithm=%s roc_auc=%s path=%s",
            model.algorithm,
            evaluation.metrics.get("roc_auc"),
            out_dir,
        )
        return model, out_dir
