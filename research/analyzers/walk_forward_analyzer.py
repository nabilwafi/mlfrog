"""Rolling walk-forward validation — retrain per window, no hyperparameter search."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from datasets.entities.dataset import Dataset
from models.registries.trainer_registry import TrainerRegistry
from models.trainers.base_trainer import BaseTrainer
from research.analyzers._common import filter_years, frame_to_markdown, slice_dataset

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class WalkForwardWindow:
    train_start_year: int
    train_end_year: int
    valid_year: int

    @property
    def name(self) -> str:
        return f"train_{self.train_start_year}_{self.train_end_year}_val_{self.valid_year}"


class WalkForwardAnalyzer:
    def __init__(
        self,
        *,
        windows: list[WalkForwardWindow] | None = None,
        algorithm: str = "lightgbm",
        trainer_params: dict[str, Any] | None = None,
        target_mode: str = "exclude_timeout",
        threshold: float = 0.5,
        random_seed: int = 42,
        dataset_version: str = "v1",
    ) -> None:
        self.windows = windows or self.default_windows()
        self.algorithm = algorithm
        self.trainer_params = dict(trainer_params or {})
        self.target_mode = target_mode
        self.threshold = threshold
        self.random_seed = random_seed
        self.dataset_version = dataset_version

    @staticmethod
    def default_windows() -> list[WalkForwardWindow]:
        return [
            WalkForwardWindow(2020, 2022, 2023),
            WalkForwardWindow(2021, 2023, 2024),
            WalkForwardWindow(2022, 2024, 2025),
            WalkForwardWindow(2023, 2025, 2026),
        ]

    def analyze(
        self,
        frame: pd.DataFrame,
        template: Dataset,
    ) -> tuple[dict[str, Any], pd.DataFrame, str]:
        TrainerRegistry.discover()
        trainer: BaseTrainer = TrainerRegistry.create(self.algorithm, self.trainer_params)
        rows: list[dict[str, Any]] = []
        importance_by_fold: dict[str, dict[str, float]] = {}

        for window in self.windows:
            train_frame = filter_years(frame, window.train_start_year, window.train_end_year)
            val_frame = filter_years(frame, window.valid_year, window.valid_year)
            if train_frame.empty or val_frame.empty:
                logger.warning(
                    "Skipping window %s | train=%s val=%s",
                    window.name,
                    len(train_frame),
                    len(val_frame),
                )
                rows.append(
                    {
                        "window": window.name,
                        "train_start_year": window.train_start_year,
                        "train_end_year": window.train_end_year,
                        "valid_year": window.valid_year,
                        "status": "skipped_empty",
                        "train_rows_raw": int(len(train_frame)),
                        "valid_rows_raw": int(len(val_frame)),
                    }
                )
                continue

            train_ds = slice_dataset(template, train_frame, split="train")
            val_ds = slice_dataset(template, val_frame, split="validation")
            context = {
                "target_mode": self.target_mode,
                "random_seed": self.random_seed,
                "dataset_version": self.dataset_version,
            }
            logger.info(
                "Walk-forward fit | %s algorithm=%s train=%s val=%s",
                window.name,
                self.algorithm,
                len(train_frame),
                len(val_frame),
            )
            model = trainer.fit(train_ds, val_ds, context=context)
            x_val, y_val, _ = trainer.prepare_xy(val_ds, target_mode=self.target_mode)
            if len(y_val) == 0:
                rows.append(
                    {
                        "window": window.name,
                        "status": "skipped_empty_target",
                        "train_rows_raw": int(len(train_frame)),
                        "valid_rows_raw": int(len(val_frame)),
                    }
                )
                continue

            proba = np.asarray(trainer.predict_proba(model, x_val), dtype=float)
            pred = (proba >= self.threshold).astype(int)
            roc = (
                float(roc_auc_score(y_val, proba))
                if len(np.unique(y_val)) >= 2
                else float("nan")
            )
            pr = (
                float(average_precision_score(y_val, proba))
                if len(np.unique(y_val)) >= 2
                else float("nan")
            )
            importance = dict(model.feature_importance)
            importance_by_fold[window.name] = importance
            top_imp = dict(
                sorted(importance.items(), key=lambda kv: kv[1], reverse=True)[:15]
            )
            rows.append(
                {
                    "window": window.name,
                    "train_start_year": window.train_start_year,
                    "train_end_year": window.train_end_year,
                    "valid_year": window.valid_year,
                    "status": "ok",
                    "train_rows_raw": int(len(train_frame)),
                    "valid_rows_raw": int(len(val_frame)),
                    "train_rows_mapped": int(model.train_rows),
                    "valid_rows_mapped": int(model.validation_rows),
                    "roc_auc": roc,
                    "pr_auc": pr,
                    "precision": float(precision_score(y_val, pred, zero_division=0)),
                    "recall": float(recall_score(y_val, pred, zero_division=0)),
                    "f1": float(f1_score(y_val, pred, zero_division=0)),
                    "log_loss": float(log_loss(y_val, proba, labels=[0, 1])),
                    "best_iteration": int(model.metadata.get("best_iteration") or 0),
                    "probability_mean": float(proba.mean()),
                    "probability_std": float(proba.std(ddof=1)) if len(proba) > 1 else 0.0,
                    "feature_importance_json": json.dumps(top_imp),
                }
            )

        metrics = pd.DataFrame(rows)
        if not metrics.empty and "status" in metrics.columns:
            ok = metrics.loc[metrics["status"] == "ok"].copy()
        else:
            ok = metrics.iloc[0:0].copy()
        summary: dict[str, Any] = {
            "n_windows": int(len(self.windows)),
            "n_ok": int(len(ok)),
            "algorithm": self.algorithm,
            "target_mode": self.target_mode,
            "mean_roc_auc": float(ok["roc_auc"].mean()) if len(ok) and "roc_auc" in ok else float("nan"),
            "std_roc_auc": float(ok["roc_auc"].std(ddof=1)) if len(ok) > 1 and "roc_auc" in ok else float("nan"),
            "best_window": None,
            "worst_window": None,
            "strong_years": [],
            "fail_years": [],
        }
        if len(ok) and "roc_auc" in ok.columns:
            best_idx = ok["roc_auc"].idxmax()
            worst_idx = ok["roc_auc"].idxmin()
            summary["best_window"] = ok.loc[best_idx].to_dict()
            summary["worst_window"] = ok.loc[worst_idx].to_dict()
            summary["strong_years"] = [
                int(r["valid_year"])
                for _, r in ok.iterrows()
                if isinstance(r.get("roc_auc"), float) and r["roc_auc"] == r["roc_auc"] and r["roc_auc"] >= 0.55
            ]
            summary["fail_years"] = [
                int(r["valid_year"])
                for _, r in ok.iterrows()
                if isinstance(r.get("roc_auc"), float)
                and r["roc_auc"] == r["roc_auc"]
                and r["roc_auc"] < 0.52
            ]

        # Aggregate mean importance across folds
        if importance_by_fold:
            all_feats = sorted({f for d in importance_by_fold.values() for f in d})
            mean_imp = {
                f: float(np.mean([d.get(f, 0.0) for d in importance_by_fold.values()]))
                for f in all_feats
            }
            summary["mean_feature_importance"] = dict(
                sorted(mean_imp.items(), key=lambda kv: kv[1], reverse=True)[:20]
            )

        md = self._markdown(summary, metrics)
        return summary, metrics, md

    def _markdown(self, summary: dict[str, Any], metrics: pd.DataFrame) -> str:
        lines = [
            "# Walk-Forward Report",
            "",
            f"- Algorithm: `{summary.get('algorithm')}`",
            f"- Windows OK: `{summary.get('n_ok')}/{summary.get('n_windows')}`",
            f"- Mean ROC-AUC: `{summary.get('mean_roc_auc')}` (std `{summary.get('std_roc_auc')}`)",
            f"- Strong valid years (ROC>=0.55): `{summary.get('strong_years')}`",
            f"- Fail valid years (ROC<0.52): `{summary.get('fail_years')}`",
            "",
            "## Per-window metrics",
            "",
        ]
        if metrics.empty:
            lines.append("No windows evaluated.")
            return "\n".join(lines)
        cols = [
            c
            for c in (
                "window",
                "status",
                "roc_auc",
                "pr_auc",
                "precision",
                "recall",
                "f1",
                "log_loss",
                "best_iteration",
                "probability_mean",
                "probability_std",
            )
            if c in metrics.columns
        ]
        lines.append(frame_to_markdown(metrics[cols]))
        lines.append("")
        return "\n".join(lines)
