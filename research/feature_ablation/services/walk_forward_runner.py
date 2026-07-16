"""Walk-forward training runner for one ablation experiment."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
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
from research.analyzers._common import filter_years
from research.analyzers.walk_forward_analyzer import WalkForwardWindow
from research.feature_ablation.entities.ablation_experiment import AblationExperiment
from research.feature_ablation.entities.experiment_result import ExperimentResult, WindowMetrics

logger = logging.getLogger(__name__)

_META = {
    "timestamp",
    "symbol",
    "timeframe",
    "feature_version",
    "label_version",
    "split",
    "side",
    "label",
}


def expected_calibration_error(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    """ECE with equal-width probability bins."""
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_proba = np.asarray(y_proba, dtype=float).ravel()
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = max(len(y_true), 1)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_proba >= lo) & (y_proba <= hi) if i == n_bins - 1 else (y_proba >= lo) & (y_proba < hi)
        if not np.any(mask):
            continue
        conf = float(y_proba[mask].mean())
        acc = float(y_true[mask].mean())
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


class AblationWalkForwardRunner:
    def __init__(
        self,
        *,
        windows: list[WalkForwardWindow] | None = None,
        trainer_params: dict[str, Any] | None = None,
        target_mode: str = "exclude_timeout",
        threshold: float = 0.5,
        random_seed: int = 42,
        algorithm: str = "lightgbm",
    ) -> None:
        self.windows = windows or [
            WalkForwardWindow(2020, 2022, 2023),
            WalkForwardWindow(2021, 2023, 2024),
            WalkForwardWindow(2022, 2024, 2025),
            WalkForwardWindow(2023, 2025, 2026),
        ]
        self.trainer_params = dict(trainer_params or {})
        self.target_mode = target_mode
        self.threshold = threshold
        self.random_seed = random_seed
        self.algorithm = algorithm

    def run(
        self,
        panel: pd.DataFrame,
        experiment: AblationExperiment,
        *,
        category_counts: dict[str, int],
    ) -> ExperimentResult:
        TrainerRegistry.discover()
        trainer = TrainerRegistry.create(self.algorithm, self.trainer_params)
        feature_names = list(experiment.feature_names)
        keep_cols = [c for c in _META if c in panel.columns] + feature_names
        work = panel.loc[:, keep_cols].copy()

        result = ExperimentResult(
            experiment_id=experiment.experiment_id,
            name=experiment.name,
            description=experiment.description,
            n_features=len(feature_names),
            feature_names=tuple(feature_names),
            feature_categories=dict(category_counts),
        )

        for window in self.windows:
            train_raw = filter_years(work, window.train_start_year, window.train_end_year)
            val_raw = filter_years(work, window.valid_year, window.valid_year)
            if train_raw.empty or val_raw.empty:
                result.windows.append(
                    self._empty_window(experiment.experiment_id, window, feature_names, train_raw, val_raw)
                )
                continue

            train_ds = self._to_dataset(train_raw, split="train")
            val_ds = self._to_dataset(val_raw, split="validation")
            context = {
                "target_mode": self.target_mode,
                "random_seed": self.random_seed,
                "dataset_version": "ablation_v1",
            }
            t0 = time.perf_counter()
            model = trainer.fit(train_ds, val_ds, context=context)
            train_seconds = time.perf_counter() - t0

            x_val, y_val, _ = trainer.prepare_xy(val_ds, target_mode=self.target_mode)
            t1 = time.perf_counter()
            proba = np.asarray(trainer.predict_proba(model, x_val), dtype=float)
            infer_seconds = time.perf_counter() - t1
            pred = (proba >= self.threshold).astype(int)

            if len(y_val) == 0 or len(np.unique(y_val)) < 2:
                status = "skipped_degenerate"
                roc = pr = ll = prec = rec = f1 = ece = float("nan")
            else:
                status = "ok"
                roc = float(roc_auc_score(y_val, proba))
                pr = float(average_precision_score(y_val, proba))
                ll = float(log_loss(y_val, proba, labels=[0, 1]))
                prec = float(precision_score(y_val, pred, zero_division=0))
                rec = float(recall_score(y_val, pred, zero_division=0))
                f1 = float(f1_score(y_val, pred, zero_division=0))
                ece = expected_calibration_error(y_val, proba)

            logger.info(
                "Ablation %s | %s roc=%.4f ece=%.4f feats=%s",
                experiment.experiment_id,
                window.name,
                roc if roc == roc else -1.0,
                ece if ece == ece else -1.0,
                len(feature_names),
            )
            result.windows.append(
                WindowMetrics(
                    experiment_id=experiment.experiment_id,
                    window=window.name,
                    valid_year=window.valid_year,
                    status=status,
                    n_features=len(feature_names),
                    roc_auc=roc,
                    pr_auc=pr,
                    log_loss=ll,
                    precision=prec,
                    recall=rec,
                    f1=f1,
                    calibration_error=ece,
                    best_iteration=int(model.metadata.get("best_iteration") or 0),
                    train_seconds=float(train_seconds),
                    infer_seconds=float(infer_seconds),
                    train_rows=int(model.train_rows),
                    valid_rows=int(len(y_val)),
                )
            )

        result.aggregate()
        return result

    def _empty_window(
        self,
        experiment_id: str,
        window: WalkForwardWindow,
        feature_names: list[str],
        train_raw: pd.DataFrame,
        val_raw: pd.DataFrame,
    ) -> WindowMetrics:
        return WindowMetrics(
            experiment_id=experiment_id,
            window=window.name,
            valid_year=window.valid_year,
            status="skipped_empty",
            n_features=len(feature_names),
            roc_auc=float("nan"),
            pr_auc=float("nan"),
            log_loss=float("nan"),
            precision=float("nan"),
            recall=float("nan"),
            f1=float("nan"),
            calibration_error=float("nan"),
            best_iteration=0,
            train_seconds=0.0,
            infer_seconds=0.0,
            train_rows=int(len(train_raw)),
            valid_rows=int(len(val_raw)),
        )

    def _to_dataset(self, frame: pd.DataFrame, *, split: str) -> Dataset:
        df = frame.copy()
        df["split"] = split
        side = str(df["side"].iloc[0]) if "side" in df.columns else "long"
        return Dataset(
            symbol=str(df["symbol"].iloc[0]),
            timeframe=str(df["timeframe"].iloc[0]),
            side=side,
            strategy="triple_barrier",
            feature_version=str(df["feature_version"].iloc[0]),
            label_version=str(df["label_version"].iloc[0]),
            split=split,
            created_at=datetime.now(tz=timezone.utc),
            frame=df.reset_index(drop=True),
        )
