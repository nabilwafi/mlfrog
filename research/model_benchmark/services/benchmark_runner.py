"""Walk-forward benchmark runner — same protocol for every algorithm."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
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
from research.feature_ablation.services.walk_forward_runner import expected_calibration_error
from research.model_benchmark.entities.benchmark_result import (
    BenchmarkModelResult,
    BenchmarkWindowResult,
)
from research.model_benchmark.services.model_catalog import default_params

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


def best_f1_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    best_t, best_f1 = 0.5, -1.0
    for t in np.round(np.arange(0.05, 0.96, 0.05), 2):
        pred = (y_proba >= t).astype(int)
        score = float(f1_score(y_true, pred, zero_division=0))
        if score > best_f1:
            best_f1 = score
            best_t = float(t)
    return best_t


class BenchmarkWalkForwardRunner:
    def __init__(
        self,
        *,
        windows: list[WalkForwardWindow] | None = None,
        target_mode: str = "exclude_timeout",
        threshold: float = 0.5,
        random_seed: int = 42,
        model_params: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.windows = windows or [
            WalkForwardWindow(2020, 2022, 2023),
            WalkForwardWindow(2021, 2023, 2024),
            WalkForwardWindow(2022, 2024, 2025),
            WalkForwardWindow(2023, 2025, 2026),
        ]
        self.target_mode = target_mode
        self.threshold = threshold
        self.random_seed = random_seed
        self.model_params = dict(model_params or {})

    def run(
        self,
        panel: pd.DataFrame,
        algorithm: str,
        *,
        feature_names: list[str],
    ) -> BenchmarkModelResult:
        TrainerRegistry.discover()
        params = default_params(algorithm)
        params.update(self.model_params.get(algorithm) or {})
        trainer = TrainerRegistry.create(algorithm, params)

        keep = [c for c in _META if c in panel.columns] + list(feature_names)
        work = panel.loc[:, keep].copy()
        result = BenchmarkModelResult(algorithm=algorithm, n_features=len(feature_names))

        for window in self.windows:
            train_raw = filter_years(work, window.train_start_year, window.train_end_year)
            val_raw = filter_years(work, window.valid_year, window.valid_year)
            if train_raw.empty or val_raw.empty:
                result.windows.append(
                    self._skipped(algorithm, window, len(train_raw), len(val_raw))
                )
                continue

            train_ds = self._to_dataset(train_raw, "train")
            val_ds = self._to_dataset(val_raw, "validation")
            context = {
                "target_mode": self.target_mode,
                "random_seed": self.random_seed,
                "dataset_version": "benchmark_v1",
            }
            try:
                t0 = time.perf_counter()
                model = trainer.fit(train_ds, val_ds, context=context)
                train_s = time.perf_counter() - t0

                x_val, y_val, _ = trainer.prepare_xy(val_ds, target_mode=self.target_mode)
                t1 = time.perf_counter()
                proba = np.asarray(trainer.predict_proba(model, x_val), dtype=float)
                infer_s = time.perf_counter() - t1

                if len(y_val) == 0 or len(np.unique(y_val)) < 2:
                    result.windows.append(
                        self._skipped(
                            algorithm,
                            window,
                            model.train_rows,
                            len(y_val),
                            status="skipped_degenerate",
                        )
                    )
                    continue

                thr = best_f1_threshold(y_val, proba)
                pred = (proba >= self.threshold).astype(int)
                cm = confusion_matrix(y_val, pred, labels=[0, 1])
                result.pooled_proba.extend(proba.tolist())
                result.pooled_y.extend(y_val.tolist())

                wr = BenchmarkWindowResult(
                    algorithm=algorithm,
                    window=window.name,
                    valid_year=window.valid_year,
                    status="ok",
                    roc_auc=float(roc_auc_score(y_val, proba)),
                    pr_auc=float(average_precision_score(y_val, proba)),
                    precision=float(precision_score(y_val, pred, zero_division=0)),
                    recall=float(recall_score(y_val, pred, zero_division=0)),
                    f1=float(f1_score(y_val, pred, zero_division=0)),
                    log_loss=float(log_loss(y_val, proba, labels=[0, 1])),
                    calibration_error=expected_calibration_error(y_val, proba),
                    best_threshold=thr,
                    best_iteration=int(model.metadata.get("best_iteration") or 0),
                    train_seconds=float(train_s),
                    infer_seconds=float(infer_s),
                    train_rows=int(model.train_rows),
                    valid_rows=int(len(y_val)),
                    proba_mean=float(proba.mean()),
                    proba_std=float(proba.std(ddof=1)) if len(proba) > 1 else 0.0,
                    proba_min=float(proba.min()),
                    proba_max=float(proba.max()),
                    tn=int(cm[0, 0]),
                    fp=int(cm[0, 1]),
                    fn=int(cm[1, 0]),
                    tp=int(cm[1, 1]),
                )
                logger.info(
                    "Benchmark %s | %s roc=%.4f ece=%.4f train=%.2fs",
                    algorithm,
                    window.name,
                    wr.roc_auc,
                    wr.calibration_error,
                    wr.train_seconds,
                )
                result.windows.append(wr)
            except Exception:
                logger.exception(
                    "Benchmark window failed | algo=%s window=%s",
                    algorithm,
                    window.name,
                )
                result.windows.append(
                    self._skipped(
                        algorithm,
                        window,
                        len(train_raw),
                        len(val_raw),
                        status="error",
                    )
                )

        result.aggregate()
        return result

    def _skipped(
        self,
        algorithm: str,
        window: WalkForwardWindow,
        train_rows: int,
        valid_rows: int,
        *,
        status: str = "skipped_empty",
    ) -> BenchmarkWindowResult:
        nan = float("nan")
        return BenchmarkWindowResult(
            algorithm=algorithm,
            window=window.name,
            valid_year=window.valid_year,
            status=status,
            roc_auc=nan,
            pr_auc=nan,
            precision=nan,
            recall=nan,
            f1=nan,
            log_loss=nan,
            calibration_error=nan,
            best_threshold=0.5,
            best_iteration=0,
            train_seconds=0.0,
            infer_seconds=0.0,
            train_rows=train_rows,
            valid_rows=valid_rows,
            proba_mean=nan,
            proba_std=nan,
            proba_min=nan,
            proba_max=nan,
            tn=0,
            fp=0,
            fn=0,
            tp=0,
        )

    def _to_dataset(self, frame: pd.DataFrame, split: str) -> Dataset:
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
