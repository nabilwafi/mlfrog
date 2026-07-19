"""Walk-forward OOF prediction collector for v2 feature sets."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from datasets.entities.dataset import Dataset
from models.registries.trainer_registry import TrainerRegistry
from research.analyzers._common import filter_years
from research.analyzers.walk_forward_analyzer import WalkForwardWindow
from research.structure_selection.entities.selection_result import SelectionExperiment

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
    "strategy",
    "realized_return",
    "exit_reason",
}


class PredictionWalkForwardRunner:
    """Same WF protocol as model v2; persists OOF y_true / y_prob / returns."""

    def __init__(
        self,
        *,
        windows: list[WalkForwardWindow] | None = None,
        trainer_params: dict[str, Any] | None = None,
        target_mode: str = "exclude_timeout",
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
        self.random_seed = random_seed
        self.algorithm = algorithm

    def collect(
        self,
        panel: pd.DataFrame,
        experiment: SelectionExperiment,
        *,
        side: str,
    ) -> pd.DataFrame:
        TrainerRegistry.discover()
        trainer = TrainerRegistry.create(self.algorithm, self.trainer_params)
        feature_names = list(experiment.feature_names)
        missing = [f for f in feature_names if f not in panel.columns]
        if missing:
            raise ValueError(f"panel missing features: {missing}")

        meta_present = [c for c in _META if c in panel.columns]
        work = panel.loc[:, meta_present + feature_names].copy()
        rows: list[pd.DataFrame] = []

        for window in self.windows:
            train_raw = filter_years(work, window.train_start_year, window.train_end_year)
            val_raw = filter_years(work, window.valid_year, window.valid_year)
            if train_raw.empty or val_raw.empty:
                logger.warning(
                    "ProbabilityQuality skip empty | %s/%s %s",
                    side,
                    experiment.experiment_id,
                    window.name,
                )
                continue

            train_ds = self._to_dataset(train_raw, "train")
            val_ds = self._to_dataset(val_raw, "validation")
            context = {
                "target_mode": self.target_mode,
                "random_seed": self.random_seed,
                "dataset_version": "probability_quality_v1",
            }
            model = trainer.fit(train_ds, val_ds, context=context)
            x_val, y_val, _ = trainer.prepare_xy(val_ds, target_mode=self.target_mode)
            proba = np.asarray(trainer.predict_proba(model, x_val), dtype=float)

            # Align returns from raw val (not Dataset frame — returns are not features)
            filtered = self._filtered_frame(val_raw)
            if len(filtered) != len(y_val):
                raise RuntimeError(
                    f"prepare_xy length mismatch: frame={len(filtered)} y={len(y_val)}"
                )

            part = pd.DataFrame(
                {
                    "side": side,
                    "experiment_id": experiment.experiment_id,
                    "window": window.name,
                    "valid_year": window.valid_year,
                    "timestamp": filtered["timestamp"].to_numpy(),
                    "y_true": y_val.astype(int),
                    "y_prob": proba.astype(float),
                    "realized_return": (
                        filtered["realized_return"].astype(float).to_numpy()
                        if "realized_return" in filtered.columns
                        else np.full(len(y_val), np.nan)
                    ),
                }
            )
            logger.info(
                "ProbabilityQuality %s | %s n=%s mean_p=%.4f",
                side,
                window.name,
                len(part),
                float(part["y_prob"].mean()) if len(part) else float("nan"),
            )
            rows.append(part)

        if not rows:
            return pd.DataFrame(
                columns=[
                    "side",
                    "experiment_id",
                    "window",
                    "valid_year",
                    "timestamp",
                    "y_true",
                    "y_prob",
                    "realized_return",
                ]
            )
        return pd.concat(rows, ignore_index=True)

    def _filtered_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        df = frame.copy()
        raw = df["label"].astype(int)
        if self.target_mode == "exclude_timeout":
            df = df.loc[raw != 0].reset_index(drop=True)
        return df

    def _to_dataset(self, frame: pd.DataFrame, split: str) -> Dataset:
        df = frame.copy()
        # ponytail: drop label aux cols so Dataset.feature_names stays numeric
        df = df.drop(columns=[c for c in ("realized_return", "exit_reason") if c in df.columns])
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
