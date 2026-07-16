"""Walk-forward collector: train+val probs for baseline and v2."""

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
}


class DiagnosisWalkForwardRunner:
    """Collect train/val OOF-style probs per WF window (validation is true OOF)."""

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
                continue

            train_ds = self._to_dataset(train_raw, "train")
            val_ds = self._to_dataset(val_raw, "validation")
            context = {
                "target_mode": self.target_mode,
                "random_seed": self.random_seed,
                "dataset_version": "probability_diagnosis_v1",
            }
            model = trainer.fit(train_ds, val_ds, context=context)

            for split_name, raw, ds in (
                ("train", train_raw, train_ds),
                ("validation", val_raw, val_ds),
            ):
                x, y, _ = trainer.prepare_xy(ds, target_mode=self.target_mode)
                proba = np.asarray(trainer.predict_proba(model, x), dtype=float)
                filtered = self._filtered_frame(raw)
                if len(filtered) != len(y):
                    raise RuntimeError(
                        f"length mismatch {split_name}: {len(filtered)} vs {len(y)}"
                    )
                part = pd.DataFrame(
                    {
                        "side": side,
                        "experiment_id": experiment.experiment_id,
                        "window": window.name,
                        "valid_year": window.valid_year,
                        "split": split_name,
                        "timestamp": filtered["timestamp"].to_numpy(),
                        "y_true": y.astype(int),
                        "y_prob": proba.astype(float),
                    }
                )
                logger.info(
                    "Diagnosis %s/%s %s %s n=%s mean_p=%.4f",
                    side,
                    experiment.experiment_id,
                    window.name,
                    split_name,
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
                    "split",
                    "timestamp",
                    "y_true",
                    "y_prob",
                ]
            )
        return pd.concat(rows, ignore_index=True)

    def _filtered_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        df = frame.copy()
        if self.target_mode == "exclude_timeout":
            df = df.loc[df["label"].astype(int) != 0].reset_index(drop=True)
        return df

    def _to_dataset(self, frame: pd.DataFrame, split: str) -> Dataset:
        df = frame.copy()
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
