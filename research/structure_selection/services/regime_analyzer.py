"""Regime-conditioned performance for structure experiments."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from datasets.entities.dataset import Dataset
from models.registries.trainer_registry import TrainerRegistry
from research.analyzers._common import filter_years
from research.analyzers.walk_forward_analyzer import WalkForwardWindow
from research.structure_selection.entities.selection_result import SelectionExperiment

logger = logging.getLogger(__name__)

_REGIME_COLS = (
    "ctx_h4_volatility_regime",
    "ctx_h4_market_regime",
    "ctx_h4_trend_direction",
)


def _bin_series(
    s: pd.Series, *, labels: tuple[str, str, str] = ("low", "mid", "high")
) -> pd.Series:
    x = s.astype(float)
    try:
        return pd.qcut(x, q=3, labels=labels, duplicates="drop")
    except ValueError:
        return pd.Series(["mid"] * len(x), index=x.index)


class RegimeAnalyzer:
    """Train experiment windows and score ROC within regime bins on validation."""

    def __init__(
        self,
        *,
        windows: list[WalkForwardWindow],
        trainer_params: dict[str, Any],
        target_mode: str = "exclude_timeout",
        random_seed: int = 42,
        algorithm: str = "lightgbm",
    ) -> None:
        self.windows = windows
        self.trainer_params = trainer_params
        self.target_mode = target_mode
        self.random_seed = random_seed
        self.algorithm = algorithm

    def analyze(
        self,
        panel: pd.DataFrame,
        experiment: SelectionExperiment,
        *,
        side: str,
    ) -> pd.DataFrame:
        present = [c for c in _REGIME_COLS if c in panel.columns]
        if not present:
            logger.warning("No regime columns on panel — skip regime analysis")
            return pd.DataFrame()

        TrainerRegistry.discover()
        trainer = TrainerRegistry.create(self.algorithm, self.trainer_params)
        feature_names = list(experiment.feature_names)
        meta = [
            "timestamp",
            "symbol",
            "timeframe",
            "feature_version",
            "label_version",
            "split",
            "side",
            "label",
            *present,
        ]
        meta = [c for c in meta if c in panel.columns]
        work = panel.loc[:, list(dict.fromkeys(meta + feature_names))].copy()

        rows: list[dict[str, Any]] = []
        for window in self.windows:
            train_raw = filter_years(work, window.train_start_year, window.train_end_year)
            val_raw = filter_years(work, window.valid_year, window.valid_year)
            if train_raw.empty or val_raw.empty:
                continue

            train_frame = train_raw.drop(columns=present, errors="ignore")
            val_frame = val_raw.drop(columns=present, errors="ignore")
            train_ds = self._to_dataset(train_frame, "train")
            val_ds = self._to_dataset(val_frame, "validation")
            model = trainer.fit(
                train_ds,
                val_ds,
                context={
                    "target_mode": self.target_mode,
                    "random_seed": self.random_seed,
                    "dataset_version": "structure_selection_regime",
                },
            )
            x_val, y_val, _ = trainer.prepare_xy(val_ds, target_mode=self.target_mode)
            if len(y_val) == 0 or len(np.unique(y_val)) < 2:
                continue
            proba = np.asarray(trainer.predict_proba(model, x_val), dtype=float)
            val_aligned = self._align_regimes_exclude_timeout(val_raw, present)
            if len(val_aligned) != len(y_val):
                continue

            for col in present:
                bins = _bin_series(val_aligned[col])
                for level in bins.dropna().unique():
                    mask = (bins == level).to_numpy()
                    if int(mask.sum()) < 30 or len(np.unique(y_val[mask])) < 2:
                        continue
                    rows.append(
                        {
                            "side": side,
                            "experiment_id": experiment.experiment_id,
                            "window": window.name,
                            "valid_year": window.valid_year,
                            "regime_family": col.replace("ctx_h4_", ""),
                            "regime_bin": str(level),
                            "n": int(mask.sum()),
                            "roc_auc": float(roc_auc_score(y_val[mask], proba[mask])),
                        }
                    )
        return pd.DataFrame(rows)

    def _align_regimes_exclude_timeout(
        self, val_raw: pd.DataFrame, present: list[str]
    ) -> pd.DataFrame:
        mask = val_raw["label"].astype(int) != 0
        return val_raw.loc[mask, present].reset_index(drop=True)

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
