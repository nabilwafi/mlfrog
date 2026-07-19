"""Trainer ABC — plug-in ML algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

from datasets.entities.dataset import Dataset
from models.entities.model import Model


class BaseTrainer(ABC):
    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params: dict[str, Any] = dict(params or {})

    @abstractmethod
    def name(self) -> str:
        """Registry key, e.g. 'lightgbm'."""

    @abstractmethod
    def fit(self, train: Dataset, validation: Dataset, *, context: dict[str, Any]) -> Model:
        """Train on Dataset frames only (no Feature/Label repos)."""

    @abstractmethod
    def predict_proba(self, model: Model, frame: pd.DataFrame) -> np.ndarray:
        """Return P(y=1) aligned to frame rows."""

    def prepare_xy(
        self,
        dataset: Dataset,
        *,
        target_mode: str,
    ) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
        """Map triple-barrier labels to binary y; return X, y, feature_names."""
        df = dataset.frame.copy()
        names = dataset.feature_names
        if not names:
            raise ValueError("dataset has no feature columns")

        raw = df["label"].astype(int)
        if target_mode == "exclude_timeout":
            mask = raw != 0
            df = df.loc[mask].reset_index(drop=True)
            y = (df["label"].astype(int) == 1).astype(int).to_numpy()
        elif target_mode == "binary_vs_rest":
            y = (raw == 1).astype(int).to_numpy()
        else:
            raise ValueError(
                f"unknown target_mode {target_mode!r}; use exclude_timeout|binary_vs_rest"
            )

        x = df[names].astype(float)
        return x, y, names
