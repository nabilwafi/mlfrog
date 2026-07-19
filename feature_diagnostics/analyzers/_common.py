"""Shared helpers for feature diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from diagnostics.analyzers._common import safe_ks, safe_psi
from feature_diagnostics.exceptions import FeatureDiagnosticsInputError

__all__ = [
    "safe_psi",
    "safe_ks",
    "load_feature_matrix",
    "load_feature_metadata",
    "align_features_labels",
    "feature_columns",
    "map_binary_y",
]


def load_feature_matrix(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FeatureDiagnosticsInputError(f"feature matrix not found: {path}")
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def load_feature_metadata(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {f["name"]: f for f in raw.get("features", [])}


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c != "timestamp"]


def map_binary_y(labels: pd.Series, *, target_mode: str) -> tuple[pd.Series, pd.Series]:
    """Return mask and binary y (1=TP, 0=SL) under target_mode."""
    raw = labels.astype(int)
    if target_mode == "exclude_timeout":
        mask = raw != 0
        y = (raw.loc[mask] == 1).astype(int)
        return mask, y
    if target_mode == "binary_vs_rest":
        mask = pd.Series(True, index=raw.index)
        y = (raw == 1).astype(int)
        return mask, y
    raise ValueError(f"unknown target_mode {target_mode!r}")


def align_features_labels(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    target_mode: str = "exclude_timeout",
) -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
    lab = labels.copy()
    lab["timestamp"] = pd.to_datetime(lab["timestamp"], utc=True)
    keep = ["timestamp", "label"]
    merged = features.merge(lab[keep], on="timestamp", how="inner")
    if merged.empty:
        raise FeatureDiagnosticsInputError("no overlap between features and labels")
    mask, y = map_binary_y(merged["label"], target_mode=target_mode)
    aligned = merged.loc[mask].reset_index(drop=True)
    y = y.reset_index(drop=True)
    feats = feature_columns(features)
    x = aligned[feats].astype(float)
    ts = aligned["timestamp"]
    return x, y.to_numpy(dtype=int), ts
