"""Shared helpers for diagnostics analyzers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from datasets.entities.dataset import Dataset


def feature_matrix(dataset: Dataset) -> pd.DataFrame:
    return dataset.frame.loc[:, dataset.feature_names].astype(float)


def map_binary_target(
    dataset: Dataset,
    *,
    target_mode: str,
) -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
    """Return X, y, timestamps after applying the same target_mode as training."""
    df = dataset.frame.copy()
    names = dataset.feature_names
    raw = df["label"].astype(int)
    if target_mode == "exclude_timeout":
        mask = raw != 0
        df = df.loc[mask].reset_index(drop=True)
        y = (df["label"].astype(int) == 1).astype(int).to_numpy()
    elif target_mode == "binary_vs_rest":
        y = (raw == 1).astype(int).to_numpy()
    else:
        raise ValueError(f"unknown target_mode {target_mode!r}")
    x = df[names].astype(float)
    ts = pd.to_datetime(df["timestamp"], utc=True)
    return x, y, ts


def safe_psi(expected: np.ndarray, actual: np.ndarray, *, bins: int = 10) -> float:
    """Population Stability Index between expected (train) and actual (val)."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if len(expected) < 2 or len(actual) < 2:
        return float("nan")
    qs = np.linspace(0, 1, bins + 1)
    cuts = np.unique(np.quantile(expected, qs))
    if len(cuts) < 3:
        return 0.0
    e_counts, _ = np.histogram(expected, bins=cuts)
    a_counts, _ = np.histogram(actual, bins=cuts)
    e_pct = e_counts / max(e_counts.sum(), 1)
    a_pct = a_counts / max(a_counts.sum(), 1)
    e_pct = np.clip(e_pct, 1e-6, None)
    a_pct = np.clip(a_pct, 1e-6, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def safe_ks(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample KS statistic without scipy dependency."""
    a = np.sort(np.asarray(a, dtype=float)[np.isfinite(a)])
    b = np.sort(np.asarray(b, dtype=float)[np.isfinite(b)])
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    data = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, data, side="right") / len(a)
    cdf_b = np.searchsorted(b, data, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))
