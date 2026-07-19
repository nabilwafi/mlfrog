"""Load panels + train fixed-param LGBM for aging experiments (research only)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from research.analyzers._common import feature_cols
from research.temporal_stability import LGBM_PARAMS, NUM_BOOST_ROUND

logger = logging.getLogger(__name__)

_META_SKIP = {
    "timestamp",
    "symbol",
    "timeframe",
    "feature_version",
    "label_version",
    "split",
    "side",
    "label",
    "strategy",
    "year",
    "sample_weight",
}


def load_ml_panel(dataset_root: Path, *, sides: tuple[str, ...] = ("long", "short")) -> pd.DataFrame:
    """Concatenate v2 train+validation(+test) for aging WF."""
    parts: list[pd.DataFrame] = []
    for side in sides:
        side_dir = dataset_root / side / "v2"
        if not side_dir.is_dir():
            side_dir = dataset_root / side
        for name in ("train.parquet", "validation.parquet", "test.parquet", "sealed.parquet"):
            p = side_dir / name
            if p.is_file():
                df = pd.read_parquet(p)
                df["side"] = side
                parts.append(df)
    if not parts:
        raise FileNotFoundError(f"no dataset parquets under {dataset_root}")
    out = pd.concat(parts, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out["year"] = out["timestamp"].dt.year.astype(int)
    out = out.sort_values("timestamp").drop_duplicates(subset=["side", "timestamp"], keep="last")
    return out.reset_index(drop=True)


def load_frozen_trades(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["year"] = df["timestamp"].dt.year.astype(int)
    return df.sort_values("timestamp").reset_index(drop=True)


def ml_feature_names(frame: pd.DataFrame) -> list[str]:
    return [c for c in feature_cols(frame) if c not in _META_SKIP and c != "year"]


def prepare_xy(
    frame: pd.DataFrame,
    features: list[str],
    *,
    weights: np.ndarray | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray | None]:
    """exclude_timeout binary mapping."""
    raw = frame["label"].astype(int)
    mask = raw != 0
    sub = frame.loc[mask].copy()
    y = (sub["label"].astype(int) == 1).astype(int).to_numpy()
    x = sub[features].astype(float)
    w = None
    if weights is not None:
        w = np.asarray(weights, dtype=float)[mask.to_numpy()]
    return x, y, w


def fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    *,
    sample_weight: np.ndarray | None = None,
    num_boost_round: int = NUM_BOOST_ROUND,
) -> tuple[np.ndarray, np.ndarray, lgb.Booster]:
    x_tr, y_tr, w = prepare_xy(train, features, weights=sample_weight)
    x_te, y_te, _ = prepare_xy(test, features)
    if len(y_tr) < 50 or len(np.unique(y_tr)) < 2:
        raise ValueError("insufficient train labels")
    n_pos = max(int((y_tr == 1).sum()), 1)
    n_neg = max(int((y_tr == 0).sum()), 1)
    params = {**LGBM_PARAMS, "scale_pos_weight": n_neg / n_pos}
    dtrain = lgb.Dataset(x_tr, label=y_tr, weight=w, feature_name=features, free_raw_data=False)
    booster = lgb.train(params, dtrain, num_boost_round=num_boost_round)
    proba = np.asarray(booster.predict(x_te), dtype=float)
    return y_te, proba, booster


def recency_weights(years: np.ndarray, *, mode: str, half_life: float = 3.0) -> np.ndarray:
    y = np.asarray(years, dtype=float)
    ymax = float(np.nanmax(y))
    age = ymax - y
    if mode == "uniform":
        return np.ones(len(y), dtype=float)
    if mode == "linear":
        span = max(float(np.nanmax(age)), 1.0)
        return (span - age) / span + 0.1
    if mode == "exponential":
        return np.exp(-age / max(half_life, 0.5))
    if mode == "half_life":
        return 0.5 ** (age / max(half_life, 0.5))
    raise ValueError(f"unknown weight mode {mode}")
