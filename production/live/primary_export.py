"""Export frozen primary B_v2_context boosters (fixed params; no HPO)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from research.model_v2.services.experiment_catalog import LONG_STRUCTURE_CONTEXT, SHORT_STRUCTURE_CONTEXT
from settings.paths import ROOT

logger = logging.getLogger(__name__)

_DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 40,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "verbosity": -1,
    "seed": 42,
}


def _frozen_dir(cfg: dict[str, Any]) -> Path:
    paper = dict(cfg.get("paper_trading") or {})
    raw = paper.get("frozen_models_dir", "./artifacts/models/frozen")
    p = Path(raw)
    return p if p.is_absolute() else (ROOT / p).resolve()


def _structure_path(cfg: dict[str, Any]) -> Path:
    h4_cfg = dict(cfg.get("h4_structure") or {})
    root = Path(str(h4_cfg.get("output_directory", "./artifacts/research/h4_structure")))
    return root if root.is_absolute() else (ROOT / root).resolve() / "structure_features.parquet"


def _feature_matrix_path(cfg: dict[str, Any], symbol: str, timeframe: str) -> Path:
    fe = dict(cfg.get("feature_engineering") or {})
    root = Path(str(fe.get("output_directory", "./artifacts/features")))
    base = root if root.is_absolute() else (ROOT / root).resolve()
    return base / symbol.upper() / timeframe.upper() / "feature_matrix.parquet"


def _context_for_side(side: str) -> tuple[str, ...]:
    return LONG_STRUCTURE_CONTEXT if side.lower() == "long" else SHORT_STRUCTURE_CONTEXT


def ensure_frozen_primary(cfg: dict[str, Any], *, symbol: str, timeframe: str, side: str) -> Path:
    """
    ponytail: one-time full fit on historical parquet when .txt missing.
    Same B_v2_context feature set as research; not walk-forward tuning.
    """
    out_dir = _frozen_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"primary_{side.lower()}.txt"
    if path.is_file():
        return path

    h1_path = _feature_matrix_path(cfg, symbol, timeframe)
    struct_path = _structure_path(cfg)
    if not h1_path.is_file() or not struct_path.is_file():
        raise FileNotFoundError(f"missing features for frozen primary export: {h1_path} / {struct_path}")

    h1 = pd.read_parquet(h1_path)
    h1["timestamp"] = pd.to_datetime(h1["timestamp"], utc=True)
    struct = pd.read_parquet(struct_path)
    struct["timestamp"] = pd.to_datetime(struct["timestamp"], utc=True)
    ctx = _context_for_side(side)
    panel = h1.merge(struct[["timestamp", *ctx]], on="timestamp", how="inner")

    ds_root = ROOT / "artifacts" / "datasets" / symbol.upper() / timeframe.upper() / side.lower() / "v2"
    parts = []
    for name in ("train.parquet", "validation.parquet"):
        p = ds_root / name
        if p.is_file():
            parts.append(pd.read_parquet(p))
    if not parts:
        raise FileNotFoundError(f"missing v2 dataset under {ds_root}")
    labels = pd.concat(parts, ignore_index=True)
    labels["timestamp"] = pd.to_datetime(labels["timestamp"], utc=True)
    merged = panel.merge(labels[["timestamp", "label"]], on="timestamp", how="inner")
    if merged.empty:
        raise RuntimeError("no rows after joining features + labels for frozen primary export")

    feat_names = [c for c in merged.columns if c not in {"timestamp", "label"}]
    x = merged[feat_names].astype(float)
    y = merged["label"].astype(int).to_numpy()
    n_pos = max(int((y == 1).sum()), 1)
    n_neg = max(int((y == 0).sum()), 1)
    params = {**_DEFAULT_PARAMS, "scale_pos_weight": n_neg / n_pos}
    dtrain = lgb.Dataset(x, label=y, feature_name=feat_names, free_raw_data=False)
    booster = lgb.train(params, dtrain, num_boost_round=300)
    booster.save_model(str(path))
    logger.info("exported_frozen_primary side=%s path=%s rows=%s", side, path, len(merged))
    return path
