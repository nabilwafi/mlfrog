"""Export frozen primary H1-native boosters (fixed params; no HPO)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import lightgbm as lgb
import pandas as pd

from production import PRIMARY_FEATURES, RESEARCH_SYMBOL
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


def _model_matches(path: Path, feat: list[str]) -> bool:
    if not path.is_file():
        return False
    try:
        booster = lgb.Booster(model_file=str(path))
        return list(booster.feature_name()) == list(feat)
    except Exception:
        return False


def ensure_frozen_primary(cfg: dict[str, Any], *, symbol: str, timeframe: str, side: str) -> Path:
    """
    ponytail: fit H1-native primary on v2 train/val parquet when .txt missing or feature set drifts.
    Uses RESEARCH_SYMBOL artifacts even when live trades XAUUSDc.
    """
    del symbol  # research artifacts only
    feat = list(PRIMARY_FEATURES)
    out_dir = _frozen_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"primary_{side.lower()}.txt"
    if _model_matches(path, feat):
        return path

    ds_root = ROOT / "artifacts" / "datasets" / str(RESEARCH_SYMBOL).upper() / timeframe.upper() / side.lower() / "v2"
    train_path = ds_root / "train.parquet"
    val_path = ds_root / "validation.parquet"
    if not train_path.is_file():
        raise FileNotFoundError(f"missing v2 train under {ds_root}")

    train = pd.read_parquet(train_path)
    train["timestamp"] = pd.to_datetime(train["timestamp"], utc=True)
    missing = [c for c in feat if c not in train.columns]
    if missing or "label" not in train.columns:
        raise RuntimeError(f"frozen primary missing columns: {missing or ['label']}")

    y_tr = train["label"].astype(int).to_numpy()
    n_pos = max(int((y_tr == 1).sum()), 1)
    n_neg = max(int((y_tr == 0).sum()), 1)
    params = {**_DEFAULT_PARAMS, "scale_pos_weight": n_neg / n_pos}
    dtrain = lgb.Dataset(train[feat].astype(float), label=y_tr, feature_name=feat, free_raw_data=False)

    if val_path.is_file():
        val = pd.read_parquet(val_path)
        val["timestamp"] = pd.to_datetime(val["timestamp"], utc=True)
        y_va = val["label"].astype(int).to_numpy()
        dval = lgb.Dataset(val[feat].astype(float), label=y_va, reference=dtrain, free_raw_data=False)
        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=500,
            valid_sets=[dval],
            valid_names=["val"],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
    else:
        booster = lgb.train(params, dtrain, num_boost_round=300)

    booster.save_model(str(path))
    logger.info(
        "exported_frozen_primary side=%s path=%s rows=%s feats=%s best_iter=%s",
        side,
        path,
        len(train),
        len(feat),
        getattr(booster, "best_iteration", None),
    )
    return path
