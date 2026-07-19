"""Sprint 28 — Temporal Stability & Dataset Aging (research only; production frozen).

Diagnoses whether the edge ages. Does NOT change Meta/Conf/Heat/TP-SL or live pipeline.
"""

from __future__ import annotations

META_GATE: float = 0.45
CONF_SKIP: float = 40.0
STARTING_EQUITY: float = 80.0
COST: float = 0.00015

# Research-only LGBM (fixed; not production HPO)
LGBM_PARAMS: dict = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 40,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "verbosity": -1,
    "seed": 42,
}
NUM_BOOST_ROUND: int = 200

HISTORY_WINDOWS: tuple[int, ...] = (3, 5, 7, 10, 0)  # 0 = full history
RETRAIN_HORIZONS_MONTHS: tuple[int, ...] = (1, 3, 6, 12, 18, 24)
THRESHOLDS: tuple[float, ...] = (0.35, 0.40, 0.45, 0.50, 0.55, 0.60)
