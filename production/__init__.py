"""Sprint 27 — Production paper trading (frozen research stack)."""

from __future__ import annotations

# Frozen production policy (research settled; DO NOT retune here).
META_THRESHOLD: float = 0.45
CONFIDENCE_SKIP: float = 40.0
RISK_PCT: float = 0.01
DAILY_LOSS_STOP_R: float = 1.0  # Heat: daily_loss_-1R
MODEL_VERSION: str = "primary_v3_frozen"
META_VERSION: str = "meta_lgbm_frozen"
FEATURE_VERSION: str = "sprint19_18feat"
LABEL_VERSION: str = "triple_barrier_v1"
PIPELINE_VERSION: str = "prod_v1_sprint27"
