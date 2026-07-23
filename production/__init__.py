"""Sprint 27 — Production paper trading policy knobs.

Locked to WF exit-engine winner (2015–2026):
  Primary top 5% + session 09–15 UTC + ATR trail 0.12 + max_open 1
  (~WR 70%, PF ~1.77 on that backtest; live may differ with spread).
"""

from __future__ import annotations

# --- Gates ---
META_AS_GATE: bool = False
CONFIDENCE_ENABLED: bool = False
META_THRESHOLD: float = 0.45  # logging / optional restore
CONFIDENCE_SKIP: float = 40.0

# Primary density (live rolling percentile gate)
PRIMARY_TOP_PCT: float = 0.05

# Session gate (UTC hour inclusive). Backtest winner: 09–15.
SESSION_HOUR_START_UTC: int = 9
SESSION_HOUR_END_UTC: int = 15
SESSION_GATE_ENABLED: bool = True

# --- Portfolio concurrency ---
MAX_OPEN_POSITIONS: int = 1
BLOCK_OPPOSITE_SIDE: bool = True

# --- Heat reference R ---
RISK_PCT: float = 0.01
RISK_BASE: float = 0.01
DAILY_LOSS_STOP_R: float = 1.0

# --- Sizing: primary proba as edge (matches exit-engine backtest) ---
SIZE_FROM_PRIMARY: bool = True
EXPECTED_R_REF: float = 0.25
RISK_MIN: float = 0.0025
RISK_MAX: float = 0.015

# --- Exit engine (ATR trail; initial TP/SL still from settings.strategy) ---
EXIT_MODE: str = "atr_trail"  # atr_trail | barrier
TRAIL_ATR_MULT: float = 0.12
TRAIL_ACTIVATE_R: float = 0.5  # activate after +0.5R (R = SL_ATR * atr)
EXIT_HORIZON_BARS: int = 16

MODEL_VERSION: str = "primary_v3_frozen"
META_VERSION: str = "meta_lgbm_frozen"
FEATURE_VERSION: str = "sprint19_18feat"
LABEL_VERSION: str = "triple_barrier_v1"
PIPELINE_VERSION: str = "prod_v1_trail012_sess0915_top5"
