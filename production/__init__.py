"""Production paper/live policy knobs (locked exit-engine path).

Locked: Primary top 5% + ATR trail 0.12 + max_open 1.
Session gate OFF (all UTC hours). Finex-style fixed lot 0.01.
"""

from __future__ import annotations

# --- Gates ---
META_AS_GATE: bool = False
CONFIDENCE_ENABLED: bool = False
META_THRESHOLD: float = 0.45  # logging / optional restore
CONFIDENCE_SKIP: float = 40.0

# Primary density (live rolling percentile gate)
PRIMARY_TOP_PCT: float = 0.05

# Session gate (UTC hour inclusive). Off = trade all hours.
SESSION_HOUR_START_UTC: int = 9
SESSION_HOUR_END_UTC: int = 15
SESSION_GATE_ENABLED: bool = False

# --- Portfolio concurrency ---
MAX_OPEN_POSITIONS: int = 1
BLOCK_OPPOSITE_SIDE: bool = True

# --- Heat reference R ---
RISK_PCT: float = 0.01
RISK_BASE: float = 0.01
DAILY_LOSS_STOP_R: float = 1.0

# --- Sizing: Finex-style fixed lot (paper/live micro) ---
SIZE_FROM_PRIMARY: bool = True  # still used for logging edge / expected_r
SIZE_MODE: str = "fixed"  # fixed | risk
FIXED_LOT: float = 0.01
EXPECTED_R_REF: float = 0.25
RISK_MIN: float = 0.0025
RISK_MAX: float = 0.015

# --- Exit engine (ATR trail; initial TP/SL still from settings.strategy) ---
EXIT_MODE: str = "atr_trail"  # atr_trail | barrier
TRAIL_ATR_MULT: float = 0.12
TRAIL_ACTIVATE_R: float = 0.5  # activate after +0.5R (R = SL_ATR * atr)
EXIT_HORIZON_BARS: int = 16
TRAIL_HORIZON: int = 16  # alias used by older call sites
SL_ATR_MULT: float = 1.5

# --- Account sync (MT5 account_info) ---
USE_ACCOUNT_EQUITY: bool = True
ACCOUNT_LEVERAGE_FALLBACK: float = 500.0

# --- Live execution safety ---
# paper = MT5 candles + paper fills (no order_send)
# live  = MT5 candles + real order_send WHEN this flag is True
EXECUTION_ENABLED: bool = False  # dry-run by default; set True to send real orders

# HF cent account: trade XAUUSDc (case-sensitive on HF). Research artifacts stay XAUUSD.
LIVE_SYMBOL: str = "XAUUSDc"
RESEARCH_SYMBOL: str = "XAUUSD"

MODEL_VERSION: str = "primary_v3_frozen"
META_VERSION: str = "meta_lgbm_frozen"
FEATURE_VERSION: str = "sprint19_18feat"
LABEL_VERSION: str = "triple_barrier_v1"
PIPELINE_VERSION: str = "prod_v1_trail012_allsess_top5_fixed001_acct"
