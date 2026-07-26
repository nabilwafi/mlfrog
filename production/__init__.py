"""Production paper/live policy knobs (research-locked stack).

Locked: FEAT7 primary + top 5% + ATR trail a0.25/d0.08
+ ATR map_conservative lot scale
+ Parallel multi-trade max_open=5, dist=0 ATR, cooldown=0, heat_budget=3R.
All UTC hours (no session gate). Finex-style base lot 0.01 × atr risk_mult.
No TP / no short time-exit (horizon = research CAP 48).
"""

from __future__ import annotations

# Canonical FS forward-selection order (LGBM determinism).
PRIMARY_FEATURES: tuple[str, ...] = (
    "hour_cos",
    "atr_percentile_252",
    "ema_trend_duration",
    "rolling_quantile",
    "hour_sin",
    "atr_percent",
    "ctx_h4_swing_quality",
)

# --- Gates ---
META_AS_GATE: bool = False
CONFIDENCE_ENABLED: bool = False
META_THRESHOLD: float = 0.45  # logging / optional restore
CONFIDENCE_SKIP: float = 40.0

# Primary density (live rolling percentile gate)
PRIMARY_TOP_PCT: float = 0.05

# --- Portfolio concurrency (Sprint 37 winner: parallel_mo5_d0.0_cd0_h3.0) ---
MAX_OPEN_POSITIONS: int = 5
BLOCK_OPPOSITE_SIDE: bool = True
PARALLEL_MIN_DISTANCE_ATR: float = 0.0  # 0 = no distance filter
PARALLEL_COOLDOWN_BARS: int = 0  # H1 bars since last entry; 0 = off
# Heat = sum(lots / FIXED_LOT) across opens; winner budget 3R slots
HEAT_BUDGET_R: float = 3.0

# --- Heat reference R (daily loss stop) ---
RISK_PCT: float = 0.01
RISK_BASE: float = 0.01
DAILY_LOSS_STOP_R: float = 1.0

# --- Sizing: Finex-style fixed lot × ATR percentile map_conservative ---
SIZE_FROM_PRIMARY: bool = True  # still used for logging edge / expected_r
SIZE_MODE: str = "fixed"  # fixed | risk
FIXED_LOT: float = 0.01
EXPECTED_R_REF: float = 0.25
RISK_MIN: float = 0.0025
RISK_MAX: float = 0.015
ATR_RISK_ENABLED: bool = True
# atr_pct edges → risk_mult buckets (Sprint 35 map_conservative)
ATR_RISK_EDGES: tuple[float, ...] = (0.30, 0.60, 0.80, 0.90)
ATR_RISK_MULTS: tuple[float, ...] = (1.0, 0.70, 0.40, 0.25, 0.10)

# --- Exit engine (ATR trail; research a0.25_d0.08, TP off) ---
EXIT_MODE: str = "atr_trail"  # atr_trail | barrier
TRAIL_ATR_MULT: float = 0.08
TRAIL_ACTIVATE_R: float = 0.25  # activate after +0.25R (R = SL_ATR * atr)
TAKE_PROFIT_ENABLED: bool = False
EXIT_HORIZON_BARS: int = 48  # research CAP when time exit disabled
TRAIL_HORIZON: int = 48  # alias used by older call sites
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

MODEL_VERSION: str = "primary_feat7_frozen"
META_VERSION: str = "meta_lgbm_frozen"
FEATURE_VERSION: str = "fs7_feat"
LABEL_VERSION: str = "triple_barrier_v1"
PIPELINE_VERSION: str = "prod_v3_feat7_a025_d008_atr_par_mo5_h3"


def atr_risk_mult(atr_percentile: float) -> float:
    """Sprint 35 map_conservative: atr_percentile_252 → lot multiplier."""
    if not ATR_RISK_ENABLED:
        return 1.0
    p = float(atr_percentile)
    if not (p == p):  # NaN
        return 1.0
    edges = ATR_RISK_EDGES
    risks = ATR_RISK_MULTS
    for i, e in enumerate(edges):
        if p <= e:
            return float(risks[i])
    return float(risks[-1])
