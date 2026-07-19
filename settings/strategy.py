"""
Frozen strategy constants — values unchanged from settled research stack.

Single source of truth for thr / barriers / regime / sizing maps.
Do NOT retune here as part of architecture work.
"""

from __future__ import annotations

# --- Barriers (label + backtest + paper) ---
SL_ATR_MULT = 1.5
TP_ATR_MULT = 2.0
HORIZON_BARS = 8
BREAKEVEN_WINRATE = SL_ATR_MULT / (SL_ATR_MULT + TP_ATR_MULT)  # 0.4286...

# --- Signal gate (LONG v3 operational) ---
# Engine default SELECTED_THR remains 0.52 for v1 runners; v3/paper use this.
LONG_V3_THR = 0.51

# --- Symbol / cost (Finex XAUUSD Method A) ---
POINT = 0.01
CONTRACT_SIZE = 100.0
FALLBACK_SPREAD_POINTS = 19
ASSUMED_SLIPPAGE_POINTS = 5

# --- Risk ---
STARTING_EQUITY = 10_000.0
RISK_PER_TRADE_PCT = 0.01
BLOWN_DRAWDOWN_PCT = 0.30
VOLUME_STEP = 0.01
VOLUME_MIN = 0.01

# --- Regime guard (Gate 4 settled) ---
ROLLING_WINDOW_BARS = 24 * 180
Z_REDUCE = 2.0
Z_BLOCK = 4.0  # settled; regime.py historical default arg was 3.5 — callers must pass 4.0
REGIME_ELEVATED_SIZE_MULT = 0.5

# --- Confidence sizing (experiment; conservative = adoption pick) ---
CONFIDENCE_SIZE_MULT_DEFAULT = {"medium": 0.75, "high": 1.25}
CONFIDENCE_SIZE_MULT_CONSERVATIVE = {"medium": 0.85, "high": 1.15}
CONFIDENCE_SIZE_MULT_AGGRESSIVE = {"medium": 0.50, "high": 1.50}
# Soft default for Risk Engine / paper when enabled
CONFIDENCE_SIZE_MULT = dict(CONFIDENCE_SIZE_MULT_CONSERVATIVE)
