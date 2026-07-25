"""Sprint 27 — End-to-end trading pipeline L1–L6 (shared by paper + backtest)."""

from __future__ import annotations

from settings.strategy import SL_ATR_MULT, TP_ATR_MULT

# --- Gates ---
META_THRESHOLD: float = 0.45
CONFIDENCE_ENABLED: bool = False  # sprint 27: skip40 off; restore later
CONFIDENCE_SKIP: float = 40.0  # kept for logging / future restore

# --- Risk / heat reference (heat uses RISK_BASE, not per-trade risk_pct) ---
RISK_BASE: float = 0.01
DAILY_LOSS_STOP_R: float = 1.0

# --- Option C sizing (expected_r → risk_pct) ---
EXPECTED_R_REF: float = 0.25
RISK_MIN: float = 0.0025
RISK_MAX: float = 0.015
STRUCTURAL_RR: float = float(TP_ATR_MULT) / float(SL_ATR_MULT)

# --- Versions ---
MODEL_VERSION: str = "primary_v3_frozen"
META_VERSION: str = "meta_lgbm_frozen"
FEATURE_VERSION: str = "sprint19_18feat"
LABEL_VERSION: str = "triple_barrier_v1"
PIPELINE_VERSION: str = "prod_v1_sprint27_e2e"

# Compat alias used by older imports
RISK_PCT: float = RISK_BASE
