"""Sprint 21 - Realistic portfolio backtest (no ML; frozen models)."""

META_THRESHOLD: float = 0.45
STARTING_EQUITY: float = 80.0

# (id, mode, size_param, enforce_volume_min)
SCENARIOS: tuple[tuple[str, str, float | None, bool], ...] = (
    ("A_fixed_1lot", "fixed", 1.0, True),
    ("B_risk_0_5pct", "risk", 0.005, True),
    ("C_risk_1pct", "risk", 0.01, True),
    ("D_risk_2pct", "risk", 0.02, True),
    # Fractional lots (no VOLUME_MIN) — analytical risk MM on $80 capital
    ("B_risk_0_5pct_frac", "risk", 0.005, False),
    ("C_risk_1pct_frac", "risk", 0.01, False),
    ("D_risk_2pct_frac", "risk", 0.02, False),
)
