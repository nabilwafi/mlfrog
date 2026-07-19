"""Sprint 23 — Trade Quality Engine (execution / capital allocation; frozen models)."""

META_GATE: float = 0.45
STARTING_EQUITY: float = 80.0

BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("0-20", 0.0, 20.0),
    ("20-40", 20.0, 40.0),
    ("40-60", 40.0, 60.0),
    ("60-80", 60.0, 80.0),
    ("80-100", 80.0, 100.01),
)

# risk_pct by TQ bucket schedule (name -> list of (lo, hi, risk))
RISK_SCHEDULES: tuple[tuple[str, tuple[tuple[float, float, float], ...]], ...] = (
    (
        "tq_skip30_step",
        ((0, 30, 0.0), (30, 50, 0.005), (50, 70, 0.01), (70, 85, 0.015), (85, 100.01, 0.02)),
    ),
    (
        "tq_skip40_flat1",
        ((0, 40, 0.0), (40, 100.01, 0.01)),
    ),
    (
        "tq_skip40_flat05",
        ((0, 40, 0.0), (40, 100.01, 0.005)),
    ),
    (
        "tq_skip40_step_05_10_20",
        ((0, 40, 0.0), (40, 60, 0.005), (60, 80, 0.01), (80, 100.01, 0.02)),
    ),
    (
        "tq_skip50_step",
        ((0, 50, 0.0), (50, 70, 0.005), (70, 85, 0.01), (85, 100.01, 0.02)),
    ),
    (
        "conf_skip40_flat1",
        ((0, 40, 0.0), (40, 100.01, 0.01)),
    ),
    (
        "always_1pct",
        ((0, 100.01, 0.01),),
    ),
    (
        "tq_aggressive_skip30",
        ((0, 30, 0.0), (30, 50, 0.01), (50, 70, 0.015), (70, 85, 0.02), (85, 100.01, 0.025)),
    ),
    (
        "tq_conservative_skip50",
        ((0, 50, 0.0), (50, 70, 0.005), (70, 100.01, 0.01)),
    ),
)
