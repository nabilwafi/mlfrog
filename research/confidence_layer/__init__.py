"""Sprint 22 — Multi-timeframe confidence layer (frozen Primary + Meta)."""

META_GATE: float = 0.45  # only evaluate candidates Meta would take (production)
STARTING_EQUITY: float = 80.0

CONFIDENCE_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("0-20", 0.0, 20.0),
    ("20-40", 20.0, 40.0),
    ("40-60", 40.0, 60.0),
    ("60-80", 60.0, 80.0),
    ("80-100", 80.0, 100.01),
)

# Dynamic risk schedules to search (confidence score 0-100)
RISK_SCHEDULES: tuple[tuple[str, tuple[tuple[float, float, float], ...]], ...] = (
    # name, ((lo, hi, risk_pct), ...)  hi exclusive-ish
    (
        "skip40_r05_10_20",
        ((0, 40, 0.0), (40, 60, 0.005), (60, 80, 0.01), (80, 100.01, 0.02)),
    ),
    (
        "skip50_r05_10_20",
        ((0, 50, 0.0), (50, 70, 0.005), (70, 85, 0.01), (85, 100.01, 0.02)),
    ),
    (
        "skip40_flat1",
        ((0, 40, 0.0), (40, 100.01, 0.01)),
    ),
    (
        "skip40_flat05",
        ((0, 40, 0.0), (40, 100.01, 0.005)),
    ),
    (
        "skip60_r10_20",
        ((0, 60, 0.0), (60, 80, 0.01), (80, 100.01, 0.02)),
    ),
    (
        "always_1pct",
        ((0, 100.01, 0.01),),
    ),
    (
        "skip40_r05_10_15",
        ((0, 40, 0.0), (40, 60, 0.005), (60, 80, 0.01), (80, 100.01, 0.015)),
    ),
)
