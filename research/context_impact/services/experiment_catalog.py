"""Experiment catalog — baseline vs context subsets (no new features)."""

from __future__ import annotations

from research.context_impact.entities.experiment import ContextImpactExperiment

# Fixed mapping from Sprint-9 context feature categories.
TREND_CONTEXT: tuple[str, ...] = (
    "ctx_h4_trend_direction",
    "ctx_h4_trend_strength",
    "ctx_h4_market_regime",
)
VOLATILITY_CONTEXT: tuple[str, ...] = (
    "ctx_h4_volatility_regime",
    "ctx_h4_compression",
    "ctx_h4_expansion",
)
STRUCTURE_CONTEXT: tuple[str, ...] = ("ctx_h4_swing_quality",)
ALL_CONTEXT: tuple[str, ...] = TREND_CONTEXT + VOLATILITY_CONTEXT + STRUCTURE_CONTEXT


def resolve_experiments(h1_features: list[str]) -> list[ContextImpactExperiment]:
    """Build the five experiments: baseline + 3 category ablations + all context."""
    h1 = tuple(h1_features)
    catalog = [
        ContextImpactExperiment(
            experiment_id="baseline",
            name="H1 Features Only",
            description="Sprint-6 H1 engineered features + LightGBM (no H4 context).",
            context_category="none",
            feature_names=h1,
        ),
        ContextImpactExperiment(
            experiment_id="trend_context_only",
            name="H1 + Trend Context",
            description="H1 features plus H4 trend_direction / trend_strength / market_regime.",
            context_category="trend",
            feature_names=h1 + TREND_CONTEXT,
        ),
        ContextImpactExperiment(
            experiment_id="volatility_context_only",
            name="H1 + Volatility Context",
            description="H1 features plus H4 volatility_regime / compression / expansion.",
            context_category="volatility",
            feature_names=h1 + VOLATILITY_CONTEXT,
        ),
        ContextImpactExperiment(
            experiment_id="structure_context_only",
            name="H1 + Structure Context",
            description="H1 features plus H4 swing_quality.",
            context_category="structure",
            feature_names=h1 + STRUCTURE_CONTEXT,
        ),
        ContextImpactExperiment(
            experiment_id="all_context",
            name="H1 + All H4 Context",
            description="H1 features plus all seven H4 context features.",
            context_category="all",
            feature_names=h1 + ALL_CONTEXT,
        ),
    ]
    return catalog
