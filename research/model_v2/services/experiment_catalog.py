"""Long/Short v2 context feature sets (from Sprint 12)."""

from __future__ import annotations

from research.model_v2.entities import SideContextSpec
from research.structure_selection.entities.selection_result import SelectionExperiment

LONG_STRUCTURE_CONTEXT: tuple[str, ...] = (
    "ctx_h4_distance_from_equilibrium",
    "ctx_h4_swing_strength",
    "ctx_h4_swing_quality",
    "ctx_h4_swing_high_distance_atr",
    "ctx_h4_rejection_strength",
)

SHORT_STRUCTURE_CONTEXT: tuple[str, ...] = ("ctx_h4_swing_quality",)

SIDE_SPECS: tuple[SideContextSpec, ...] = (
    SideContextSpec(
        side="long",
        context_features=LONG_STRUCTURE_CONTEXT,
        label="Long Structure Context (Sprint-12 top-5)",
    ),
    SideContextSpec(
        side="short",
        context_features=SHORT_STRUCTURE_CONTEXT,
        label="Short swing_quality only",
    ),
)


def resolve_side_experiments(
    h1_features: list[str],
    *,
    side: str,
    context_features: tuple[str, ...],
) -> list[SelectionExperiment]:
    h1 = tuple(h1_features)
    return [
        SelectionExperiment(
            experiment_id="A_baseline",
            name="A: H1 Features Only",
            description=f"{side}: H1 engineered features only.",
            feature_names=h1,
        ),
        SelectionExperiment(
            experiment_id="B_v2_context",
            name="B: H1 + v2 Context",
            description=f"{side}: H1 plus validated structure context.",
            feature_names=h1 + context_features,
        ),
    ]
