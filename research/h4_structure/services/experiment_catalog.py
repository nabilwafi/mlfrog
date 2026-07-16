"""Ablation catalog for H4 structure enhancement."""

from __future__ import annotations

from market_context.builders.h4_structure_builder import (
    ALL_STRUCTURE_FEATURES,
    NEW_STRUCTURE_FEATURES,
    SWING_QUALITY_ONLY,
)
from research.h4_structure.entities.experiment_result import StructureExperiment


def resolve_experiments(h1_features: list[str]) -> list[StructureExperiment]:
    h1 = tuple(h1_features)
    return [
        StructureExperiment(
            experiment_id="baseline",
            name="H1 Features Only",
            description="Sprint-6 H1 features, no H4 structure.",
            feature_names=h1,
        ),
        StructureExperiment(
            experiment_id="swing_quality_only",
            name="H1 + swing_quality",
            description="H1 plus legacy ctx_h4_swing_quality.",
            feature_names=h1 + SWING_QUALITY_ONLY,
        ),
        StructureExperiment(
            experiment_id="new_structure",
            name="H1 + New Structure",
            description="H1 plus Sprint-11 structure features (excluding swing_quality).",
            feature_names=h1 + NEW_STRUCTURE_FEATURES,
        ),
        StructureExperiment(
            experiment_id="all_structure",
            name="H1 + All Structure",
            description="H1 plus swing_quality and all new structure features.",
            feature_names=h1 + ALL_STRUCTURE_FEATURES,
        ),
    ]
