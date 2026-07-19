"""Experiment catalog A–E for structure feature selection."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from research.structure_selection.entities.selection_result import SelectionExperiment

SWING_QUALITY = "ctx_h4_swing_quality"
DISTANCE_EQ = "ctx_h4_distance_from_equilibrium"

# Fallback top set from Sprint-11 mean gain importance
DEFAULT_TOP_STRUCTURE: tuple[str, ...] = (
    DISTANCE_EQ,
    "ctx_h4_swing_strength",
    SWING_QUALITY,
    "ctx_h4_swing_high_distance_atr",
    "ctx_h4_rejection_strength",
)


def load_top_structure_features(
    feature_statistics_path: Path | None,
    *,
    top_k: int = 5,
) -> tuple[str, ...]:
    if feature_statistics_path is None or not feature_statistics_path.is_file():
        return DEFAULT_TOP_STRUCTURE[:top_k]
    stats = pd.read_csv(feature_statistics_path)
    if "feature" not in stats.columns:
        return DEFAULT_TOP_STRUCTURE[:top_k]
    imp_cols = [c for c in stats.columns if c.startswith("importance_")]
    if not imp_cols:
        return DEFAULT_TOP_STRUCTURE[:top_k]
    ranked = stats.copy()
    ranked["_avg"] = ranked[imp_cols].mean(axis=1)
    ranked = ranked.sort_values("_avg", ascending=False)
    names = [str(x) for x in ranked["feature"].head(top_k).tolist()]
    return tuple(names) if names else DEFAULT_TOP_STRUCTURE[:top_k]


def resolve_experiments(
    h1_features: list[str],
    *,
    top_structure: tuple[str, ...],
) -> list[SelectionExperiment]:
    h1 = tuple(h1_features)
    return [
        SelectionExperiment(
            experiment_id="A_baseline",
            name="A: H1 only",
            description="H1 engineered features only.",
            feature_names=h1,
        ),
        SelectionExperiment(
            experiment_id="B_swing_quality",
            name="B: H1 + swing_quality",
            description="H1 plus ctx_h4_swing_quality.",
            feature_names=h1 + (SWING_QUALITY,),
        ),
        SelectionExperiment(
            experiment_id="C_distance_eq",
            name="C: H1 + distance_from_equilibrium",
            description="H1 plus ctx_h4_distance_from_equilibrium.",
            feature_names=h1 + (DISTANCE_EQ,),
        ),
        SelectionExperiment(
            experiment_id="D_swing_plus_eq",
            name="D: H1 + swing_quality + distance_eq",
            description="H1 plus swing_quality and distance_from_equilibrium.",
            feature_names=h1 + (SWING_QUALITY, DISTANCE_EQ),
        ),
        SelectionExperiment(
            experiment_id="E_top_structure",
            name="E: H1 + top structure",
            description=f"H1 plus top-{len(top_structure)} Sprint-11 importance features.",
            feature_names=h1 + top_structure,
        ),
    ]
