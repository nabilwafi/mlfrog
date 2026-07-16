"""Model v2 validation experiment entities."""

from __future__ import annotations

from dataclasses import dataclass

from research.structure_selection.entities.selection_result import (
    ExperimentResult,
    SelectionExperiment,
    WindowMetrics,
)

__all__ = ["ExperimentResult", "SelectionExperiment", "WindowMetrics", "SideContextSpec"]


@dataclass(frozen=True, slots=True)
class SideContextSpec:
    side: str
    context_features: tuple[str, ...]
    label: str
