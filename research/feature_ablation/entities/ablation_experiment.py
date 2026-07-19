"""Feature ablation experiment definition."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ExperimentKind = Literal[
    "baseline",
    "remove_category",
    "stable_only",
    "keep_only",
    "top_shap",
    "top_mi",
]


@dataclass(frozen=True, slots=True)
class AblationExperiment:
    experiment_id: str
    name: str
    kind: ExperimentKind
    description: str
    remove_category: str | None = None
    top_k: int | None = None
    feature_names: tuple[str, ...] = field(default_factory=tuple)

    def with_features(self, names: list[str] | tuple[str, ...]) -> AblationExperiment:
        return AblationExperiment(
            experiment_id=self.experiment_id,
            name=self.name,
            kind=self.kind,
            description=self.description,
            remove_category=self.remove_category,
            top_k=self.top_k,
            feature_names=tuple(names),
        )
