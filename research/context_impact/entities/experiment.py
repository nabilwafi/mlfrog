"""Context impact experiment definition."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ContextImpactExperiment:
    experiment_id: str
    name: str
    description: str
    context_category: str  # none | trend | volatility | structure | all
    feature_names: tuple[str, ...]

    def with_features(self, names: list[str] | tuple[str, ...]) -> ContextImpactExperiment:
        return ContextImpactExperiment(
            experiment_id=self.experiment_id,
            name=self.name,
            description=self.description,
            context_category=self.context_category,
            feature_names=tuple(names),
        )
