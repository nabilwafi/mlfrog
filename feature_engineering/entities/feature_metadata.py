"""Feature metadata — required for every engineered column."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class FeatureMetadata:
    name: str
    category: str
    stationary: bool
    normalized: bool
    drift_sensitive: bool
    depends_on: tuple[str, ...]
    description: str
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("feature name must be non-empty")
        if not self.category.strip():
            raise ValueError("category must be non-empty")
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "category", self.category.strip().lower())
        object.__setattr__(self, "depends_on", tuple(self.depends_on))
        object.__setattr__(self, "description", self.description.strip())
        object.__setattr__(self, "extras", dict(self.extras))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "stationary": self.stationary,
            "normalized": self.normalized,
            "drift_sensitive": self.drift_sensitive,
            "depends_on": list(self.depends_on),
            "description": self.description,
            **({"extras": self.extras} if self.extras else {}),
        }
