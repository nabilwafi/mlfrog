"""Context feature metadata entity."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ContextFeatureSpec:
    name: str
    category: str
    description: str
    value_range: str
    depends_on: tuple[str, ...] = ()
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "value_range": self.value_range,
            "depends_on": list(self.depends_on),
            **({"extras": self.extras} if self.extras else {}),
        }
