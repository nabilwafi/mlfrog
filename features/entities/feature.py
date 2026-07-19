"""Single feature column."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class Feature:
    name: str
    dtype: str
    values: tuple[float, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Feature.name must be non-empty")
        if not self.dtype.strip():
            raise ValueError("Feature.dtype must be non-empty")
        object.__setattr__(self, "metadata", dict(self.metadata))
