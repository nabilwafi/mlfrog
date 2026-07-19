"""Single engineered feature (series + metadata)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from feature_engineering.entities.feature_metadata import FeatureMetadata


@dataclass(frozen=True, slots=True)
class Feature:
    metadata: FeatureMetadata
    values: pd.Series

    def __post_init__(self) -> None:
        if self.values.name is None:
            object.__setattr__(self, "values", self.values.rename(self.metadata.name))
        elif str(self.values.name) != self.metadata.name:
            object.__setattr__(
                self, "values", self.values.rename(self.metadata.name)
            )
