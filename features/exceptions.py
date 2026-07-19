"""Feature-engineering exceptions."""

from __future__ import annotations


class FeatureError(Exception):
    """Base error for the feature layer."""


class IndicatorError(FeatureError):
    """Indicator calculation failed."""


class FeatureValidationError(FeatureError):
    """FeatureSet failed validation."""


class FeatureRepositoryError(FeatureError):
    """Feature persistence failure."""


class RegistryError(FeatureError):
    """Unknown or duplicate indicator registration."""
