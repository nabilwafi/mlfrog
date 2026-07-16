"""Exceptions for feature engineering."""

from __future__ import annotations


class FeatureEngineeringError(Exception):
    """Base error."""


class BuilderError(FeatureEngineeringError):
    """Builder failed."""


class RegistryError(FeatureEngineeringError):
    """Registry failure."""
