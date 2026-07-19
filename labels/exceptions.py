"""Label-engineering exceptions."""

from __future__ import annotations


class LabelError(Exception):
    """Base error for the label layer."""


class StrategyError(LabelError):
    """Label strategy failed."""


class LabelValidationError(LabelError):
    """LabelSet failed validation."""


class LabelRepositoryError(LabelError):
    """Label persistence failure."""


class RegistryError(LabelError):
    """Unknown or duplicate strategy registration."""
