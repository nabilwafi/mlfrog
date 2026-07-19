"""Model-training exceptions."""

from __future__ import annotations


class ModelError(Exception):
    """Base error for the model training layer."""


class TrainerError(ModelError):
    """Trainer failed."""


class TrainingValidationError(ModelError):
    """Training inputs failed validation."""


class ModelRepositoryError(ModelError):
    """Model persistence failure."""


class RegistryError(ModelError):
    """Unknown or duplicate trainer registration."""
