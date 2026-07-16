"""Dataset-layer exceptions."""

from __future__ import annotations


class DatasetError(Exception):
    """Base error for the dataset layer."""


class DatasetValidationError(DatasetError):
    """Dataset failed validation."""


class DatasetRepositoryError(DatasetError):
    """Dataset persistence failure."""


class AlignmentError(DatasetError):
    """Feature/label timestamp alignment failed."""
