"""Diagnostics-layer exceptions."""

from __future__ import annotations


class DiagnosticsError(Exception):
    """Base error for the diagnostics layer."""


class DiagnosticsRepositoryError(DiagnosticsError):
    """Persistence failure for diagnostics artifacts."""


class DiagnosticsInputError(DiagnosticsError):
    """Missing or invalid training/dataset inputs."""
