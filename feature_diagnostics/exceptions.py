"""Feature-diagnostics exceptions."""

from __future__ import annotations


class FeatureDiagnosticsError(Exception):
    """Base error."""


class FeatureDiagnosticsInputError(FeatureDiagnosticsError):
    """Missing features/labels."""
