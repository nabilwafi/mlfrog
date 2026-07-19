"""Market-data domain exceptions."""

from __future__ import annotations


class MarketDataError(Exception):
    """Base error for the market-data layer."""


class ProviderConnectionError(MarketDataError):
    """Provider cannot connect or lost connectivity."""


class ValidationError(MarketDataError):
    """Candle / MarketData failed validation."""


class RepositoryError(MarketDataError):
    """Persistence load/save failure."""
