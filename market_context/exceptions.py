"""Market context layer exceptions."""


class MarketContextError(Exception):
    """Base error for the market context pipeline."""


class ContextInputError(MarketContextError):
    """Missing or invalid inputs (raw OHLCV, H1 frame, etc.)."""


class ContextJoinError(MarketContextError):
    """Causal asof-join failed."""
