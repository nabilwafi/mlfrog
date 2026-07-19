"""H4 context feature builder — regime / structure signals (not entry signals)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from feature_engineering.builders._transforms import atr, ema, rolling_percentile, safe_div, slope
from market_context.entities.context_feature import ContextFeatureSpec


class H4ContextBuilder:
    """Deterministic H4 context 'model' — trend / vol / regime / swing quality."""

    PREFIX = "ctx_h4_"

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = dict(params or {})

    def build(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, list[ContextFeatureSpec]]:
        if frame.empty:
            raise ValueError("empty H4 frame")
        required = {"timestamp", "open", "high", "low", "close"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"H4 frame missing columns: {sorted(missing)}")

        df = frame.sort_values("timestamp").reset_index(drop=True).copy()
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)

        ema_fast_n = int(self.params.get("ema_fast", 20))
        ema_slow_n = int(self.params.get("ema_slow", 50))
        atr_n = int(self.params.get("atr_period", 14))
        vol_window = int(self.params.get("vol_percentile_window", 50))
        bb_window = int(self.params.get("bb_window", 20))
        swing_window = int(self.params.get("swing_window", 5))
        slope_lb = int(self.params.get("slope_lookback", 3))

        ema_fast = ema(close, ema_fast_n)
        ema_slow = ema(close, ema_slow_n)
        atr14 = atr(high, low, close, atr_n)
        range_ = (high - low).astype(float)

        # --- Trend Direction: -1 bear / 0 flat / +1 bull (continuous soft sign) ---
        cross = safe_div(ema_fast - ema_slow, atr14)
        trend_direction = np.tanh(cross.fillna(0.0) * 2.0)

        # --- Trend Strength: |distance to slow EMA| in ATR units, capped ---
        trend_strength = safe_div((close - ema_slow).abs(), atr14).clip(upper=5.0) / 5.0

        # --- Volatility Regime: ATR percentile in [0, 1] ---
        volatility_regime = rolling_percentile(atr14, vol_window)

        # --- Compression / Expansion: BB width vs its own history ---
        mid = close.rolling(bb_window, min_periods=max(5, bb_window // 5)).mean()
        std = close.rolling(bb_window, min_periods=max(5, bb_window // 5)).std(ddof=0)
        bb_width = safe_div(2.0 * std, mid)
        width_pct = rolling_percentile(bb_width, vol_window)
        compression = (1.0 - width_pct).clip(0.0, 1.0)
        # Expansion = recent ATR growth rate (positive when expanding)
        atr_slope = slope(atr14, slope_lb)
        expansion = ((atr_slope + 1.0) / 2.0).clip(0.0, 1.0)

        # --- Market Regime: trending vs ranging in [-1, 1] ---
        # High |trend| + low compression → trend; high compression → range
        market_regime = (trend_strength * trend_direction.abs() * (1.0 - compression)).clip(0.0, 1.0)
        market_regime = market_regime * np.sign(trend_direction.replace(0.0, 1.0))

        # --- Swing Quality: clarity of local swing structure ---
        roll_high = high.rolling(swing_window, min_periods=swing_window).max()
        roll_low = low.rolling(swing_window, min_periods=swing_window).min()
        swing_span = safe_div(roll_high - roll_low, atr14)
        # Prefer swings that are not tiny noise and not chaotic vs ATR
        body = (close - df["open"].astype(float)).abs()
        body_share = safe_div(body, range_.replace(0.0, np.nan))
        swing_quality = (0.5 * swing_span.clip(upper=3.0) / 3.0 + 0.5 * body_share.clip(0.0, 1.0)).clip(
            0.0, 1.0
        )

        p = self.PREFIX
        out = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(df["timestamp"], utc=True),
                f"{p}trend_direction": trend_direction.astype(float),
                f"{p}trend_strength": trend_strength.astype(float),
                f"{p}volatility_regime": volatility_regime.astype(float),
                f"{p}market_regime": market_regime.astype(float),
                f"{p}compression": compression.astype(float),
                f"{p}expansion": expansion.astype(float),
                f"{p}swing_quality": swing_quality.astype(float),
            }
        )

        specs = [
            ContextFeatureSpec(
                name=f"{p}trend_direction",
                category="trend_direction",
                description="Soft H4 trend direction from EMA fast/slow vs ATR (tanh).",
                value_range="[-1, 1]",
                depends_on=("ema_fast", "ema_slow", "atr"),
            ),
            ContextFeatureSpec(
                name=f"{p}trend_strength",
                category="trend_strength",
                description="Normalized |close - EMA slow| / ATR (capped).",
                value_range="[0, 1]",
                depends_on=("close", "ema_slow", "atr"),
            ),
            ContextFeatureSpec(
                name=f"{p}volatility_regime",
                category="volatility_regime",
                description="Rolling ATR percentile (low→high vol).",
                value_range="[0, 1]",
                depends_on=("atr",),
            ),
            ContextFeatureSpec(
                name=f"{p}market_regime",
                category="market_regime",
                description="Signed trending-vs-ranging score (strength x direction x anti-compression).",
                value_range="[-1, 1]",
                depends_on=("trend_strength", "trend_direction", "compression"),
            ),
            ContextFeatureSpec(
                name=f"{p}compression",
                category="compression",
                description="1 - BB-width percentile (high = squeezed).",
                value_range="[0, 1]",
                depends_on=("bb_width",),
            ),
            ContextFeatureSpec(
                name=f"{p}expansion",
                category="expansion",
                description="Normalized ATR slope (high = expanding volatility).",
                value_range="[0, 1]",
                depends_on=("atr",),
            ),
            ContextFeatureSpec(
                name=f"{p}swing_quality",
                category="swing_quality",
                description="Local swing span clarity vs ATR plus body share of range.",
                value_range="[0, 1]",
                depends_on=("high", "low", "atr", "open", "close"),
            ),
        ]
        return out, specs
