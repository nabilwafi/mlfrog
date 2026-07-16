"""M15 research-only context features (not for primary model).

Causal bar transforms; H1 join must use ContextJoinService (available_at).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from feature_engineering.builders._transforms import atr, safe_div


class M15ContextBuilder:
    """Lean M15 structure/momentum pack for meta-feature research."""

    PREFIX = "ctx_m15_"

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = dict(params or {})

    def build(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            raise ValueError("empty M15 frame")
        required = {"timestamp", "open", "high", "low", "close"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"M15 frame missing: {sorted(missing)}")

        df = frame.sort_values("timestamp").reset_index(drop=True).copy()
        o = df["open"].astype(float)
        h = df["high"].astype(float)
        l = df["low"].astype(float)
        c = df["close"].astype(float)
        vol = (
            df["tick_volume"].astype(float)
            if "tick_volume" in df.columns
            else pd.Series(np.nan, index=df.index)
        )

        atr_n = int(self.params.get("atr_period", 14))
        swing_n = int(self.params.get("swing_window", 5))
        range_n = int(self.params.get("range_window", 20))
        mom_n = int(self.params.get("momentum_window", 8))
        atr14 = atr(h, l, c, atr_n)
        bar_range = (h - l).astype(float)

        ema_fast = c.ewm(span=8, adjust=False).mean()
        ema_slow = c.ewm(span=21, adjust=False).mean()
        ema_alignment = safe_div(ema_fast - ema_slow, atr14)

        swing_high = h.rolling(swing_n, min_periods=swing_n).max()
        swing_low = l.rolling(swing_n, min_periods=swing_n).min()
        swing_span = safe_div(swing_high - swing_low, atr14)
        body = (c - o).abs()
        body_share = safe_div(body, bar_range.replace(0.0, np.nan))
        swing_quality = (
            0.5 * swing_span.clip(upper=3.0) / 3.0 + 0.5 * body_share.clip(0.0, 1.0)
        ).clip(0.0, 1.0)

        range_high = h.rolling(range_n, min_periods=max(5, range_n // 5)).max()
        range_low = l.rolling(range_n, min_periods=max(5, range_n // 5)).min()
        range_mid = (range_high + range_low) / 2.0
        distance_equilibrium = safe_div(c - range_mid, atr14)

        upper_wick = h - np.maximum(o, c)
        lower_wick = np.minimum(o, c) - l
        rejection_strength = safe_div(
            np.maximum(upper_wick, lower_wick), bar_range.replace(0.0, np.nan)
        ).clip(0.0, 1.0)

        momentum = safe_div(c - c.shift(mom_n), atr14)
        # Trend strength: |EMA alignment| clipped
        trend_strength = ema_alignment.abs().clip(upper=3.0) / 3.0
        # Breakout: close vs prior range high/low
        prev_hi = range_high.shift(1)
        prev_lo = range_low.shift(1)
        breakout_strength = np.where(
            c > prev_hi,
            safe_div(c - prev_hi, atr14).to_numpy(),
            np.where(
                c < prev_lo,
                safe_div(prev_lo - c, atr14).to_numpy(),
                0.0,
            ),
        )
        breakout_strength = pd.Series(breakout_strength, index=df.index, dtype=float)

        vol_ma = vol.rolling(20, min_periods=5).mean()
        volume_spike = safe_div(vol, vol_ma.replace(0.0, np.nan)).clip(upper=5.0)

        p = self.PREFIX
        return pd.DataFrame(
            {
                "timestamp": pd.to_datetime(df["timestamp"], utc=True),
                f"{p}atr_percent": safe_div(atr14, c).astype(float),
                f"{p}trend_strength": trend_strength.astype(float),
                f"{p}momentum": momentum.astype(float),
                f"{p}ema_alignment": ema_alignment.astype(float),
                f"{p}swing_quality": swing_quality.astype(float),
                f"{p}rejection_strength": rejection_strength.astype(float),
                f"{p}distance_equilibrium": distance_equilibrium.astype(float),
                f"{p}breakout_strength": breakout_strength.astype(float),
                f"{p}volume_spike": volume_spike.astype(float),
            }
        )
