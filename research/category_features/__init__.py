"""Category feature sets for ablation vs H1 native research baseline."""

from __future__ import annotations

from typing import Any

import pandas as pd

from data.entities.market_data import MarketData
from feature_engineering.services.feature_engineering_service import FeatureEngineeringService, market_to_frame
from research.mtf_h4_h1_m5.h1_features import H1_NATIVE

# Existing H1 feature engine (6 builders, pre-extension count).
H1_ENGINE_V1: tuple[str, ...] = (
    "ema20_distance_atr", "ema50_distance_atr", "ema20_distance_percent", "ema50_distance_percent",
    "ema_cross_distance", "ema20_slope", "ema50_slope", "ema_trend_duration", "ema_alignment_score",
    "atr_percent", "atr_percentile_252", "rolling_volatility", "volatility_rank", "volatility_regime_score",
    "macd_normalized", "macd_histogram_zscore", "macd_signal_distance", "momentum_rank", "rsi_percentile",
    "body_percent", "upper_wick_percent", "lower_wick_percent", "close_position", "range_percent", "body_rank",
    "hour_sin", "hour_cos", "day_sin", "day_cos",
    "rolling_zscore", "rolling_percentile", "rolling_rank", "rolling_quantile", "rolling_std", "rolling_mean_distance",
)

NEW_CATEGORY_FEATURES: tuple[str, ...] = (
    "return_1_atr", "return_5_atr", "return_20_atr",
    "volume_zscore", "volume_ratio",
    "spread_zscore", "spread_ratio", "spread_bps",
    "price_zscore", "bb_position",
    "roc_3", "roc_12", "momentum_acceleration",
)

LIBRARY_MOMENTUM: tuple[str, ...] = (
    "macd_normalized", "macd_histogram_zscore", "macd_signal_distance", "momentum_rank", "rsi_percentile",
)
LIBRARY_CANDLE: tuple[str, ...] = (
    "body_percent", "upper_wick_percent", "lower_wick_percent", "close_position", "range_percent", "body_rank",
)
LIBRARY_TREND: tuple[str, ...] = (
    "ema20_distance_atr", "ema50_distance_atr", "ema_cross_distance", "ema20_slope", "ema50_slope", "ema_alignment_score",
)
LIBRARY_VOL: tuple[str, ...] = (
    "rolling_volatility", "volatility_rank", "volatility_regime_score",
)

CATEGORY_MAP: dict[str, tuple[str, ...]] = {
    "price_return_new": ("return_1_atr", "return_5_atr", "return_20_atr"),
    "volume_new": ("volume_zscore", "volume_ratio"),
    "liquidity_new": ("spread_zscore", "spread_ratio", "spread_bps"),
    "mean_reversion_new": ("price_zscore", "bb_position"),
    "momentum_new": ("roc_3", "roc_12", "momentum_acceleration"),
    "momentum_lib": LIBRARY_MOMENTUM,
    "candle_lib": LIBRARY_CANDLE,
    "trend_lib": LIBRARY_TREND,
    "vol_lib": LIBRARY_VOL,
}

UNAVAILABLE_CATEGORIES: tuple[str, ...] = (
    "order_flow", "cross_asset", "funding_derivatives",
)


def compute_extended_features(h1: pd.DataFrame) -> pd.DataFrame:
    """Run full feature engine including new category builders on H1 OHLCV."""
    from data.entities.candle import Candle
    from data.entities.market_data import MarketData

    frame = h1.sort_values("timestamp").reset_index(drop=True)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    candles = tuple(
        Candle(
            timestamp=row.timestamp.to_pydatetime() if hasattr(row.timestamp, "to_pydatetime") else row.timestamp,
            open=float(row.open),
            high=float(row.high),
            low=float(row.low),
            close=float(row.close),
            tick_volume=float(getattr(row, "tick_volume", 0) or 0),
            spread=float(getattr(row, "spread", 0) or 0),
            real_volume=float(getattr(row, "real_volume", 0) or 0),
        )
        for row in frame.itertuples(index=False)
    )
    md = MarketData(symbol="XAUUSD", timeframe="H1", timezone="UTC", candles=candles)
    cfg: dict[str, Any] = {
        "builders": [
            "trend", "volatility", "momentum", "candle", "session", "statistical",
            "return", "volume", "liquidity", "mean_reversion",
        ],
        "drop_na": False,
    }
    matrix, _, _ = FeatureEngineeringService(cfg, output_root="artifacts/features").run(md)
    return matrix


def attach_extended_features(df: pd.DataFrame, ext: pd.DataFrame) -> pd.DataFrame:
    ext = ext.copy()
    ext["timestamp"] = pd.to_datetime(ext["timestamp"], utc=True)
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    new_cols = [c for c in ext.columns if c not in out.columns and c != "timestamp"]
    if not new_cols:
        return out
    return out.merge(ext[["timestamp", *new_cols]], on="timestamp", how="left")
