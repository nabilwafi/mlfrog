from __future__ import annotations

"""
Feature engineering for XAUUSD using:
- H1 as base (entry timeframe)
- H4 + D1 as higher timeframe context

No-lookahead guarantee:
We interpret the input CSV `Date` as the bar OPEN time (as MT5 provides).
For feature availability at decision time, we shift each timeframe's `Date`
to the bar CLOSE time:
- H1: Date + 1 hour
- H4: Date + 4 hours
- D1: Date + 1 day

Then we broadcast H4/D1 features to H1 using merge_asof with direction='backward'
on these CLOSE times. This ensures the H4/D1 candle used for a given H1 row
is always CLOSED before the H1 CLOSE time.
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator, EMAIndicator
from ta.volatility import AverageTrueRange


@dataclass(frozen=True)
class FeatureConfig:
    # Base indicators on H1
    adx_period: int = 14
    atr_period: int = 14
    rsi_period: int = 14

    ema_fast: int = 12
    ema_slow: int = 26
    ema_cross_lookback_bars: int = 3  # N default

    # Higher timeframe indicators
    h4_ema_period: int = 20
    d1_ema_period: int = 20

    # Slope of EMA(period) over N bars (linear regression slope, normalized)
    slope_window_bars: int = 10

    # trend_direction_d1 thresholds (normalized slope)
    trend_slope_threshold: float = 1e-4


def _require_columns(df: pd.DataFrame, cols: Iterable[str], df_name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} missing columns: {missing}")


def load_ohlc_csv(path: Path) -> pd.DataFrame:
    """Load historical OHLC CSV with Date column formatted in UTC.

    Expected columns:
    Date, Open, High, Low, Close, Tick Volume, Volume, Spread
    """
    df = pd.read_csv(path, parse_dates=["Date"])
    _require_columns(
        df,
        ["Date", "Open", "High", "Low", "Close", "Tick Volume", "Volume", "Spread"],
        df_name=str(path),
    )

    # Ensure timezone awareness (CSV already includes +00:00, but be defensive).
    if getattr(df["Date"].dtype, "tz", None) is None:
        df["Date"] = pd.to_datetime(df["Date"], utc=True)

    df = df.sort_values("Date").reset_index(drop=True)
    return df


def _assign_session_utc(date_utc: pd.Series) -> pd.Series:
    """Assign session label based on hour in UTC.

    Windows (common reference; configurable in code comment):
    - asia: 00:00-08:00 UTC
    - london: 08:00-13:00 UTC
    - london_ny_overlap: 13:00-16:00 UTC
    - ny: 16:00-21:00 UTC
    Remaining 21:00-24:00 is mapped to 'asia' (wrap-around).
    """
    hour = date_utc.dt.hour
    out = np.empty(len(hour), dtype=object)

    # Define with explicit boundaries (start inclusive, end exclusive).
    out[(hour >= 0) & (hour < 8)] = "asia"
    out[(hour >= 8) & (hour < 13)] = "london"
    out[(hour >= 13) & (hour < 16)] = "london_ny_overlap"
    out[(hour >= 16) & (hour < 21)] = "ny"
    out[(hour >= 21) & (hour <= 23)] = "asia"
    return pd.Series(out, index=date_utc.index)


def _rolling_linear_regression_slope(series: pd.Series, window: int) -> pd.Series:
    """Compute rolling linear regression slope for a series.

    Uses y = series values, x = 0..window-1. Returns slope per step.
    """

    x = np.arange(window, dtype=float)

    def _slope(y: np.ndarray) -> float:
        # y length == window due to rolling.
        if np.any(np.isnan(y)):
            return np.nan
        slope, _intercept = np.polyfit(x, y.astype(float), 1)
        return float(slope)

    return series.rolling(window=window, min_periods=window).apply(_slope, raw=True)


def compute_h1_features(df_h1: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """Compute H1 base features.

    Output columns (requested):
    adx_h1, atr_h1, rsi_h1,
    ema_fast_h1, ema_slow_h1,
    ema_cross_signal_h1,
    session,
    return_1h, return_4h
    """
    _require_columns(df_h1, ["Date", "High", "Low", "Close"], "df_h1")

    df = df_h1.copy()

    adx = ADXIndicator(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        window=cfg.adx_period,
        fillna=False,
    ).adx()
    atr = AverageTrueRange(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        window=cfg.atr_period,
        fillna=False,
    ).average_true_range()
    rsi = RSIIndicator(close=df["Close"], window=cfg.rsi_period, fillna=False).rsi()

    ema_fast = EMAIndicator(close=df["Close"], window=cfg.ema_fast, fillna=False).ema_indicator()
    ema_slow = EMAIndicator(close=df["Close"], window=cfg.ema_slow, fillna=False).ema_indicator()

    # Cross detection at the exact bar:
    bull_cross = (ema_fast.shift(1) <= ema_slow.shift(1)) & (ema_fast > ema_slow)
    bear_cross = (ema_fast.shift(1) >= ema_slow.shift(1)) & (ema_fast < ema_slow)

    cross_dir = pd.Series(0, index=df.index, dtype=int)
    cross_dir[bull_cross] = 1
    cross_dir[bear_cross] = -1

    # ema_cross_signal_h1: last cross direction within last N bars (0 if none).
    # Conservative definition for "baru saja": take the most recent non-zero cross_dir
    # within the last N bars.
    def _last_non_zero(arr: np.ndarray) -> float:
        # arr length == window; raw=True gives ndarray.
        nz = arr[arr != 0]
        if nz.size == 0:
            return 0.0
        return float(nz[-1])

    ema_cross_signal = (
        cross_dir.rolling(window=cfg.ema_cross_lookback_bars, min_periods=cfg.ema_cross_lookback_bars)
        .apply(_last_non_zero, raw=True)
        .fillna(0.0)
        .astype(int)
    )

    session = _assign_session_utc(df["Date"])
    return_1h = np.log(df["Close"] / df["Close"].shift(1))
    return_4h = np.log(df["Close"] / df["Close"].shift(4))

    out = pd.DataFrame(
        {
            "Date": df["Date"],
            "adx_h1": adx,
            "atr_h1": atr,
            "rsi_h1": rsi,
            "ema_fast_h1": ema_fast,
            "ema_slow_h1": ema_slow,
            "ema_cross_signal_h1": ema_cross_signal,
            "session": session,
            "return_1h": return_1h,
            "return_4h": return_4h,
        }
    )
    return out


def compute_h4_features(df_h4: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """Compute H4 higher timeframe features.

    Output columns:
    adx_h4, atr_h4, ema_slope_h4
    """
    _require_columns(df_h4, ["Date", "High", "Low", "Close"], "df_h4")
    df = df_h4.copy()

    adx = ADXIndicator(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        window=cfg.adx_period,
        fillna=False,
    ).adx()
    atr = AverageTrueRange(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        window=cfg.atr_period,
        fillna=False,
    ).average_true_range()

    ema = EMAIndicator(close=df["Close"], window=cfg.h4_ema_period, fillna=False).ema_indicator()
    raw_slope = _rolling_linear_regression_slope(ema, window=cfg.slope_window_bars)

    # Normalize to make thresholds portable across price levels.
    ema_slope_h4 = raw_slope / ema.replace(0, np.nan)

    out = pd.DataFrame(
        {
            "Date": df["Date"],
            "adx_h4": adx,
            "atr_h4": atr,
            "ema_slope_h4": ema_slope_h4,
        }
    )
    return out


def compute_d1_features(df_d1: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """Compute D1 higher timeframe features.

    Output columns:
    ema_slope_d1, trend_direction_d1
    """
    _require_columns(df_d1, ["Date", "High", "Low", "Close"], "df_d1")
    df = df_d1.copy()

    ema = EMAIndicator(close=df["Close"], window=cfg.d1_ema_period, fillna=False).ema_indicator()
    raw_slope = _rolling_linear_regression_slope(ema, window=cfg.slope_window_bars)
    ema_slope_d1 = raw_slope / ema.replace(0, np.nan)

    thr = cfg.trend_slope_threshold
    trend_direction = pd.Series(index=df.index, dtype=object)
    trend_direction[ema_slope_d1 > thr] = "uptrend"
    trend_direction[ema_slope_d1 < -thr] = "downtrend"
    trend_direction[(ema_slope_d1 >= -thr) & (ema_slope_d1 <= thr)] = "sideways"

    out = pd.DataFrame(
        {
            "Date": df["Date"],
            "ema_slope_d1": ema_slope_d1,
            "trend_direction_d1": trend_direction,
        }
    )
    return out


def merge_multi_tf_features(
    df_h1_feat: pd.DataFrame,
    df_h4_feat: pd.DataFrame,
    df_d1_feat: pd.DataFrame,
    *,
    drop_na: bool = True,
    assert_no_lookahead_samples: int = 8,
    seed: int = 42,
) -> pd.DataFrame:
    """Broadcast H4 and D1 features to H1 rows using no-lookahead merge_asof.

    Assumption (enforced by upstream build):
    - df_*['Date'] is shifted to bar CLOSE time for each timeframe.
    Then merge_asof(direction='backward') selects the last closed higher-TF candle
    not later than each H1 close time.
    """
    _require_columns(df_h1_feat, ["Date"], "df_h1_feat")
    _require_columns(df_h4_feat, ["Date"], "df_h4_feat")
    _require_columns(df_d1_feat, ["Date"], "df_d1_feat")

    left = df_h1_feat.sort_values("Date").reset_index(drop=True)
    right_h4 = df_h4_feat.sort_values("Date").reset_index(drop=True).rename(columns={"Date": "h4_closed_time"})
    right_d1 = df_d1_feat.sort_values("Date").reset_index(drop=True).rename(columns={"Date": "d1_closed_time"})

    merged = pd.merge_asof(
        left,
        right_h4,
        left_on="Date",
        right_on="h4_closed_time",
        direction="backward",
    )
    merged = pd.merge_asof(
        merged,
        right_d1,
        left_on="Date",
        right_on="d1_closed_time",
        direction="backward",
    )

    # Lookahead assertion (debug-time safety check)
    if assert_no_lookahead_samples > 0:
        _assert_no_lookahead(
            merged_df=merged,
            h4_closed_time_series=right_h4["h4_closed_time"],
            d1_closed_time_series=right_d1["d1_closed_time"],
            n_samples=assert_no_lookahead_samples,
            seed=seed,
        )

    if drop_na:
        feat_cols = [
            c
            for c in merged.columns
            if c
            not in {"Date", "h4_closed_time", "d1_closed_time"}
        ]
        merged = merged.dropna(subset=feat_cols).reset_index(drop=True)

    # Drop debug columns in final output
    merged = merged.drop(columns=[c for c in ["h4_closed_time", "d1_closed_time"] if c in merged.columns])
    return merged


def _assert_no_lookahead(
    *,
    merged_df: pd.DataFrame,
    h4_closed_time_series: pd.Series,
    d1_closed_time_series: pd.Series,
    n_samples: int,
    seed: int,
) -> None:
    """Assert and print that the merged H4/D1 source candles were closed before each sampled H1 row."""
    import bisect
    # IMPORTANT:
    # Pandas datetime with tz may use underlying resolution (e.g. us vs ns).
    # `Series.astype("int64")` therefore may yield microseconds, causing false mismatches.
    # We normalize everything to nanoseconds using `pd.Timestamp(x).value`.
    h4_times_ns = np.array([pd.Timestamp(x).value for x in h4_closed_time_series.dropna()], dtype=np.int64)
    d1_times_ns = np.array([pd.Timestamp(x).value for x in d1_closed_time_series.dropna()], dtype=np.int64)
    h4_times_ns.sort()
    d1_times_ns.sort()

    if len(h4_times_ns) == 0 or len(d1_times_ns) == 0:
        raise AssertionError("No H4/D1 closed-time data available for lookahead assertion.")

    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(merged_df), size=min(n_samples, len(merged_df)), replace=False)

    for i in idxs:
        h1_time = merged_df.loc[i, "Date"]
        h4_used = merged_df.loc[i, "h4_closed_time"]
        d1_used = merged_df.loc[i, "d1_closed_time"]

        if pd.isna(h4_used) or pd.isna(d1_used):
            raise AssertionError("Found NaN in merged closed-time columns during lookahead assertion.")

        # Ensure closed_time_used <= h1_time (no future candle usage).
        if h4_used > h1_time:
            raise AssertionError(f"Lookahead: H4 used after H1 close. h4_used={h4_used}, h1_time={h1_time}")
        if d1_used > h1_time:
            raise AssertionError(f"Lookahead: D1 used after H1 close. d1_used={d1_used}, h1_time={h1_time}")

        # Ensure it's the last closed candle <= h1_time.
        h1_ns = pd.Timestamp(h1_time).value
        h4_pos = bisect.bisect_right(h4_times_ns, h1_ns) - 1
        d1_pos = bisect.bisect_right(d1_times_ns, h1_ns) - 1
        expected_h4_ns = h4_times_ns[h4_pos]
        expected_d1_ns = d1_times_ns[d1_pos]

        if expected_h4_ns != pd.Timestamp(h4_used).value:
            raise AssertionError("Lookahead mismatch: H4 source not equal to last closed candle.")
        if expected_d1_ns != pd.Timestamp(d1_used).value:
            raise AssertionError("Lookahead mismatch: D1 source not equal to last closed candle.")

        # Claude-style no-lookahead invariant in OPEN-time terms:
        # H4 valid iff h4_open_time + 4h <= H1_T
        # D1 valid iff d1_open_time + 1d <= H1_T
        h4_open_time = pd.Timestamp(h4_used) - pd.Timedelta(hours=4)
        d1_open_time = pd.Timestamp(d1_used) - pd.Timedelta(days=1)

        print(
            "Lookahead check | H1_T=",
            h1_time,
            "| h4_open=",
            h4_open_time,
            "| h4_close_used=",
            h4_used,
            "| d1_open=",
            d1_open_time,
            "| d1_close_used=",
            d1_used,
        )


def build_feature_table(
    *,
    h1_path: Path,
    h4_path: Path,
    d1_path: Path,
    out_path: Path,
    cfg: FeatureConfig = FeatureConfig(),
    assert_no_lookahead_samples: int = 8,
) -> None:
    """Build and save the H1+H4+D1 feature table (one row per H1 candle)."""
    df_h1 = load_ohlc_csv(h1_path)
    df_h4 = load_ohlc_csv(h4_path)
    df_d1 = load_ohlc_csv(d1_path)

    # Shift Date to bar CLOSE time so merge_asof uses only closed candles.
    df_h1["Date"] = df_h1["Date"] + pd.Timedelta(hours=1)
    df_h4["Date"] = df_h4["Date"] + pd.Timedelta(hours=4)
    df_d1["Date"] = df_d1["Date"] + pd.Timedelta(days=1)

    df_h1_feat = compute_h1_features(df_h1, cfg)
    df_h4_feat = compute_h4_features(df_h4, cfg)
    df_d1_feat = compute_d1_features(df_d1, cfg)

    merged = merge_multi_tf_features(
        df_h1_feat,
        df_h4_feat,
        df_d1_feat,
        assert_no_lookahead_samples=assert_no_lookahead_samples,
    )

    # Final NaN safety check (after warm-up dropping)
    nan_counts = merged.isna().sum()
    nan_total = int(nan_counts.sum())
    if nan_total != 0:
        raise AssertionError(f"Expected 0 NaNs after dropping warm-up, but got: {nan_counts[nan_counts > 0]}")

    start = merged["Date"].min()
    end = merged["Date"].max()
    print(
        f"Feature table saved | rows={len(merged)} | range={start} -> {end} | nan_total={nan_total}"
    )
    print("NaN per column (expected all zeros):")
    print(nan_counts.to_string())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path, index=False)


def main() -> None:
    # Paths: adjust if your raw data filenames differ.
    base = Path("data/raw")
    out = Path("data/features/xauusd_h1_h4_d1_features.parquet")

    build_feature_table(
        h1_path=base / "XAUUSD_H1.csv",
        h4_path=base / "XAUUSD_H4.csv",
        d1_path=base / "XAUUSD_D1.csv",
        out_path=out,
        assert_no_lookahead_samples=8,
    )


if __name__ == "__main__":
    main()

