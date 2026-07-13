"""Data preparation for baseline classifiers.

Time-aware split (NOT random):
Horizon-based labels look forward up to HORIZON_BARS bars. Consecutive rows
therefore share overlapping future windows. A random train/test split would
leak future outcomes into training via those overlaps. We split by calendar
time and apply an embargo of HORIZON_BARS hours around each split boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

FEATURE_COLS = [
    "adx_h1",
    "atr_h1",
    "rsi_h1",
    "ema_fast_h1",
    "ema_slow_h1",
    "ema_cross_signal_h1",
    "session",
    "return_1h",
    "return_4h",
    "adx_h4",
    "atr_h4",
    "ema_slope_h4",
    "ema_slope_d1",
    "trend_direction_d1",
]
CAT_COLS = ["session", "trend_direction_d1"]
NUM_COLS = [c for c in FEATURE_COLS if c not in CAT_COLS]

SL_MULT = 1.5
TP_MULT = 2.0
HORIZON_BARS = 8
BREAKEVEN_WINRATE = SL_MULT / (SL_MULT + TP_MULT)  # 0.4286

Direction = Literal["long", "short"]


@dataclass(frozen=True)
class SplitFrames:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def load_joined_dataset(
    features_path: Path,
    labels_path: Path,
) -> pd.DataFrame:
    """Join feature table with label columns on Date."""
    feat = pd.read_parquet(features_path)
    lab = pd.read_parquet(labels_path)

    label_keep = [
        "Date",
        "label_long",
        "label_short",
        "bars_to_resolution_long",
        "bars_to_resolution_short",
    ]
    missing = [c for c in FEATURE_COLS + ["Date"] if c not in feat.columns]
    if missing:
        raise ValueError(f"Feature table missing columns: {missing}")
    missing_lab = [c for c in label_keep if c not in lab.columns]
    if missing_lab:
        raise ValueError(f"Label table missing columns: {missing_lab}")

    merged = feat.merge(lab[label_keep], on="Date", how="inner")
    merged = merged.sort_values("Date").reset_index(drop=True)
    return merged


def drop_timeouts(df: pd.DataFrame, direction: Direction) -> tuple[pd.DataFrame, dict]:
    """Keep only clear win/loss outcomes (drop label == -1)."""
    label_col = f"label_{direction}"
    n_total = len(df)
    n_timeout = int((df[label_col] == -1).sum())
    kept = df[df[label_col] != -1].copy().reset_index(drop=True)
    stats = {
        "direction": direction,
        "n_total": n_total,
        "n_timeout": n_timeout,
        "timeout_pct": 100.0 * n_timeout / n_total if n_total else 0.0,
        "n_kept": len(kept),
        "kept_pct": 100.0 * len(kept) / n_total if n_total else 0.0,
        "n_win": int((kept[label_col] == 1).sum()),
        "n_loss": int((kept[label_col] == 0).sum()),
    }
    return kept, stats


def time_aware_split(
    df: pd.DataFrame,
    *,
    train_end: str = "2023-01-01",
    val_end: str = "2024-01-01",
    embargo_hours: int = HORIZON_BARS,
) -> SplitFrames:
    """Split by calendar time with embargo around boundaries.

    Why not random split?
    Labels are formed by scanning up to HORIZON_BARS future H1 candles.
    Rows at T and T+k (k < horizon) share overlapping future windows, so a
    random split would put correlated future outcomes into both train and
    test -> leakage. Time-ordered split + embargo reduces that leakage.
    """
    df = df.sort_values("Date").reset_index(drop=True)
    train_end_ts = pd.Timestamp(train_end, tz="UTC")
    val_end_ts = pd.Timestamp(val_end, tz="UTC")
    embargo = pd.Timedelta(hours=embargo_hours)

    # Provisional assignment by date.
    train = df[df["Date"] < train_end_ts].copy()
    val = df[(df["Date"] >= train_end_ts) & (df["Date"] < val_end_ts)].copy()
    test = df[df["Date"] >= val_end_ts].copy()

    # Embargo: drop rows within +/- embargo around each boundary.
    b1, b2 = train_end_ts, val_end_ts
    train = train[train["Date"] < (b1 - embargo)].copy()
    val = val[(val["Date"] >= (b1 + embargo)) & (val["Date"] < (b2 - embargo))].copy()
    test = test[test["Date"] >= (b2 + embargo)].copy()

    return SplitFrames(
        train=train.reset_index(drop=True),
        val=val.reset_index(drop=True),
        test=test.reset_index(drop=True),
    )


def describe_split(name: str, frame: pd.DataFrame) -> str:
    if frame.empty:
        return f"- {name}: 0 rows"
    return (
        f"- {name}: rows={len(frame)} | "
        f"range={frame['Date'].min()} -> {frame['Date'].max()}"
    )
