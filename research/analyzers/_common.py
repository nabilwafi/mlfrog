"""Shared research helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from datasets.entities.dataset import Dataset
from diagnostics.analyzers._common import safe_ks, safe_psi

__all__ = [
    "safe_ks",
    "safe_psi",
    "combine_splits",
    "slice_dataset",
    "feature_cols",
    "drift_severity",
    "frame_to_markdown",
]


def feature_cols(frame: pd.DataFrame) -> list[str]:
    skip = {
        "timestamp",
        "symbol",
        "timeframe",
        "feature_version",
        "label_version",
        "split",
        "side",
        "label",
        "strategy",
    }
    return [c for c in frame.columns if c not in skip]


def combine_splits(splits: dict[str, Dataset]) -> pd.DataFrame:
    frames = []
    for name in ("train", "validation", "test", "sealed"):
        if name not in splits:
            continue
        df = splits[name].frame.copy()
        df["split"] = name
        frames.append(df)
    if not frames:
        raise ValueError("no dataset splits to combine")
    out = pd.concat(frames, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out = out.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    return out.reset_index(drop=True)


def slice_dataset(
    template: Dataset,
    frame: pd.DataFrame,
    *,
    split: str,
) -> Dataset:
    if frame.empty:
        raise ValueError(f"empty frame for split={split}")
    return Dataset(
        symbol=template.symbol,
        timeframe=template.timeframe,
        side=template.side,
        strategy=template.strategy,
        feature_version=template.feature_version,
        label_version=template.label_version,
        split=split,
        created_at=datetime.now(tz=timezone.utc),
        frame=frame.reset_index(drop=True),
        metadata={"source": "research_slice"},
    )


def drift_severity(psi: float, *, mild: float = 0.1, severe: float = 0.25) -> str:
    if psi != psi:  # NaN
        return "unknown"
    if psi >= severe:
        return "severe"
    if psi >= mild:
        return "moderate"
    return "stable"


def year_bounds(year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(year=year, month=1, day=1, tz="UTC")
    end = pd.Timestamp(year=year, month=12, day=31, hour=23, minute=59, second=59, tz="UTC")
    return start, end


def filter_years(frame: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    ts = pd.to_datetime(frame["timestamp"], utc=True)
    lo, _ = year_bounds(start_year)
    _, hi = year_bounds(end_year)
    return frame.loc[(ts >= lo) & (ts <= hi)].copy()


def frame_to_markdown(frame: pd.DataFrame) -> str:
    """Minimal markdown table without optional tabulate dependency."""
    if frame.empty:
        return "_(empty)_"
    cols = [str(c) for c in frame.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in frame.iterrows():
        cells = []
        for c in frame.columns:
            v = row[c]
            if isinstance(v, float):
                cells.append("n/a" if v != v else f"{v:.6g}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
