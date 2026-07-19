"""Align FeatureSet + LabelSet into a chronological ML frame, then time-split."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from datasets.entities.dataset import Dataset
from datasets.exceptions import AlignmentError, DatasetError
from features.entities.feature_set import FeatureSet
from labels.entities.label_set import LabelSet

logger = logging.getLogger(__name__)


class DatasetBuilder:
    def __init__(self, config: dict[str, Any]) -> None:
        self._cfg = config

    def build_aligned_frame(
        self,
        feature_set: FeatureSet,
        label_set: LabelSet,
    ) -> pd.DataFrame:
        if feature_set.symbol != label_set.symbol or feature_set.timeframe != label_set.timeframe:
            raise AlignmentError(
                f"symbol/timeframe mismatch features={feature_set.symbol}/{feature_set.timeframe} "
                f"labels={label_set.symbol}/{label_set.timeframe}"
            )

        feat = pd.DataFrame(feature_set.to_frame_dict())
        feat["timestamp"] = pd.to_datetime(feat["timestamp"], utc=True)

        lab_rows = [
            {
                "timestamp": lab.timestamp,
                "label": lab.label,
                "side": lab.side,
                "expire_timestamp": lab.expire_timestamp,
                "holding_bars": lab.holding_bars,
                "exit_reason": lab.exit_reason,
            }
            for lab in label_set.labels
        ]
        if not lab_rows:
            raise AlignmentError("LabelSet is empty")
        labels = pd.DataFrame(lab_rows)
        labels["timestamp"] = pd.to_datetime(labels["timestamp"], utc=True)
        labels["expire_timestamp"] = pd.to_datetime(labels["expire_timestamp"], utc=True)

        # Point-in-time join: features as-of label timestamp (exact match after inner join)
        merged = feat.merge(labels, on="timestamp", how="inner", validate="one_to_one")
        if merged.empty:
            raise AlignmentError("no overlapping timestamps between FeatureSet and LabelSet")

        # Leakage: feature row must not encode information after expire; exact as-of join
        # is enforced by timestamp equality. Expire must be >= timestamp.
        bad = merged["expire_timestamp"] < merged["timestamp"]
        if bad.any():
            raise AlignmentError(f"look-ahead leakage: {int(bad.sum())} rows with expire < timestamp")

        drop_na = bool(self._cfg.get("drop_na", True))
        feature_cols = [c for c in feature_set.feature_names]
        before = len(merged)
        if drop_na:
            merged = merged.dropna(subset=feature_cols + ["label"]).reset_index(drop=True)
        nan_removed = before - len(merged)
        logger.info(
            "Aligned features+labels | rows=%s nan_removed=%s side=%s",
            len(merged),
            nan_removed,
            label_set.side,
        )

        merged = merged.sort_values("timestamp").reset_index(drop=True)
        merged["symbol"] = feature_set.symbol
        merged["timeframe"] = feature_set.timeframe
        merged["feature_version"] = feature_set.feature_version
        merged["label_version"] = label_set.label_version
        merged["strategy"] = label_set.strategy
        merged["side"] = label_set.side
        return merged

    def split_time(
        self,
        frame: pd.DataFrame,
        *,
        symbol: str,
        timeframe: str,
        side: str,
        strategy: str,
        feature_version: str,
        label_version: str,
    ) -> dict[str, Dataset]:
        splits_cfg = self._cfg.get("splits")
        if not splits_cfg:
            raise DatasetError("datasets.splits missing from config")

        out: dict[str, Dataset] = {}
        created = datetime.now(tz=timezone.utc)
        ts = pd.to_datetime(frame["timestamp"], utc=True)

        for split_name, bounds in splits_cfg.items():
            start = pd.Timestamp(str(bounds["start"]), tz="UTC")
            end = pd.Timestamp(str(bounds["end"]), tz="UTC")
            # inclusive range on calendar dates
            mask = (ts >= start) & (ts <= end + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1))
            # simpler: date inclusive
            mask = (ts >= start) & (ts < end + pd.Timedelta(days=1))
            part = frame.loc[mask].copy()
            if part.empty:
                logger.warning(
                    "Empty split | split=%s start=%s end=%s", split_name, start, end
                )
                continue
            part["split"] = split_name
            # Column order: meta then features then label
            meta_cols = [
                "timestamp",
                "symbol",
                "timeframe",
                "feature_version",
                "label_version",
                "split",
                "side",
                "strategy",
            ]
            feat_cols = [
                c
                for c in part.columns
                if c not in set(meta_cols) | {"label", "expire_timestamp", "holding_bars", "exit_reason"}
            ]
            ordered = meta_cols + feat_cols + ["label"]
            ordered = [c for c in ordered if c in part.columns]
            part = part[ordered]
            ds = Dataset(
                symbol=symbol,
                timeframe=timeframe,
                side=side,
                strategy=strategy,
                feature_version=feature_version,
                label_version=label_version,
                split=str(split_name),
                created_at=created,
                frame=part,
                metadata={"start": str(start), "end": str(end), "rows": len(part)},
            )
            out[str(split_name)] = ds
            logger.info("Split built | split=%s rows=%s", split_name, len(part))
        if not out:
            raise DatasetError("all time splits are empty")
        return out
