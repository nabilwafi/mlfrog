"""Persist datasets under artifacts/datasets/{symbol}/{tf}/{side}/{split}.parquet."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from datasets.entities.dataset import Dataset
from datasets.exceptions import DatasetRepositoryError

logger = logging.getLogger(__name__)


class DatasetRepository:
    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    def _dir(self, symbol: str, timeframe: str, side: str) -> Path:
        return self._root / symbol.upper() / timeframe.upper() / side.lower()

    def parquet_path(self, symbol: str, timeframe: str, side: str, split: str) -> Path:
        return self._dir(symbol, timeframe, side) / f"{split.lower()}.parquet"

    def meta_path(self, symbol: str, timeframe: str, side: str, split: str) -> Path:
        return self._dir(symbol, timeframe, side) / f"{split.lower()}.meta.json"

    def save_parquet(self, dataset: Dataset) -> Path:
        path = self.parquet_path(dataset.symbol, dataset.timeframe, dataset.side, dataset.split)
        logger.info("Saving Dataset | path=%s rows=%s", path, dataset.size)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            dataset.frame.to_parquet(path, index=False)
            meta = {
                "symbol": dataset.symbol,
                "timeframe": dataset.timeframe,
                "side": dataset.side,
                "strategy": dataset.strategy,
                "feature_version": dataset.feature_version,
                "label_version": dataset.label_version,
                "split": dataset.split,
                "created_at": dataset.created_at.isoformat(),
                "rows": dataset.size,
                "feature_names": dataset.feature_names,
                "metadata": dataset.metadata,
            }
            self.meta_path(
                dataset.symbol, dataset.timeframe, dataset.side, dataset.split
            ).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        except Exception as exc:
            raise DatasetRepositoryError(f"save_parquet failed: {exc}") from exc
        logger.info("Saved Dataset | path=%s", path)
        return path

    def load_parquet(
        self,
        symbol: str,
        timeframe: str,
        side: str,
        split: str,
        *,
        strategy: str = "",
        feature_version: str = "",
        label_version: str = "",
    ) -> Dataset:
        path = self.parquet_path(symbol, timeframe, side, split)
        logger.info("Loading Dataset | path=%s", path)
        if not path.is_file():
            raise DatasetRepositoryError(f"dataset parquet not found: {path}")
        try:
            df = pd.read_parquet(path)
            meta_file = self.meta_path(symbol, timeframe, side, split)
            meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.is_file() else {}
            created_raw = meta.get("created_at")
            created_at = (
                datetime.fromisoformat(created_raw)
                if created_raw
                else datetime.utcnow()
            )
            return Dataset(
                symbol=symbol,
                timeframe=timeframe,
                side=side,
                strategy=str(meta.get("strategy", strategy)),
                feature_version=str(meta.get("feature_version", feature_version)),
                label_version=str(meta.get("label_version", label_version)),
                split=split,
                created_at=created_at,
                frame=df,
                metadata=dict(meta.get("metadata") or {}),
            )
        except DatasetRepositoryError:
            raise
        except Exception as exc:
            raise DatasetRepositoryError(f"load_parquet failed: {exc}") from exc
