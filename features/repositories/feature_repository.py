"""Persist FeatureSet under artifacts/features/{symbol}/{timeframe}/."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from features.entities.feature import Feature
from features.entities.feature_set import FeatureSet
from features.exceptions import FeatureRepositoryError

logger = logging.getLogger(__name__)


class FeatureRepository:
    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    def _dir(self, symbol: str, timeframe: str) -> Path:
        return self._root / symbol.upper() / timeframe.upper()

    def parquet_path(self, symbol: str, timeframe: str, feature_version: str) -> Path:
        safe = feature_version.replace("/", "_")
        return self._dir(symbol, timeframe) / f"features_{safe}.parquet"

    def meta_path(self, symbol: str, timeframe: str, feature_version: str) -> Path:
        safe = feature_version.replace("/", "_")
        return self._dir(symbol, timeframe) / f"features_{safe}.meta.json"

    def save_parquet(self, feature_set: FeatureSet) -> Path:
        path = self.parquet_path(
            feature_set.symbol, feature_set.timeframe, feature_set.feature_version
        )
        logger.info(
            "Saving FeatureSet parquet | path=%s features=%s rows=%s",
            path,
            feature_set.feature_count,
            feature_set.row_count,
        )
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame(feature_set.to_frame_dict())
            df.to_parquet(path, index=False)
            meta = {
                "symbol": feature_set.symbol,
                "timeframe": feature_set.timeframe,
                "feature_version": feature_set.feature_version,
                "pipeline_version": feature_set.pipeline_version,
                "created_at": feature_set.created_at.isoformat(),
                "feature_names": list(feature_set.feature_names),
                "dtypes": {f.name: f.dtype for f in feature_set.features},
                "metadata": {f.name: dict(f.metadata) for f in feature_set.features},
            }
            self.meta_path(
                feature_set.symbol, feature_set.timeframe, feature_set.feature_version
            ).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        except Exception as exc:
            raise FeatureRepositoryError(f"save_parquet failed: {exc}") from exc
        logger.info("Saved FeatureSet | path=%s", path)
        return path

    def load_parquet(
        self,
        symbol: str,
        timeframe: str,
        feature_version: str,
        *,
        timezone: str = "UTC",
    ) -> FeatureSet:
        path = self.parquet_path(symbol, timeframe, feature_version)
        logger.info("Loading FeatureSet | path=%s", path)
        if not path.is_file():
            raise FeatureRepositoryError(f"feature parquet not found: {path}")
        try:
            df = pd.read_parquet(path)
            meta_file = self.meta_path(symbol, timeframe, feature_version)
            meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.is_file() else {}
            tz = ZoneInfo(timezone)
            ts = pd.to_datetime(df["timestamp"], utc=True)
            timestamps = tuple(t.to_pydatetime().astimezone(tz) for t in ts)
            features: list[Feature] = []
            dtypes = meta.get("dtypes", {})
            metadatas = meta.get("metadata", {})
            for col in df.columns:
                if col == "timestamp":
                    continue
                series = df[col].astype(float)
                features.append(
                    Feature(
                        name=col,
                        dtype=str(dtypes.get(col, series.dtype)),
                        values=tuple(float(x) if pd.notna(x) else float("nan") for x in series),
                        metadata=metadatas.get(col, {}),
                    )
                )
            created_raw = meta.get("created_at")
            created_at = (
                datetime.fromisoformat(created_raw)
                if created_raw
                else datetime.now(tz=tz)
            )
            return FeatureSet(
                symbol=symbol,
                timeframe=timeframe,
                feature_version=str(meta.get("feature_version", feature_version)),
                pipeline_version=str(meta.get("pipeline_version", "unknown")),
                created_at=created_at,
                features=tuple(features),
                timestamps=timestamps,
            )
        except FeatureRepositoryError:
            raise
        except Exception as exc:
            raise FeatureRepositoryError(f"load_parquet failed: {exc}") from exc
