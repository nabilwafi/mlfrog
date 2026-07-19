"""Persist trained models under artifacts/models/{symbol}/{tf}/{side}/."""

from __future__ import annotations

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from models.entities.model import Model
from models.exceptions import ModelRepositoryError

logger = logging.getLogger(__name__)


class ModelRepository:
    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    def model_dir(self, symbol: str, timeframe: str, side: str) -> Path:
        return self._root / symbol.upper() / timeframe.upper() / side.lower()

    def model_path(self, symbol: str, timeframe: str, side: str) -> Path:
        return self.model_dir(symbol, timeframe, side) / "model.pkl"

    def metadata_path(self, symbol: str, timeframe: str, side: str) -> Path:
        return self.model_dir(symbol, timeframe, side) / "metadata.json"

    def importance_path(self, symbol: str, timeframe: str, side: str) -> Path:
        return self.model_dir(symbol, timeframe, side) / "feature_importance.parquet"

    def report_path(self, symbol: str, timeframe: str, side: str) -> Path:
        return self.model_dir(symbol, timeframe, side) / "training_report.md"

    def save(
        self,
        model: Model,
        *,
        importance_frame: pd.DataFrame,
        report_markdown: str,
    ) -> Path:
        out = self.model_dir(model.symbol, model.timeframe, model.side)
        logger.info("Saving Model | path=%s algorithm=%s", out, model.algorithm)
        try:
            out.mkdir(parents=True, exist_ok=True)
            with self.model_path(model.symbol, model.timeframe, model.side).open("wb") as fh:
                pickle.dump(model.estimator, fh, protocol=pickle.HIGHEST_PROTOCOL)

            meta = model.to_metadata()
            self.metadata_path(model.symbol, model.timeframe, model.side).write_text(
                json.dumps(meta, indent=2, default=str),
                encoding="utf-8",
            )
            importance_frame.to_parquet(
                self.importance_path(model.symbol, model.timeframe, model.side),
                index=False,
            )
            self.report_path(model.symbol, model.timeframe, model.side).write_text(
                report_markdown,
                encoding="utf-8",
            )
        except Exception as exc:
            raise ModelRepositoryError(f"save failed: {exc}") from exc
        logger.info("Saved Model | dir=%s", out)
        return out

    def load_estimator(self, symbol: str, timeframe: str, side: str) -> Any:
        path = self.model_path(symbol, timeframe, side)
        if not path.is_file():
            raise ModelRepositoryError(f"model.pkl not found: {path}")
        try:
            with path.open("rb") as fh:
                return pickle.load(fh)
        except Exception as exc:
            raise ModelRepositoryError(f"load_estimator failed: {exc}") from exc

    def load_metadata(self, symbol: str, timeframe: str, side: str) -> dict[str, Any]:
        path = self.metadata_path(symbol, timeframe, side)
        if not path.is_file():
            raise ModelRepositoryError(f"metadata.json not found: {path}")
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ModelRepositoryError(f"load_metadata failed: {exc}") from exc

    def load_model(self, symbol: str, timeframe: str, side: str) -> Model:
        meta = self.load_metadata(symbol, timeframe, side)
        estimator = self.load_estimator(symbol, timeframe, side)
        importance_path = self.importance_path(symbol, timeframe, side)
        importance: dict[str, float] = {}
        if importance_path.is_file():
            frame = pd.read_parquet(importance_path)
            importance = {
                str(r["feature"]): float(r["importance"]) for _, r in frame.iterrows()
            }
        trained_raw = meta.get("training_timestamp")
        trained_at = (
            datetime.fromisoformat(trained_raw) if trained_raw else datetime.utcnow()
        )
        return Model(
            symbol=str(meta.get("symbol", symbol)),
            timeframe=str(meta.get("timeframe", timeframe)),
            side=str(meta.get("side", side)),
            algorithm=str(meta["algorithm"]),
            feature_version=str(meta["feature_version"]),
            label_version=str(meta["label_version"]),
            dataset_version=str(meta.get("dataset_version", "v1")),
            trained_at=trained_at,
            hyperparameters=dict(meta.get("hyperparameters") or {}),
            feature_names=tuple(meta.get("feature_names") or []),
            train_rows=int(meta.get("training_rows", 0)),
            validation_rows=int(meta.get("validation_rows", 0)),
            metrics=dict(meta.get("metrics") or {}),
            feature_importance=importance,
            metadata=dict(meta.get("metadata") or {}),
            estimator=estimator,
        )
