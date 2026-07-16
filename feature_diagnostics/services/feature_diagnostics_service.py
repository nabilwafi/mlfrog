"""FeatureDiagnosticsService — load Sprint-6 features + labels, run pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from feature_diagnostics.analyzers.shap_analyzer import ShapWindow
from feature_diagnostics.pipelines.feature_diagnostics_pipeline import FeatureDiagnosticsPipeline
from feature_diagnostics.repositories.feature_diagnostics_repository import (
    FeatureDiagnosticsRepository,
)
from settings.paths import FEATURES

logger = logging.getLogger(__name__)


class FeatureDiagnosticsService:
    def __init__(
        self,
        research_repository: FeatureDiagnosticsRepository,
        *,
        features_root: str | Path = FEATURES,
        config: dict[str, Any] | None = None,
        label_loader: Any | None = None,
    ) -> None:
        self._repo = research_repository
        self._features_root = Path(features_root)
        self._cfg = dict(config or {})
        self._label_loader = label_loader
        self._pipeline = FeatureDiagnosticsPipeline(research_repository)

    def run(self, symbol: str, timeframe: str, side: str) -> tuple[Any, Path]:
        feat_dir = self._features_root / symbol.upper() / timeframe.upper()
        matrix_path = feat_dir / "feature_matrix.parquet"
        meta_path = feat_dir / "feature_metadata.json"
        if self._label_loader is None:
            raise ValueError("label_loader is required")
        labels = self._label_loader(symbol, timeframe, side)
        windows = self._parse_windows(self._cfg.get("windows"))

        logger.info(
            "FeatureDiagnosticsService | %s %s %s matrix=%s",
            symbol,
            timeframe,
            side,
            matrix_path,
        )
        return self._pipeline.run(
            symbol=symbol,
            timeframe=timeframe,
            side=side,
            feature_matrix_path=matrix_path,
            feature_metadata_path=meta_path,
            labels=labels,
            target_mode=str(self._cfg.get("target_mode", "exclude_timeout")),
            corr_threshold=float(self._cfg.get("corr_threshold", 0.95)),
            shap_windows=windows,
            random_state=int(self._cfg.get("random_seed", 42)),
        )

    @staticmethod
    def _parse_windows(raw: Any) -> list[ShapWindow] | None:
        if not raw:
            return None
        return [
            ShapWindow(
                int(w["train_start_year"]),
                int(w["train_end_year"]),
                int(w["valid_year"]),
            )
            for w in raw
        ]
