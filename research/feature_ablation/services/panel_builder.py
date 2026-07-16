"""Align Sprint-6 feature matrix with labels into an ablation panel."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from feature_diagnostics.exceptions import FeatureDiagnosticsInputError


class AblationPanelBuilder:
    def build(
        self,
        *,
        feature_matrix_path: Path,
        feature_metadata_path: Path,
        labels: pd.DataFrame,
        symbol: str,
        timeframe: str,
        side: str = "long",
    ) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
        if not feature_matrix_path.is_file():
            raise FeatureDiagnosticsInputError(f"missing feature matrix: {feature_matrix_path}")
        feats = pd.read_parquet(feature_matrix_path)
        feats["timestamp"] = pd.to_datetime(feats["timestamp"], utc=True)
        feature_names = [c for c in feats.columns if c != "timestamp"]

        category_by_feature: dict[str, str] = {}
        if feature_metadata_path.is_file():
            meta = json.loads(feature_metadata_path.read_text(encoding="utf-8"))
            for item in meta.get("features", []):
                category_by_feature[str(item["name"])] = str(item.get("category", "unknown"))

        lab = labels.copy()
        lab["timestamp"] = pd.to_datetime(lab["timestamp"], utc=True)
        merged = feats.merge(lab[["timestamp", "label"]], on="timestamp", how="inner")
        if merged.empty:
            raise FeatureDiagnosticsInputError("no overlap between features and labels")

        panel = merged.copy().reset_index(drop=True)
        panel["label"] = panel["label"].astype(int)
        panel["symbol"] = symbol.upper()
        panel["timeframe"] = timeframe.upper()
        panel["feature_version"] = "engineered_v1"
        panel["label_version"] = "v1"
        panel["split"] = "train"
        panel["side"] = side.lower()
        return panel, feature_names, category_by_feature
