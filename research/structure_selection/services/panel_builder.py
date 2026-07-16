"""Panel = H1 features + Sprint-11 structure columns + labels."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from research.structure_selection.exceptions import StructureSelectionInputError


class SelectionPanelBuilder:
    def build(
        self,
        *,
        feature_matrix_path: Path,
        structure_features_path: Path,
        labels: pd.DataFrame,
        symbol: str,
        timeframe: str,
        side: str,
        regime_context_path: Path | None = None,
    ) -> tuple[pd.DataFrame, list[str], list[str]]:
        if not feature_matrix_path.is_file():
            raise StructureSelectionInputError(f"missing features: {feature_matrix_path}")
        if not structure_features_path.is_file():
            raise StructureSelectionInputError(
                f"missing structure features: {structure_features_path} "
                "(run apps/run_h4_structure.py first)"
            )

        feats = pd.read_parquet(feature_matrix_path)
        feats["timestamp"] = pd.to_datetime(feats["timestamp"], utc=True)
        h1_features = [c for c in feats.columns if c != "timestamp"]

        struct = pd.read_parquet(structure_features_path)
        struct["timestamp"] = pd.to_datetime(struct["timestamp"], utc=True)
        structure_cols = [c for c in struct.columns if c.startswith("ctx_h4_")]
        if not structure_cols:
            raise StructureSelectionInputError("no ctx_h4_* columns in structure_features")

        keep_struct = ["timestamp", *structure_cols]
        # do not carry context_bar_timestamp into ML panel (would become a feature)
        lab = labels.copy()
        lab["timestamp"] = pd.to_datetime(lab["timestamp"], utc=True)

        merged = feats.merge(struct[keep_struct], on="timestamp", how="inner")
        merged = merged.merge(lab[["timestamp", "label"]], on="timestamp", how="inner")
        merged = merged.dropna(subset=structure_cols).reset_index(drop=True)
        if merged.empty:
            raise StructureSelectionInputError("empty panel after joins")

        # Optional Sprint-9 regime columns for regime analysis
        if regime_context_path is not None and regime_context_path.is_file():
            ctx = pd.read_parquet(regime_context_path)
            ctx["timestamp"] = pd.to_datetime(ctx["timestamp"], utc=True)
            regime_cols = [
                c
                for c in (
                    "ctx_h4_volatility_regime",
                    "ctx_h4_market_regime",
                    "ctx_h4_trend_direction",
                )
                if c in ctx.columns
            ]
            if regime_cols:
                merged = merged.merge(
                    ctx[["timestamp", *regime_cols]],
                    on="timestamp",
                    how="left",
                )

        merged["label"] = merged["label"].astype(int)
        merged["symbol"] = symbol.upper()
        merged["timeframe"] = timeframe.upper()
        merged["feature_version"] = "structure_selection_v1"
        merged["label_version"] = "v1"
        merged["split"] = "train"
        merged["side"] = side.lower()
        return merged, h1_features, structure_cols
