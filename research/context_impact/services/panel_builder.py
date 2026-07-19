"""Build H1 + context + labels panel (context already causally joined in Sprint 9)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from feature_diagnostics.exceptions import FeatureDiagnosticsInputError

_CONTEXT_PREFIX = "ctx_h4_"


class ContextImpactPanelBuilder:
    def build(
        self,
        *,
        feature_matrix_path: Path,
        context_path: Path,
        labels: pd.DataFrame,
        symbol: str,
        timeframe: str,
        side: str,
    ) -> tuple[pd.DataFrame, list[str], list[str]]:
        if not feature_matrix_path.is_file():
            raise FeatureDiagnosticsInputError(f"missing feature matrix: {feature_matrix_path}")
        if not context_path.is_file():
            raise FeatureDiagnosticsInputError(f"missing context parquet: {context_path}")

        feats = pd.read_parquet(feature_matrix_path)
        feats["timestamp"] = pd.to_datetime(feats["timestamp"], utc=True)
        h1_features = [c for c in feats.columns if c != "timestamp"]

        ctx = pd.read_parquet(context_path)
        ctx["timestamp"] = pd.to_datetime(ctx["timestamp"], utc=True)
        context_cols = [c for c in ctx.columns if c.startswith(_CONTEXT_PREFIX)]
        if not context_cols:
            raise FeatureDiagnosticsInputError("no ctx_h4_* columns in context.parquet")

        lab = labels.copy()
        lab["timestamp"] = pd.to_datetime(lab["timestamp"], utc=True)

        merged = feats.merge(
            ctx[["timestamp", *context_cols]],
            on="timestamp",
            how="inner",
        )
        merged = merged.merge(lab[["timestamp", "label"]], on="timestamp", how="inner")
        if merged.empty:
            raise FeatureDiagnosticsInputError("no overlap between features, context, and labels")

        # Drop rows without context (warmup / edge)
        merged = merged.dropna(subset=context_cols).reset_index(drop=True)

        panel = merged.copy()
        panel["label"] = panel["label"].astype(int)
        panel["symbol"] = symbol.upper()
        panel["timeframe"] = timeframe.upper()
        panel["feature_version"] = "context_impact_v1"
        panel["label_version"] = "v1"
        panel["split"] = "train"
        panel["side"] = side.lower()
        return panel, h1_features, context_cols
