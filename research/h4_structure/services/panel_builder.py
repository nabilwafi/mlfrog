"""Build H4 structure features, causally join to H1, align labels."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from data.repositories.market_repository import MarketRepository
from feature_engineering.services.feature_engineering_service import market_to_frame
from market_context.builders.h4_structure_builder import H4StructureBuilder
from market_context.entities.context_feature import ContextFeatureSpec
from market_context.services.context_join_service import ContextJoinService
from research.h4_structure.exceptions import H4StructureInputError


class StructurePanelBuilder:
    def __init__(self, builder_params: dict[str, Any] | None = None) -> None:
        self._builder = H4StructureBuilder(builder_params)
        self._join = ContextJoinService()

    def build_structure_on_h4(
        self,
        *,
        market_repo: MarketRepository,
        symbol: str,
        context_timeframe: str = "H4",
        timezone: str = "UTC",
        drop_na: bool = True,
    ) -> tuple[pd.DataFrame, list[ContextFeatureSpec]]:
        market = market_repo.load_parquet(symbol, context_timeframe, timezone=timezone)
        frame = market_to_frame(market)
        structure, specs = self._builder.build(frame)
        cols = [s.name for s in specs]
        if drop_na:
            structure = structure.dropna(subset=cols).reset_index(drop=True)
        return structure, specs

    def join_to_h1(
        self,
        *,
        market_repo: MarketRepository,
        structure_h4: pd.DataFrame,
        structure_cols: list[str],
        symbol: str,
        base_timeframe: str = "H1",
        context_timeframe: str = "H4",
        timezone: str = "UTC",
    ) -> pd.DataFrame:
        h1 = market_to_frame(
            market_repo.load_parquet(symbol, base_timeframe, timezone=timezone)
        )
        return self._join.join(
            h1["timestamp"],
            structure_h4,
            context_timeframe=context_timeframe,
            context_cols=structure_cols,
        )

    def build_ml_panel(
        self,
        *,
        feature_matrix_path: Path,
        structure_on_h1: pd.DataFrame,
        structure_cols: list[str],
        labels: pd.DataFrame,
        symbol: str,
        timeframe: str,
        side: str,
    ) -> tuple[pd.DataFrame, list[str]]:
        if not feature_matrix_path.is_file():
            raise H4StructureInputError(f"missing H1 features: {feature_matrix_path}")
        feats = pd.read_parquet(feature_matrix_path)
        feats["timestamp"] = pd.to_datetime(feats["timestamp"], utc=True)
        h1_features = [c for c in feats.columns if c != "timestamp"]

        ctx = structure_on_h1[["timestamp", *structure_cols]].copy()
        ctx["timestamp"] = pd.to_datetime(ctx["timestamp"], utc=True)
        ctx = ctx.dropna(subset=structure_cols)

        lab = labels.copy()
        lab["timestamp"] = pd.to_datetime(lab["timestamp"], utc=True)

        merged = feats.merge(ctx, on="timestamp", how="inner")
        merged = merged.merge(lab[["timestamp", "label"]], on="timestamp", how="inner")
        if merged.empty:
            raise H4StructureInputError("no overlap between H1 features, structure, labels")

        panel = merged.reset_index(drop=True)
        panel["label"] = panel["label"].astype(int)
        panel["symbol"] = symbol.upper()
        panel["timeframe"] = timeframe.upper()
        panel["feature_version"] = "h4_structure_v1"
        panel["label_version"] = "v1"
        panel["split"] = "train"
        panel["side"] = side.lower()
        return panel, h1_features
