"""RunMarketContextUseCase — H4 context → join H1 → optional dataset v2."""

from __future__ import annotations

import logging
from datetime import datetime, timezone as tz_utc
from pathlib import Path
from typing import Any

import pandas as pd

from data.repositories.market_repository import MarketRepository
from feature_engineering.services.feature_engineering_service import (
    FeatureEngineeringService,
    market_to_frame,
)
from market_context.builders.h4_context_builder import H4ContextBuilder
from market_context.exceptions import ContextInputError
from market_context.reports.context_report import ContextReportBuilder
from market_context.repositories.context_repository import ContextRepository
from market_context.services.context_join_service import ContextJoinService
from market_context.services.dataset_v2_builder import DatasetV2Builder

logger = logging.getLogger(__name__)


class RunMarketContextUseCase:
    def __init__(
        self,
        *,
        market_repo: MarketRepository,
        context_repo: ContextRepository,
        config: dict[str, Any] | None = None,
        feature_engineering_config: dict[str, Any] | None = None,
        features_output_root: Path | None = None,
        datasets_root: Path | None = None,
        dataset_config: dict[str, Any] | None = None,
    ) -> None:
        self._market_repo = market_repo
        self._context_repo = context_repo
        self._cfg = dict(config or {})
        self._fe_cfg = dict(feature_engineering_config or {})
        self._features_root = features_output_root
        self._datasets_root = datasets_root
        self._dataset_cfg = dict(dataset_config or {})
        self._join = ContextJoinService()
        self._report = ContextReportBuilder()

    def execute(
        self,
        *,
        symbol: str,
        base_timeframe: str = "H1",
        context_timeframe: str = "H4",
        timezone: str = "UTC",
        labels_by_side: dict[str, pd.DataFrame] | None = None,
        h1_feature_matrix_path: Path | None = None,
    ) -> Path:
        symbol = symbol.upper()
        base_timeframe = base_timeframe.upper()
        context_timeframe = context_timeframe.upper()

        # 1. Load context TF OHLCV
        h4_market = self._market_repo.load_parquet(
            symbol, context_timeframe, timezone=timezone
        )
        h4_frame = market_to_frame(h4_market)

        # 2. Optional: engineer standard features on H4 (pipeline step "feature")
        if bool(self._cfg.get("run_feature_engineering", True)) and self._features_root is not None:
            logger.info("Engineering %s features | symbol=%s", context_timeframe, symbol)
            FeatureEngineeringService(self._fe_cfg, output_root=self._features_root).run(
                h4_market
            )

        # 3. Build H4 context signals
        builder = H4ContextBuilder(dict(self._cfg.get("builder_params") or {}))
        h4_context, specs = builder.build(h4_frame)
        if bool(self._cfg.get("drop_na", True)):
            before = len(h4_context)
            cols = [s.name for s in specs]
            h4_context = h4_context.dropna(subset=cols).reset_index(drop=True)
            logger.info("Dropped H4 context NA | before=%s after=%s", before, len(h4_context))

        # 4. Base TF timestamps
        h1_market = self._market_repo.load_parquet(
            symbol, base_timeframe, timezone=timezone
        )
        h1_frame = market_to_frame(h1_market)
        context_cols = [s.name for s in specs]
        joined = self._join.join(
            h1_frame["timestamp"],
            h4_context,
            context_timeframe=context_timeframe,
            context_cols=context_cols,
        )

        # 5. Dataset v2 (optional)
        dataset_sides: list[str] = []
        if (
            bool(self._cfg.get("build_dataset_v2", True))
            and self._datasets_root is not None
            and labels_by_side
            and h1_feature_matrix_path is not None
        ):
            if not h1_feature_matrix_path.is_file():
                raise ContextInputError(
                    f"H1 feature matrix required for dataset v2: {h1_feature_matrix_path}"
                )
            h1_feats = pd.read_parquet(h1_feature_matrix_path)
            v2 = DatasetV2Builder(
                dataset_config=self._dataset_cfg,
                datasets_root=self._datasets_root,
                feature_version=str(self._cfg.get("feature_version", "v2")),
            )
            strategy = str(self._cfg.get("strategy", "triple_barrier"))
            label_version = str(self._cfg.get("label_version", "v1"))
            for side, labs in labels_by_side.items():
                v2.build_and_save(
                    h1_features=h1_feats,
                    context_on_h1=joined,
                    labels=labs,
                    symbol=symbol,
                    timeframe=base_timeframe,
                    side=side,
                    strategy=strategy,
                    label_version=label_version,
                    context_cols=context_cols,
                )
                dataset_sides.append(side)

        report_md = self._report.build(
            symbol=symbol,
            base_tf=base_timeframe,
            context_tf=context_timeframe,
            specs=specs,
            joined=joined,
            h4_rows=len(h4_context),
            dataset_v2_sides=dataset_sides,
        )
        metadata = {
            "symbol": symbol,
            "base_timeframe": base_timeframe,
            "context_timeframe": context_timeframe,
            "created_at": datetime.now(tz=tz_utc.utc).isoformat(),
            "context_version": str(self._cfg.get("context_version", "v1")),
            "feature_version": str(self._cfg.get("feature_version", "v2")),
            "n_base_rows": len(joined),
            "n_context_rows": len(h4_context),
            "context_features": [s.to_dict() for s in specs],
            "dataset_v2_sides": dataset_sides,
            "leakage_rule": "available_at = context_bar_open + bar_duration; asof backward",
        }
        return self._context_repo.save(
            symbol=symbol,
            base_timeframe=base_timeframe,
            context_tf=context_timeframe,
            joined=joined,
            specs=specs,
            report_md=report_md,
            metadata=metadata,
        )
