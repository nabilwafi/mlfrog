"""Build frozen-stack feature rows from live MT5 candles."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from feature_engineering.services.feature_engineering_service import FeatureEngineeringService, market_to_frame
from market_context.builders.h4_structure_builder import H4StructureBuilder
from market_context.services.context_join_service import ContextJoinService
from production.live.mt5_candles import frame_to_market
from research.confidence_layer.services.mtf_context import asof_join, build_d1_features, build_m5_entry_quality
from research.meta_feature_research.services.panel_builder import _session_flags
from research.model_v2.services.experiment_catalog import LONG_STRUCTURE_CONTEXT, SHORT_STRUCTURE_CONTEXT
from settings.paths import ROOT

logger = logging.getLogger(__name__)


class LiveFeatureBuilder:
    def __init__(self, cfg: dict[str, Any], *, symbol: str, timezone: str = "UTC") -> None:
        self._cfg = cfg
        self._symbol = str(symbol)  # preserve broker case (XAUUSDc)
        self._tz = timezone
        fe_cfg = dict(cfg.get("feature_engineering") or {})
        out = fe_cfg.get("output_directory", "./artifacts/features")
        self._fe = FeatureEngineeringService(fe_cfg, output_root=ROOT / out if not str(out).startswith("/") else out)
        self._h4 = H4StructureBuilder(dict((cfg.get("market_context") or {}).get("builder_params") or {}))
        self._join = ContextJoinService()

    def build_panel(
        self,
        *,
        h1: pd.DataFrame,
        h4: pd.DataFrame,
        d1: pd.DataFrame,
        m5: pd.DataFrame,
    ) -> pd.DataFrame:
        if h1.empty:
            return pd.DataFrame()
        h1m = frame_to_market(self._symbol, "H1", h1, tz=self._tz)
        matrix, _, _ = self._fe.run(h1m)
        matrix["timestamp"] = pd.to_datetime(matrix["timestamp"], utc=True)

        h4_frame = market_to_frame(frame_to_market(self._symbol, "H4", h4, tz=self._tz))
        structure, _ = self._h4.build(h4_frame)
        ctx_cols = [c for c in structure.columns if c != "timestamp"]
        joined = self._join.join(matrix["timestamp"], structure, context_timeframe="H4", context_cols=ctx_cols)
        panel = matrix.merge(joined, on="timestamp", how="left")

        sess = _session_flags(panel["timestamp"])
        for col in sess.columns:
            panel[col] = sess[col].to_numpy()

        d1_feat = build_d1_features(d1.rename(columns={"timestamp": "timestamp"}))
        m5_feat = build_m5_entry_quality(m5.rename(columns={"timestamp": "timestamp"}))
        panel = asof_join(panel, d1_feat, ["d1_ema_slope", "d1_trend_dist", "d1_atr_z", "d1_regime"])
        panel = asof_join(
            panel,
            m5_feat[["available_at", "m5_entry_quality", "m5_body_ratio", "m5_atr_expansion", "m5_dist_swing_atr"]],
            ["m5_entry_quality", "m5_body_ratio", "m5_atr_expansion", "m5_dist_swing_atr"],
        )
        return panel.sort_values("timestamp").reset_index(drop=True)

    def context_features(self, side: str) -> tuple[str, ...]:
        return LONG_STRUCTURE_CONTEXT if str(side).lower() == "long" else SHORT_STRUCTURE_CONTEXT

    def h1_feature_names(self, panel: pd.DataFrame) -> list[str]:
        skip = {"timestamp", "context_bar_timestamp"}
        return [c for c in panel.columns if c not in skip and not c.startswith("ctx_h4_") and not c.startswith("d1_") and not c.startswith("m5_") and not c.startswith("session_") and c not in {"hour_of_day", "day_of_week", "month"}]
