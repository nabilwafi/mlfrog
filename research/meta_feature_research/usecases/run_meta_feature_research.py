"""RunMetaFeatureResearchUseCase — feature research only (no meta training)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from research.meta_feature_research.reports import (
    MetaFeatureChartBuilder,
    build_answers,
    build_report,
)
from research.meta_feature_research.repositories import MetaFeatureResearchRepository
from research.meta_feature_research.services import (
    MetaFeaturePanelBuilder,
    categorical_outcome_table,
    interaction_mi,
    score_all_features,
    walkforward_stability,
)

logger = logging.getLogger(__name__)


def _assign_group(feature: str, groups: dict[str, list[str]]) -> str:
    for g, cols in groups.items():
        if feature in cols:
            return g
    return "other"


class RunMetaFeatureResearchUseCase:
    def __init__(
        self,
        repository: MetaFeatureResearchRepository,
        *,
        config: dict[str, Any] | None = None,
    ) -> None:
        self._repo = repository
        self._cfg = dict(config or {})
        self._panel = MetaFeaturePanelBuilder()
        self._charts = MetaFeatureChartBuilder()

    def execute(
        self,
        *,
        symbol: str,
        timeframe: str,
        candidates_path: Path,
        h1_features_path: Path,
        h4_structure_path: Path,
        m15_candles_path: Path | None,
    ) -> tuple[pd.DataFrame, Path]:
        percentile = float(self._cfg.get("percentile", 0.03))
        candidates = pd.read_parquet(candidates_path)
        # Ensure meta_label + net_return for categorical expectancy
        if "meta_label" not in candidates.columns and "net_return" in candidates.columns:
            candidates["meta_label"] = (candidates["net_return"] > 0).astype(int)

        panel, groups = self._panel.build(
            candidates=candidates,
            h1_features_path=h1_features_path,
            h4_structure_path=h4_structure_path,
            m15_candles_path=m15_candles_path,
            percentile=percentile,
        )

        feature_cols = [c for cols in groups.values() for c in cols]
        feature_cols = list(dict.fromkeys(feature_cols))  # preserve order, unique

        logger.info("Scoring %s features (model-free)", len(feature_cols))
        univariate = score_all_features(panel, feature_cols)
        univariate["group"] = univariate["feature"].map(lambda f: _assign_group(f, groups))
        univariate = univariate.sort_values("mutual_info", ascending=False).reset_index(drop=True)

        group_rows = []
        for g, cols in groups.items():
            sub = univariate.loc[univariate["feature"].isin(cols)]
            group_rows.append(
                {
                    "group": g,
                    "n_features": len(cols),
                    "mean_mutual_info": float(sub["mutual_info"].mean()) if not sub.empty else float("nan"),
                    "max_mutual_info": float(sub["mutual_info"].max()) if not sub.empty else float("nan"),
                    "mean_iv": float(sub["information_value"].mean()) if not sub.empty else float("nan"),
                }
            )
        group_importance = pd.DataFrame(group_rows).sort_values(
            "mean_mutual_info", ascending=False
        )

        logger.info("Walk-forward stability")
        stability = walkforward_stability(panel, feature_cols)

        # Interactions: probability × selected partners
        partners = []
        for name in (
            "ctx_h4_rejection_strength",
            "ctx_m15_momentum",
            "session_london_ny_overlap",
            "atr_percent",
            "ema_alignment_score",
            "rsi_percentile",
            "ctx_m15_trend_strength",
            "ctx_h4_distance_from_equilibrium",
            "hour_of_day",
            "volatility_regime_score",
        ):
            if name in panel.columns:
                partners.append(("raw_probability", name))
        # also a few non-prob pairs from top MI
        top_feats = univariate["feature"].head(8).tolist()
        for i, a in enumerate(top_feats):
            for b in top_feats[i + 1 : i + 3]:
                if a != "raw_probability" and b != "raw_probability":
                    partners.append((a, b))
        interactions = interaction_mi(panel, partners)
        if not interactions.empty:
            interactions = interactions.sort_values(
                "lift_vs_best_single", ascending=False
            ).reset_index(drop=True)

        # Session categorical tables
        sess_parts = []
        for feat in ("hour_of_day", "day_of_week", "month"):
            if feat in panel.columns:
                # Attach net_return from candidates if needed
                if "net_return" not in panel.columns and "net_return" in candidates.columns:
                    tmp = candidates.loc[candidates["percentile"] == percentile][
                        ["side", "timestamp", "window", "net_return"]
                    ].drop_duplicates(["side", "timestamp", "window"])
                    panel = panel.merge(tmp, on=["side", "timestamp", "window"], how="left")
                sess_parts.append(categorical_outcome_table(panel, feat))
        for feat in (
            "session_asia",
            "session_london",
            "session_newyork",
            "session_london_ny_overlap",
        ):
            if feat in panel.columns:
                sess_parts.append(categorical_outcome_table(panel, feat))
        session_table = (
            pd.concat(sess_parts, ignore_index=True) if sess_parts else pd.DataFrame()
        )

        answers = build_answers(
            univariate=univariate,
            group_importance=group_importance,
            stability=stability,
            interactions=interactions,
            session_table=session_table,
            groups=groups,
        )
        report = build_report(
            symbol=symbol,
            timeframe=timeframe,
            groups=groups,
            univariate=univariate,
            group_importance=group_importance,
            stability=stability,
            interactions=interactions,
            session_table=session_table,
            answers=answers,
        )

        out = self._repo.save(
            panel=panel,
            univariate=univariate,
            group_importance=group_importance,
            stability=stability,
            interactions=interactions,
            session_table=session_table,
            report_md=report,
        )
        self._charts.write_all(
            out_dir=out,
            univariate=univariate,
            group_importance=group_importance,
            stability=stability,
            interactions=interactions,
            session_table=session_table,
            answers=answers,
        )
        return panel, out
