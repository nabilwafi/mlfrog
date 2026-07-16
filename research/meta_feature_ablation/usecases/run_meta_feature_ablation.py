"""RunMetaFeatureAblationUseCase — fixed-param ablation (not final meta)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from research.meta_feature_ablation.reports.ablation_charts import MetaAblationChartBuilder
from research.meta_feature_ablation.reports.ablation_report import (
    build_answers,
    build_report,
    summarize_stages,
)
from research.meta_feature_ablation.repositories.meta_feature_ablation_repository import (
    MetaFeatureAblationRepository,
)
from research.meta_feature_ablation.services import (
    ABLATION_STAGES,
    MetaAblationRunner,
    available_features,
    correlation_clusters,
    interaction_study,
    pearson_spearman,
    removal_recommendations,
    variance_inflation_factors,
)
from research.meta_feature_ablation.services.feature_groups import (
    GROUP_B_H1,
    GROUP_C_H4,
    GROUP_E_SESSION,
    GROUP_F_ENTRY,
    drop_near_constant,
)

logger = logging.getLogger(__name__)


def _importance_stability(importance: pd.DataFrame) -> pd.DataFrame:
    if importance.empty:
        return pd.DataFrame()
    g = importance.groupby("feature", as_index=False).agg(
        gain_mean=("gain", "mean"),
        gain_std=("gain", "std"),
        split_mean=("split", "mean"),
        split_std=("split", "std"),
        shap_mean=("shap", "mean"),
        shap_std=("shap", "std"),
        permutation_mean=("permutation", "mean"),
        permutation_std=("permutation", "std"),
        n_windows=("valid_year", "nunique"),
    )
    g["gain_cv"] = g["gain_std"] / (g["gain_mean"].abs() + 1e-9)
    return g.sort_values("gain_mean", ascending=False).reset_index(drop=True)


def _enrich_holding(panel: pd.DataFrame, candidates_path: Path) -> pd.DataFrame:
    """Join holding_bars from candidate trades if missing on panel."""
    out = panel.copy()
    if "holding_bars" in out.columns and out["holding_bars"].notna().all():
        return out
    cand = pd.read_parquet(candidates_path)
    if "percentile" in cand.columns and (cand["percentile"] == 0.03).any():
        cand = cand.loc[cand["percentile"] == 0.03]
    keys = [k for k in ("side", "timestamp", "valid_year", "window") if k in out.columns and k in cand.columns]
    cols = [c for c in ("holding_bars", "net_return", "realized_return") if c in cand.columns]
    if not keys or not cols:
        return out
    add = cand[keys + cols].drop_duplicates(keys)
    merged = out.merge(add, on=keys, how="left", suffixes=("", "_cand"))
    if "holding_bars" not in merged.columns and "holding_bars_cand" in merged.columns:
        merged = merged.rename(columns={"holding_bars_cand": "holding_bars"})
    elif "holding_bars_cand" in merged.columns:
        merged["holding_bars"] = merged["holding_bars"].fillna(merged["holding_bars_cand"])
        merged = merged.drop(columns=["holding_bars_cand"])
    if "net_return" not in merged.columns and "net_return_cand" in merged.columns:
        merged = merged.rename(columns={"net_return_cand": "net_return"})
    elif "net_return_cand" in merged.columns:
        merged["net_return"] = merged["net_return"].fillna(merged["net_return_cand"])
        merged = merged.drop(columns=["net_return_cand"])
    return merged


def _recommend_features(
    *,
    stage_features: dict[str, list[str]],
    minimal_stage: str,
    removal: pd.DataFrame,
    stability: pd.DataFrame,
    answers: dict[str, Any],
) -> list[str]:
    feats = list(stage_features.get(minimal_stage, []))
    # Pairwise Pearson drops only (VIF>10 alone is too aggressive for tree models)
    drop: set[str] = set()
    if not removal.empty:
        for r in removal.itertuples():
            if str(r.reason).startswith("corr("):
                drop.add(str(r.feature))
    # Drop low-stable, low-gain tails when stage is large
    if not stability.empty and len(feats) > 12:
        keep_rank = stability.loc[stability["feature"].isin(feats)].copy()
        if not keep_rank.empty:
            med = float(keep_rank["gain_mean"].median())
            keep_rank = keep_rank.loc[keep_rank["gain_mean"] >= med * 0.25]
            feats = [f for f in feats if f in set(keep_rank["feature"])]
    # Always lean H1: keep top-gain H1 (trees tolerate moderate VIF)
    h1 = set(GROUP_B_H1)
    if not stability.empty and any(f in h1 for f in feats):
        top_h1 = (
            stability.loc[stability["feature"].isin(h1)]
            .sort_values("gain_mean", ascending=False)
            .head(5)["feature"]
            .tolist()
        )
        feats = [f for f in feats if f not in h1 or f in top_h1]
    # H4: drop exact duplicates only; keep high-gain survivors
    h4 = set(GROUP_C_H4)
    if not stability.empty and any(f in h4 for f in feats):
        top_h4 = (
            stability.loc[stability["feature"].isin(h4)]
            .sort_values("gain_mean", ascending=False)
            .head(6)["feature"]
            .tolist()
        )
        feats = [f for f in feats if f not in h4 or f in top_h4]
    # Session: keep only if helped, else drop session flags (keep hour optionally)
    if answers.get("session_helps") != "yes":
        sess = set(GROUP_E_SESSION) - {"hour_of_day", "day_of_week"}
        feats = [f for f in feats if f not in sess]
    if answers.get("entry_helps") not in ("yes",):
        entry = set(GROUP_F_ENTRY) - {"atr_percent"}
        feats = [f for f in feats if f not in entry]
    feats = [f for f in feats if f not in drop]
    # Force-include calendar anchors when session helps trading
    if answers.get("session_helps") == "yes":
        for must in ("hour_of_day", "day_of_week"):
            if must in stage_features.get(minimal_stage, []) and must not in feats:
                feats.append(must)
    # rolling_std vs rolling_volatility are duplicates — keep higher gain
    twins = {"rolling_std", "rolling_volatility"}
    if not stability.empty and twins.intersection(feats):
        scores = []
        for t in twins:
            sub = stability.loc[stability["feature"] == t, "gain_mean"]
            scores.append((float(sub.iloc[0]) if len(sub) else -1.0, t))
        scores.sort(reverse=True)
        keep_t = scores[0][1]
        feats = [f for f in feats if f not in twins]
        if keep_t in stage_features.get(minimal_stage, []):
            feats.append(keep_t)
    return list(dict.fromkeys(feats))


class RunMetaFeatureAblationUseCase:
    def __init__(
        self,
        repository: MetaFeatureAblationRepository,
        *,
        config: dict[str, Any] | None = None,
    ) -> None:
        self._repo = repository
        self._cfg = dict(config or {})
        self._charts = MetaAblationChartBuilder()

    def execute(
        self,
        *,
        symbol: str,
        timeframe: str,
        panel_path: Path,
        candidates_path: Path,
    ) -> Path:
        panel = pd.read_parquet(panel_path)
        panel = _enrich_holding(panel, candidates_path)
        if "meta_label" not in panel.columns and "net_return" in panel.columns:
            panel["meta_label"] = (panel["net_return"] > 0).astype(int)

        # Feature universe for correlation = union of ablation stage features
        all_wanted: list[str] = []
        for _, feats in ABLATION_STAGES:
            all_wanted.extend(feats)
        corr_feats = drop_near_constant(
            panel, available_features(list(panel.columns), tuple(dict.fromkeys(all_wanted)))
        )
        logger.info("Correlation / VIF on %s features", len(corr_feats))
        pearson, spearman = pearson_spearman(panel, corr_feats)
        vif = variance_inflation_factors(panel, corr_feats)
        clusters = correlation_clusters(pearson)
        removal = removal_recommendations(pearson, vif)

        runner = MetaAblationRunner(
            lgbm_params=dict(self._cfg.get("lgbm_params") or {}),
            num_boost_round=int(self._cfg.get("num_boost_round", 200)),
            early_stopping_rounds=int(self._cfg.get("early_stopping_rounds", 30)),
            threshold=float(self._cfg.get("threshold", 0.5)),
            random_seed=int(self._cfg.get("random_seed", 42)),
            cost=float(self._cfg.get("transaction_cost", 0.00015)),
            compute_shap=bool(self._cfg.get("compute_shap", True)),
        )

        stage_features: dict[str, list[str]] = {}
        win_parts: list[pd.DataFrame] = []
        oof_parts: list[pd.DataFrame] = []
        imp_parts: list[pd.DataFrame] = []

        for stage_id, wanted in ABLATION_STAGES:
            feats = drop_near_constant(panel, available_features(list(panel.columns), wanted))
            stage_features[stage_id] = feats
            logger.info("Ablation stage %s | n_features=%s", stage_id, len(feats))
            wm, oof, imp = runner.run_stage(panel, feats, stage_id=stage_id)
            win_parts.append(wm)
            if not oof.empty:
                oof_parts.append(oof)
            if not imp.empty:
                imp_parts.append(imp)

        window_metrics = pd.concat(win_parts, ignore_index=True) if win_parts else pd.DataFrame()
        oof = pd.concat(oof_parts, ignore_index=True) if oof_parts else pd.DataFrame()
        importance = pd.concat(imp_parts, ignore_index=True) if imp_parts else pd.DataFrame()

        stage_summary = summarize_stages(window_metrics)

        # Stability on fullest stage
        full_stage = list(stage_features.keys())[-1] if stage_features else ""
        full_imp = (
            importance.loc[importance["stage_id"] == full_stage]
            if full_stage and not importance.empty
            else importance
        )
        stability = _importance_stability(full_imp)

        full_feats = stage_features.get(full_stage, corr_feats)
        interactions = interaction_study(panel, full_feats, full_imp)

        answers = build_answers(
            stage_summary=stage_summary,
            removal=removal,
            stability=stability,
            groups_delta={},
        )

        # H1 / H4 reducibility narratives
        h1_delta = float(answers.get("h1_delta_E") or float("nan"))
        h4_delta = float(answers.get("h4_delta_E") or float("nan"))
        rem_h1 = [f for f in answers.get("remove_set", []) if f in GROUP_B_H1]
        rem_h4 = [f for f in answers.get("remove_set", []) if f in GROUP_C_H4]
        answers["h1_reduce"] = (
            f"yes - dE={h1_delta:.5f}; drop corr/VIF: {rem_h1 or 'none'}; "
            f"{'H1 add flat/negative - keep <=3 by gain' if h1_delta == h1_delta and h1_delta <= 0 else 'keep lean H1 subset'}"
        )
        answers["h4_reduce"] = (
            f"{'partial' if rem_h4 or (h4_delta == h4_delta and h4_delta > 0) else 'yes'} - "
            f"dE={h4_delta:.5f}; drop corr/VIF: {rem_h4 or 'none'}; "
            f"{'H4 is main lift - reduce via corr clusters only' if h4_delta == h4_delta and h4_delta > 0 else 'H4 alone weak on thr=0.5 trading; keep lean structure set'}"
        )

        recommended = _recommend_features(
            stage_features=stage_features,
            minimal_stage=str(answers.get("minimal_stage") or full_stage),
            removal=removal,
            stability=stability,
            answers=answers,
        )
        answers["recommended"] = recommended
        answers["minimal_set"] = recommended

        report_md = build_report(
            symbol=symbol,
            timeframe=timeframe,
            stage_summary=stage_summary,
            window_metrics=window_metrics,
            removal=removal,
            vif=vif,
            stability=stability,
            interactions=interactions,
            answers=answers,
            recommended=recommended,
            stage_features=stage_features,
        )

        out = self._repo.save(
            window_metrics=window_metrics,
            stage_summary=stage_summary,
            oof=oof,
            importance=importance,
            stability=stability,
            pearson=pearson,
            spearman=spearman,
            vif=vif,
            clusters=clusters,
            removal=removal,
            interactions=interactions,
            recommended=pd.DataFrame({"feature": recommended}),
            report_md=report_md,
        )
        self._charts.write_all(
            out_dir=out,
            pearson=pearson,
            stage_summary=stage_summary,
            window_metrics=window_metrics,
            stability=stability,
            interactions=interactions,
        )
        return out
