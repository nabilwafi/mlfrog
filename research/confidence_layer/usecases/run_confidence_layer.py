"""RunConfidenceLayerUseCase — frozen Primary/Meta; execution research only."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from research.confidence_layer import META_GATE, STARTING_EQUITY
from research.confidence_layer.reports.charts import ConfidenceChartBuilder
from research.confidence_layer.reports.report import build_answers, build_report
from research.confidence_layer.repositories.repository import ConfidenceLayerRepository
from research.confidence_layer.services.calibration import calibration_report
from research.confidence_layer.services.confidence import fit_loo_confidence, shap_components
from research.confidence_layer.services.mtf_context import (
    asof_join,
    build_d1_features,
    build_m5_entry_quality,
)
from research.confidence_layer.services.research_ops import (
    assign_bucket,
    bucket_stats,
    dynamic_tp_research,
    dynamic_trail_research,
    m5_quality_research,
    regime_performance,
    run_dynamic_risk_portfolio,
    search_risk_schedules,
)
from research.confidence_layer import RISK_SCHEDULES

logger = logging.getLogger(__name__)


class RunConfidenceLayerUseCase:
    def __init__(
        self,
        repository: ConfidenceLayerRepository,
        *,
        config: dict[str, Any] | None = None,
    ) -> None:
        self._repo = repository
        self._cfg = dict(config or {})
        self._charts = ConfidenceChartBuilder()

    def execute(
        self,
        *,
        symbol: str,
        timeframe: str,
        oof_path: Path,
        candidates_path: Path | None,
        d1_path: Path,
        m5_path: Path,
    ) -> Path:
        meta_gate = float(self._cfg.get("meta_gate", META_GATE))
        starting = float(self._cfg.get("starting_equity", STARTING_EQUITY))

        oof = pd.read_parquet(oof_path)
        # Production pipeline operates on Meta-accepted candidates
        panel = oof.loc[oof["meta_proba"].astype(float) >= meta_gate].copy()
        logger.info("Meta-gated candidates=%s / %s", len(panel), len(oof))

        # Join MAE/MFE for TP/trail research
        if candidates_path and Path(candidates_path).is_file():
            cand = pd.read_parquet(candidates_path)
            if "percentile" in cand.columns:
                cand = cand.loc[cand["percentile"] == 0.03]
            keys = [k for k in ("side", "timestamp", "valid_year", "window") if k in panel.columns and k in cand.columns]
            extra = [c for c in ("mae", "mfe", "holding_bars", "net_return") if c in cand.columns]
            if keys and extra:
                panel = panel.merge(cand[keys + extra].drop_duplicates(keys), on=keys, how="left", suffixes=("", "_c"))
                for c in ("mae", "mfe", "holding_bars", "net_return"):
                    cc = f"{c}_c"
                    if cc in panel.columns:
                        panel[c] = panel[c].fillna(panel[cc]) if c in panel.columns else panel[cc]
                        panel = panel.drop(columns=[cc])

        # D1 + M5 context (causal)
        d1 = pd.read_parquet(d1_path)
        m5 = pd.read_parquet(m5_path)
        d1_feat = build_d1_features(d1)
        m5_feat = build_m5_entry_quality(m5)
        # Downsample M5 features for merge speed: keep last bar per H1 already via asof
        panel = asof_join(
            panel,
            d1_feat,
            ["d1_ema_slope", "d1_trend_dist", "d1_atr_z", "d1_regime"],
        )
        panel = asof_join(
            panel,
            m5_feat[["available_at", "m5_entry_quality", "m5_body_ratio", "m5_atr_expansion", "m5_dist_swing_atr"]],
            ["m5_entry_quality", "m5_body_ratio", "m5_atr_expansion", "m5_dist_swing_atr"],
        )

        # Confidence LOO
        scored, weights = fit_loo_confidence(panel)
        scored["confidence_bucket"] = assign_bucket(scored["confidence"])

        buckets = bucket_stats(scored)
        regimes = regime_performance(scored)
        m5_research = m5_quality_research(scored)

        # Baseline: flat 1% fractional on all meta-gated trades (always_1pct schedule)
        baseline_sched = dict(RISK_SCHEDULES)["always_1pct"]
        _, _, base_m = run_dynamic_risk_portfolio(scored, baseline_sched, starting_equity=starting)

        risk_search = search_risk_schedules(scored, starting_equity=starting)
        best_row = risk_search.iloc[0] if not risk_search.empty else None
        best_cagr = float(best_row["cagr"]) if best_row is not None else float("nan")
        best_dd = float(best_row["max_drawdown"]) if best_row is not None else float("nan")

        tp_research = dynamic_tp_research(scored)
        trail_research = dynamic_trail_research(scored)
        cal_bins, cal_sum = calibration_report(scored)
        shap_imp, shap_ix = shap_components(scored)

        answers = build_answers(
            weights=weights,
            buckets=buckets,
            regimes=regimes,
            risk_search=risk_search,
            tp_research=tp_research,
            trail_research=trail_research,
            m5_research=m5_research,
            calibration_summary=cal_sum,
            baseline_cagr=float(base_m.get("cagr", float("nan"))),
            best_cagr=best_cagr,
            baseline_dd=float(base_m.get("max_drawdown", float("nan"))),
            best_dd=best_dd,
        )
        report_md = build_report(
            symbol=symbol,
            timeframe=timeframe,
            answers=answers,
            weights=weights,
            buckets=buckets,
            regimes=regimes,
            risk_search=risk_search,
            tp_research=tp_research,
            trail_research=trail_research,
            calibration_bins=cal_bins,
            shap_importance=shap_imp,
            m5_research=m5_research,
        )

        out = self._repo.save(
            panel=scored,
            weights=weights,
            buckets=buckets,
            regimes=regimes,
            risk_search=risk_search,
            tp_research=tp_research,
            trail_research=trail_research,
            calibration_bins=cal_bins,
            calibration_summary=cal_sum,
            shap_importance=shap_imp,
            shap_interactions=shap_ix,
            m5_research=m5_research,
            answers=answers,
            report_md=report_md,
        )
        self._charts.write_all(
            out_dir=out,
            panel=scored,
            buckets=buckets,
            regimes=regimes,
            risk_search=risk_search,
            calibration_bins=cal_bins,
            shap_importance=shap_imp,
        )
        return out
