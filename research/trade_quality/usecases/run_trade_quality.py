"""RunTradeQualityUseCase."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from research.confidence_layer.services.research_ops import run_dynamic_risk_portfolio
from research.trade_quality import RISK_SCHEDULES, STARTING_EQUITY
from research.trade_quality.reports.charts import TradeQualityChartBuilder
from research.trade_quality.reports.report import build_answers, build_report
from research.trade_quality.repositories.repository import TradeQualityRepository
from research.trade_quality.services.analysis import (
    bucket_performance,
    capital_allocation_table,
    explainability,
    interaction_analysis,
    monotonicity_ok,
    search_tq_risk,
    yearly_stability,
)
from research.trade_quality.services.scorers import loo_trade_quality

logger = logging.getLogger(__name__)


class RunTradeQualityUseCase:
    def __init__(
        self,
        repository: TradeQualityRepository,
        *,
        config: dict[str, Any] | None = None,
    ) -> None:
        self._repo = repository
        self._cfg = dict(config or {})
        self._charts = TradeQualityChartBuilder()

    def execute(self, *, symbol: str, timeframe: str, confidence_panel_path: Path) -> Path:
        starting = float(self._cfg.get("starting_equity", STARTING_EQUITY))
        panel_in = pd.read_parquet(confidence_panel_path)
        logger.info("Loaded confidence panel rows=%s", len(panel_in))

        scored, stability = loo_trade_quality(panel_in)
        logger.info("Best TQ method=%s", scored["tq_best_method"].iloc[0] if len(scored) else "n/a")

        tq_buckets = bucket_performance(scored, score_col="trade_quality")
        mono = monotonicity_ok(tq_buckets)
        conf_buckets = (
            bucket_performance(scored, score_col="confidence")
            if "confidence" in scored.columns
            else pd.DataFrame()
        )
        interactions = interaction_analysis(scored, score_col="trade_quality")
        capital = capital_allocation_table(scored, score_col="trade_quality")
        yearly = yearly_stability(scored, score_col="trade_quality")
        risk_search = search_tq_risk(scored, score_col="trade_quality", starting_equity=starting)
        shap_df, perm_df = explainability(scored)

        # Baseline always 1%
        base_sched = dict(RISK_SCHEDULES)["always_1pct"]
        work = scored.copy()
        work["confidence"] = work["trade_quality"]
        _, _, base_m = run_dynamic_risk_portfolio(work, base_sched, starting_equity=starting)

        answers = build_answers(
            stability=stability,
            tq_buckets=tq_buckets,
            conf_buckets=conf_buckets,
            risk_search=risk_search,
            interactions=interactions,
            capital=capital,
            yearly=yearly,
            shap_df=shap_df,
            mono=mono,
            baseline_cagr=float(base_m.get("cagr", float("nan"))),
            baseline_dd=float(base_m.get("max_drawdown", float("nan"))),
        )
        report_md = build_report(
            symbol=symbol,
            timeframe=timeframe,
            answers=answers,
            stability=stability,
            tq_buckets=tq_buckets,
            conf_buckets=conf_buckets,
            risk_search=risk_search,
            interactions=interactions,
            capital=capital,
            yearly=yearly,
            shap_df=shap_df,
        )
        out = self._repo.save(
            panel=scored,
            stability=stability,
            tq_buckets=tq_buckets,
            conf_buckets=conf_buckets,
            risk_search=risk_search,
            interactions=interactions,
            capital=capital,
            yearly=yearly,
            shap_df=shap_df,
            perm_df=perm_df,
            answers=answers,
            report_md=report_md,
        )
        self._charts.write_all(
            out_dir=out,
            panel=scored,
            tq_buckets=tq_buckets,
            yearly=yearly,
            risk_search=risk_search,
            shap_df=shap_df,
            interactions=interactions,
            capital=capital,
        )
        return out
