"""RunTemporalStabilityUseCase — diagnose aging; no production changes."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from research.temporal_stability import STARTING_EQUITY
from research.temporal_stability.reports import (
    TemporalStabilityChartBuilder,
    build_answers,
    build_report,
    year_summary_md,
)
from research.temporal_stability.repositories import TemporalStabilityRepository
from research.temporal_stability.services.data import load_frozen_trades, load_ml_panel
from research.temporal_stability.services.drift import (
    feature_importance_over_time,
    feature_stability_summary,
    monte_carlo_year,
    year_pair_drift,
)
from research.temporal_stability.services.experiments import (
    dataset_aging,
    probability_stability,
    recency_weight_experiment,
    retrain_frequency,
    rolling_performance,
    threshold_by_year,
    year_by_year_frozen,
    year_by_year_ml,
)

logger = logging.getLogger(__name__)


class RunTemporalStabilityUseCase:
    def __init__(
        self,
        repository: TemporalStabilityRepository,
        *,
        config: dict[str, Any] | None = None,
    ) -> None:
        self._repo = repository
        self._cfg = dict(config or {})
        self._charts = TemporalStabilityChartBuilder()

    def execute(
        self,
        *,
        symbol: str,
        timeframe: str,
        dataset_root: Path,
        frozen_trades_path: Path | None,
    ) -> Path:
        starting = float(self._cfg.get("starting_equity", STARTING_EQUITY))
        n_mc = int(self._cfg.get("n_monte_carlo", 300))
        min_train = int(self._cfg.get("min_train_years", 5))

        logger.info("Loading ML panel from %s", dataset_root)
        panel = load_ml_panel(dataset_root)
        logger.info("ML panel rows=%s years=%s-%s", len(panel), panel["year"].min(), panel["year"].max())

        # Optional side filter for speed
        side = self._cfg.get("side")
        if side:
            panel = panel.loc[panel["side"] == str(side).lower()].copy()
            logger.info("Filtered side=%s rows=%s", side, len(panel))

        # Subsample years for smoke if configured
        max_years = self._cfg.get("max_years")
        if max_years:
            years = sorted(panel["year"].unique())[-int(max_years) :]
            panel = panel.loc[panel["year"].isin(years)].copy()
            logger.info("Limited to years=%s", years)

        logger.info("Part1 rolling performance")
        rolling = rolling_performance(panel, min_train_years=min_train)

        logger.info("Part2 year-by-year")
        year_ml = year_by_year_ml(panel)
        year_frozen = pd.DataFrame()
        frozen = None
        if frozen_trades_path and Path(frozen_trades_path).is_file():
            frozen = load_frozen_trades(Path(frozen_trades_path))
            # keep accepted / non-skipped if column exists
            if "skipped" in frozen.columns:
                frozen = frozen.loc[~frozen["skipped"].astype(bool)].copy()
            year_frozen = year_by_year_frozen(frozen)

        logger.info("Part3 concept drift")
        drift, drift_detail = year_pair_drift(panel)

        logger.info("Part4 dataset aging")
        aging = dataset_aging(panel)

        logger.info("Part5 recency weights")
        recency = recency_weight_experiment(panel)

        logger.info("Part6 retrain frequency")
        retrain = retrain_frequency(panel)

        logger.info("Part7 feature stability")
        feat_imp = feature_importance_over_time(panel)
        feat_stab = feature_stability_summary(feat_imp)

        logger.info("Part8 threshold stability")
        threshold = threshold_by_year(panel)

        logger.info("Part9 probability stability")
        probability = probability_stability(panel)

        logger.info("Part10 Monte Carlo")
        mc = pd.DataFrame()
        if frozen is not None and not frozen.empty:
            mc = monte_carlo_year(frozen, n_sims=n_mc, starting=starting)

        arts: dict[str, Any] = {
            "rolling": rolling,
            "year_ml": year_ml,
            "year_frozen": year_frozen,
            "aging": aging,
            "recency": recency,
            "retrain": retrain,
            "drift": drift,
            "drift_detail": drift_detail,
            "feature_importance": feat_imp,
            "feature_stability": feat_stab,
            "threshold": threshold,
            "probability": probability,
            "monte_carlo": mc,
            "year_summary_md": year_summary_md(year_frozen, year_ml),
        }

        answers = build_answers(arts)
        report_md = build_report(symbol=symbol, timeframe=timeframe, answers=answers, arts=arts)
        out = self._repo.save(arts, report_md=report_md, answers=answers)
        # also save drift detail
        if isinstance(drift_detail, pd.DataFrame) and not drift_detail.empty:
            drift_detail.to_csv(out / "drift_detail.csv", index=False)
        self._charts.write_all(out, arts)
        logger.info("Recommendation: %s", answers.get("recommendation"))
        return out
