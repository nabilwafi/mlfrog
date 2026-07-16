"""RunMetaModelUseCase — production Trade/Skip meta (primary frozen)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from research.meta_model import FINAL_FEATURES, REFERENCE_THRESHOLD, THRESHOLDS
from research.meta_model.reports.meta_charts import MetaModelChartBuilder
from research.meta_model.reports.meta_report import build_report
from research.meta_model.repositories.meta_model_repository import MetaModelRepository
from research.meta_model.services.evaluator import (
    all_equity_curves,
    build_answers,
    evaluate_thresholds,
    statistical_meaning,
)
from research.meta_model.services.trainer import MetaModelTrainer

logger = logging.getLogger(__name__)


def _load_panel(panel_path: Path, candidates_path: Path, percentile: float) -> pd.DataFrame:
    panel = pd.read_parquet(panel_path)
    cand = pd.read_parquet(candidates_path)
    if "percentile" in cand.columns:
        cand = cand.loc[cand["percentile"] == percentile]
    keys = [k for k in ("side", "timestamp", "valid_year", "window") if k in panel.columns and k in cand.columns]
    extra = [c for c in ("net_return", "holding_bars", "meta_label", "realized_return") if c in cand.columns]
    if keys and extra:
        add = cand[keys + extra].drop_duplicates(keys)
        panel = panel.merge(add, on=keys, how="left", suffixes=("", "_c"))
        for c in ("net_return", "holding_bars", "meta_label"):
            cc = f"{c}_c"
            if cc in panel.columns:
                if c not in panel.columns:
                    panel = panel.rename(columns={cc: c})
                else:
                    panel[c] = panel[c].fillna(panel[cc])
                    panel = panel.drop(columns=[cc])
    if "meta_label" not in panel.columns and "net_return" in panel.columns:
        panel["meta_label"] = (panel["net_return"] > 0).astype(int)
    missing = [f for f in FINAL_FEATURES if f not in panel.columns]
    if missing:
        raise ValueError(f"panel missing Final features: {missing}")
    return panel


class RunMetaModelUseCase:
    def __init__(
        self,
        repository: MetaModelRepository,
        *,
        config: dict[str, Any] | None = None,
    ) -> None:
        self._repo = repository
        self._cfg = dict(config or {})
        self._charts = MetaModelChartBuilder()

    def execute(
        self,
        *,
        symbol: str,
        timeframe: str,
        panel_path: Path,
        candidates_path: Path,
    ) -> Path:
        percentile = float(self._cfg.get("percentile", 0.03))
        cost = float(self._cfg.get("transaction_cost", 0.00015))
        thr_ref = float(self._cfg.get("reference_threshold", REFERENCE_THRESHOLD))
        starting = float(self._cfg.get("starting_equity", 80.0))
        features = tuple(self._cfg.get("features") or FINAL_FEATURES)

        panel = _load_panel(panel_path, candidates_path, percentile)
        logger.info("Meta panel rows=%s years=%s", len(panel), sorted(panel["valid_year"].unique()))

        trainer = MetaModelTrainer(
            features=features,
            lgbm_params=dict(self._cfg.get("lgbm_params") or {}),
            num_boost_round=int(self._cfg.get("num_boost_round", 200)),
            early_stopping_rounds=int(self._cfg.get("early_stopping_rounds", 30)),
            random_seed=int(self._cfg.get("random_seed", 42)),
        )
        oof, loo_models = trainer.leave_one_year_out(panel)
        if oof.empty:
            raise RuntimeError("no OOF predictions — check panel years/labels")

        by_year, summary = evaluate_thresholds(oof, thresholds=THRESHOLDS, cost=cost)
        equity = all_equity_curves(by_year, thresholds=THRESHOLDS, starting_equity=starting)
        stats_out = statistical_meaning(oof, by_year, threshold=thr_ref, seed=int(self._cfg.get("random_seed", 42)))
        answers = build_answers(summary, by_year, equity, stats_out, threshold=thr_ref)

        full_model, medians = trainer.fit_full(panel)
        report_md = build_report(
            symbol=symbol,
            timeframe=timeframe,
            summary=summary,
            by_year=by_year,
            equity=equity,
            answers=answers,
            features=features,
        )

        out = self._repo.save(
            oof=oof,
            by_year=by_year,
            summary=summary,
            equity=equity,
            answers=answers,
            report_md=report_md,
            loo_models=loo_models,
            full_model=full_model,
            feature_medians=medians,
            features=list(features),
        )
        self._charts.write_all(
            out_dir=out,
            summary=summary,
            by_year=by_year,
            equity=equity,
            threshold_ref=thr_ref,
        )
        return out
