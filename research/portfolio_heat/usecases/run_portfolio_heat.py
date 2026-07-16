"""RunPortfolioHeatUseCase."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from research.portfolio_heat import STARTING_EQUITY, HeatPolicy, build_policy_grid
from research.portfolio_heat.reports.charts import PortfolioHeatChartBuilder
from research.portfolio_heat.reports.report import build_answers, build_report, research_question_lookup
from research.portfolio_heat.repositories.repository import PortfolioHeatRepository
from research.portfolio_heat.services.engine import run_heat_portfolio
from research.portfolio_heat.services.monte_carlo import monte_carlo_heat
from research.portfolio_heat.services.search import evaluate_policies, score_policies, walk_forward_years
from research.portfolio_heat.services.stats_tests import compare_policies

logger = logging.getLogger(__name__)


class RunPortfolioHeatUseCase:
    def __init__(
        self,
        repository: PortfolioHeatRepository,
        *,
        config: dict[str, Any] | None = None,
    ) -> None:
        self._repo = repository
        self._cfg = dict(config or {})
        self._charts = PortfolioHeatChartBuilder()

    def execute(self, *, symbol: str, timeframe: str, confidence_panel_path: Path) -> Path:
        starting = float(self._cfg.get("starting_equity", STARTING_EQUITY))
        n_mc = int(self._cfg.get("n_monte_carlo", 500))
        n_boot = int(self._cfg.get("n_bootstrap", 1000))
        n_perm = int(self._cfg.get("n_permutation", 1000))

        panel = pd.read_parquet(confidence_panel_path)
        logger.info("Loaded panel rows=%s", len(panel))

        policies = build_policy_grid()
        table, arts = evaluate_policies(panel, policies, starting_equity=starting)
        base_log, base_curve, base_m = arts["baseline_skip40_flat1"]
        base_row = table.loc[table["policy"] == "baseline_skip40_flat1"].iloc[0]
        min_trades = int(self._cfg.get("min_trades", max(100, int(0.25 * int(base_m.get("trades") or 400)))))
        scored = score_policies(table, base_row, min_trades=min_trades)

        # Production best: production_ok first; never pick toy trade-count policies
        prod = scored.loc[scored.get("production_ok", False) == True]  # noqa: E712
        if not prod.empty:
            best_row = prod.iloc[0]
        else:
            accepted = scored.loc[
                (scored["accepted"] == True) & (scored["trades"] >= min_trades)  # noqa: E712
            ]
            best_row = accepted.iloc[0] if not accepted.empty else scored.loc[scored["policy"] == "baseline_skip40_flat1"].iloc[0]

        best_name = str(best_row["policy"])
        best_log, best_curve, best_m = arts[best_name]
        logger.info(
            "Best policy=%s cagr=%s dd=%s trades=%s min_trades=%s",
            best_name,
            best_m.get("cagr"),
            best_m.get("max_drawdown"),
            best_m.get("trades"),
            min_trades,
        )

        pol_map = {p.name: p for p in policies}
        best_pol = pol_map[best_name]

        wf = walk_forward_years(panel, best_pol, starting_equity=starting)

        # Stats vs baseline for top production policies
        stats_rows = []
        top = prod.head(5) if not prod.empty else scored.head(3)
        for _, r in top.iterrows():
            name = str(r["policy"])
            if name not in arts:
                continue
            clog = arts[name][0]
            st = compare_policies(base_log, clog, n_boot=n_boot, n_perm=n_perm)
            st["policy"] = name
            st["accepted"] = bool(r.get("accepted"))
            st["production_ok"] = bool(r.get("production_ok"))
            stats_rows.append(st)
        stats_df = pd.DataFrame(stats_rows)

        mc_detail, mc_summary = monte_carlo_heat(
            panel, best_pol, starting_equity=starting, n_sims=n_mc
        )

        q_lookup = research_question_lookup(scored)
        answers = build_answers(
            scored=scored,
            baseline=base_m,
            best_row=best_row,
            wf=wf,
            mc_summary=mc_summary,
            q_lookup=q_lookup,
        )
        report_md = build_report(
            symbol=symbol,
            timeframe=timeframe,
            answers=answers,
            scored=scored,
            wf=wf,
            stats_df=stats_df,
            mc_summary=mc_summary,
        )
        out = self._repo.save(
            scored=scored,
            wf=wf,
            stats_df=stats_df,
            mc_detail=mc_detail,
            mc_summary=mc_summary,
            best_log=best_log,
            best_curve=best_curve,
            answers=answers,
            report_md=report_md,
        )
        self._charts.write_all(
            out_dir=out,
            scored=scored,
            best_curve=best_curve,
            baseline_curve=base_curve,
            wf=wf,
            mc_detail=mc_detail,
            best_log=best_log,
        )
        # also store baseline metrics echo
        pd.DataFrame([base_m]).to_csv(out / "baseline_metrics.csv", index=False)
        return out
