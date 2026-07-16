"""RunAtreUseCase."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from research.atre import STARTING_EQUITY
from research.atre.reports.charts import AtreChartBuilder
from research.atre.reports.report import build_answers, build_report
from research.atre.repositories.repository import AtreRepository
from research.atre.services.diagnostics import (
    mae_sl_rate,
    momentum_collapse_edge,
    recovery_probability,
    session_recovery,
    side_recovery,
    structure_fail_edge,
    underwater_analysis,
    vol_recovery,
)
from research.atre.services.evaluate import evaluate_atre, run_fixed_portfolio
from research.atre.services.path_diag import build_path_library
from research.atre.services.recovery_score import fit_loo_recovery_scores, score_map
from research.portfolio_backtest.services.metrics import yearly_report
from research.position_mgmt.services.monte_carlo import monte_carlo_fixed
from research.position_mgmt.services.paths import load_h1, load_production_trades

logger = logging.getLogger(__name__)


def _wf_stable(wf: pd.DataFrame, starting: float) -> bool:
    if wf.empty:
        return False
    stand = wf.loc[wf["mode"] == "standalone_year"] if "mode" in wf.columns else wf
    if stand.empty or "final_equity" not in stand.columns:
        return False
    # every year ends above 90% of start (no ruin) and at least 3 years
    ok = (stand["final_equity"].astype(float) >= 0.9 * starting).all() and len(stand) >= 3
    return bool(ok)


class RunAtreUseCase:
    def __init__(self, repository: AtreRepository, *, config: dict[str, Any] | None = None) -> None:
        self._repo = repository
        self._cfg = dict(config or {})
        self._charts = AtreChartBuilder()

    def execute(self, *, symbol: str, timeframe: str, heat_trades_path: Path) -> Path:
        starting = float(self._cfg.get("starting_equity", STARTING_EQUITY))
        n_mc = int(self._cfg.get("n_monte_carlo", 500))
        n_boot = int(self._cfg.get("n_bootstrap", 1000))
        n_perm = int(self._cfg.get("n_permutation", 1000))

        heat = pd.read_parquet(heat_trades_path)
        trades = load_production_trades(heat)
        logger.info("ATRE trades=%s", len(trades))
        h1 = load_h1(symbol)
        lib = build_path_library(trades, h1)

        # Diagnostics
        mae_df = mae_sl_rate(lib)
        uw_df = underwater_analysis(lib)
        rec_df = recovery_probability(lib)
        mom = momentum_collapse_edge(lib)
        struct = structure_fail_edge(lib)
        sess_df = session_recovery(lib)
        vol_df = vol_recovery(lib)
        side_df = side_recovery(lib)

        events, importance = fit_loo_recovery_scores(lib, min_mae=0.4)
        scores = score_map(events)
        logger.info("Recovery events=%s scored=%s", len(events), len(scores))

        table, arts = evaluate_atre(
            trades, lib, scores, starting_equity=starting, n_boot=n_boot, n_perm=n_perm
        )
        base_m = arts["baseline_metrics"]
        sig = table.loc[table["accepted_sig"] == True] if "accepted_sig" in table.columns else pd.DataFrame()  # noqa: E712
        best_row = sig.iloc[0] if not sig.empty else table.loc[table["policy"] == "baseline"].iloc[0]
        best_name = str(best_row["policy"])
        logger.info("Best ATRE policy=%s accepted_sig=%s", best_name, best_row.get("accepted_sig"))

        best_log = arts["logs"].get(best_name, arts["logs"]["baseline"])
        best_panel = arts["panels"].get(best_name, arts["panels"]["baseline"])
        chron = yearly_report(best_log, starting_equity=starting)
        stand_rows = []
        for year, g in best_panel.groupby("valid_year"):
            _, _, m = run_fixed_portfolio(g, starting_equity=starting)
            stand_rows.append(
                {
                    "valid_year": int(year),
                    "mode": "standalone_year",
                    "cagr": m.get("cagr"),
                    "max_drawdown": m.get("max_drawdown"),
                    "final_equity": m.get("final_equity"),
                    "trades": m.get("trades"),
                }
            )
        stand = pd.DataFrame(stand_rows)
        if not chron.empty:
            chron = chron.copy()
            chron["mode"] = "chronological"
        wf = pd.concat([stand, chron], ignore_index=True) if not stand.empty else chron
        wf_ok = _wf_stable(wf, starting)

        _, mc_base = monte_carlo_fixed(arts["panels"]["baseline"], starting_equity=starting, n_sims=n_mc)
        mc_detail, mc_best = monte_carlo_fixed(best_panel, starting_equity=starting, n_sims=n_mc)
        mc_base["policy"] = "baseline"
        mc_best["policy"] = best_name

        diagnostics = {
            "momentum": mom,
            "structure": struct,
            "recovery_prob": rec_df,
            "side_recovery": side_df,
        }
        answers = build_answers(
            table=table,
            baseline=base_m,
            best_row=best_row,
            diagnostics=diagnostics,
            importance=importance,
            mc_base=mc_base,
            mc_best=mc_best,
            wf_ok=wf_ok,
        )
        # If verdict says no, force best to baseline for charts/MC labeling
        if not answers["q1_engine_exist"]:
            best_name = "baseline"
            best_panel = arts["panels"]["baseline"]
            mc_detail, mc_best = monte_carlo_fixed(best_panel, starting_equity=starting, n_sims=n_mc)
            mc_best["policy"] = "baseline"
            answers["mc_best"] = mc_best

        diag_tables = {
            "mae_sl_rate": mae_df,
            "underwater": uw_df,
            "recovery_probability": rec_df,
            "session_recovery": sess_df,
            "vol_recovery": vol_df,
            "side_recovery": side_df,
        }
        report_md = build_report(
            symbol=symbol,
            timeframe=timeframe,
            answers=answers,
            table=table,
            wf=wf,
            diag_tables=diag_tables,
        )
        out = self._repo.save(
            table=table,
            wf=wf,
            answers=answers,
            report_md=report_md,
            diag={**diag_tables, "momentum": mom, "structure": struct},
            events=events,
            importance=importance,
            mc_base=mc_base,
            mc_best=mc_best,
            mc_detail=mc_detail,
        )
        self._charts.write_all(
            out_dir=out,
            lib=lib,
            recovery_prob=rec_df,
            underwater=uw_df,
            table=table,
            events=events,
            importance=importance,
            wf=wf,
            mc=mc_detail,
            base_curve=arts["curves"]["baseline"],
            best_curve=arts["curves"].get(best_name, arts["curves"]["baseline"]),
        )
        return out
