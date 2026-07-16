"""RunPositionMgmtUseCase."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from research.portfolio_backtest.services.metrics import yearly_report
from research.position_mgmt import STARTING_EQUITY
from research.position_mgmt.reports.charts import PositionMgmtChartBuilder
from research.position_mgmt.reports.report import build_answers, build_report
from research.position_mgmt.repositories.repository import PositionMgmtRepository
from research.position_mgmt.services.evaluate import evaluate_all, run_fixed_portfolio
from research.position_mgmt.services.monte_carlo import monte_carlo_fixed
from research.position_mgmt.services.paths import load_h1, load_production_trades
from research.position_mgmt.services.simulator import ManageConfig, simulate_panel

logger = logging.getLogger(__name__)


def _rank_techniques(table: pd.DataFrame) -> list[dict[str, Any]]:
    ranks = []
    order = ["be", "partial", "trail", "time", "vol", "pyramid", "scale_in"]
    for fam in order:
        sub = table.loc[table["family"] == fam]
        if sub.empty:
            ranks.append({"technique": fam, "score": -1.0, "best": "off", "accepted_sig": False})
            continue
        sig = sub.loc[sub["accepted_sig"] == True] if "accepted_sig" in sub.columns else pd.DataFrame()  # noqa: E712
        if sig.empty:
            best = sub.iloc[0]
            ranks.append(
                {
                    "technique": fam,
                    "score": float(best.get("calmar") or -1),
                    "best": str(best["policy"]),
                    "accepted_sig": False,
                }
            )
        else:
            best = sig.iloc[0]
            ranks.append(
                {
                    "technique": fam,
                    "score": float(best.get("calmar") or 0),
                    "best": str(best["policy"]),
                    "accepted_sig": True,
                }
            )
    ranks.sort(key=lambda r: (r["accepted_sig"], r["score"]), reverse=True)
    for i, r in enumerate(ranks, 1):
        r["rank"] = i
    return ranks


class RunPositionMgmtUseCase:
    def __init__(self, repository: PositionMgmtRepository, *, config: dict[str, Any] | None = None) -> None:
        self._repo = repository
        self._cfg = dict(config or {})
        self._charts = PositionMgmtChartBuilder()

    def execute(self, *, symbol: str, timeframe: str, heat_trades_path: Path) -> Path:
        starting = float(self._cfg.get("starting_equity", STARTING_EQUITY))
        n_mc = int(self._cfg.get("n_monte_carlo", 500))
        n_boot = int(self._cfg.get("n_bootstrap", 1000))
        n_perm = int(self._cfg.get("n_permutation", 1000))

        heat = pd.read_parquet(heat_trades_path)
        logger.info("Loaded heat log rows=%s accepted=%s", len(heat), int((~heat["skipped"]).sum()))
        h1 = load_h1(symbol)

        table, trades, arts, logs = evaluate_all(
            heat,
            h1,
            starting_equity=starting,
            n_boot=n_boot,
            n_perm=n_perm,
        )
        base_m = arts["baseline_metrics"]
        ranks = _rank_techniques(table)

        sig = table.loc[table["accepted_sig"] == True] if "accepted_sig" in table.columns else pd.DataFrame()  # noqa: E712
        if not sig.empty:
            best_row = sig.iloc[0]
        else:
            best_row = table.loc[table["policy"] == "baseline"].iloc[0]

        best_name = str(best_row["policy"])
        logger.info("Best policy=%s accepted_sig=%s", best_name, best_row.get("accepted_sig"))

        base_curve = arts["curves"]["baseline"]
        best_curve = arts["curves"].get(best_name, base_curve)
        best_panel = arts["panels"].get(best_name)
        if best_name != "baseline" and best_panel is None:
            best_panel = trades

        # Walk-forward on best: yearly chronological from its trade log
        best_log = logs.get(best_name, logs["baseline"])
        wf = yearly_report(best_log, starting_equity=starting)
        # also standalone-year CAGR-ish via final equity from year slice of panel
        stand_rows = []
        panel_for_wf = arts["panels"].get(best_name, trades)
        for year, g in panel_for_wf.groupby("valid_year"):
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
        if not wf.empty:
            wf = wf.copy()
            wf["mode"] = "chronological"
        wf_all = pd.concat([stand, wf], ignore_index=True) if not stand.empty else wf

        _, mc_base = monte_carlo_fixed(trades, starting_equity=starting, n_sims=n_mc)
        mc_panel = best_panel if best_panel is not None else trades
        mc_detail, mc_best = monte_carlo_fixed(mc_panel, starting_equity=starting, n_sims=n_mc)
        mc_best["policy"] = best_name
        mc_base["policy"] = "baseline"

        answers = build_answers(
            table=table,
            baseline=base_m,
            best_row=best_row,
            mc_base=mc_base,
            mc_best=mc_best,
            ranks=ranks,
        )
        report_md = build_report(symbol=symbol, timeframe=timeframe, answers=answers, table=table, wf=wf_all)
        out = self._repo.save(
            table=table,
            wf=wf_all,
            answers=answers,
            report_md=report_md,
            mc_base=mc_base,
            mc_best=mc_best,
            mc_detail=mc_detail,
            best_panel=best_panel if best_name != "baseline" else None,
            baseline_metrics=base_m,
        )
        self._charts.write_all(
            out_dir=out,
            table=table,
            base_curve=base_curve,
            best_curve=best_curve,
            mc_detail=mc_detail,
            base_panel=trades,
            best_panel=best_panel if best_name != "baseline" else None,
            ranks=ranks,
        )
        # baseline path fidelity check
        from research.position_mgmt.services.simulator import ManageConfig

        chk = simulate_panel(trades.head(50), h1, ManageConfig(name="baseline", family="baseline"))
        corr = float(chk["net_return_sim"].corr(chk["net_return_base"]))
        (out / "path_fidelity.txt").write_text(
            f"corr(sim_baseline, stored_net_return) on 50 trades = {corr}\n", encoding="utf-8"
        )
        logger.info("Path fidelity corr=%s", corr)
        return out
