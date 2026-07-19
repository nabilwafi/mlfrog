"""RunPortfolioBacktestUseCase."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from research.portfolio_backtest import META_THRESHOLD, SCENARIOS, STARTING_EQUITY
from research.portfolio_backtest.reports.charts import PortfolioChartBuilder
from research.portfolio_backtest.reports.report import (
    build_answers,
    build_backtest_report,
    build_risk_report,
)
from research.portfolio_backtest.repositories.repository import PortfolioBacktestRepository
from research.portfolio_backtest.services.engine import prepare_trade_universe, run_portfolio
from research.portfolio_backtest.services.metrics import (
    compute_metrics,
    monthly_returns_table,
    yearly_report,
)
from research.portfolio_backtest.services.monte_carlo import monte_carlo

logger = logging.getLogger(__name__)


class RunPortfolioBacktestUseCase:
    def __init__(
        self,
        repository: PortfolioBacktestRepository,
        *,
        config: dict[str, Any] | None = None,
    ) -> None:
        self._repo = repository
        self._cfg = dict(config or {})
        self._charts = PortfolioChartBuilder()

    def execute(self, *, symbol: str, timeframe: str, oof_path: Path) -> Path:
        starting = float(self._cfg.get("starting_equity", STARTING_EQUITY))
        meta_thr = float(self._cfg.get("meta_threshold", META_THRESHOLD))
        n_sims = int(self._cfg.get("n_monte_carlo", 1000))
        seed = int(self._cfg.get("random_seed", 42))
        # Prefer fractional 1% for narrative once risk min-lot is shown broken at $80
        preferred = str(self._cfg.get("preferred_scenario", "C_risk_1pct_frac"))

        oof = pd.read_parquet(oof_path)
        if "net_return" not in oof.columns:
            raise ValueError("oof missing net_return")
        if "meta_proba" not in oof.columns:
            raise ValueError("oof missing meta_proba — run meta model first")

        universes = {
            "primary": prepare_trade_universe(oof, system="primary", meta_thr=meta_thr),
            "primary_plus_meta": prepare_trade_universe(
                oof, system="primary_plus_meta", meta_thr=meta_thr
            ),
        }
        logger.info(
            "Universe primary=%s meta@%.2f=%s",
            len(universes["primary"]),
            meta_thr,
            len(universes["primary_plus_meta"]),
        )

        metric_rows: list[dict] = []
        mc_rows: list[dict] = []
        yearly_by_key: dict[str, pd.DataFrame] = {}
        # Persist primary artifacts for preferred compare + full meta scenario dumps
        artifacts: dict[str, Any] = {}

        for system, trades in universes.items():
            for scen_id, mode, param, enforce_min in SCENARIOS:
                fixed = param if mode == "fixed" else None
                risk = param if mode == "risk" else None
                log, curve = run_portfolio(
                    trades,
                    starting_equity=starting,
                    mode=mode,
                    fixed_lots=fixed,
                    risk_pct=risk,
                    enforce_volume_min=enforce_min,
                )
                metrics = compute_metrics(log, curve, starting_equity=starting)
                metrics.update(
                    {
                        "system": system,
                        "scenario": scen_id,
                        "mode": mode,
                        "param": param,
                        "enforce_volume_min": enforce_min,
                        "meta_threshold": meta_thr,
                        "starting_equity": starting,
                    }
                )
                metric_rows.append(metrics)
                yrep = yearly_report(log, starting_equity=starting)
                yearly_by_key[f"{system}__{scen_id}"] = yrep
                monthly = monthly_returns_table(curve)

                run_mc = False
                if len(trades) > 0:
                    # Skip MC when realistic risk cannot size any trade at $80
                    if mode == "risk" and enforce_min:
                        run_mc = False
                    elif mode == "fixed":
                        run_mc = system == "primary_plus_meta"  # light check only
                    else:
                        run_mc = True
                mc_detail = pd.DataFrame()
                mc_sum: dict = {"n_sims": 0}
                if run_mc:
                    sims = min(n_sims, 200) if mode == "fixed" else n_sims
                    mc_detail, mc_sum = monte_carlo(
                        trades,
                        starting_equity=starting,
                        mode=mode,
                        fixed_lots=fixed,
                        risk_pct=risk,
                        n_sims=sims,
                        seed=seed,
                        enforce_volume_min=enforce_min,
                    )
                mc_rows.append({"system": system, "scenario": scen_id, **mc_sum})

                key = f"{system}__{scen_id}"
                artifacts[key] = {
                    "trade_log": log,
                    "equity_curve": curve,
                    "monthly": monthly,
                    "mc_detail": mc_detail,
                }
                logger.info(
                    "%s %s final=%.2f trades=%s dd=%.3f",
                    system,
                    scen_id,
                    metrics["final_equity"],
                    metrics["trades"],
                    metrics["max_drawdown"],
                )

        metrics_table = pd.DataFrame(metric_rows)
        mc_table = pd.DataFrame(mc_rows)
        answers = build_answers(
            metrics_table=metrics_table,
            yearly=yearly_by_key.get(f"primary_plus_meta__{preferred}", pd.DataFrame()),
            mc_rows=mc_table,
            preferred_scenario=preferred,
        )
        report_md = build_backtest_report(
            symbol=symbol,
            timeframe=timeframe,
            metrics_table=metrics_table,
            answers=answers,
            yearly_by_key=yearly_by_key,
        )
        risk_md = build_risk_report(answers=answers, mc_rows=mc_table)

        out = self._repo.save(
            metrics_table=metrics_table,
            mc_table=mc_table,
            answers=answers,
            report_md=report_md,
            risk_md=risk_md,
            artifacts=artifacts,
            preferred_key=f"primary_plus_meta__{preferred}",
            primary_compare_key=f"primary__{preferred}",
            yearly_by_key=yearly_by_key,
        )

        # Charts for preferred meta + primary compare
        charts_root = out / "charts"
        charts_root.mkdir(parents=True, exist_ok=True)
        for label, key in (
            ("meta", f"primary_plus_meta__{preferred}"),
            ("primary", f"primary__{preferred}"),
        ):
            art = artifacts.get(key)
            if not art:
                continue
            self._charts.write_all(
                out_dir=out,
                equity_curve=art["equity_curve"],
                trade_log=art["trade_log"],
                monthly=art["monthly"],
                mc_detail=art["mc_detail"] if not art["mc_detail"].empty else None,
                prefix=label,
            )
        # Also chart A_fixed and realistic C for meta
        for scen in ("A_fixed_1lot", "C_risk_1pct", "C_risk_1pct_frac"):
            key = f"primary_plus_meta__{scen}"
            art = artifacts.get(key)
            if art:
                self._charts.write_all(
                    out_dir=out,
                    equity_curve=art["equity_curve"],
                    trade_log=art["trade_log"],
                    monthly=art["monthly"],
                    mc_detail=art["mc_detail"] if not art["mc_detail"].empty else None,
                    prefix=f"meta_{scen}",
                )
        self._charts.compare_systems(
            charts_root,
            {
                "primary": artifacts[f"primary__{preferred}"]["equity_curve"],
                "primary_plus_meta": artifacts[f"primary_plus_meta__{preferred}"]["equity_curve"],
            },
            title=f"Primary vs Meta ({preferred}, start ${starting:.0f})",
        )
        return out
