"""Persist portfolio backtest artifacts."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class PortfolioBacktestRepository:
    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    @property
    def root(self) -> Path:
        return self._root

    def save(
        self,
        *,
        metrics_table: pd.DataFrame,
        mc_table: pd.DataFrame,
        answers: dict[str, Any],
        report_md: str,
        risk_md: str,
        artifacts: dict[str, Any],
        preferred_key: str,
        primary_compare_key: str,
        yearly_by_key: dict[str, pd.DataFrame],
    ) -> Path:
        out = self._root
        out.mkdir(parents=True, exist_ok=True)

        metrics_table.to_csv(out / "portfolio_metrics.csv", index=False)
        mc_table.to_csv(out / "monte_carlo_summary.csv", index=False)
        (out / "backtest_report.md").write_text(report_md, encoding="utf-8")
        (out / "risk_report.md").write_text(risk_md, encoding="utf-8")
        (out / "answers.json").write_text(json.dumps(answers, indent=2, default=str), encoding="utf-8")

        # Preferred deliverables at root (user-requested filenames)
        pref = artifacts.get(preferred_key) or {}
        if pref:
            pref["equity_curve"].to_csv(out / "equity_curve.csv", index=False)
            pref["trade_log"].to_csv(out / "trade_log.csv", index=False)
            pref["monthly"].to_csv(out / "monthly_returns.csv", index=False)

        # Dump all scenario logs under scenarios/
        scen_dir = out / "scenarios"
        scen_dir.mkdir(parents=True, exist_ok=True)
        for key, art in artifacts.items():
            d = scen_dir / key
            d.mkdir(parents=True, exist_ok=True)
            art["equity_curve"].to_csv(d / "equity_curve.csv", index=False)
            art["trade_log"].to_csv(d / "trade_log.csv", index=False)
            art["monthly"].to_csv(d / "monthly_returns.csv", index=False)
            if art.get("mc_detail") is not None and not art["mc_detail"].empty:
                art["mc_detail"].to_csv(d / "monte_carlo_paths.csv", index=False)

        ydir = out / "yearly"
        ydir.mkdir(parents=True, exist_ok=True)
        for key, ydf in yearly_by_key.items():
            if not ydf.empty:
                ydf.to_csv(ydir / f"{key}.csv", index=False)

        logger.info("Saved portfolio backtest | dir=%s", out)
        return out
