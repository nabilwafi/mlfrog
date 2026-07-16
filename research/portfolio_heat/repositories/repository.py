"""Persist portfolio heat artifacts."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class PortfolioHeatRepository:
    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    def save(
        self,
        *,
        scored: pd.DataFrame,
        wf: pd.DataFrame,
        stats_df: pd.DataFrame,
        mc_detail: pd.DataFrame,
        mc_summary: dict[str, Any],
        best_log: pd.DataFrame,
        best_curve: pd.DataFrame,
        answers: dict[str, Any],
        report_md: str,
    ) -> Path:
        out = self._root
        out.mkdir(parents=True, exist_ok=True)
        scored.to_csv(out / "policy_leaderboard.csv", index=False)
        wf.to_csv(out / "walk_forward.csv", index=False)
        stats_df.to_csv(out / "statistical_validation.csv", index=False)
        mc_detail.to_csv(out / "monte_carlo_detail.csv", index=False)
        (out / "monte_carlo_summary.json").write_text(json.dumps(mc_summary, indent=2), encoding="utf-8")
        best_log.to_parquet(out / "best_policy_trades.parquet", index=False)
        best_curve.to_parquet(out / "best_policy_equity.parquet", index=False)
        (out / "answers.json").write_text(json.dumps(answers, indent=2, default=str), encoding="utf-8")
        (out / "portfolio_heat_report.md").write_text(report_md, encoding="utf-8")
        logger.info("Saved portfolio heat artifacts | dir=%s", out)
        return out
