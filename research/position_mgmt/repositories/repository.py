"""Persist position management artifacts."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class PositionMgmtRepository:
    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    def save(
        self,
        *,
        table: pd.DataFrame,
        wf: pd.DataFrame,
        answers: dict[str, Any],
        report_md: str,
        mc_base: dict[str, Any],
        mc_best: dict[str, Any],
        mc_detail: pd.DataFrame,
        best_panel: pd.DataFrame | None,
        baseline_metrics: dict[str, Any],
    ) -> Path:
        out = self._root
        out.mkdir(parents=True, exist_ok=True)
        table.to_csv(out / "policy_leaderboard.csv", index=False)
        wf.to_csv(out / "walk_forward.csv", index=False)
        (out / "answers.json").write_text(json.dumps(answers, indent=2, default=str), encoding="utf-8")
        (out / "position_mgmt_report.md").write_text(report_md, encoding="utf-8")
        (out / "monte_carlo_baseline.json").write_text(json.dumps(mc_base, indent=2), encoding="utf-8")
        (out / "monte_carlo_best.json").write_text(json.dumps(mc_best, indent=2), encoding="utf-8")
        mc_detail.to_csv(out / "monte_carlo_detail.csv", index=False)
        pd.DataFrame([baseline_metrics]).to_csv(out / "baseline_metrics.csv", index=False)
        if best_panel is not None:
            best_panel.to_parquet(out / "best_policy_panel.parquet", index=False)
        logger.info("Saved position mgmt artifacts | dir=%s", out)
        return out
