"""Persist ATRE artifacts."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class AtreRepository:
    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    def save(self, *, table: pd.DataFrame, wf: pd.DataFrame, answers: dict[str, Any], report_md: str, diag: dict[str, Any], events: pd.DataFrame, importance: pd.DataFrame, mc_base: dict, mc_best: dict, mc_detail: pd.DataFrame) -> Path:
        out = self._root
        out.mkdir(parents=True, exist_ok=True)
        table.to_csv(out / "policy_leaderboard.csv", index=False)
        wf.to_csv(out / "walk_forward.csv", index=False)
        (out / "answers.json").write_text(json.dumps(answers, indent=2, default=str), encoding="utf-8")
        (out / "atre_report.md").write_text(report_md, encoding="utf-8")
        for k, v in diag.items():
            if isinstance(v, pd.DataFrame):
                v.to_csv(out / f"{k}.csv", index=False)
            elif isinstance(v, dict):
                (out / f"{k}.json").write_text(json.dumps(v, indent=2, default=str), encoding="utf-8")
        if not events.empty:
            events.to_csv(out / "recovery_events.csv", index=False)
        if not importance.empty:
            importance.to_csv(out / "feature_importance.csv", index=False)
        (out / "monte_carlo_baseline.json").write_text(json.dumps(mc_base, indent=2), encoding="utf-8")
        (out / "monte_carlo_best.json").write_text(json.dumps(mc_best, indent=2), encoding="utf-8")
        mc_detail.to_csv(out / "monte_carlo_detail.csv", index=False)
        logger.info("Saved ATRE artifacts | dir=%s", out)
        return out
