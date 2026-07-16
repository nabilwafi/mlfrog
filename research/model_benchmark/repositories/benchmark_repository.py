"""Persist model benchmark CSVs and markdown."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class BenchmarkRepository:
    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    @property
    def root(self) -> Path:
        return self._root

    def save(
        self,
        *,
        summary: pd.DataFrame,
        metrics: pd.DataFrame,
        rankings: pd.DataFrame,
        report_md: str,
    ) -> Path:
        out = self._root
        out.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out / "model_benchmark.csv", index=False)
        metrics.to_csv(out / "model_benchmark_windows.csv", index=False)
        rankings.to_csv(out / "model_rankings.csv", index=False)
        (out / "model_benchmark.md").write_text(report_md, encoding="utf-8")
        logger.info("Saved model benchmark artifacts | dir=%s", out)
        return out
