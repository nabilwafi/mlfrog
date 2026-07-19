"""Persist feature-diagnostics artifacts under research/{symbol}/{tf}/{side}/."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from feature_diagnostics.entities.feature_diagnostics_report import FeatureDiagnosticsReport

logger = logging.getLogger(__name__)


class FeatureDiagnosticsRepository:
    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    def report_dir(self, symbol: str, timeframe: str, side: str) -> Path:
        return self._root / symbol.upper() / timeframe.upper() / side.lower()

    def save(
        self,
        report: FeatureDiagnosticsReport,
        *,
        tables: dict[str, pd.DataFrame],
        markdown: str,
    ) -> Path:
        out = self.report_dir(report.symbol, report.timeframe, report.side)
        out.mkdir(parents=True, exist_ok=True)
        logger.info("Saving feature diagnostics | path=%s", out)
        for name, frame in tables.items():
            path = out / name
            if not str(name).endswith(".csv"):
                path = out / f"{name}.csv"
            frame.to_csv(path, index=False)
        (out / "feature_diagnostics_report.md").write_text(markdown, encoding="utf-8")
        logger.info("Saved feature diagnostics | files=%s", sorted(tables) + ["feature_diagnostics_report.md"])
        return out
