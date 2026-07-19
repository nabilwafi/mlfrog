"""Persist diagnostics artifacts under artifacts/diagnostics/{symbol}/{tf}/{side}/."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from diagnostics.entities.diagnostics_report import DiagnosticsReport
from diagnostics.exceptions import DiagnosticsRepositoryError

logger = logging.getLogger(__name__)


class DiagnosticsRepository:
    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    def report_dir(self, symbol: str, timeframe: str, side: str) -> Path:
        return self._root / symbol.upper() / timeframe.upper() / side.lower()

    def save(
        self,
        report: DiagnosticsReport,
        *,
        markdown: str,
        tables: dict[str, pd.DataFrame],
    ) -> Path:
        out = self.report_dir(report.symbol, report.timeframe, report.side)
        logger.info("Saving diagnostics | path=%s warnings=%s", out, len(report.warnings))
        try:
            out.mkdir(parents=True, exist_ok=True)
            (out / "diagnostics_report.md").write_text(markdown, encoding="utf-8")
            (out / "metadata.json").write_text(
                json.dumps(report.to_metadata(), indent=2, default=str),
                encoding="utf-8",
            )
            for name, frame in tables.items():
                path = out / name
                if not name.endswith(".csv"):
                    path = out / f"{name}.csv"
                frame.to_csv(path, index=False)
        except Exception as exc:
            raise DiagnosticsRepositoryError(f"save failed: {exc}") from exc
        logger.info("Saved diagnostics | dir=%s", out)
        return out

    def load_metadata(self, symbol: str, timeframe: str, side: str) -> dict[str, Any]:
        path = self.report_dir(symbol, timeframe, side) / "metadata.json"
        if not path.is_file():
            raise DiagnosticsRepositoryError(f"metadata.json not found: {path}")
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise DiagnosticsRepositoryError(f"load_metadata failed: {exc}") from exc
