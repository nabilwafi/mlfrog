"""Persist research artifacts under artifacts/research/{symbol}/{tf}/{side}/."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from research.entities.research_report import ResearchReport

logger = logging.getLogger(__name__)


class ResearchRepository:
    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    def report_dir(self, symbol: str, timeframe: str, side: str) -> Path:
        return self._root / symbol.upper() / timeframe.upper() / side.lower()

    def save(
        self,
        report: ResearchReport,
        *,
        files: dict[str, str | pd.DataFrame],
    ) -> Path:
        out = self.report_dir(report.symbol, report.timeframe, report.side)
        logger.info("Saving research | path=%s", out)
        out.mkdir(parents=True, exist_ok=True)
        for name, payload in files.items():
            path = out / name
            if isinstance(payload, pd.DataFrame):
                payload.to_csv(path, index=False)
            else:
                path.write_text(str(payload), encoding="utf-8")
        logger.info("Saved research | dir=%s files=%s", out, sorted(files))
        return out
