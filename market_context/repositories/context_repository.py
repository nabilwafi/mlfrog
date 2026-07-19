"""Persist market context artifacts."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from market_context.entities.context_feature import ContextFeatureSpec

logger = logging.getLogger(__name__)


class ContextRepository:
    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    def output_dir(self, symbol: str, base_timeframe: str) -> Path:
        return self._root / symbol.upper() / base_timeframe.upper()

    def save(
        self,
        *,
        symbol: str,
        base_timeframe: str,
        context_tf: str,
        joined: pd.DataFrame,
        specs: list[ContextFeatureSpec],
        report_md: str,
        metadata: dict[str, Any],
    ) -> Path:
        out = self.output_dir(symbol, base_timeframe)
        out.mkdir(parents=True, exist_ok=True)

        joined.to_parquet(out / "context.parquet", index=False)
        (out / "context_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        (out / "context_report.md").write_text(report_md, encoding="utf-8")

        feat_rows = [s.to_dict() for s in specs]
        pd.DataFrame(feat_rows).to_csv(out / "context_features.csv", index=False)

        logger.info("Saved market context artifacts | dir=%s rows=%s", out, len(joined))
        return out
