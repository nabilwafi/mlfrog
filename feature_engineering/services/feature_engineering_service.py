"""FeatureEngineeringService — OHLCV -> stationary feature matrix + metadata."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from data.entities.market_data import MarketData
from feature_engineering.builders._transforms import atr, ema, macd, rsi
from feature_engineering.entities.feature import Feature
from feature_engineering.entities.feature_metadata import FeatureMetadata
from feature_engineering.exceptions import FeatureEngineeringError
from feature_engineering.registry.feature_registry import FeatureRegistry

logger = logging.getLogger(__name__)


def market_to_frame(market_data: MarketData) -> pd.DataFrame:
    rows = [
        {
            "timestamp": c.timestamp,
            "open": c.open,
            "high": c.high,
            "low": c.low,
            "close": c.close,
            "tick_volume": c.tick_volume,
            "spread": c.spread,
            "real_volume": c.real_volume,
        }
        for c in market_data.candles
    ]
    return pd.DataFrame(rows)


class FeatureEngineeringService:
    def __init__(self, config: dict[str, Any], *, output_root: str | Path) -> None:
        self._cfg = dict(config)
        self._output_root = Path(output_root)
        FeatureRegistry.discover()

    def run(self, market_data: MarketData) -> tuple[pd.DataFrame, list[FeatureMetadata], Path]:
        if market_data.total_candles == 0:
            raise FeatureEngineeringError("empty MarketData")

        frame = market_to_frame(market_data)
        intermediates = self._build_intermediates(frame)
        context = {"intermediates": intermediates, "params": self._cfg}

        builder_keys = list(
            self._cfg.get("builders")
            or ["trend", "volatility", "momentum", "candle", "session", "statistical"]
        )
        builder_params = dict(self._cfg.get("builder_params") or {})

        features: list[Feature] = []
        for key in builder_keys:
            builder = FeatureRegistry.create(key, builder_params.get(key))
            produced = builder.build(frame, context=context)
            logger.info("Builder %s produced %s features", key, len(produced))
            features.extend(produced)

        names = [f.metadata.name for f in features]
        if len(names) != len(set(names)):
            dup = sorted({n for n in names if names.count(n) > 1})
            raise FeatureEngineeringError(f"duplicate feature names: {dup}")

        matrix = pd.DataFrame({"timestamp": frame["timestamp"]})
        for feat in features:
            matrix[feat.metadata.name] = feat.values.to_numpy()

        if bool(self._cfg.get("drop_na", True)):
            before = len(matrix)
            matrix = matrix.dropna().reset_index(drop=True)
            logger.info("Dropped NA rows | before=%s after=%s", before, len(matrix))

        meta_list = [f.metadata for f in features]
        out_dir = (
            self._output_root
            / market_data.symbol.upper()
            / market_data.timeframe.upper()
        )
        self._save(out_dir, matrix, meta_list, market_data)
        return matrix, meta_list, out_dir

    def _build_intermediates(self, frame: pd.DataFrame) -> dict[str, pd.Series]:
        close = frame["close"].astype(float)
        high = frame["high"].astype(float)
        low = frame["low"].astype(float)
        macd_line, macd_sig, macd_hist = macd(
            close,
            fast=int(self._cfg.get("macd_fast", 12)),
            slow=int(self._cfg.get("macd_slow", 26)),
            signal=int(self._cfg.get("macd_signal", 9)),
        )
        return {
            "ema_20": ema(close, int(self._cfg.get("ema_fast", 20))),
            "ema_50": ema(close, int(self._cfg.get("ema_slow", 50))),
            "atr_14": atr(high, low, close, int(self._cfg.get("atr_period", 14))),
            "rsi_14": rsi(close, int(self._cfg.get("rsi_period", 14))),
            "macd": macd_line,
            "macd_signal": macd_sig,
            "macd_hist": macd_hist,
        }

    def _save(
        self,
        out_dir: Path,
        matrix: pd.DataFrame,
        metadata: list[FeatureMetadata],
        market_data: MarketData,
    ) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = out_dir / "feature_matrix.parquet"
        meta_path = out_dir / "feature_metadata.json"
        report_path = out_dir / "feature_report.md"

        matrix.to_parquet(parquet_path, index=False)
        payload = {
            "symbol": market_data.symbol.upper(),
            "timeframe": market_data.timeframe.upper(),
            "rows": int(len(matrix)),
            "n_features": int(len(metadata)),
            "features": [m.to_dict() for m in metadata],
        }
        meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        report_path.write_text(self._report_md(payload, matrix, metadata), encoding="utf-8")
        logger.info("Saved feature engineering artifacts | dir=%s", out_dir)

    def _report_md(
        self,
        payload: dict[str, Any],
        matrix: pd.DataFrame,
        metadata: list[FeatureMetadata],
    ) -> str:
        by_cat: dict[str, list[str]] = {}
        for m in metadata:
            by_cat.setdefault(m.category, []).append(m.name)

        lines = [
            f"# Feature Engineering Report — {payload['symbol']} {payload['timeframe']}",
            "",
            f"- Rows: `{payload['rows']}`",
            f"- Features: `{payload['n_features']}`",
            f"- Stationary features: `{sum(1 for m in metadata if m.stationary)}`",
            f"- Normalized features: `{sum(1 for m in metadata if m.normalized)}`",
            f"- Drift-sensitive features: `{sum(1 for m in metadata if m.drift_sensitive)}`",
            "",
            "## Categories",
            "",
        ]
        for cat, names in sorted(by_cat.items()):
            lines.append(f"### {cat} ({len(names)})")
            lines.append("")
            for n in names:
                lines.append(f"- `{n}`")
            lines.append("")

        lines.extend(
            [
                "## Design notes",
                "",
                "- Raw EMA / ATR / MACD levels are intermediates only — not exported.",
                "- Distances are ATR- or percent-scaled for cross-year generalization.",
                "- Calendar effects use cyclic encodings (`hour_sin/cos`, `day_sin/cos`).",
                "",
                "## Feature list",
                "",
                "| name | category | stationary | normalized | drift_sensitive |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for m in metadata:
            lines.append(
                f"| `{m.name}` | {m.category} | {m.stationary} | {m.normalized} | {m.drift_sensitive} |"
            )
        lines.append("")
        return "\n".join(lines)
