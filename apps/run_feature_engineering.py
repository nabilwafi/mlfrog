"""CLI: engineer stationary/normalized features from MarketData.

Example:
  python apps/run_feature_engineering.py --symbol XAUUSD --timeframe H1
"""

from __future__ import annotations

import argparse
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import yaml

from data.repositories.market_repository import MarketRepository
from feature_engineering.services.feature_engineering_service import FeatureEngineeringService
from settings.paths import FEATURES, LOGS, MT5_CONFIG, MT5_CONFIG_EXAMPLE, RAW, ROOT


def _load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"config not found: {path} (copy {MT5_CONFIG_EXAMPLE.name} -> {path.name})"
        )
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError("config root must be a mapping")
    return data


def _resolve_path(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (ROOT / p).resolve()


def _setup_logging(cfg: dict[str, Any]) -> None:
    log_cfg = cfg.get("logging", {})
    level = getattr(logging, str(log_cfg.get("level", "INFO")).upper(), logging.INFO)
    log_dir = _resolve_path(str(log_cfg.get("directory", LOGS)))
    log_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    fh = RotatingFileHandler(
        log_dir / str(log_cfg.get("filename", "feature_engineering.log")),
        maxBytes=int(log_cfg.get("max_bytes", 10_485_760)),
        backupCount=int(log_cfg.get("backup_count", 5)),
        encoding="utf-8",
    )
    fh.setFormatter(fmt)
    root.addHandler(fh)
    if log_cfg.get("console", True):
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        root.addHandler(ch)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Engineer stationary/normalized features")
    p.add_argument("--config", type=Path, default=MT5_CONFIG)
    p.add_argument("--symbol", default=None)
    p.add_argument("--timeframe", default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = _load_config(Path(args.config))
    _setup_logging(cfg)

    fe_cfg = dict(cfg.get("feature_engineering") or {})
    symbol = args.symbol or str(cfg.get("symbol", "XAUUSD"))
    timeframe = args.timeframe or str(cfg.get("timeframe", "H1"))
    timezone = str(cfg.get("timezone", "UTC"))

    raw_root = _resolve_path(str((cfg.get("output") or {}).get("directory", RAW)))
    # prefer dedicated raw root; fall back to artifacts/raw
    if not (raw_root / symbol.upper() / timeframe.upper()).exists():
        raw_root = RAW
    out_root = _resolve_path(str(fe_cfg.get("output_directory", FEATURES)))

    market = MarketRepository(raw_root).load_parquet(
        symbol, timeframe, timezone=timezone
    )
    matrix, metadata, out_dir = FeatureEngineeringService(
        fe_cfg, output_root=out_root
    ).run(market)

    print()
    print(f"rows={len(matrix):,} features={len(metadata)}")
    print(f"stationary={sum(1 for m in metadata if m.stationary)}")
    print(f"saved -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
