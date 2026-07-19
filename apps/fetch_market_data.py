"""CLI: fetch → validate → save market data.

Example (from repo root):
  python -m apps.fetch_market_data --provider mt5 --symbol XAUUSD --timeframe H1 --start 2020-01-01 --end 2025-12-31
  python apps/fetch_market_data.py --provider mt5 --symbol XAUUSD --timeframe H1 --start 2020-01-01 --end 2025-12-31
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

# Allow `python apps/fetch_market_data.py` without installing a package
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import yaml

from data.providers.base_provider import BaseMarketProvider
from data.providers.csv_provider import CSVProvider
from data.providers.mt5_provider import MT5Provider
from data.providers.parquet_provider import ParquetProvider
from data.repositories.market_repository import MarketRepository
from data.services.market_data_service import MarketDataService
from data.validators.candle_validator import CandleValidator
from settings.paths import LOGS, MT5_CONFIG, MT5_CONFIG_EXAMPLE, RAW, ROOT


def _load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"config not found: {path} (copy {MT5_CONFIG_EXAMPLE.name} → {path.name})"
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
    filename = str(log_cfg.get("filename", "market_data.log"))
    log_path = log_dir / filename

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=int(log_cfg.get("max_bytes", 10_485_760)),
        backupCount=int(log_cfg.get("backup_count", 5)),
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    if log_cfg.get("console", True):
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(fmt)
        root.addHandler(console)


def _parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d")


def _build_provider(name: str, cfg: dict[str, Any], source: str | None) -> BaseMarketProvider:
    key = name.strip().lower()
    if key == "mt5":
        return MT5Provider(cfg)
    if key == "csv":
        if not source:
            raise ValueError("--source is required for provider=csv")
        return CSVProvider(_resolve_path(source))
    if key == "parquet":
        if not source:
            raise ValueError("--source is required for provider=parquet")
        return ParquetProvider(_resolve_path(source))
    raise ValueError(f"unknown provider: {name!r} (mt5|csv|parquet)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fetch / validate / save market data")
    p.add_argument("--config", type=Path, default=MT5_CONFIG, help="Path to config.yaml")
    p.add_argument("--provider", required=True, choices=("mt5", "csv", "parquet"))
    p.add_argument("--symbol", default=None)
    p.add_argument("--timeframe", default=None)
    p.add_argument("--start", default=None, help="YYYY-MM-DD")
    p.add_argument("--end", default=None, help="YYYY-MM-DD")
    p.add_argument("--source", default=None, help="Input file for csv/parquet providers")
    p.add_argument("--format", default=None, choices=("parquet", "csv"), dest="fmt")
    p.add_argument(
        "--check-continuity",
        action="store_true",
        help="Fail validation on timeframe gaps (strict; weekends will fail for FX)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = _load_config(Path(args.config))
    _setup_logging(cfg)

    symbol = args.symbol or str(cfg.get("symbol", "XAUUSD"))
    timeframe = args.timeframe or str(cfg.get("timeframe", "H1"))
    dr = cfg.get("date_range", {})
    start = _parse_date(args.start or str(dr.get("start_date", "2020-01-01")))
    end = _parse_date(args.end or str(dr.get("end_date", "2025-12-31")))
    timezone = str(cfg.get("timezone", "UTC"))
    out = cfg.get("output", {})
    raw_root = _resolve_path(str(out.get("directory", RAW)))
    fmt = args.fmt or str(out.get("format", "parquet"))

    validation_cfg = cfg.get("validation", {})
    check_continuity = bool(args.check_continuity or validation_cfg.get("check_continuity", False))

    provider = _build_provider(args.provider, cfg, args.source)
    repo = MarketRepository(raw_root)
    validator = CandleValidator(check_continuity=check_continuity)
    service = MarketDataService(
        provider, repo, validator, default_format=fmt, timezone=timezone
    )

    try:
        market, path = service.fetch_validate_save(symbol, timeframe, start, end, fmt=fmt)
    finally:
        service.close()

    print()
    print(f"Fetched : {market.total_candles:,} candles")
    print("Validation : PASS")
    print(f"Saved :")
    print(f"  {path}")
    if market.start_time and market.end_time:
        print(f"Range : {market.start_time.isoformat()} → {market.end_time.isoformat()}")
    print(f"Symbol : {market.symbol} | Timeframe : {market.timeframe} | TZ : {market.timezone}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
