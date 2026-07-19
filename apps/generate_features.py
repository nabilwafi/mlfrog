"""CLI: generate FeatureSet from stored MarketData.

Example:
  python apps/generate_features.py --symbol XAUUSD --timeframe H1
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
from features.repositories.feature_repository import FeatureRepository
from features.services.feature_service import FeatureService
from settings.paths import FEATURES, LOGS, MT5_CONFIG, MT5_CONFIG_EXAMPLE, RAW, ROOT


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
    filename = str(log_cfg.get("filename", "features.log"))
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    fh = RotatingFileHandler(
        log_dir / filename,
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
    p = argparse.ArgumentParser(description="Generate features from MarketData")
    p.add_argument("--config", type=Path, default=MT5_CONFIG)
    p.add_argument("--symbol", default=None)
    p.add_argument("--timeframe", default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = _load_config(Path(args.config))
    _setup_logging(cfg)

    symbol = args.symbol or str(cfg.get("symbol", "XAUUSD"))
    timeframe = args.timeframe or str(cfg.get("timeframe", "H1"))
    timezone = str(cfg.get("timezone", "UTC"))
    feature_cfg = cfg.get("features")
    if not feature_cfg:
        raise SystemExit("config missing 'features' section — see config.example.yaml")

    out = cfg.get("output", {})
    raw_root = _resolve_path(str(out.get("directory", RAW)))
    feat_root = _resolve_path(str(cfg.get("features", {}).get("output_directory", FEATURES)))

    market_repo = MarketRepository(raw_root)
    feature_repo = FeatureRepository(feat_root)
    service = FeatureService(
        market_repo, feature_repo, feature_cfg, timezone=timezone
    )
    feature_set, path, stats = service.run(symbol, timeframe)

    print()
    print(f"Raw Candles : {stats['raw_candles']:,}")
    print(f"Indicators : {stats['indicators']}")
    print(f"Interaction Features : {stats['interactions']}")
    print(f"Total Features : {stats['total_features']}")
    print(f"NaN Removed : {stats['nan_removed']}")
    print(f"Rows : {stats['row_count']:,}")
    print("Saved :")
    print(f"  {path}")
    print(f"Version : {feature_set.feature_version} | Pipeline : {feature_set.pipeline_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
