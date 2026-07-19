"""CLI: generate LabelSet(s) from stored MarketData.

Example:
  python apps/generate_labels.py --symbol XAUUSD --timeframe H1 --strategy triple_barrier --sides long short
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
from labels.repositories.label_repository import LabelRepository
from labels.services.label_service import LabelService
from settings.paths import LABELS, LOGS, MT5_CONFIG, MT5_CONFIG_EXAMPLE, RAW, ROOT


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
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    fh = RotatingFileHandler(
        log_dir / str(log_cfg.get("filename", "labels.log")),
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
    p = argparse.ArgumentParser(description="Generate labels from MarketData")
    p.add_argument("--config", type=Path, default=MT5_CONFIG)
    p.add_argument("--symbol", default=None)
    p.add_argument("--timeframe", default=None)
    p.add_argument("--strategy", default=None)
    p.add_argument(
        "--sides",
        nargs="+",
        default=None,
        choices=("long", "short"),
        help="One or more sides (default from config)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = _load_config(Path(args.config))
    _setup_logging(cfg)

    symbol = args.symbol or str(cfg.get("symbol", "XAUUSD"))
    timeframe = args.timeframe or str(cfg.get("timeframe", "H1"))
    timezone = str(cfg.get("timezone", "UTC"))
    label_cfg = cfg.get("labels")
    if not label_cfg:
        raise SystemExit("config missing 'labels' section — see config.example.yaml")

    strategy = args.strategy or str(label_cfg.get("strategy", "triple_barrier"))
    sides = args.sides or list(label_cfg.get("sides") or ["long"])
    out = cfg.get("output", {})
    raw_root = _resolve_path(str(out.get("directory", RAW)))
    label_root = _resolve_path(str(label_cfg.get("output_directory", LABELS)))

    service = LabelService(
        MarketRepository(raw_root),
        LabelRepository(label_root),
        label_cfg,
        timezone=timezone,
    )
    results = service.run(symbol, timeframe, strategy=strategy, sides=sides)

    print()
    print(f"Strategy : {strategy}")
    for label_set, path, report in results:
        print(f"--- side={label_set.side} ---")
        print(f"Labels : {label_set.size:,}")
        print(f"Class counts : {label_set.class_counts}")
        print(f"Class balance : {report.get('class_balance')}")
        print(f"Saved : {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
