"""CLI: build time-split datasets from FeatureSet + LabelSet.

Example:
  python apps/build_dataset.py --symbol XAUUSD --timeframe H1
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

from datasets.repositories.dataset_repository import DatasetRepository
from datasets.services.dataset_service import DatasetService
from features.repositories.feature_repository import FeatureRepository
from labels.repositories.label_repository import LabelRepository
from settings.paths import (
    DATASETS,
    FEATURES,
    LABELS,
    LOGS,
    MT5_CONFIG,
    MT5_CONFIG_EXAMPLE,
    ROOT,
)


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
        log_dir / str(log_cfg.get("filename", "datasets.log")),
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
    p = argparse.ArgumentParser(description="Build ML datasets from features + labels")
    p.add_argument("--config", type=Path, default=MT5_CONFIG)
    p.add_argument("--symbol", default=None)
    p.add_argument("--timeframe", default=None)
    p.add_argument("--sides", nargs="+", default=None, choices=("long", "short"))
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = _load_config(Path(args.config))
    _setup_logging(cfg)

    symbol = args.symbol or str(cfg.get("symbol", "XAUUSD"))
    timeframe = args.timeframe or str(cfg.get("timeframe", "H1"))
    timezone = str(cfg.get("timezone", "UTC"))

    ds_cfg = cfg.get("datasets")
    if not ds_cfg:
        raise SystemExit("config missing 'datasets' section — see config.example.yaml")

    # Inherit versions/strategy/sides from sibling sections when omitted
    feat_cfg = cfg.get("features") or {}
    lab_cfg = cfg.get("labels") or {}
    merged = {
        "feature_version": ds_cfg.get("feature_version", feat_cfg.get("feature_version", "v1")),
        "label_version": ds_cfg.get("label_version", lab_cfg.get("label_version", "v1")),
        "strategy": ds_cfg.get("strategy", lab_cfg.get("strategy", "triple_barrier")),
        "sides": ds_cfg.get("sides", lab_cfg.get("sides", ["long"])),
        "drop_na": ds_cfg.get("drop_na", True),
        "splits": ds_cfg.get("splits"),
        "output_directory": ds_cfg.get("output_directory", DATASETS),
    }
    if not merged["splits"]:
        raise SystemExit("datasets.splits is required")

    sides = args.sides or list(merged["sides"])
    feat_root = _resolve_path(str(feat_cfg.get("output_directory", FEATURES)))
    label_root = _resolve_path(str(lab_cfg.get("output_directory", LABELS)))
    ds_root = _resolve_path(str(merged["output_directory"]))

    service = DatasetService(
        FeatureRepository(feat_root),
        LabelRepository(label_root),
        DatasetRepository(ds_root),
        merged,
        timezone=timezone,
    )
    results = service.run(symbol, timeframe, sides=sides)

    print()
    for side, splits in results.items():
        print(f"=== side={side} ===")
        for split_name, (ds, path) in splits.items():
            print(f"{split_name}: {ds.size:,} rows -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
