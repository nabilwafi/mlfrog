"""CLI: train a model from Dataset Builder outputs.

Example:
  python apps/train_model.py --symbol XAUUSD --timeframe H1 --side long --algorithm lightgbm
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
from models.repositories.model_repository import ModelRepository
from models.services.model_service import ModelService
from settings.paths import (
    DATASETS,
    LOGS,
    MODELS,
    MT5_CONFIG,
    MT5_CONFIG_EXAMPLE,
    ROOT,
)


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
        log_dir / str(log_cfg.get("filename", "train_model.log")),
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
    p = argparse.ArgumentParser(description="Train ML model from dataset splits")
    p.add_argument("--config", type=Path, default=MT5_CONFIG)
    p.add_argument("--symbol", default=None)
    p.add_argument("--timeframe", default=None)
    p.add_argument("--side", default=None, choices=("long", "short"))
    p.add_argument(
        "--algorithm",
        default=None,
        help="Trainer key (lightgbm, xgboost, catboost)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = _load_config(Path(args.config))
    _setup_logging(cfg)

    models_cfg = dict(cfg.get("models") or {})
    if not models_cfg:
        raise SystemExit("config missing 'models' section — see config.example.yaml")

    ds_cfg = cfg.get("datasets") or {}
    symbol = args.symbol or str(cfg.get("symbol", "XAUUSD"))
    timeframe = args.timeframe or str(cfg.get("timeframe", "H1"))
    side = args.side or str(models_cfg.get("side", "long"))
    algorithm = args.algorithm or str(models_cfg.get("algorithm", "lightgbm"))

    # Inherit dataset_version / versions from datasets section when omitted
    if "dataset_version" not in models_cfg:
        models_cfg["dataset_version"] = "v1"

    dataset_root = _resolve_path(
        str(ds_cfg.get("output_directory", models_cfg.get("dataset_directory", DATASETS)))
    )
    model_root = _resolve_path(str(models_cfg.get("output_directory", MODELS)))

    service = ModelService(
        DatasetRepository(dataset_root),
        ModelRepository(model_root),
        models_cfg,
    )
    model, out_dir = service.run(symbol, timeframe, side, algorithm)

    metrics = model.metrics
    print()
    print(f"algorithm={model.algorithm} side={model.side}")
    print(f"train_rows={model.train_rows} validation_rows={model.validation_rows}")
    print(
        "roc_auc={roc} pr_auc={pr} f1={f1} log_loss={ll}".format(
            roc=metrics.get("roc_auc"),
            pr=metrics.get("pr_auc"),
            f1=metrics.get("f1"),
            ll=metrics.get("log_loss"),
        )
    )
    print(f"saved -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
