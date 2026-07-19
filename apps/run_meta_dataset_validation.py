"""CLI: Meta dataset suitability validation (research only — no meta training).

Example:
  python apps/run_meta_dataset_validation.py --symbol XAUUSD --timeframe H1
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

import pandas as pd
import yaml

from labels.repositories.label_repository import LabelRepository
from research.meta_dataset_validation.repositories import MetaDatasetValidationRepository
from research.meta_dataset_validation.usecases import RunMetaDatasetValidationUseCase
from settings.paths import (
    LABELS,
    LOGS,
    MT5_CONFIG,
    MT5_CONFIG_EXAMPLE,
    RAW,
    RESEARCH,
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
        log_dir / str(log_cfg.get("filename", "meta_dataset_validation.log")),
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
    p = argparse.ArgumentParser(description="Validate meta-dataset suitability from v2 OOF preds")
    p.add_argument("--config", type=Path, default=MT5_CONFIG)
    p.add_argument("--symbol", default=None)
    p.add_argument("--timeframe", default=None)
    p.add_argument("--sides", default=None)
    p.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="Override predictions parquet (defaults to calibration then diagnosis)",
    )
    return p


def _resolve_predictions(cfg: dict[str, Any], override: Path | None) -> Path:
    if override is not None:
        p = override if override.is_absolute() else (ROOT / override).resolve()
        if p.is_file():
            return p
        raise FileNotFoundError(f"predictions not found: {p}")

    md_cfg = dict(cfg.get("meta_dataset_validation") or {})
    if md_cfg.get("predictions_path"):
        p = _resolve_path(str(md_cfg["predictions_path"]))
        if p.is_file():
            return p

    cal = dict(cfg.get("probability_calibration") or {})
    cal_root = _resolve_path(str(cal.get("output_directory", RESEARCH / "probability_calibration")))
    cal_preds = cal_root / "calibrated_predictions.parquet"
    if cal_preds.is_file():
        return cal_preds

    diag = dict(cfg.get("probability_diagnosis") or {})
    diag_root = _resolve_path(str(diag.get("output_directory", RESEARCH / "probability_diagnosis")))
    diag_preds = diag_root / "predictions.parquet"
    if diag_preds.is_file():
        return diag_preds

    raise FileNotFoundError(
        "No OOF predictions found. Run Sprint 15/16 first "
        "(probability_diagnosis or probability_calibration)."
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = _load_config(Path(args.config))
    _setup_logging(cfg)

    md_cfg = dict(cfg.get("meta_dataset_validation") or {})
    pc_cfg = dict(cfg.get("probability_calibration") or {})
    lab_cfg = dict(cfg.get("labels") or {})

    symbol = args.symbol or str(cfg.get("symbol", "XAUUSD"))
    timeframe = args.timeframe or str(md_cfg.get("timeframe") or "H1")
    sides_raw = args.sides or md_cfg.get("sides") or "long,short"
    if isinstance(sides_raw, list):
        sides = [str(s).lower() for s in sides_raw]
    else:
        sides = [s.strip().lower() for s in str(sides_raw).split(",") if s.strip()]

    cost = float(md_cfg.get("transaction_cost", pc_cfg.get("transaction_cost", 0.00015)))
    out_root = _resolve_path(
        str(md_cfg.get("output_directory", RESEARCH / "meta_dataset_validation"))
    )

    pred_path = _resolve_predictions(cfg, args.predictions)
    logging.getLogger(__name__).info("Using predictions | %s", pred_path)
    predictions = pd.read_parquet(pred_path)

    label_root = _resolve_path(str(lab_cfg.get("output_directory", LABELS)))
    label_repo = LabelRepository(label_root)
    strategy = str(lab_cfg.get("strategy") or "triple_barrier")
    version = str(lab_cfg.get("label_version") or "v1")
    labels_by_side: dict[str, pd.DataFrame] = {}
    for side in sides:
        path = label_repo.parquet_path(symbol, timeframe, side, strategy, version)
        if path.is_file():
            labels_by_side[side] = pd.read_parquet(path)

    candles_path = RAW / symbol.upper() / timeframe.upper() / "data.parquet"
    candles = pd.read_parquet(candles_path) if candles_path.is_file() else None

    candidates, out = RunMetaDatasetValidationUseCase(
        MetaDatasetValidationRepository(out_root),
        config={"transaction_cost": cost},
    ).execute(
        symbol=symbol,
        timeframe=timeframe,
        predictions=predictions,
        labels_by_side=labels_by_side,
        candles=candles,
        sides=sides,
    )

    print()
    print(f"candidates={len(candidates)} sides={sides} cost={cost}")
    print(f"saved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
