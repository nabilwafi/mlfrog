"""CLI: Market Context Layer (H4 context → join H1 → dataset v2).

Example:
  python apps/run_market_context.py --symbol XAUUSD --base H1 --context H4
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

from data.repositories.market_repository import MarketRepository
from labels.repositories.label_repository import LabelRepository
from market_context.repositories.context_repository import ContextRepository
from market_context.usecases.run_market_context import RunMarketContextUseCase
from settings.paths import (
    CONTEXT,
    DATASETS,
    FEATURES,
    LABELS,
    LOGS,
    MT5_CONFIG,
    MT5_CONFIG_EXAMPLE,
    RAW,
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
        log_dir / str(log_cfg.get("filename", "market_context.log")),
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
    p = argparse.ArgumentParser(description="Build H4 market context and join to H1")
    p.add_argument("--config", type=Path, default=MT5_CONFIG)
    p.add_argument("--symbol", default=None)
    p.add_argument("--base", default=None, help="Base timeframe (default H1)")
    p.add_argument("--context", default=None, help="Context timeframe (default H4)")
    p.add_argument(
        "--sides",
        default=None,
        help="Comma-separated sides for dataset v2 (default: long,short)",
    )
    p.add_argument(
        "--skip-dataset-v2",
        action="store_true",
        help="Only write context artifacts",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = _load_config(Path(args.config))
    _setup_logging(cfg)

    mc_cfg = dict(cfg.get("market_context") or {})
    fe_cfg = dict(cfg.get("feature_engineering") or {})
    ds_cfg = dict(cfg.get("datasets") or {})
    lab_cfg = dict(cfg.get("labels") or {})

    symbol = args.symbol or str(mc_cfg.get("symbol") or cfg.get("symbol", "XAUUSD"))
    base_tf = args.base or str(mc_cfg.get("base_timeframe", "H1"))
    context_tf = args.context or str(mc_cfg.get("context_timeframe", "H4"))
    timezone = str(cfg.get("timezone", "UTC"))

    if args.skip_dataset_v2:
        mc_cfg["build_dataset_v2"] = False

    sides_raw = args.sides or mc_cfg.get("sides") or "long,short"
    if isinstance(sides_raw, list):
        sides = [str(s).lower() for s in sides_raw]
    else:
        sides = [s.strip().lower() for s in str(sides_raw).split(",") if s.strip()]

    raw_root = _resolve_path(str((cfg.get("output") or {}).get("directory", RAW)))
    if not (raw_root / symbol.upper() / context_tf.upper()).exists():
        raw_root = RAW
    features_root = _resolve_path(str(fe_cfg.get("output_directory", FEATURES)))
    context_root = _resolve_path(str(mc_cfg.get("output_directory", CONTEXT)))
    dataset_root = _resolve_path(str(ds_cfg.get("output_directory", DATASETS)))
    label_root = _resolve_path(str(lab_cfg.get("output_directory", LABELS)))

    mc_cfg.setdefault("strategy", lab_cfg.get("strategy", "triple_barrier"))
    mc_cfg.setdefault("label_version", lab_cfg.get("label_version", "v1"))

    labels_by_side: dict[str, pd.DataFrame] = {}
    if bool(mc_cfg.get("build_dataset_v2", True)):
        label_repo = LabelRepository(label_root)
        strategy = str(mc_cfg["strategy"])
        version = str(mc_cfg["label_version"])
        for side in sides:
            path = label_repo.parquet_path(symbol, base_tf, side, strategy, version)
            if path.is_file():
                labels_by_side[side] = pd.read_parquet(path)
            else:
                logging.getLogger(__name__).warning("Labels missing for side=%s path=%s", side, path)

    h1_feat_path = features_root / symbol.upper() / base_tf.upper() / "feature_matrix.parquet"

    out = RunMarketContextUseCase(
        market_repo=MarketRepository(raw_root),
        context_repo=ContextRepository(context_root),
        config=mc_cfg,
        feature_engineering_config=fe_cfg,
        features_output_root=features_root,
        datasets_root=dataset_root,
        dataset_config=ds_cfg,
    ).execute(
        symbol=symbol,
        base_timeframe=base_tf,
        context_timeframe=context_tf,
        timezone=timezone,
        labels_by_side=labels_by_side or None,
        h1_feature_matrix_path=h1_feat_path if h1_feat_path.is_file() else None,
    )

    print()
    print(f"saved -> {out}")
    print(f"context.parquet | features={len(list((out / 'context_features.csv').read_text().splitlines())) - 1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
