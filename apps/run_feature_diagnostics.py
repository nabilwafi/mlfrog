"""CLI: feature diagnostics on Sprint-6 feature matrix + labels.

Example:
  python apps/run_feature_diagnostics.py --symbol XAUUSD --timeframe H1 --side long
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

from feature_diagnostics.repositories.feature_diagnostics_repository import (
    FeatureDiagnosticsRepository,
)
from feature_diagnostics.services.feature_diagnostics_service import FeatureDiagnosticsService
from labels.repositories.label_repository import LabelRepository
from settings.paths import (
    FEATURES,
    LABELS,
    LOGS,
    MT5_CONFIG,
    MT5_CONFIG_EXAMPLE,
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
        log_dir / str(log_cfg.get("filename", "feature_diagnostics.log")),
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
    p = argparse.ArgumentParser(description="Run Sprint-6 feature diagnostics")
    p.add_argument("--config", type=Path, default=MT5_CONFIG)
    p.add_argument("--symbol", default=None)
    p.add_argument("--timeframe", default=None)
    p.add_argument("--side", default=None, choices=("long", "short"))
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = _load_config(Path(args.config))
    _setup_logging(cfg)

    fd_cfg = dict(cfg.get("feature_diagnostics") or {})
    research_cfg = dict(cfg.get("research") or {})
    models_cfg = dict(cfg.get("models") or {})
    lab_cfg = dict(cfg.get("labels") or {})
    fe_cfg = dict(cfg.get("feature_engineering") or {})

    symbol = args.symbol or str(cfg.get("symbol", "XAUUSD"))
    timeframe = args.timeframe or str(cfg.get("timeframe", "H1"))
    side = args.side or str(
        fd_cfg.get("side") or research_cfg.get("side") or models_cfg.get("side") or "long"
    )

    merged = {
        "target_mode": fd_cfg.get(
            "target_mode",
            research_cfg.get("target_mode", models_cfg.get("target_mode", "exclude_timeout")),
        ),
        "corr_threshold": float(fd_cfg.get("corr_threshold", 0.95)),
        "random_seed": int(fd_cfg.get("random_seed", research_cfg.get("random_seed", 42))),
        "windows": fd_cfg.get("windows") or research_cfg.get("windows"),
    }

    features_root = _resolve_path(str(fe_cfg.get("output_directory", FEATURES)))
    research_root = _resolve_path(
        str(fd_cfg.get("output_directory", research_cfg.get("output_directory", RESEARCH)))
    )
    label_root = _resolve_path(str(lab_cfg.get("output_directory", LABELS)))
    label_repo = LabelRepository(label_root)

    def _label_loader(sym: str, tf: str, sd: str) -> pd.DataFrame:
        strategy = str(lab_cfg.get("strategy") or "triple_barrier")
        version = str(lab_cfg.get("label_version") or "v1")
        path = label_repo.parquet_path(sym, tf, sd, strategy, version)
        if not path.is_file():
            raise FileNotFoundError(path)
        return pd.read_parquet(path)

    report, out = FeatureDiagnosticsService(
        FeatureDiagnosticsRepository(research_root),
        features_root=features_root,
        config=merged,
        label_loader=_label_loader,
    ).run(symbol, timeframe, side)

    print()
    print(f"keep={len(report.keep)} remove={len(report.remove)}")
    print(f"library_v3={len(report.candidate_library_v3)}")
    print(f"top categories={list(report.category_power.items())[:5]}")
    print(f"saved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
