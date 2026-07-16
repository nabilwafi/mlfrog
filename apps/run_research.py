"""CLI: run research hypothesis engine on dataset artifacts.

Example:
  python apps/run_research.py --symbol XAUUSD --timeframe H1 --side long
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

from datasets.repositories.dataset_repository import DatasetRepository
from labels.repositories.label_repository import LabelRepository
from research.repositories.research_repository import ResearchRepository
from research.services.research_service import ResearchService
from settings.paths import (
    DATASETS,
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
        log_dir / str(log_cfg.get("filename", "research.log")),
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
    p = argparse.ArgumentParser(description="Run research hypothesis engine")
    p.add_argument("--config", type=Path, default=MT5_CONFIG)
    p.add_argument("--symbol", default=None)
    p.add_argument("--timeframe", default=None)
    p.add_argument("--side", default=None, choices=("long", "short"))
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = _load_config(Path(args.config))
    _setup_logging(cfg)

    research_cfg = dict(cfg.get("research") or {})
    models_cfg = dict(cfg.get("models") or {})
    ds_cfg = dict(cfg.get("datasets") or {})
    lab_cfg = dict(cfg.get("labels") or {})

    symbol = args.symbol or str(cfg.get("symbol", "XAUUSD"))
    timeframe = args.timeframe or str(cfg.get("timeframe", "H1"))
    side = args.side or str(
        research_cfg.get("side") or models_cfg.get("side") or "long"
    )

    algorithm = str(
        research_cfg.get("algorithm") or models_cfg.get("algorithm") or "lightgbm"
    )
    algos = models_cfg.get("algorithms") or {}
    algo_cfg = dict(algos.get(algorithm) or {})
    trainer_params = dict(
        research_cfg.get("trainer_params") or algo_cfg.get("params") or {}
    )

    merged = {
        "algorithm": algorithm,
        "trainer_params": trainer_params,
        "target_mode": research_cfg.get(
            "target_mode", models_cfg.get("target_mode", "exclude_timeout")
        ),
        "threshold": float(
            research_cfg.get("threshold", models_cfg.get("threshold", 0.5))
        ),
        "random_seed": int(
            research_cfg.get("random_seed", models_cfg.get("random_seed", 42))
        ),
        "dataset_version": str(
            research_cfg.get("dataset_version", models_cfg.get("dataset_version", "v1"))
        ),
        "windows": research_cfg.get("windows"),
        "splits": research_cfg.get(
            "splits", ["train", "validation", "test", "sealed"]
        ),
    }

    dataset_root = _resolve_path(str(ds_cfg.get("output_directory", DATASETS)))
    research_root = _resolve_path(
        str(research_cfg.get("output_directory", RESEARCH))
    )
    label_root = _resolve_path(str(lab_cfg.get("output_directory", LABELS)))
    label_repo = LabelRepository(label_root)

    def _label_loader(sym: str, tf: str, sd: str) -> pd.DataFrame:
        strategy = str(
            lab_cfg.get("strategy") or ds_cfg.get("strategy") or "triple_barrier"
        )
        version = str(
            lab_cfg.get("label_version")
            or ds_cfg.get("label_version")
            or "v1"
        )
        path = label_repo.parquet_path(sym, tf, sd, strategy, version)
        if not path.is_file():
            raise FileNotFoundError(path)
        return pd.read_parquet(path)

    service = ResearchService(
        DatasetRepository(dataset_root),
        ResearchRepository(research_root),
        config=merged,
        label_loader=_label_loader,
    )
    report, out_dir = service.run(symbol, timeframe, side)
    c = report.conclusions

    print()
    print(f"algorithm={report.algorithm} side={report.side}")
    print(f"failure_driver={c.get('failure_driver')}")
    print(f"strong_years={c.get('strong_years')} fail_years={c.get('fail_years')}")
    print(f"recommendation={c.get('model_strategy_recommendation')}")
    print(f"saved -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
