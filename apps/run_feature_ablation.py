"""CLI: feature ablation research (category / subset walk-forward).

Example:
  python apps/run_feature_ablation.py --symbol XAUUSD --timeframe H1 --side long
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
from research.feature_ablation.repositories.ablation_repository import AblationRepository
from research.feature_ablation.usecases.run_feature_ablation import RunFeatureAblationUseCase
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
        log_dir / str(log_cfg.get("filename", "feature_ablation.log")),
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
    p = argparse.ArgumentParser(description="Run feature ablation research")
    p.add_argument("--config", type=Path, default=MT5_CONFIG)
    p.add_argument("--symbol", default=None)
    p.add_argument("--timeframe", default=None)
    p.add_argument("--side", default=None, choices=("long", "short"))
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = _load_config(Path(args.config))
    _setup_logging(cfg)

    ab_cfg = dict(cfg.get("feature_ablation") or {})
    models_cfg = dict(cfg.get("models") or {})
    research_cfg = dict(cfg.get("research") or {})
    lab_cfg = dict(cfg.get("labels") or {})
    fe_cfg = dict(cfg.get("feature_engineering") or {})

    symbol = args.symbol or str(cfg.get("symbol", "XAUUSD"))
    timeframe = args.timeframe or str(cfg.get("timeframe", "H1"))
    side = args.side or str(
        ab_cfg.get("side") or research_cfg.get("side") or models_cfg.get("side") or "long"
    )

    algorithm = str(ab_cfg.get("algorithm") or models_cfg.get("algorithm") or "lightgbm")
    algo_params = dict((models_cfg.get("algorithms") or {}).get(algorithm) or {})
    trainer_params = dict(
        ab_cfg.get("trainer_params") or algo_params.get("params") or {}
    )

    merged = {
        "algorithm": algorithm,
        "trainer_params": trainer_params,
        "target_mode": ab_cfg.get(
            "target_mode",
            research_cfg.get("target_mode", models_cfg.get("target_mode", "exclude_timeout")),
        ),
        "threshold": float(ab_cfg.get("threshold", models_cfg.get("threshold", 0.5))),
        "random_seed": int(ab_cfg.get("random_seed", research_cfg.get("random_seed", 42))),
        "windows": ab_cfg.get("windows") or research_cfg.get("windows"),
    }

    features_root = _resolve_path(str(fe_cfg.get("output_directory", FEATURES)))
    feat_dir = features_root / symbol.upper() / timeframe.upper()
    research_root = _resolve_path(str(research_cfg.get("output_directory", RESEARCH)))
    diagnostics_dir = research_root / symbol.upper() / timeframe.upper() / side.lower()
    out_root = _resolve_path(
        str(ab_cfg.get("output_directory", RESEARCH / "feature_ablation"))
    )

    label_root = _resolve_path(str(lab_cfg.get("output_directory", LABELS)))
    label_repo = LabelRepository(label_root)
    strategy = str(lab_cfg.get("strategy") or "triple_barrier")
    version = str(lab_cfg.get("label_version") or "v1")
    labels = pd.read_parquet(
        label_repo.parquet_path(symbol, timeframe, side, strategy, version)
    )

    results, out = RunFeatureAblationUseCase(
        AblationRepository(out_root),
        config=merged,
    ).execute(
        symbol=symbol,
        timeframe=timeframe,
        side=side,
        feature_matrix_path=feat_dir / "feature_matrix.parquet",
        feature_metadata_path=feat_dir / "feature_metadata.json",
        diagnostics_dir=diagnostics_dir,
        labels=labels,
    )

    summary = sorted(results, key=lambda r: r.mean_roc_auc if r.mean_roc_auc == r.mean_roc_auc else -1, reverse=True)
    print()
    print(f"experiments={len(results)}")
    if summary:
        best = summary[0]
        print(
            f"best={best.experiment_id} roc={best.mean_roc_auc:.4f} "
            f"ece={best.mean_calibration_error:.4f} feats={best.n_features}"
        )
    print(f"saved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
