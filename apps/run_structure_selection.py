"""CLI: H4 structure feature selection & stability (Sprint 12).

Example:
  python apps/run_structure_selection.py --symbol XAUUSD --timeframe H1
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
from research.structure_selection.repositories.structure_selection_repository import (
    StructureSelectionRepository,
)
from research.structure_selection.usecases.run_structure_selection import (
    RunStructureSelectionUseCase,
)
from settings.paths import (
    CONTEXT,
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
        log_dir / str(log_cfg.get("filename", "structure_selection.log")),
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
    p = argparse.ArgumentParser(description="Select robust H4 structure features")
    p.add_argument("--config", type=Path, default=MT5_CONFIG)
    p.add_argument("--symbol", default=None)
    p.add_argument("--timeframe", default=None)
    p.add_argument("--sides", default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = _load_config(Path(args.config))
    _setup_logging(cfg)

    ss_cfg = dict(cfg.get("structure_selection") or {})
    hs_cfg = dict(cfg.get("h4_structure") or {})
    mc_cfg = dict(cfg.get("market_context") or {})
    models_cfg = dict(cfg.get("models") or {})
    research_cfg = dict(cfg.get("research") or {})
    lab_cfg = dict(cfg.get("labels") or {})
    fe_cfg = dict(cfg.get("feature_engineering") or {})

    symbol = args.symbol or str(cfg.get("symbol", "XAUUSD"))
    timeframe = args.timeframe or str(
        ss_cfg.get("timeframe") or hs_cfg.get("base_timeframe") or "H1"
    )

    sides_raw = args.sides or ss_cfg.get("sides") or "long,short"
    if isinstance(sides_raw, list):
        sides = [str(s).lower() for s in sides_raw]
    else:
        sides = [s.strip().lower() for s in str(sides_raw).split(",") if s.strip()]

    algorithm = str(ss_cfg.get("algorithm") or models_cfg.get("algorithm") or "lightgbm")
    algo_params = dict((models_cfg.get("algorithms") or {}).get(algorithm) or {})
    trainer_params = dict(ss_cfg.get("trainer_params") or algo_params.get("params") or {})

    merged = {
        "algorithm": algorithm,
        "trainer_params": trainer_params,
        "target_mode": ss_cfg.get(
            "target_mode",
            research_cfg.get("target_mode", models_cfg.get("target_mode", "exclude_timeout")),
        ),
        "threshold": float(ss_cfg.get("threshold", models_cfg.get("threshold", 0.5))),
        "random_seed": int(ss_cfg.get("random_seed", research_cfg.get("random_seed", 42))),
        "windows": ss_cfg.get("windows") or research_cfg.get("windows"),
        "top_k": int(ss_cfg.get("top_k", 5)),
        "compute_shap": bool(ss_cfg.get("compute_shap", True)),
        "top_structure_features": ss_cfg.get("top_structure_features"),
    }

    features_root = _resolve_path(str(fe_cfg.get("output_directory", FEATURES)))
    feat_path = features_root / symbol.upper() / timeframe.upper() / "feature_matrix.parquet"

    h4_out = _resolve_path(str(hs_cfg.get("output_directory", RESEARCH / "h4_structure")))
    structure_path = h4_out / "structure_features.parquet"
    stats_path = h4_out / "feature_statistics.csv"

    context_root = _resolve_path(str(mc_cfg.get("output_directory", CONTEXT)))
    regime_path = context_root / symbol.upper() / timeframe.upper() / "context.parquet"

    out_root = _resolve_path(
        str(ss_cfg.get("output_directory", RESEARCH / "structure_selection"))
    )
    label_root = _resolve_path(str(lab_cfg.get("output_directory", LABELS)))
    label_repo = LabelRepository(label_root)
    strategy = str(lab_cfg.get("strategy") or "triple_barrier")
    version = str(lab_cfg.get("label_version") or "v1")
    labels_by_side: dict[str, pd.DataFrame] = {}
    for side in sides:
        path = label_repo.parquet_path(symbol, timeframe, side, strategy, version)
        if path.is_file():
            labels_by_side[side] = pd.read_parquet(path)

    results, out = RunStructureSelectionUseCase(
        StructureSelectionRepository(out_root),
        config=merged,
    ).execute(
        symbol=symbol,
        timeframe=timeframe,
        sides=sides,
        feature_matrix_path=feat_path,
        structure_features_path=structure_path,
        labels_by_side=labels_by_side,
        feature_statistics_path=stats_path if stats_path.is_file() else None,
        regime_context_path=regime_path if regime_path.is_file() else None,
    )

    print()
    print(f"experiments={len(results)} sides={sides}")
    for side in sides:
        side_res = [r for r in results if r.side == side]
        base = next((r for r in side_res if r.experiment_id == "A_baseline"), None)
        best = max(
            (r for r in side_res if r.experiment_id != "A_baseline"),
            key=lambda r: r.mean_roc_auc if r.mean_roc_auc == r.mean_roc_auc else -1,
            default=None,
        )
        if base and best:
            print(
                f"{side}: baseline={base.mean_roc_auc:.4f} "
                f"best={best.experiment_id}:{best.mean_roc_auc:.4f} "
                f"delta={best.mean_roc_auc - base.mean_roc_auc:+.4f}"
            )
    print(f"saved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
