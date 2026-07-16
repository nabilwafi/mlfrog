"""CLI: context impact & ablation research (Sprint 10).

Example:
  python apps/run_context_impact.py --symbol XAUUSD --timeframe H1
  python apps/run_context_impact.py --symbol XAUUSD --timeframe H1 --sides long
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
from research.context_impact.repositories.context_impact_repository import ContextImpactRepository
from research.context_impact.usecases.run_context_impact import RunContextImpactUseCase
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
        log_dir / str(log_cfg.get("filename", "context_impact.log")),
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
    p = argparse.ArgumentParser(description="Run H4 context impact / ablation research")
    p.add_argument("--config", type=Path, default=MT5_CONFIG)
    p.add_argument("--symbol", default=None)
    p.add_argument("--timeframe", default=None)
    p.add_argument(
        "--sides",
        default=None,
        help="Comma-separated sides (default: long,short — run independently)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = _load_config(Path(args.config))
    _setup_logging(cfg)

    ci_cfg = dict(cfg.get("context_impact") or {})
    models_cfg = dict(cfg.get("models") or {})
    research_cfg = dict(cfg.get("research") or {})
    lab_cfg = dict(cfg.get("labels") or {})
    fe_cfg = dict(cfg.get("feature_engineering") or {})
    mc_cfg = dict(cfg.get("market_context") or {})

    symbol = args.symbol or str(cfg.get("symbol", "XAUUSD"))
    timeframe = args.timeframe or str(
        ci_cfg.get("timeframe") or mc_cfg.get("base_timeframe") or cfg.get("timeframe", "H1")
    )

    sides_raw = args.sides or ci_cfg.get("sides") or "long,short"
    if isinstance(sides_raw, list):
        sides = [str(s).lower() for s in sides_raw]
    else:
        sides = [s.strip().lower() for s in str(sides_raw).split(",") if s.strip()]

    algorithm = str(ci_cfg.get("algorithm") or models_cfg.get("algorithm") or "lightgbm")
    algo_params = dict((models_cfg.get("algorithms") or {}).get(algorithm) or {})
    trainer_params = dict(ci_cfg.get("trainer_params") or algo_params.get("params") or {})

    merged = {
        "algorithm": algorithm,
        "trainer_params": trainer_params,
        "target_mode": ci_cfg.get(
            "target_mode",
            research_cfg.get("target_mode", models_cfg.get("target_mode", "exclude_timeout")),
        ),
        "threshold": float(ci_cfg.get("threshold", models_cfg.get("threshold", 0.5))),
        "random_seed": int(ci_cfg.get("random_seed", research_cfg.get("random_seed", 42))),
        "windows": ci_cfg.get("windows") or research_cfg.get("windows"),
    }

    features_root = _resolve_path(str(fe_cfg.get("output_directory", FEATURES)))
    feat_path = features_root / symbol.upper() / timeframe.upper() / "feature_matrix.parquet"
    context_root = _resolve_path(str(mc_cfg.get("output_directory", CONTEXT)))
    context_path = context_root / symbol.upper() / timeframe.upper() / "context.parquet"
    out_root = _resolve_path(
        str(ci_cfg.get("output_directory", RESEARCH / "context_impact"))
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
        else:
            logging.getLogger(__name__).warning("Labels missing | side=%s path=%s", side, path)

    results, out = RunContextImpactUseCase(
        ContextImpactRepository(out_root),
        config=merged,
    ).execute(
        symbol=symbol,
        timeframe=timeframe,
        sides=sides,
        feature_matrix_path=feat_path,
        context_path=context_path,
        labels_by_side=labels_by_side,
    )

    print()
    print(f"experiments={len(results)} sides={sides}")
    for side in sides:
        side_res = [r for r in results if r.side == side]
        base = next((r for r in side_res if r.experiment_id == "baseline"), None)
        all_c = next((r for r in side_res if r.experiment_id == "all_context"), None)
        if base and all_c:
            delta = all_c.mean_roc_auc - base.mean_roc_auc
            print(
                f"{side}: baseline_roc={base.mean_roc_auc:.4f} "
                f"all_context={all_c.mean_roc_auc:.4f} delta={delta:+.4f}"
            )
    print(f"saved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
