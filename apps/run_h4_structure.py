"""CLI: H4 structure enhancement + ablation (Sprint 11).

Example:
  python apps/run_h4_structure.py --symbol XAUUSD --base H1 --context H4
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
from research.h4_structure.repositories.h4_structure_repository import H4StructureRepository
from research.h4_structure.usecases.run_h4_structure import RunH4StructureUseCase
from settings.paths import (
    FEATURES,
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
        log_dir / str(log_cfg.get("filename", "h4_structure.log")),
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
    p = argparse.ArgumentParser(description="Build H4 structure features and run ablation")
    p.add_argument("--config", type=Path, default=MT5_CONFIG)
    p.add_argument("--symbol", default=None)
    p.add_argument("--base", default=None, help="Base TF (default H1)")
    p.add_argument("--context", default=None, help="Context TF (default H4)")
    p.add_argument("--sides", default=None, help="Comma-separated sides (default long,short)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = _load_config(Path(args.config))
    _setup_logging(cfg)

    hs_cfg = dict(cfg.get("h4_structure") or {})
    mc_cfg = dict(cfg.get("market_context") or {})
    models_cfg = dict(cfg.get("models") or {})
    research_cfg = dict(cfg.get("research") or {})
    lab_cfg = dict(cfg.get("labels") or {})
    fe_cfg = dict(cfg.get("feature_engineering") or {})

    symbol = args.symbol or str(cfg.get("symbol", "XAUUSD"))
    base_tf = args.base or str(hs_cfg.get("base_timeframe") or mc_cfg.get("base_timeframe", "H1"))
    context_tf = args.context or str(
        hs_cfg.get("context_timeframe") or mc_cfg.get("context_timeframe", "H4")
    )
    timezone = str(cfg.get("timezone", "UTC"))

    sides_raw = args.sides or hs_cfg.get("sides") or "long,short"
    if isinstance(sides_raw, list):
        sides = [str(s).lower() for s in sides_raw]
    else:
        sides = [s.strip().lower() for s in str(sides_raw).split(",") if s.strip()]

    algorithm = str(hs_cfg.get("algorithm") or models_cfg.get("algorithm") or "lightgbm")
    algo_params = dict((models_cfg.get("algorithms") or {}).get(algorithm) or {})
    trainer_params = dict(hs_cfg.get("trainer_params") or algo_params.get("params") or {})

    builder_params = dict(mc_cfg.get("builder_params") or {})
    builder_params.update(dict(hs_cfg.get("builder_params") or {}))

    merged = {
        "algorithm": algorithm,
        "trainer_params": trainer_params,
        "builder_params": builder_params,
        "drop_na": bool(hs_cfg.get("drop_na", True)),
        "target_mode": hs_cfg.get(
            "target_mode",
            research_cfg.get("target_mode", models_cfg.get("target_mode", "exclude_timeout")),
        ),
        "threshold": float(hs_cfg.get("threshold", models_cfg.get("threshold", 0.5))),
        "random_seed": int(hs_cfg.get("random_seed", research_cfg.get("random_seed", 42))),
        "windows": hs_cfg.get("windows") or research_cfg.get("windows"),
    }

    raw_root = _resolve_path(str((cfg.get("output") or {}).get("directory", RAW)))
    if not (raw_root / symbol.upper() / context_tf.upper()).exists():
        raw_root = RAW
    features_root = _resolve_path(str(fe_cfg.get("output_directory", FEATURES)))
    feat_path = features_root / symbol.upper() / base_tf.upper() / "feature_matrix.parquet"
    out_root = _resolve_path(str(hs_cfg.get("output_directory", RESEARCH / "h4_structure")))
    label_root = _resolve_path(str(lab_cfg.get("output_directory", LABELS)))

    label_repo = LabelRepository(label_root)
    strategy = str(lab_cfg.get("strategy") or "triple_barrier")
    version = str(lab_cfg.get("label_version") or "v1")
    labels_by_side: dict[str, pd.DataFrame] = {}
    for side in sides:
        path = label_repo.parquet_path(symbol, base_tf, side, strategy, version)
        if path.is_file():
            labels_by_side[side] = pd.read_parquet(path)

    results, out = RunH4StructureUseCase(
        H4StructureRepository(out_root),
        MarketRepository(raw_root),
        config=merged,
    ).execute(
        symbol=symbol,
        base_timeframe=base_tf,
        context_timeframe=context_tf,
        sides=sides,
        feature_matrix_path=feat_path,
        labels_by_side=labels_by_side,
        timezone=timezone,
    )

    print()
    print(f"experiments={len(results)} sides={sides}")
    for side in sides:
        side_res = [r for r in results if r.side == side]
        base = next((r for r in side_res if r.experiment_id == "baseline"), None)
        all_s = next((r for r in side_res if r.experiment_id == "all_structure"), None)
        if base and all_s:
            print(
                f"{side}: baseline={base.mean_roc_auc:.4f} "
                f"all_structure={all_s.mean_roc_auc:.4f} "
                f"delta={all_s.mean_roc_auc - base.mean_roc_auc:+.4f}"
            )
    print(f"saved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
