"""CLI: Probability collapse diagnosis for Long/Short model v2 (Sprint 15).

Example:
  python apps/run_probability_diagnosis.py --symbol XAUUSD --timeframe H1
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
from research.probability_diagnosis.repositories import ProbabilityDiagnosisRepository
from research.probability_diagnosis.usecases import RunProbabilityDiagnosisUseCase
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
        log_dir / str(log_cfg.get("filename", "probability_diagnosis.log")),
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
    p = argparse.ArgumentParser(description="Diagnose probability collapse for model v2")
    p.add_argument("--config", type=Path, default=MT5_CONFIG)
    p.add_argument("--symbol", default=None)
    p.add_argument("--timeframe", default=None)
    p.add_argument("--sides", default=None, help="Comma-separated (default long,short)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = _load_config(Path(args.config))
    _setup_logging(cfg)

    pd_cfg = dict(cfg.get("probability_diagnosis") or {})
    pq_cfg = dict(cfg.get("probability_quality") or {})
    mv_cfg = dict(cfg.get("model_v2") or {})
    hs_cfg = dict(cfg.get("h4_structure") or {})
    models_cfg = dict(cfg.get("models") or {})
    research_cfg = dict(cfg.get("research") or {})
    lab_cfg = dict(cfg.get("labels") or {})
    fe_cfg = dict(cfg.get("feature_engineering") or {})

    symbol = args.symbol or str(cfg.get("symbol", "XAUUSD"))
    timeframe = args.timeframe or str(
        pd_cfg.get("timeframe") or pq_cfg.get("timeframe") or mv_cfg.get("timeframe") or "H1"
    )

    sides_raw = (
        args.sides
        or pd_cfg.get("sides")
        or pq_cfg.get("sides")
        or mv_cfg.get("sides")
        or "long,short"
    )
    if isinstance(sides_raw, list):
        sides = [str(s).lower() for s in sides_raw]
    else:
        sides = [s.strip().lower() for s in str(sides_raw).split(",") if s.strip()]

    algorithm = str(
        pd_cfg.get("algorithm")
        or mv_cfg.get("algorithm")
        or models_cfg.get("algorithm")
        or "lightgbm"
    )
    algo_params = dict((models_cfg.get("algorithms") or {}).get(algorithm) or {})
    trainer_params = dict(
        pd_cfg.get("trainer_params")
        or mv_cfg.get("trainer_params")
        or algo_params.get("params")
        or {}
    )

    merged = {
        "algorithm": algorithm,
        "trainer_params": trainer_params,
        "target_mode": pd_cfg.get(
            "target_mode",
            mv_cfg.get(
                "target_mode",
                research_cfg.get("target_mode", models_cfg.get("target_mode", "exclude_timeout")),
            ),
        ),
        "random_seed": int(
            pd_cfg.get(
                "random_seed",
                mv_cfg.get("random_seed", research_cfg.get("random_seed", 42)),
            )
        ),
        "windows": pd_cfg.get("windows")
        or mv_cfg.get("windows")
        or research_cfg.get("windows"),
    }

    features_root = _resolve_path(str(fe_cfg.get("output_directory", FEATURES)))
    feat_path = features_root / symbol.upper() / timeframe.upper() / "feature_matrix.parquet"
    h4_out = _resolve_path(str(hs_cfg.get("output_directory", RESEARCH / "h4_structure")))
    structure_path = h4_out / "structure_features.parquet"
    out_root = _resolve_path(
        str(pd_cfg.get("output_directory", RESEARCH / "probability_diagnosis"))
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

    preds, out = RunProbabilityDiagnosisUseCase(
        ProbabilityDiagnosisRepository(out_root),
        config=merged,
    ).execute(
        symbol=symbol,
        timeframe=timeframe,
        feature_matrix_path=feat_path,
        structure_features_path=structure_path,
        labels_by_side=labels_by_side,
        sides=sides,
    )

    print()
    print(f"predictions={len(preds)} sides={sides}")
    if not preds.empty:
        val = preds.loc[
            (preds["split"] == "validation") & (preds["experiment_id"] == "B_v2_context")
        ]
        for side in sides:
            sub = val.loc[val["side"] == side]
            if sub.empty:
                continue
            print(
                f"{side} v2 val: n={len(sub)} mean_p={sub['y_prob'].mean():.4f} "
                f"std={sub['y_prob'].std():.4f} pos_rate={sub['y_true'].mean():.4f}"
            )
    print(f"saved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
