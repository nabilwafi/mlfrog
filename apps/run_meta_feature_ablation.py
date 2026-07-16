"""CLI: Meta feature ablation (Sprint 19) — fixed LGBM, not final meta.

Example:
  python apps/run_meta_feature_ablation.py --symbol XAUUSD --timeframe H1
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

from research.meta_feature_ablation.repositories import MetaFeatureAblationRepository
from research.meta_feature_ablation.usecases import RunMetaFeatureAblationUseCase
from settings.paths import LOGS, MT5_CONFIG, MT5_CONFIG_EXAMPLE, RESEARCH, ROOT


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
        log_dir / str(log_cfg.get("filename", "meta_feature_ablation.log")),
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
    p = argparse.ArgumentParser(description="Meta feature ablation (research; not final meta)")
    p.add_argument("--config", type=Path, default=MT5_CONFIG)
    p.add_argument("--symbol", default=None)
    p.add_argument("--timeframe", default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = _load_config(Path(args.config))
    _setup_logging(cfg)

    ab_cfg = dict(cfg.get("meta_feature_ablation") or {})
    mf_cfg = dict(cfg.get("meta_feature_research") or {})
    md_cfg = dict(cfg.get("meta_dataset_validation") or {})
    cal_cfg = dict(cfg.get("probability_calibration") or {})

    symbol = args.symbol or str(cfg.get("symbol", "XAUUSD"))
    timeframe = args.timeframe or str(ab_cfg.get("timeframe") or "H1")

    out_root = _resolve_path(
        str(ab_cfg.get("output_directory", RESEARCH / "meta_feature_ablation"))
    )
    panel_root = _resolve_path(
        str(mf_cfg.get("output_directory", RESEARCH / "meta_feature_research"))
    )
    panel_path = panel_root / "meta_feature_panel.parquet"
    if not panel_path.is_file():
        raise FileNotFoundError(
            f"missing {panel_path} — run apps/run_meta_feature_research.py first"
        )

    cand_root = _resolve_path(
        str(md_cfg.get("output_directory", RESEARCH / "meta_dataset_validation"))
    )
    candidates_path = cand_root / "candidate_trades.parquet"
    if not candidates_path.is_file():
        raise FileNotFoundError(
            f"missing {candidates_path} — run apps/run_meta_dataset_validation.py first"
        )

    use_cfg = {
        "random_seed": int(ab_cfg.get("random_seed", 42)),
        "threshold": float(ab_cfg.get("threshold", 0.5)),
        "num_boost_round": int(ab_cfg.get("num_boost_round", 200)),
        "early_stopping_rounds": int(ab_cfg.get("early_stopping_rounds", 30)),
        "transaction_cost": float(
            ab_cfg.get(
                "transaction_cost",
                cal_cfg.get("transaction_cost", md_cfg.get("transaction_cost", 0.00015)),
            )
        ),
        "compute_shap": bool(ab_cfg.get("compute_shap", True)),
        "lgbm_params": dict(ab_cfg.get("lgbm_params") or {}),
    }

    out = RunMetaFeatureAblationUseCase(
        MetaFeatureAblationRepository(out_root),
        config=use_cfg,
    ).execute(
        symbol=symbol,
        timeframe=timeframe,
        panel_path=panel_path,
        candidates_path=candidates_path,
    )

    print()
    print(f"ablation artifacts -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
