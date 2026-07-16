"""CLI: Meta feature research (Sprint 18) — no meta-model training.

Example:
  python apps/run_meta_feature_research.py --symbol XAUUSD --timeframe H1
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

from research.meta_feature_research.repositories import MetaFeatureResearchRepository
from research.meta_feature_research.usecases import RunMetaFeatureResearchUseCase
from settings.paths import (
    FEATURES,
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
        log_dir / str(log_cfg.get("filename", "meta_feature_research.log")),
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
    p = argparse.ArgumentParser(description="Meta feature research (no meta training)")
    p.add_argument("--config", type=Path, default=MT5_CONFIG)
    p.add_argument("--symbol", default=None)
    p.add_argument("--timeframe", default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = _load_config(Path(args.config))
    _setup_logging(cfg)

    mf_cfg = dict(cfg.get("meta_feature_research") or {})
    md_cfg = dict(cfg.get("meta_dataset_validation") or {})
    hs_cfg = dict(cfg.get("h4_structure") or {})
    fe_cfg = dict(cfg.get("feature_engineering") or {})

    symbol = args.symbol or str(cfg.get("symbol", "XAUUSD"))
    timeframe = args.timeframe or str(mf_cfg.get("timeframe") or "H1")

    out_root = _resolve_path(
        str(mf_cfg.get("output_directory", RESEARCH / "meta_feature_research"))
    )
    cand_root = _resolve_path(
        str(md_cfg.get("output_directory", RESEARCH / "meta_dataset_validation"))
    )
    candidates_path = cand_root / "candidate_trades.parquet"
    if not candidates_path.is_file():
        raise FileNotFoundError(
            f"missing {candidates_path} — run apps/run_meta_dataset_validation.py first"
        )

    features_root = _resolve_path(str(fe_cfg.get("output_directory", FEATURES)))
    h1_path = features_root / symbol.upper() / timeframe.upper() / "feature_matrix.parquet"
    h4_out = _resolve_path(str(hs_cfg.get("output_directory", RESEARCH / "h4_structure")))
    h4_path = h4_out / "structure_features.parquet"
    m15_path = RAW / symbol.upper() / "M15" / "data.parquet"

    panel, out = RunMetaFeatureResearchUseCase(
        MetaFeatureResearchRepository(out_root),
        config={"percentile": float(mf_cfg.get("percentile", 0.03))},
    ).execute(
        symbol=symbol,
        timeframe=timeframe,
        candidates_path=candidates_path,
        h1_features_path=h1_path,
        h4_structure_path=h4_path,
        m15_candles_path=m15_path if m15_path.is_file() else None,
    )

    print()
    print(f"panel_rows={len(panel)} saved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
