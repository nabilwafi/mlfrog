"""CLI: Temporal Stability & Dataset Aging (Sprint 28).

Pure research. Production Meta/Conf/Heat/TP-SL frozen — no strategy changes.

Example:
  python apps/run_temporal_stability.py --symbol XAUUSD --timeframe H1
  python apps/run_temporal_stability.py --max-years 8 --side long
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

from research.temporal_stability.repositories import TemporalStabilityRepository
from research.temporal_stability.usecases import RunTemporalStabilityUseCase
from settings.paths import DATASETS, LOGS, MT5_CONFIG, MT5_CONFIG_EXAMPLE, RESEARCH, ROOT


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
        log_dir / str(log_cfg.get("filename", "temporal_stability.log")),
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
    p = argparse.ArgumentParser(description="Temporal Stability & Dataset Aging (Sprint 28)")
    p.add_argument("--config", type=Path, default=MT5_CONFIG)
    p.add_argument("--symbol", default=None)
    p.add_argument("--timeframe", default=None)
    p.add_argument("--side", default=None, help="optional: long|short (faster)")
    p.add_argument("--max-years", type=int, default=None, help="limit to last N years (smoke)")
    p.add_argument("--min-train-years", type=int, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = _load_config(Path(args.config))
    _setup_logging(cfg)

    ts_cfg = dict(cfg.get("temporal_stability") or {})
    symbol = args.symbol or str(cfg.get("symbol", "XAUUSD"))
    timeframe = args.timeframe or str(ts_cfg.get("timeframe") or "H1")

    out_root = _resolve_path(str(ts_cfg.get("output_directory", RESEARCH / "temporal_stability")))
    dataset_root = _resolve_path(str(ts_cfg.get("dataset_directory", DATASETS / symbol.upper() / timeframe.upper())))

    heat_root = _resolve_path(
        str((cfg.get("portfolio_heat") or {}).get("output_directory", RESEARCH / "portfolio_heat"))
    )
    frozen_path = heat_root / "best_policy_trades.parquet"
    if not frozen_path.is_file():
        conf_root = _resolve_path(
            str((cfg.get("confidence_layer") or {}).get("output_directory", RESEARCH / "confidence_layer"))
        )
        alt = conf_root / "confidence_panel.parquet"
        frozen_path = alt if alt.is_file() else None

    use_cfg = {
        "starting_equity": float(ts_cfg.get("starting_equity", 80.0)),
        "n_monte_carlo": int(ts_cfg.get("n_monte_carlo", 300)),
        "min_train_years": int(args.min_train_years or ts_cfg.get("min_train_years", 5)),
        "side": args.side or ts_cfg.get("side"),
        "max_years": args.max_years or ts_cfg.get("max_years"),
    }

    out = RunTemporalStabilityUseCase(TemporalStabilityRepository(out_root), config=use_cfg).execute(
        symbol=symbol,
        timeframe=timeframe,
        dataset_root=dataset_root,
        frozen_trades_path=frozen_path,
    )
    print()
    print(f"temporal stability artifacts -> {out}")
    print(f"report -> {out / 'temporal_stability_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
