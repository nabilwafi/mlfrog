"""CLI: Portfolio Heat Management (Sprint 24).

Frozen Primary + Meta + Confidence. Research execution heat only.

Example:
  python apps/run_portfolio_heat.py --symbol XAUUSD --timeframe H1
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

from research.portfolio_heat.repositories.repository import PortfolioHeatRepository
from research.portfolio_heat.usecases.run_portfolio_heat import RunPortfolioHeatUseCase
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
        log_dir / str(log_cfg.get("filename", "portfolio_heat.log")),
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
    p = argparse.ArgumentParser(description="Portfolio Heat Management (Sprint 24)")
    p.add_argument("--config", type=Path, default=MT5_CONFIG)
    p.add_argument("--symbol", default=None)
    p.add_argument("--timeframe", default=None)
    p.add_argument("--confidence-panel", type=Path, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = _load_config(Path(args.config))
    _setup_logging(cfg)

    ph_cfg = dict(cfg.get("portfolio_heat") or {})
    cl_cfg = dict(cfg.get("confidence_layer") or {})

    symbol = args.symbol or str(cfg.get("symbol", "XAUUSD"))
    timeframe = args.timeframe or str(ph_cfg.get("timeframe") or "H1")

    out_root = _resolve_path(str(ph_cfg.get("output_directory", RESEARCH / "portfolio_heat")))
    conf_root = _resolve_path(str(cl_cfg.get("output_directory", RESEARCH / "confidence_layer")))
    panel_path = (
        _resolve_path(str(args.confidence_panel))
        if args.confidence_panel
        else conf_root / "confidence_panel.parquet"
    )
    if not panel_path.is_file():
        raise FileNotFoundError(
            f"missing {panel_path} — run apps/run_confidence_layer.py first"
        )

    use_cfg = {
        "starting_equity": float(ph_cfg.get("starting_equity", 80.0)),
        "n_monte_carlo": int(ph_cfg.get("n_monte_carlo", 500)),
        "n_bootstrap": int(ph_cfg.get("n_bootstrap", 1000)),
        "n_permutation": int(ph_cfg.get("n_permutation", 1000)),
    }

    out = RunPortfolioHeatUseCase(
        PortfolioHeatRepository(out_root), config=use_cfg
    ).execute(symbol=symbol, timeframe=timeframe, confidence_panel_path=panel_path)
    print()
    print(f"portfolio heat -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
