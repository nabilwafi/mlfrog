"""CLI: Realistic portfolio backtest (Sprint 21).

No ML retrain. Meta gate fixed at 0.45. Uses Sprint-20 OOF trades.

Example:
  python apps/run_portfolio_backtest.py --symbol XAUUSD --timeframe H1
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

from research.portfolio_backtest.repositories import PortfolioBacktestRepository
from research.portfolio_backtest.usecases import RunPortfolioBacktestUseCase
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
        log_dir / str(log_cfg.get("filename", "portfolio_backtest.log")),
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
    p = argparse.ArgumentParser(description="Realistic portfolio backtest (Sprint 21)")
    p.add_argument("--config", type=Path, default=MT5_CONFIG)
    p.add_argument("--symbol", default=None)
    p.add_argument("--timeframe", default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = _load_config(Path(args.config))
    _setup_logging(cfg)

    pb_cfg = dict(cfg.get("portfolio_backtest") or {})
    mm_cfg = dict(cfg.get("meta_model") or {})

    symbol = args.symbol or str(cfg.get("symbol", "XAUUSD"))
    timeframe = args.timeframe or str(pb_cfg.get("timeframe") or "H1")

    out_root = _resolve_path(
        str(pb_cfg.get("output_directory", RESEARCH / "portfolio_backtest"))
    )
    meta_root = _resolve_path(str(mm_cfg.get("output_directory", RESEARCH / "meta_model")))
    oof_path = meta_root / "oof_predictions.parquet"
    if not oof_path.is_file():
        raise FileNotFoundError(f"missing {oof_path} — run apps/run_meta_model.py first")

    use_cfg = {
        "starting_equity": float(pb_cfg.get("starting_equity", 80.0)),
        "meta_threshold": float(pb_cfg.get("meta_threshold", 0.45)),
        "n_monte_carlo": int(pb_cfg.get("n_monte_carlo", 1000)),
        "random_seed": int(pb_cfg.get("random_seed", 42)),
        "preferred_scenario": str(pb_cfg.get("preferred_scenario", "C_risk_1pct_frac")),
    }

    out = RunPortfolioBacktestUseCase(
        PortfolioBacktestRepository(out_root), config=use_cfg
    ).execute(symbol=symbol, timeframe=timeframe, oof_path=oof_path)

    print()
    print(f"portfolio backtest -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
