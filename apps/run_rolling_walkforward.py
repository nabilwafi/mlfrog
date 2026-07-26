"""Rolling Walk-Forward Backtest — train 5y / val 1y / test 1y, retrain yearly.

Windows (example):
  Train 2015-2019 | Val 2020 | Test 2021
  Train 2016-2020 | Val 2021 | Test 2022
  ...
  Train 2020-2024 | Val 2025 | Test 2026

Rules:
  - Model trained on train years only; early-stop on validation year.
  - Frozen booster scored on test year only (no leakage).
  - Existing FE / labels / LGBM params / top5% gate / ATR trail exit unchanged.
  - Postgres (research.wf_*) is the source of truth; files only for debug.

Examples:
  python apps/run_rolling_walkforward.py --apply-schema
  python apps/run_rolling_walkforward.py
  python apps/run_rolling_walkforward.py --starting 80 --debug-files
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import tools.pyc_path_hook  # noqa: F401

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml

from production import FEATURE_VERSION, LABEL_VERSION, PIPELINE_VERSION
from production.db.wf_writer import WalkForwardWriter
from simulation.wf.sim import (
    ACTIVATE_R,
    CONTRACT_SIZE,
    COST,
    FIXED_LOT,
    HORIZON,
    MAX_OPEN,
    SL_ATR,
    TRAIL,
    _feat_names,
    _load_side,
    entry_indices,
    load_h1,
    max_dd_from_equity,
    prepare_market,
    profit_factor_pnl,
    replay_trail,
    run_portfolio,
    wilder_atr,
)

logger = logging.getLogger(__name__)

TRAIN_YEARS = 5
VAL_YEARS = 1
TEST_YEARS = 1
FIRST_TEST_YEAR = 2021
LAST_TEST_YEAR = 2026
TOP_PCT = 0.05
DEFAULT_STARTING = 80.0
MODEL_DIR = _ROOT / "artifacts" / "models" / "rolling_wf"
DEBUG_DIR = _ROOT / "artifacts" / "pipeline_backtest" / "rolling_wf"


@dataclass(frozen=True)
class WFWindow:
    train_start: int
    train_end: int  # inclusive
    val_year: int
    test_year: int

    @property
    def window_id(self) -> str:
        return f"t{self.train_start}-{self.train_end}_v{self.val_year}_te{self.test_year}"


def build_windows(
    *,
    first_test: int = FIRST_TEST_YEAR,
    last_test: int = LAST_TEST_YEAR,
    train_years: int = TRAIN_YEARS,
) -> list[WFWindow]:
    """test=T → train=[T-train_years-1 .. T-2], val=T-1."""
    out: list[WFWindow] = []
    for test in range(first_test, last_test + 1):
        val = test - 1
        train_end = test - 2
        train_start = train_end - (train_years - 1)
        out.append(
            WFWindow(
                train_start=train_start,
                train_end=train_end,
                val_year=val,
                test_year=test,
            )
        )
    return out


def _slice_years(df: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    y = df["timestamp"].dt.year
    return df[(y >= start) & (y <= end)].copy()


def _slice_year(df: pd.DataFrame, year: int) -> pd.DataFrame:
    return df[df["timestamp"].dt.year == year].copy()


def train_frozen(
    train: pd.DataFrame,
    val: pd.DataFrame,
    feat: list[str],
) -> tuple[lgb.Booster, dict[str, Any]]:
    """Train on train years; early-stop on calendar validation year (no test peek)."""
    y_tr = train["label"].astype(int).to_numpy()
    y_va = val["label"].astype(int).to_numpy()
    spw = max((y_tr == 0).sum(), 1) / max((y_tr == 1).sum(), 1)
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 40,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": float(spw),
        "verbosity": -1,
        "seed": 42,
    }
    dtr = lgb.Dataset(train[feat], label=y_tr, feature_name=list(feat), free_raw_data=False)
    dva = lgb.Dataset(val[feat], label=y_va, reference=dtr, free_raw_data=False)
    evals: dict[str, list] = {}
    booster = lgb.train(
        params,
        dtr,
        num_boost_round=500,
        valid_sets=[dva],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(0),
            lgb.record_evaluation(evals),
        ],
    )
    best = int(booster.best_iteration or 0)
    val_losses = (evals.get("val") or {}).get("binary_logloss") or []
    val_ll = float(val_losses[best - 1]) if val_losses and best > 0 else float("nan")
    return booster, {
        "best_iteration": best,
        "val_logloss": val_ll,
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "scale_pos_weight": float(spw),
        "params": params,
    }


def score_test(booster: lgb.Booster, test: pd.DataFrame, feat: list[str]) -> np.ndarray:
    return booster.predict(test[feat], num_iteration=booster.best_iteration)


def build_test_entries(
    *,
    long_df: pd.DataFrame,
    short_df: pd.DataFrame,
    h1: pd.DataFrame,
    feat: list[str],
    window: WFWindow,
    boosters: dict[str, lgb.Booster],
    top_pct: float = TOP_PCT,
) -> pd.DataFrame:
    """Score frozen models on test year only; top_pct unique bars (best side)."""
    rows = []
    for side, df in (("long", long_df), ("short", short_df)):
        test = _slice_year(df, window.test_year)
        if test.empty or side not in boosters:
            continue
        prob = score_test(boosters[side], test, feat)
        g = test[["timestamp", "realized_return", "holding_bars", "entry_price"]].copy()
        g["side"] = side
        g["y_prob"] = prob
        rows.append(g)
    if not rows:
        return pd.DataFrame()
    p = pd.concat(rows, ignore_index=True)
    p = p.loc[p.groupby("timestamp")["y_prob"].idxmax()].copy()
    k = max(1, int(round(len(p) * float(top_pct))))
    u = p.nlargest(k, "y_prob").sort_values("timestamp")
    h = h1.copy()
    h["timestamp"] = pd.to_datetime(h["timestamp"], utc=True)
    h["atr"] = wilder_atr(h)
    m = u.merge(h[["timestamp", "close", "atr"]], on="timestamp", how="left")
    m["entry_price"] = m["entry_price"].fillna(m["close"]).astype(float)
    m["atr_price"] = m["atr"].astype(float)
    m["atr_entry"] = m["atr_price"]
    return m.dropna(subset=["entry_price", "atr_price"])


def apply_trail(entries: pd.DataFrame, mkt: dict) -> pd.DataFrame:
    if entries.empty:
        return pd.DataFrame()
    eis = entry_indices(entries, mkt["ts"])
    rows = []
    for i in range(len(entries)):
        src = entries.iloc[i]
        r = replay_trail(
            side=str(src["side"]),
            entry=float(src["entry_price"]),
            atr=float(src["atr_entry"]),
            ei=int(eis[i]),
            mkt=mkt,
            trail=TRAIL,
        )
        if r is None:
            continue
        rows.append(
            {
                "timestamp": src["timestamp"],
                "side": src["side"],
                "y_prob": float(src["y_prob"]),
                "entry_price": float(src["entry_price"]),
                "atr_price": float(src["atr_entry"]),
                "net_return": r["net_return"],
                "holding_bars": r["holding_bars"],
                "exit_reason": r["exit_reason"],
            }
        )
    return pd.DataFrame(rows)


def _load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError("config root must be a mapping")
    return data


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Rolling walk-forward backtest (5y/1y/1y)")
    p.add_argument("--config", type=Path, default=_ROOT / "configs" / "config.yaml")
    p.add_argument("--starting", type=float, default=DEFAULT_STARTING)
    p.add_argument("--first-test", type=int, default=FIRST_TEST_YEAR)
    p.add_argument("--last-test", type=int, default=LAST_TEST_YEAR)
    p.add_argument("--train-years", type=int, default=TRAIN_YEARS)
    p.add_argument("--top-pct", type=float, default=TOP_PCT)
    p.add_argument("--leverage", type=float, default=500.0)
    p.add_argument("--apply-schema", action="store_true")
    p.add_argument("--no-db", action="store_true", help="Skip Postgres writes (debug only)")
    p.add_argument("--debug-files", action="store_true", help="Also write debug parquet/md")
    p.add_argument("--run-id", type=str, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = _load_config(args.config)
    dsn = None if args.no_db else (cfg.get("paper_trading") or {}).get("postgres_dsn")
    db = WalkForwardWriter(str(dsn) if dsn else None)

    if args.apply_schema:
        if not db.enabled:
            raise SystemExit("--apply-schema requires a working postgres_dsn (and not --no-db)")
        schema = (_ROOT / "sql" / "research_wf_schema.sql").read_text(encoding="utf-8")
        db.apply_schema(schema)
        print("schema applied: research.wf_*")
        return 0

    windows = build_windows(
        first_test=args.first_test,
        last_test=args.last_test,
        train_years=args.train_years,
    )
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    policy = {
        "train_years": args.train_years,
        "val_years": VAL_YEARS,
        "test_years": TEST_YEARS,
        "top_pct": args.top_pct,
        "exit": f"atr_trail_{TRAIL}",
        "trail_activate_r": ACTIVATE_R,
        "sl_atr": SL_ATR,
        "horizon": HORIZON,
        "max_open": MAX_OPEN,
        "fixed_lot": FIXED_LOT,
        "cost": COST,
        "contract_size": CONTRACT_SIZE,
        "starting_equity": args.starting,
        "leverage": args.leverage,
        "ruin_stop": True,
        "model": "lightgbm_binary_frozen_per_window",
        "gate": "primary_top_pct_best_side",
    }
    print(f"run_id={run_id}")
    print(f"windows={len(windows)} first_test={windows[0].test_year} last_test={windows[-1].test_year}")
    print(f"policy={json.dumps(policy, default=float)}")
    if db.enabled:
        print("db=ON (research.wf_*)")
    else:
        print("db=OFF — results will NOT be persisted to Postgres")

    db.upsert_run(
        {
            "run_id": run_id,
            "symbol": "XAUUSD",
            "timeframe": "H1",
            "train_years": args.train_years,
            "val_years": VAL_YEARS,
            "test_years": TEST_YEARS,
            "first_test_year": windows[0].test_year,
            "last_test_year": windows[-1].test_year,
            "policy": policy,
            "feature_version": FEATURE_VERSION,
            "label_version": LABEL_VERSION,
            "pipeline_version": PIPELINE_VERSION,
            "status": "running",
            "notes": None,
        }
    )

    print("Loading long/short v2 datasets + H1…")
    long_df = _load_side("long")
    short_df = _load_side("short")
    feat = _feat_names(long_df)
    h1 = load_h1(_ROOT / "artifacts/raw/XAUUSD/H1/data.parquet")
    mkt = prepare_market(h1)
    print(f"features={len(feat)} long_rows={len(long_df)} short_rows={len(short_df)}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if args.debug_files:
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    yearly_summaries: list[dict[str, Any]] = []
    continuous_panels: list[pd.DataFrame] = []

    try:
        for w in windows:
            print(
                f"\n=== {w.window_id} | train {w.train_start}-{w.train_end} "
                f"| val {w.val_year} | test {w.test_year} ==="
            )
            boosters: dict[str, lgb.Booster] = {}
            for side, df in (("long", long_df), ("short", short_df)):
                train = _slice_years(df, w.train_start, w.train_end)
                val = _slice_year(df, w.val_year)
                if len(train) < 500 or len(val) < 50:
                    print(f"  skip {side}: train={len(train)} val={len(val)}")
                    continue
                # Leakage guard: train/val must not touch test year.
                assert train["timestamp"].dt.year.max() < w.val_year
                assert val["timestamp"].dt.year.max() == w.val_year
                assert train["timestamp"].dt.year.max() < w.test_year
                assert val["timestamp"].dt.year.max() < w.test_year

                booster, meta = train_frozen(train, val, feat)
                model_version = f"rwf_{side}_te{w.test_year}_{run_id[:8]}"
                model_path = MODEL_DIR / run_id / f"primary_{side}_te{w.test_year}.txt"
                model_path.parent.mkdir(parents=True, exist_ok=True)
                booster.save_model(str(model_path))
                boosters[side] = booster
                print(
                    f"  trained {side}: n_train={meta['n_train']} n_val={meta['n_val']} "
                    f"best_iter={meta['best_iteration']} val_ll={meta['val_logloss']:.4f}"
                )
                db.upsert_training(
                    {
                        "run_id": run_id,
                        "window_id": w.window_id,
                        "side": side,
                        "train_start_year": w.train_start,
                        "train_end_year": w.train_end,
                        "val_year": w.val_year,
                        "test_year": w.test_year,
                        "model_version": model_version,
                        "feature_version": FEATURE_VERSION,
                        "label_version": LABEL_VERSION,
                        "n_train": meta["n_train"],
                        "n_val": meta["n_val"],
                        "n_features": len(feat),
                        "best_iteration": meta["best_iteration"],
                        "val_logloss": meta["val_logloss"],
                        "feature_names": feat,
                        "model_path": str(model_path.relative_to(_ROOT)),
                        "metrics": meta,
                    }
                )

            if not boosters:
                print("  no models — skip window")
                continue

            entries = build_test_entries(
                long_df=long_df,
                short_df=short_df,
                h1=h1,
                feat=feat,
                window=w,
                boosters=boosters,
                top_pct=args.top_pct,
            )
            # Leakage guard on scored panel.
            if not entries.empty:
                assert set(entries["timestamp"].dt.year.unique()) == {w.test_year}

            panel = apply_trail(entries, mkt)
            print(f"  candidates={len(entries)} trailed={len(panel)}")
            traded, metrics = run_portfolio(
                panel,
                starting=float(args.starting),
                ruin_stop=True,
                leverage=float(args.leverage),
            )
            # Enrich traded with signal fields for DB.
            if not traded.empty and not panel.empty:
                # merge on timestamp+side (max_open=1 so unique enough)
                traded = traded.merge(
                    panel[["timestamp", "side", "y_prob", "entry_price", "net_return", "exit_reason"]],
                    on=["timestamp", "side"],
                    how="left",
                    suffixes=("", "_p"),
                )

            summary = {
                "year": w.test_year,
                "window_id": w.window_id,
                "n_candidates": int(len(entries)),
                **metrics,
            }
            yearly_summaries.append(summary)
            print(
                f"  trades={metrics['n_trades']} ret={metrics['total_return']:+.1%} "
                f"final=${metrics['final_equity']:.1f} dd={metrics['max_drawdown']:.1%} "
                f"pf={metrics['profit_factor']:.2f} wr={metrics['win_rate']:.1%}"
            )

            db.upsert_backtest(
                {
                    "run_id": run_id,
                    "window_id": w.window_id,
                    "test_year": w.test_year,
                    "scope": "year",
                    "n_candidates": int(len(entries)),
                    "n_trades": metrics["n_trades"],
                    "total_return": metrics["total_return"],
                    "final_equity": metrics["final_equity"],
                    "max_drawdown": metrics["max_drawdown"],
                    "profit_factor": metrics["profit_factor"] if np.isfinite(metrics["profit_factor"]) else None,
                    "win_rate": metrics["win_rate"],
                    "avg_holding_bars": metrics["avg_holding_bars"],
                    "starting_equity": float(args.starting),
                    "metrics": metrics,
                }
            )
            if not traded.empty:
                trade_rows = []
                equity_rows = []
                for i, r in enumerate(traded.itertuples(index=False), start=1):
                    trade_rows.append(
                        {
                            "run_id": run_id,
                            "window_id": w.window_id,
                            "test_year": w.test_year,
                            "trade_seq": i,
                            "timestamp": pd.Timestamp(r.timestamp).to_pydatetime(),
                            "side": str(r.side),
                            "entry_price": float(getattr(r, "entry_price", 0.0) or 0.0),
                            "lots": float(r.lots),
                            "y_prob": float(getattr(r, "y_prob", float("nan"))),
                            "net_return": float(getattr(r, "net_return", float("nan"))),
                            "pnl": float(r.pnl),
                            "equity": float(r.equity),
                            "holding_bars": int(r.holding_bars),
                            "exit_reason": str(getattr(r, "exit_reason", "") or ""),
                        }
                    )
                    equity_rows.append(
                        {
                            "run_id": run_id,
                            "scope": "year",
                            "test_year": w.test_year,
                            "timestamp": pd.Timestamp(r.timestamp).to_pydatetime(),
                            "equity": float(r.equity),
                            "pnl": float(r.pnl),
                        }
                    )
                db.insert_trades(trade_rows)
                db.insert_equity(equity_rows)
                continuous_panels.append(panel.assign(test_year=w.test_year))

            if args.debug_files and not panel.empty:
                panel.to_parquet(DEBUG_DIR / f"{run_id}_{w.window_id}_panel.parquet", index=False)

        # Combined OOS equity (compound across test years, models still per-year frozen).
        cont_metrics: dict[str, Any] = {}
        if continuous_panels:
            all_panel = pd.concat(continuous_panels, ignore_index=True).sort_values("timestamp")
            traded_c, cont_metrics = run_portfolio(
                all_panel.drop(columns=["test_year"], errors="ignore"),
                starting=float(args.starting),
                ruin_stop=True,
                leverage=float(args.leverage),
            )
            print("\n=== COMBINED OOS (compounded across test years) ===")
            print(
                f"trades={cont_metrics['n_trades']} ret={cont_metrics['total_return']:+.1%} "
                f"final=${cont_metrics['final_equity']:.1f} dd={cont_metrics['max_drawdown']:.1%} "
                f"pf={cont_metrics['profit_factor']:.2f} wr={cont_metrics['win_rate']:.1%}"
            )
            db.upsert_backtest(
                {
                    "run_id": run_id,
                    "window_id": "combined",
                    "test_year": windows[-1].test_year,
                    "scope": "combined",
                    "n_candidates": int(len(all_panel)),
                    "n_trades": cont_metrics["n_trades"],
                    "total_return": cont_metrics["total_return"],
                    "final_equity": cont_metrics["final_equity"],
                    "max_drawdown": cont_metrics["max_drawdown"],
                    "profit_factor": (
                        cont_metrics["profit_factor"]
                        if np.isfinite(cont_metrics["profit_factor"])
                        else None
                    ),
                    "win_rate": cont_metrics["win_rate"],
                    "avg_holding_bars": cont_metrics["avg_holding_bars"],
                    "starting_equity": float(args.starting),
                    "metrics": cont_metrics,
                }
            )
            if not traded_c.empty:
                eq_rows = [
                    {
                        "run_id": run_id,
                        "scope": "combined",
                        "test_year": None,
                        "timestamp": pd.Timestamp(r.timestamp).to_pydatetime(),
                        "equity": float(r.equity),
                        "pnl": float(r.pnl),
                    }
                    for r in traded_c.itertuples(index=False)
                ]
                db.insert_equity(eq_rows)

        db.set_run_status(run_id, "completed")

        # Console report
        print("\n=== PER-TEST-YEAR REPORT ===")
        print(f"{'year':>6} {'trades':>7} {'return':>10} {'final':>10} {'dd':>8} {'pf':>6} {'wr':>7}")
        for s in yearly_summaries:
            print(
                f"{s['year']:6d} {s['n_trades']:7d} {s['total_return']:+10.1%} "
                f"${s['final_equity']:9.1f} {s['max_drawdown']:8.1%} "
                f"{s['profit_factor']:6.2f} {s['win_rate']:7.1%}"
            )

        if args.debug_files:
            report = {
                "run_id": run_id,
                "policy": policy,
                "yearly": yearly_summaries,
                "combined": cont_metrics,
            }
            (DEBUG_DIR / f"{run_id}_report.json").write_text(
                json.dumps(report, indent=2, default=float), encoding="utf-8"
            )
            lines = [
                f"# Rolling WF {run_id}",
                "",
                f"Train {args.train_years}y / Val 1y / Test 1y | top {args.top_pct:.0%} | "
                f"ATR trail {TRAIL} | max_open={MAX_OPEN} | lot={FIXED_LOT} | start=${args.starting}",
                "",
                "| Year | Trades | Return | Final | DD | PF | WR |",
                "|---:|---:|---:|---:|---:|---:|---:|",
            ]
            for s in yearly_summaries:
                lines.append(
                    f"| {s['year']} | {s['n_trades']} | {s['total_return']:+.1%} | "
                    f"${s['final_equity']:.1f} | {s['max_drawdown']:.1%} | "
                    f"{s['profit_factor']:.2f} | {s['win_rate']:.1%} |"
                )
            if cont_metrics:
                lines += [
                    "",
                    (
                        f"Combined OOS: trades={cont_metrics['n_trades']} "
                        f"ret={cont_metrics['total_return']:+.1%} "
                        f"final=${cont_metrics['final_equity']:.1f} "
                        f"dd={cont_metrics['max_drawdown']:.1%} "
                        f"pf={cont_metrics['profit_factor']:.2f} "
                        f"wr={cont_metrics['win_rate']:.1%}"
                    ),
                ]
            (DEBUG_DIR / f"{run_id}_report.md").write_text("\n".join(lines), encoding="utf-8")
            print(f"\ndebug files -> {DEBUG_DIR}")

        print(f"\nDONE run_id={run_id}")
        if db.enabled:
            print("Query examples:")
            print(f"  SELECT * FROM research.wf_backtests WHERE run_id='{run_id}' ORDER BY test_year, scope;")
            print(f"  SELECT * FROM research.wf_equity WHERE run_id='{run_id}' AND scope='combined' ORDER BY timestamp;")
        return 0
    except Exception as exc:
        logger.exception("rolling_wf_failed")
        db.set_run_status(run_id, "failed", notes=str(exc)[:500])
        raise


if __name__ == "__main__":
    raise SystemExit(main())
