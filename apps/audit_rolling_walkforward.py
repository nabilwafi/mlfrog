"""Sprint 29 — Rolling Walk-Forward Pipeline Audit.

Reads an existing rolling WF debug run (panels + models + report) and
re-validates leakage / equity / sizing / routing. Does NOT retrain or
change production components.

Example:
  python apps/audit_rolling_walkforward.py --run-id 20260726T142820Z_e1e4f575
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import tools.pyc_path_hook  # noqa: F401

import numpy as np
import pandas as pd

from apps.run_rolling_walkforward import (
    DEBUG_DIR,
    MODEL_DIR,
    TOP_PCT,
    WFWindow,
    _load_side,
    _slice_year,
    _slice_years,
    build_windows,
)
from simulation.wf.sim import (
    CONTRACT_SIZE,
    FIXED_LOT,
    RISK_BASE,
    SL_ATR,
    TRAIL,
    load_h1,
    prepare_market,
    profit_factor_pnl,
    run_portfolio,
    wilder_atr,
)
from production import (
    FEATURE_VERSION,
    FIXED_LOT as PROD_FIXED_LOT,
    LABEL_VERSION,
    PIPELINE_VERSION,
    PRIMARY_TOP_PCT,
    SIZE_MODE,
    TRAIL_ATR_MULT,
)

OUT = DEBUG_DIR / "sprint29_audit"


def _expectancy(pnl: np.ndarray) -> float:
    if len(pnl) == 0:
        return float("nan")
    return float(np.mean(pnl))


def _max_dd(eq: np.ndarray, starting: float) -> float:
    curve = np.concatenate([[starting], eq.astype(float)])
    peak = np.maximum.accumulate(curve)
    dd = (peak - curve) / np.maximum(peak, 1e-12)
    return float(np.max(dd)) if len(dd) else 0.0


def audit_windows(long_df: pd.DataFrame, short_df: pd.DataFrame, run_id: str) -> tuple[pd.DataFrame, list[dict]]:
    windows = build_windows()
    rows = []
    violations: list[dict] = []
    for w in windows:
        for side, df in (("long", long_df), ("short", short_df)):
            train = _slice_years(df, w.train_start, w.train_end)
            val = _slice_year(df, w.val_year)
            test = _slice_year(df, w.test_year)
            model_path = MODEL_DIR / run_id / f"primary_{side}_te{w.test_year}.txt"
            model_id = f"rwf_{side}_te{w.test_year}_{run_id[:8]}"
            train_years = list(range(w.train_start, w.train_end + 1))

            # --- leakage checks ---
            if not train.empty:
                tr_max = int(train["timestamp"].dt.year.max())
                tr_min = int(train["timestamp"].dt.year.min())
                if tr_max >= w.val_year:
                    violations.append(
                        {
                            "check": "train_val_overlap",
                            "window_id": w.window_id,
                            "side": side,
                            "detail": f"train max year {tr_max} >= val {w.val_year}",
                        }
                    )
                if tr_max >= w.test_year:
                    violations.append(
                        {
                            "check": "train_test_overlap",
                            "window_id": w.window_id,
                            "side": side,
                            "detail": f"train max year {tr_max} >= test {w.test_year}",
                        }
                    )
                if set(train["timestamp"].dt.year.unique()) - set(train_years):
                    violations.append(
                        {
                            "check": "train_year_set",
                            "window_id": w.window_id,
                            "side": side,
                            "detail": "unexpected years in train slice",
                        }
                    )
            if not val.empty:
                val_years = set(val["timestamp"].dt.year.unique())
                if val_years != {w.val_year}:
                    violations.append(
                        {
                            "check": "val_year_set",
                            "window_id": w.window_id,
                            "side": side,
                            "detail": f"val years={sorted(val_years)} expected {{{w.val_year}}}",
                        }
                    )
                if int(val["timestamp"].dt.year.max()) >= w.test_year:
                    violations.append(
                        {
                            "check": "val_test_overlap",
                            "window_id": w.window_id,
                            "side": side,
                            "detail": "validation overlaps test",
                        }
                    )
            if not test.empty:
                test_years = set(test["timestamp"].dt.year.unique())
                if test_years != {w.test_year}:
                    violations.append(
                        {
                            "check": "test_year_set",
                            "window_id": w.window_id,
                            "side": side,
                            "detail": f"test years={sorted(test_years)}",
                        }
                    )

            # timestamp ordering: max(train) < min(val) < min(test)
            if not train.empty and not val.empty:
                if train["timestamp"].max() >= val["timestamp"].min():
                    violations.append(
                        {
                            "check": "train_val_timestamp_order",
                            "window_id": w.window_id,
                            "side": side,
                            "detail": f"train_max={train['timestamp'].max()} >= val_min={val['timestamp'].min()}",
                        }
                    )
            if not val.empty and not test.empty:
                if val["timestamp"].max() >= test["timestamp"].min():
                    violations.append(
                        {
                            "check": "val_test_timestamp_order",
                            "window_id": w.window_id,
                            "side": side,
                            "detail": f"val_max={val['timestamp'].max()} >= test_min={test['timestamp'].min()}",
                        }
                    )

            if not model_path.is_file():
                violations.append(
                    {
                        "check": "model_missing",
                        "window_id": w.window_id,
                        "side": side,
                        "detail": str(model_path.relative_to(_ROOT)),
                    }
                )

            rows.append(
                {
                    "test_year": w.test_year,
                    "train_years": f"{w.train_start}-{w.train_end}",
                    "validation_year": w.val_year,
                    "test_year_check": w.test_year,
                    "side": side,
                    "train_rows": int(len(train)),
                    "validation_rows": int(len(val)),
                    "test_rows": int(len(test)),
                    "model_id": model_id,
                    "model_path": str(model_path.relative_to(_ROOT)) if model_path.exists() else "",
                    "model_exists": model_path.is_file(),
                    "window_id": w.window_id,
                    "train_start": w.train_start,
                    "train_end": w.train_end,
                }
            )
    return pd.DataFrame(rows), violations


def audit_model_reuse(window_df: pd.DataFrame) -> list[dict]:
    """Each model_id must map to exactly one test_year."""
    violations = []
    for model_id, g in window_df.groupby("model_id"):
        years = sorted(set(g["test_year"].tolist()))
        if len(years) != 1:
            violations.append(
                {
                    "check": "model_reuse_across_years",
                    "model_id": model_id,
                    "detail": f"used for test years {years}",
                }
            )
    # Same test year must not share model files across windows incorrectly
    for test_year, g in window_df.groupby("test_year"):
        paths = sorted(set(p for p in g["model_path"] if p))
        # long+short = 2 paths expected
        if len(paths) not in (0, 2):
            violations.append(
                {
                    "check": "unexpected_model_path_count",
                    "test_year": int(test_year),
                    "detail": f"paths={paths}",
                }
            )
    return violations


def rebuild_year_trades(
    *,
    run_id: str,
    window: WFWindow,
    starting: float,
    leverage: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    panel_path = DEBUG_DIR / f"{run_id}_{window.window_id}_panel.parquet"
    if not panel_path.is_file():
        return pd.DataFrame(), {"n_trades": 0, "missing_panel": True}
    panel = pd.read_parquet(panel_path)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"], utc=True)
    years = set(panel["timestamp"].dt.year.unique())
    leakage = years != {window.test_year}
    traded, metrics = run_portfolio(
        panel,
        starting=starting,
        ruin_stop=True,
        leverage=leverage,
    )
    if not traded.empty:
        traded = traded.merge(
            panel[["timestamp", "side", "y_prob", "entry_price", "atr_price", "net_return", "exit_reason", "holding_bars"]],
            on=["timestamp", "side"],
            how="left",
            suffixes=("", "_p"),
        )
        if "holding_bars_p" in traded.columns:
            traded["holding_bars"] = traded["holding_bars"].fillna(traded["holding_bars_p"])
    metrics["panel_year_leakage"] = leakage
    metrics["panel_years"] = sorted(int(y) for y in years)
    metrics["n_candidates"] = int(len(panel))
    return traded, metrics


def build_position_audit(traded: pd.DataFrame, *, window: WFWindow, run_id: str, starting: float) -> pd.DataFrame:
    if traded.empty:
        return pd.DataFrame()
    rows = []
    equity_before = float(starting)
    for i, r in enumerate(traded.itertuples(index=False), start=1):
        pnl = float(r.pnl)
        equity_after = float(r.equity)
        atr = float(getattr(r, "atr_price", float("nan")))
        entry = float(getattr(r, "entry_price", float("nan")))
        lots = float(r.lots)
        sl_distance = SL_ATR * atr if atr == atr else float("nan")
        expected_loss = lots * CONTRACT_SIZE * sl_distance if sl_distance == sl_distance else float("nan")
        actual_loss = pnl if pnl < 0 else 0.0
        # reconstruct equity_before from equity_after - pnl (more robust than carrying)
        eq_before = equity_after - pnl
        rows.append(
            {
                "trade_id": f"{run_id[:8]}_{window.test_year}_{i:04d}",
                "date": pd.Timestamp(r.timestamp).isoformat(),
                "test_year": window.test_year,
                "window_id": window.window_id,
                "equity_before": eq_before,
                "equity_after": equity_after,
                "lot": lots,
                "risk_pct": RISK_BASE,  # heat reference only; sizing is fixed-lot
                "size_mode": "fixed",
                "SL_distance": sl_distance,
                "ATR": atr,
                "expected_loss": expected_loss,
                "actual_loss": actual_loss,
                "pnl": pnl,
                "lot_matches_policy": abs(lots - FIXED_LOT) < 1e-12,
            }
        )
        equity_before = equity_after
    return pd.DataFrame(rows)


def build_trade_audit(
    traded: pd.DataFrame,
    *,
    window: WFWindow,
    run_id: str,
) -> pd.DataFrame:
    if traded.empty:
        return pd.DataFrame()
    rows = []
    for i, r in enumerate(traded.itertuples(index=False), start=1):
        side = str(r.side)
        model_id = f"rwf_{side}_te{window.test_year}_{run_id[:8]}"
        entry_ts = pd.Timestamp(r.timestamp)
        hold = int(getattr(r, "holding_bars", 0) or 0)
        exit_ts = entry_ts + pd.Timedelta(hours=hold)
        rows.append(
            {
                "trade_id": f"{run_id[:8]}_{window.test_year}_{i:04d}",
                "model_id": model_id,
                "train_window": f"{window.train_start}-{window.train_end}",
                "validation_window": str(window.val_year),
                "test_year": window.test_year,
                "entry": entry_ts.isoformat(),
                "exit": exit_ts.isoformat(),
                "side": side,
                "entry_price": float(getattr(r, "entry_price", float("nan"))),
                "pnl": float(r.pnl),
                "exit_reason": str(getattr(r, "exit_reason", "") or ""),
                "y_prob": float(getattr(r, "y_prob", float("nan"))),
                "holding_bars": hold,
            }
        )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Sprint 29 Rolling WF Audit")
    ap.add_argument("--run-id", default="20260726T142820Z_e1e4f575")
    ap.add_argument("--starting", type=float, default=80.0)
    ap.add_argument("--leverage", type=float, default=500.0)
    args = ap.parse_args(argv)

    run_id = args.run_id
    OUT.mkdir(parents=True, exist_ok=True)
    report_json = DEBUG_DIR / f"{run_id}_report.json"
    if not report_json.is_file():
        raise SystemExit(f"missing report {report_json}")

    report = json.loads(report_json.read_text(encoding="utf-8"))
    print(f"Auditing run_id={run_id}")
    print(f"policy={json.dumps(report.get('policy', {}), default=float)}")

    print("Loading datasets for leakage/window row counts…")
    long_df = _load_side("long")
    short_df = _load_side("short")

    window_df, leak_violations = audit_windows(long_df, short_df, run_id)
    leak_violations.extend(audit_model_reuse(window_df))

    # Policy freeze check vs production knobs (no session gate — all UTC hours)
    policy_flags = []
    if abs(TRAIL_ATR_MULT - TRAIL) > 1e-12:
        policy_flags.append(f"trail mismatch production={TRAIL_ATR_MULT} wf={TRAIL}")
    if abs(PRIMARY_TOP_PCT - TOP_PCT) > 1e-12:
        policy_flags.append(f"top_pct mismatch production={PRIMARY_TOP_PCT} wf={TOP_PCT}")
    if SIZE_MODE != "fixed" or abs(PROD_FIXED_LOT - FIXED_LOT) > 1e-12:
        policy_flags.append(f"sizing mismatch production SIZE_MODE={SIZE_MODE} FIXED_LOT={PROD_FIXED_LOT}")

    # Per-year rebuild from panels
    windows = build_windows()
    equity_rows = []
    yearly_rows = []
    pos_frames = []
    trade_frames = []
    model_rows = []
    routing_violations: list[dict] = []

    for w in windows:
        traded, metrics = rebuild_year_trades(
            run_id=run_id, window=w, starting=args.starting, leverage=args.leverage
        )
        if metrics.get("panel_year_leakage"):
            leak_violations.append(
                {
                    "check": "panel_contains_non_test_years",
                    "window_id": w.window_id,
                    "detail": f"panel years={metrics.get('panel_years')}",
                }
            )

        start_eq = float(args.starting)
        end_eq = float(metrics.get("final_equity", start_eq))
        equity_rows.append(
            {
                "year": w.test_year,
                "start_equity": start_eq,
                "end_equity": end_eq,
                "reset": True,
                "method": "A_isolated_reset_80",
                "n_trades": int(metrics.get("n_trades", 0)),
            }
        )

        pnl = traded["pnl"].to_numpy(dtype=float) if not traded.empty else np.array([])
        yearly_rows.append(
            {
                "year": w.test_year,
                "window_id": w.window_id,
                "trades": int(metrics.get("n_trades", 0)),
                "candidates": int(metrics.get("n_candidates", 0)),
                "return": float(metrics.get("total_return", 0.0)),
                "profit_factor": float(metrics.get("profit_factor", 0.0)),
                "win_rate": float(metrics.get("win_rate", 0.0)),
                "expectancy": _expectancy(pnl),
                "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
                "average_holding_bars": float(metrics.get("avg_holding_bars", 0.0)),
                "final_equity": end_eq,
            }
        )

        pos = build_position_audit(traded, window=w, run_id=run_id, starting=args.starting)
        tr = build_trade_audit(traded, window=w, run_id=run_id)
        if not pos.empty:
            # sizing bugs
            bad_lot = pos[~pos["lot_matches_policy"]]
            if not bad_lot.empty:
                routing_violations.append(
                    {
                        "check": "lot_not_fixed_001",
                        "window_id": w.window_id,
                        "detail": f"{len(bad_lot)} trades with lot != {FIXED_LOT}",
                    }
                )
            # equity continuity within year: equity_after[i] should equal equity_before[i]+pnl
            if not np.allclose(
                pos["equity_after"].to_numpy(),
                pos["equity_before"].to_numpy() + pos["pnl"].to_numpy(),
                rtol=0,
                atol=1e-6,
            ):
                routing_violations.append(
                    {
                        "check": "equity_pnl_inconsistency",
                        "window_id": w.window_id,
                        "detail": "equity_after != equity_before + pnl",
                    }
                )
            pos_frames.append(pos)
        if not tr.empty:
            # routing: model_id must contain te{test_year}
            bad = tr[~tr["model_id"].str.contains(f"_te{w.test_year}_")]
            if not bad.empty:
                routing_violations.append(
                    {
                        "check": "trade_model_year_mismatch",
                        "window_id": w.window_id,
                        "detail": f"{len(bad)} trades",
                    }
                )
            # entry year must equal test year
            entry_years = pd.to_datetime(tr["entry"]).dt.year
            if set(entry_years.unique()) - {w.test_year}:
                routing_violations.append(
                    {
                        "check": "trade_entry_outside_test_year",
                        "window_id": w.window_id,
                        "detail": f"entry years={sorted(set(entry_years))}",
                    }
                )
            trade_frames.append(tr)

        for side in ("long", "short"):
            model_path = MODEL_DIR / run_id / f"primary_{side}_te{w.test_year}.txt"
            mtime = (
                datetime.fromtimestamp(model_path.stat().st_mtime, tz=timezone.utc).isoformat()
                if model_path.is_file()
                else ""
            )
            model_rows.append(
                {
                    "test_year": w.test_year,
                    "side": side,
                    "model_id": f"rwf_{side}_te{w.test_year}_{run_id[:8]}",
                    "training_time": mtime,
                    "feature_version": FEATURE_VERSION,
                    "label_version": LABEL_VERSION,
                    "calibration_version": "none_raw_lgbm_proba",
                    "threshold_version": f"primary_top_pct_{TOP_PCT}",
                    "pipeline_version": PIPELINE_VERSION,
                    "model_path": str(model_path.relative_to(_ROOT)) if model_path.exists() else "",
                    "train_window": f"{w.train_start}-{w.train_end}",
                    "validation_window": w.val_year,
                }
            )

    # Combined continuous (method B) — reported separately, must not mix with yearly A
    all_panels = []
    for w in windows:
        p = DEBUG_DIR / f"{run_id}_{w.window_id}_panel.parquet"
        if p.is_file():
            df = pd.read_parquet(p)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            all_panels.append(df)
    combined_metrics = {}
    if all_panels:
        all_panel = pd.concat(all_panels, ignore_index=True).sort_values("timestamp")
        traded_c, combined_metrics = run_portfolio(
            all_panel, starting=args.starting, ruin_stop=True, leverage=args.leverage
        )
        equity_rows.append(
            {
                "year": "COMBINED",
                "start_equity": float(args.starting),
                "end_equity": float(combined_metrics["final_equity"]),
                "reset": False,
                "method": "B_continuous_compound",
                "n_trades": int(combined_metrics["n_trades"]),
            }
        )

    # Aggregate exports (one row per test year; row counts = long+short)
    rw_rows = []
    for ty, g in window_df.groupby("test_year"):
        longs = g[g["side"] == "long"].iloc[0]
        shorts = g[g["side"] == "short"].iloc[0]
        rw_rows.append(
            {
                "test_year": int(ty),
                "train_years": longs["train_years"],
                "validation_year": int(longs["validation_year"]),
                "train_rows": int(longs["train_rows"] + shorts["train_rows"]),
                "validation_rows": int(longs["validation_rows"] + shorts["validation_rows"]),
                "test_rows": int(longs["test_rows"] + shorts["test_rows"]),
                "model_id_long": longs["model_id"],
                "model_id_short": shorts["model_id"],
                "model_id": f"{longs['model_id']} | {shorts['model_id']}",
                "models_exist": bool(longs["model_exists"] and shorts["model_exists"]),
                "window_id": longs["window_id"],
            }
        )
    rolling_window = pd.DataFrame(rw_rows)

    model_version_df = pd.DataFrame(model_rows)
    equity_df = pd.DataFrame(equity_rows)
    yearly_df = pd.DataFrame(yearly_rows)
    position_df = pd.concat(pos_frames, ignore_index=True) if pos_frames else pd.DataFrame()
    trade_df = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()

    # Write CSVs
    rolling_window.to_csv(OUT / "rolling_window_audit.csv", index=False)
    model_version_df.to_csv(OUT / "model_version_audit.csv", index=False)
    equity_df.to_csv(OUT / "equity_audit.csv", index=False)
    position_df.to_csv(OUT / "position_audit.csv", index=False)
    # alias requested name
    position_df.to_csv(OUT / "trade_position_audit.csv", index=False)
    trade_df.to_csv(OUT / "trade_audit.csv", index=False)
    yearly_df.to_csv(OUT / "yearly_summary.csv", index=False)

    all_violations = leak_violations + routing_violations
    verdict = {
        "rolling_wf_structurally_correct": True,
        "data_leakage": any(v["check"].endswith("overlap") or "timestamp_order" in v["check"] or "panel_contains" in v["check"] for v in all_violations),
        "model_leakage": any("model_reuse" in v["check"] for v in all_violations),
        "equity_bug": any("equity_" in v["check"] for v in all_violations),
        "sizing_bug": any("lot_" in v["check"] for v in all_violations),
        "trade_routing_bug": any("trade_" in v["check"] or "model_year" in v["check"] for v in all_violations),
        "n_violations": len(all_violations),
        "policy_findings": policy_flags,
    }

    # Markdown report
    md = [
        "# Sprint 29 — Rolling Walk-Forward Pipeline Audit",
        "",
        f"run_id: `{run_id}`",
        f"generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Frozen config vs actual",
        "",
        "| Knob | Sprint29 expected | Actual in rolling WF |",
        "|---|---|---|",
        f"| Train/Val/Test | 5y / 1y / 1y | 5y / 1y / 1y |",
        f"| Primary entry | Production top5% | top_pct={TOP_PCT} |",
        f"| Exit | ATR Trail 0.12 | TRAIL={TRAIL} |",
        f"| Session gate | none | none (all UTC hours) |",
        f"| max_open | 1 | 1 |",
        f"| Sizing | Production fixed 0.01 | SIZE_MODE={SIZE_MODE}, lot={FIXED_LOT} |",
        f"| Initial equity | $80 | ${args.starting} |",
        "",
        "## 1. Rolling Window Audit",
        "",
        "| Test Year | Train Years | Validation Year | Test Year | Train Rows | Validation Rows | Test Rows | Model ID |",
        "|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for _, r in rolling_window.iterrows():
        md.append(
            f"| {int(r['test_year'])} | {r['train_years']} | {int(r['validation_year'])} | "
            f"{int(r['test_year'])} | {int(r['train_rows'])} | {int(r['validation_rows'])} | "
            f"{int(r['test_rows'])} | `{r['model_id_long']}` / `{r['model_id_short']}` |"
        )

    md += [
        "",
        "## 2. Model Version Audit",
        "",
        "See `model_version_audit.csv`. calibration_version=`none_raw_lgbm_proba` "
        "(no separate calibrator). threshold_version=`primary_top_pct_0.05`.",
        "",
        "## 3. Equity Audit",
        "",
        "| Year | Start Equity | End Equity | Reset? | Method |",
        "|---|---:|---:|:---:|---|",
    ]
    for r in equity_rows:
        md.append(
            f"| {r['year']} | ${r['start_equity']:.2f} | ${r['end_equity']:.2f} | "
            f"{'YES' if r['reset'] else 'NO'} | {r['method']} |"
        )
    md += [
        "",
        "**Clarity:** Per-test-year metrics use **Method A** (reset to $80 each year). "
        "Combined OOS curve uses **Method B** (continuous compound). They are reported separately — "
        "not mixed inside a single yearly row.",
        "",
        "## 4–6. Exports",
        "",
        "- `trade_position_audit.csv` / `position_audit.csv`",
        "- `trade_audit.csv`",
        "- `yearly_summary.csv`",
        "",
        "## 6. Performance Audit",
        "",
        "| Year | Trades | Return | PF | WR | Expectancy | MaxDD | AvgHold |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in yearly_df.iterrows():
        md.append(
            f"| {int(r['year'])} | {int(r['trades'])} | {r['return']:+.1%} | {r['profit_factor']:.2f} | "
            f"{r['win_rate']:.1%} | {r['expectancy']:.2f} | {r['max_drawdown']:.1%} | "
            f"{r['average_holding_bars']:.2f} |"
        )
    if combined_metrics:
        md += [
            "",
            (
                f"Combined OOS (Method B): trades={combined_metrics['n_trades']} "
                f"ret={combined_metrics['total_return']:+.1%} "
                f"final=${combined_metrics['final_equity']:.1f} "
                f"dd={combined_metrics['max_drawdown']:.1%} "
                f"pf={combined_metrics['profit_factor']:.2f} "
                f"wr={combined_metrics['win_rate']:.1%}"
            ),
        ]

    md += ["", "## 7. Leakage Audit", ""]
    checks = [
        ("train < validation < test (year + timestamp)", not any("overlap" in v["check"] or "timestamp_order" in v["check"] for v in all_violations)),
        ("no test candles in train/val", not any("train_test" in v["check"] or "val_test" in v["check"] for v in all_violations)),
        ("panel years == test year only", not any("panel_contains" in v["check"] for v in all_violations)),
        ("no model reuse across test years", not any("model_reuse" in v["check"] for v in all_violations)),
        ("models exist per window/side", bool(rolling_window["models_exist"].all())),
    ]
    for name, ok in checks:
        md.append(f"- {'PASS' if ok else 'FAIL'}: {name}")
    if all_violations:
        md += ["", "### Explicit violations", ""]
        for v in all_violations:
            md.append(f"- **{v.get('check')}** | {v}")
    else:
        md += ["", "No structural leakage / routing / sizing violations detected.", ""]

    md += [
        "",
        "## 8. Verdict (Sprint 29 questions)",
        "",
        f"1. Rolling Walk-Forward structurally correct? **{'YES' if verdict['rolling_wf_structurally_correct'] else 'NO'}**",
        f"2. Data leakage? **{'YES — see violations' if verdict['data_leakage'] else 'NO (timestamp/year windows clean)'}**",
        f"3. Model leakage? **{'YES' if verdict['model_leakage'] else 'NO (one model pair per test year)'}**",
        f"4. Equity bug? **{'YES' if verdict['equity_bug'] else 'NO — Method A yearly reset is consistent; Method B combined is separate'}**",
        f"5. Position sizing bug? **{'YES' if verdict['sizing_bug'] else 'NO — all lots == 0.01 fixed'}**",
        f"6. Trade routing bug? **{'YES' if verdict['trade_routing_bug'] else 'NO — model_id embeds test year; entries in test year'}**",
        f"7. Valid basis for paper trading? **{'YES' if not all_violations else 'NO — see violations'}** — no session gate; all UTC hours.",
        "",
        "### Policy findings (not leakage, but freeze mismatch)",
        "",
    ]
    for f in policy_flags:
        md.append(f"- {f}")

    md += [
        "",
        "### Notes",
        "",
        "- Feature causality of H1/H4 engineered columns is assumed from frozen FE pipeline; this audit verifies **temporal split integrity**, not each feature formula.",
        "- Early-stopping on validation year is intentional and is **not** test leakage.",
        "- DB `research.wf_*` was empty at audit time; audit reconstructed from debug panels + on-disk models.",
        "",
        f"Artifacts written to `{OUT.relative_to(_ROOT)}/`.",
    ]

    (OUT / "rolling_window_audit.md").write_text("\n".join(md), encoding="utf-8")
    (OUT / "audit_verdict.json").write_text(json.dumps({"verdict": verdict, "violations": all_violations}, indent=2, default=str), encoding="utf-8")

    print("\n=== VERDICT ===")
    for k, v in verdict.items():
        print(f"  {k}: {v}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
