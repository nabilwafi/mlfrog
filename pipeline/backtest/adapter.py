"""Scored-panel backtest adapter — Meta-only + expected_r sizing + heat −1R."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from pipeline import DAILY_LOSS_STOP_R, META_THRESHOLD, RISK_BASE
from pipeline.l4_meta_edge.edge import decide_meta
from pipeline.l5_risk.engine import build_risk
from research.portfolio_backtest.services.metrics import max_dd_from_equity, profit_factor_pnl
from research.portfolio_heat import H1_HOURS, STARTING_EQUITY
from settings.strategy import CONTRACT_SIZE


def _to_ts(v: Any) -> pd.Timestamp:
    t = pd.Timestamp(v)
    if t.tzinfo is None:
        return t.tz_localize("UTC")
    return t.tz_convert("UTC")


def prepare_scored_universe(panel: pd.DataFrame) -> pd.DataFrame:
    t = panel.sort_values("timestamp").reset_index(drop=True).copy()
    if "meta_proba" not in t.columns and "meta_probability" in t.columns:
        t["meta_proba"] = t["meta_probability"]
    if "atr_price" not in t.columns:
        if "atr_entry" in t.columns:
            t["atr_price"] = t["atr_entry"].astype(float)
        elif "atr" in t.columns:
            t["atr_price"] = t["atr"].astype(float)
        else:
            t["atr_price"] = t["entry_price"].astype(float) * t["distance_to_sl"].astype(float)
    hold = t["holding_bars"].astype(float).fillna(4.0) if "holding_bars" in t.columns else 4.0
    t["exit_ts"] = pd.to_datetime(t["timestamp"], utc=True) + pd.to_timedelta(hold * H1_HOURS, unit="h")
    return t


def _simple_metrics(traded: pd.DataFrame, equity_curve: pd.DataFrame, *, starting_equity: float) -> dict[str, Any]:
    if traded.empty:
        return {
            "n_trades": 0,
            "final_equity": float(starting_equity),
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "profit_factor": float("nan"),
            "win_rate": float("nan"),
        }
    pnl = traded["pnl"].to_numpy(dtype=float)
    eq = equity_curve["equity"].to_numpy(dtype=float)
    final = float(eq[-1])
    wins = float(np.mean(pnl > 0)) if len(pnl) else float("nan")
    return {
        "n_trades": int(len(traded)),
        "final_equity": final,
        "total_return": final / float(starting_equity) - 1.0,
        "max_drawdown": float(max_dd_from_equity(eq)),
        "profit_factor": float(profit_factor_pnl(pnl)),
        "win_rate": wins,
    }


def run_meta_expected_r_backtest(
    panel: pd.DataFrame,
    *,
    starting_equity: float = STARTING_EQUITY,
    meta_threshold: float = META_THRESHOLD,
    daily_loss_r: float = DAILY_LOSS_STOP_R,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Cara A: scored rows only.
    Gate = Meta ≥ threshold (Confidence ignored).
    Size = option C expected_r → risk_pct.
    Heat = day_pnl <= -daily_loss_r * (equity * RISK_BASE).
    Settle = lots * CONTRACT_SIZE * entry * net_return (research convention).
    """
    trades_in = prepare_scored_universe(panel)
    equity = float(starting_equity)
    day = None
    day_pnl = 0.0
    rows: list[dict[str, Any]] = []
    eq_rows: list[dict[str, Any]] = []

    for _, row in trades_in.iterrows():
        ts = _to_ts(row["timestamp"])
        d = ts.date()
        if day != d:
            day = d
            day_pnl = 0.0

        meta = decide_meta(float(row["meta_proba"]), threshold=meta_threshold)
        if not meta.trade:
            continue

        r_unit = equity * RISK_BASE
        if r_unit <= 0 or day_pnl <= -daily_loss_r * r_unit:
            continue

        side = str(row.get("side", "long")).lower()
        entry = float(row["entry_price"])
        atr = float(row["atr_price"])
        risk = build_risk(side=side, entry=entry, atr=atr, equity=equity, expected_r=meta.expected_r)
        if risk is None:
            continue

        net_ret = float(row.get("net_return", 0.0) or 0.0)
        pnl = float(risk.lot) * float(CONTRACT_SIZE) * entry * net_ret
        equity += pnl
        day_pnl += pnl

        rows.append(
            {
                "timestamp": ts,
                "side": side,
                "entry_price": entry,
                "atr": atr,
                "meta_proba": meta.edge_score,
                "expected_r": meta.expected_r,
                "risk_pct": risk.risk_pct,
                "lot": risk.lot,
                "pnl": pnl,
                "net_return": net_ret,
                "equity": equity,
            }
        )
        eq_rows.append({"timestamp": ts, "equity": equity})

    traded = pd.DataFrame(rows)
    equity_curve = pd.DataFrame(eq_rows)
    metrics = _simple_metrics(traded, equity_curve, starting_equity=starting_equity)
    metrics["policy"] = "meta_expected_r_heat1R"
    metrics["meta_threshold"] = float(meta_threshold)
    metrics["confidence_enabled"] = False
    return traded, equity_curve, metrics


def golden_metrics_from_synthetic() -> dict[str, Any]:
    """Tiny deterministic panel for regression fixture (no artifact dependency)."""
    rows = []
    base = pd.Timestamp("2024-01-02 10:00:00", tz="UTC")
    for i, (meta, net) in enumerate([(0.55, 0.01), (0.70, 0.02), (0.46, -0.01)]):
        rows.append(
            {
                "timestamp": base + pd.Timedelta(hours=i),
                "side": "long",
                "entry_price": 2000.0,
                "atr_price": 4.0,
                "meta_proba": meta,
                "net_return": net,
                "holding_bars": 4,
            }
        )
    panel = pd.DataFrame(rows)
    _, _, m = run_meta_expected_r_backtest(panel, starting_equity=80.0)
    return {
        "n_trades": int(m["n_trades"]),
        "final_equity": round(float(m["final_equity"]), 6),
        "total_return": round(float(m["total_return"]), 6),
        "max_drawdown": round(float(m["max_drawdown"]), 6),
    }
