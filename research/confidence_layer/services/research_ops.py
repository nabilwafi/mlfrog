"""Bucket analysis, regime performance, dynamic risk/TP/trail research."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from research.confidence_layer import CONFIDENCE_BUCKETS, RISK_SCHEDULES, STARTING_EQUITY
from research.portfolio_backtest.services.engine import run_portfolio
from research.portfolio_backtest.services.metrics import compute_metrics, profit_factor_pnl


def assign_bucket(confidence: pd.Series) -> pd.Series:
    c = confidence.astype(float)
    labels = np.array(["0-20"] * len(c), dtype=object)
    for name, lo, hi in CONFIDENCE_BUCKETS:
        labels[(c >= lo) & (c < hi)] = name
    return pd.Series(labels, index=c.index)


def bucket_stats(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, lo, hi in CONFIDENCE_BUCKETS:
        sub = panel.loc[(panel["confidence"] >= lo) & (panel["confidence"] < hi)]
        if sub.empty:
            rows.append({"bucket": name, "trades": 0})
            continue
        net = sub["net_return"].to_numpy(dtype=float)
        rows.append(
            {
                "bucket": name,
                "trades": int(len(sub)),
                "win_rate": float((net > 0).mean()),
                "profit_factor": profit_factor_pnl(net),
                "expectancy": float(np.mean(net)),
                "annual_return": float(np.sum(net)),
                "avg_rr": float(sub["initial_rr"].mean()) if "initial_rr" in sub.columns else float("nan"),
                "avg_holding": float(sub["holding_bars"].mean()) if "holding_bars" in sub.columns else float("nan"),
                "max_drawdown": _dd_from_returns(net),
            }
        )
    return pd.DataFrame(rows)


def _dd_from_returns(r: np.ndarray) -> float:
    eq = np.cumsum(r)
    peak = np.maximum.accumulate(eq)
    dd = peak - eq
    return float(dd.max()) if len(dd) else float("nan")


def regime_performance(panel: pd.DataFrame) -> pd.DataFrame:
    if "d1_regime" not in panel.columns:
        return pd.DataFrame()
    rows = []
    for reg, sub in panel.groupby("d1_regime"):
        net = sub["net_return"].to_numpy(dtype=float)
        rows.append(
            {
                "d1_regime": reg,
                "trades": int(len(sub)),
                "win_rate": float((net > 0).mean()),
                "expectancy": float(np.mean(net)),
                "profit_factor": profit_factor_pnl(net),
                "annual_return": float(np.sum(net)),
                "avg_confidence": float(sub["confidence"].mean()) if "confidence" in sub.columns else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values("expectancy", ascending=False)


def m5_quality_research(panel: pd.DataFrame) -> dict[str, Any]:
    if "m5_entry_quality" not in panel.columns:
        return {"available": False}
    q = panel["m5_entry_quality"].astype(float)
    hi = panel.loc[q >= q.median()]
    lo = panel.loc[q < q.median()]
    return {
        "available": True,
        "high_quality_expectancy": float(hi["net_return"].mean()) if len(hi) else float("nan"),
        "low_quality_expectancy": float(lo["net_return"].mean()) if len(lo) else float("nan"),
        "high_n": int(len(hi)),
        "low_n": int(len(lo)),
        "improves_expectancy": bool(
            len(hi) and len(lo) and float(hi["net_return"].mean()) > float(lo["net_return"].mean())
        ),
        "corr_quality_net": float(q.corr(panel["net_return"].astype(float))),
    }


def risk_for_confidence(confidence: float, schedule: tuple[tuple[float, float, float], ...]) -> float:
    for lo, hi, risk in schedule:
        if lo <= confidence < hi:
            return float(risk)
    return 0.0


def run_dynamic_risk_portfolio(
    panel: pd.DataFrame,
    schedule: tuple[tuple[float, float, float], ...],
    *,
    starting_equity: float = STARTING_EQUITY,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Per-trade risk from confidence; fractional lots (research capital)."""
    trades = panel.sort_values("timestamp").reset_index(drop=True).copy()
    # Need atr for sizing — reuse portfolio engine path by injecting per-row via sequential loop
    from research.portfolio_backtest.services.engine import _lots_from_equity
    from settings.strategy import CONTRACT_SIZE, SL_ATR_MULT

    if "atr_price" not in trades.columns:
        if "atr_entry" in trades.columns:
            trades["atr_price"] = trades["atr_entry"].astype(float)
        else:
            trades["atr_price"] = trades["entry_price"].astype(float) * trades["distance_to_sl"].astype(float)

    eq = float(starting_equity)
    peak = eq
    logs = []
    curve = [{"timestamp": None, "equity": eq, "drawdown": 0.0, "trade_i": -1}]
    for i, row in trades.iterrows():
        conf = float(row["confidence"])
        risk = risk_for_confidence(conf, schedule)
        if risk <= 0 or eq <= 0:
            logs.append({**row.to_dict(), "lots": 0.0, "pnl": 0.0, "skipped": True, "risk_pct": risk,
                         "equity_before": eq, "equity_after": eq})
            continue
        lots = _lots_from_equity(
            eq, float(row["atr_price"]), mode="risk", fixed_lots=None, risk_pct=risk, enforce_volume_min=False
        )
        pnl = lots * CONTRACT_SIZE * float(row["entry_price"]) * float(row["net_return"])
        eq_b = eq
        eq = eq + pnl
        peak = max(peak, eq)
        dd = (peak - eq) / peak if peak > 0 else 0.0
        logs.append({**row.to_dict(), "lots": lots, "pnl": pnl, "skipped": False, "risk_pct": risk,
                     "equity_before": eq_b, "equity_after": eq})
        curve.append({"timestamp": row["timestamp"], "equity": eq, "drawdown": dd, "trade_i": len([x for x in logs if not x["skipped"]])-1})
        if eq <= 0:
            break
    log_df = pd.DataFrame(logs)
    curve_df = pd.DataFrame(curve)
    metrics = compute_metrics(log_df, curve_df, starting_equity=starting_equity)
    return log_df, curve_df, metrics


def search_risk_schedules(
    panel: pd.DataFrame,
    *,
    starting_equity: float = STARTING_EQUITY,
) -> pd.DataFrame:
    """A priori schedule grid on chronological panel (schedules not fit on labels)."""
    rows = []
    for name, schedule in RISK_SCHEDULES:
        _, _, m = run_dynamic_risk_portfolio(panel, schedule, starting_equity=starting_equity)
        rows.append(
            {
                "schedule": name,
                "final_equity": m.get("final_equity"),
                "cagr": m.get("cagr"),
                "max_drawdown": m.get("max_drawdown"),
                "calmar": m.get("calmar"),
                "recovery_factor": m.get("recovery_factor"),
                "trades": m.get("trades"),
                "win_rate": m.get("win_rate"),
                "profit_factor": m.get("profit_factor"),
            }
        )
    return pd.DataFrame(rows).sort_values(["calmar", "cagr"], ascending=False)


def dynamic_tp_research(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Approximate policies using realized MFE/MAE (outcome research).
    Not used as a live signal — compares counterfactual PnL proxies.
    """
    if "mfe" not in panel.columns or "mae" not in panel.columns:
        return pd.DataFrame()
    cost = 0.00015
    rows = []
    # Baseline: full trade net_return
    base = panel["net_return"].to_numpy(dtype=float)
    rows.append({"policy": "baseline_full", "expectancy": float(np.mean(base)), "sum_return": float(np.sum(base)), "n": len(panel)})

    conf = panel["confidence"].to_numpy(dtype=float)
    mfe = panel["mfe"].to_numpy(dtype=float)
    mae = panel["mae"].to_numpy(dtype=float)
    side = panel["side"].astype(str).to_numpy()

    def apply_policy(hi_thr: float, low_take: float, high_take: float) -> np.ndarray:
        # If MFE reaches take fraction of path — use that as return proxy minus cost; else use net_return
        out = []
        for i in range(len(panel)):
            take = high_take if conf[i] >= hi_thr else low_take
            # early TP if mfe >= take
            if mfe[i] >= take:
                out.append(take - cost)
            else:
                out.append(float(base[i]))
        return np.asarray(out, dtype=float)

    for name, hi, low_t, high_t in (
        ("low_tp_quick_high_run", 60.0, 0.003, 0.008),
        ("low_tp_quick_high_run_v2", 70.0, 0.002, 0.010),
        ("uniform_mid_tp", 0.0, 0.005, 0.005),
    ):
        r = apply_policy(hi, low_t, high_t)
        rows.append({"policy": name, "expectancy": float(np.mean(r)), "sum_return": float(np.sum(r)), "n": len(r)})
    return pd.DataFrame(rows).sort_values("expectancy", ascending=False)


def dynamic_trail_research(panel: pd.DataFrame) -> pd.DataFrame:
    """Trail proxy: after MFE peaks, give back trail distance; exit if MAE beyond trail."""
    if "mfe" not in panel.columns or "mae" not in panel.columns:
        return pd.DataFrame()
    cost = 0.00015
    base = panel["net_return"].to_numpy(dtype=float)
    conf = panel["confidence"].to_numpy(dtype=float)
    mfe = panel["mfe"].to_numpy(dtype=float)
    mae = panel["mae"].to_numpy(dtype=float)
    atr_pct = panel["atr_percent"].to_numpy(dtype=float) if "atr_percent" in panel.columns else np.full(len(panel), 0.002)

    rows = [{"policy": "baseline_full", "expectancy": float(np.mean(base)), "sum_return": float(np.sum(base))}]

    for name, trail_mult, conf_scale in (
        ("trail_0_5atr", 0.5, False),
        ("trail_1atr", 1.0, False),
        ("trail_conf_scaled", 1.0, True),
    ):
        out = []
        for i in range(len(panel)):
            trail = trail_mult * max(atr_pct[i], 1e-4)
            if conf_scale:
                # high conf wider trail
                trail = trail * (0.5 + conf[i] / 100.0)
            # If adverse mae exceeds trail before mfe meaningful — stop at -trail
            if mae[i] >= trail and mfe[i] < trail:
                out.append(-trail - cost)
            elif mfe[i] > trail:
                # lock trail: mfe - trail
                out.append(max(mfe[i] - trail, -trail) - cost)
            else:
                out.append(float(base[i]))
        r = np.asarray(out, dtype=float)
        rows.append({"policy": name, "expectancy": float(np.mean(r)), "sum_return": float(np.sum(r))})
    return pd.DataFrame(rows).sort_values("expectancy", ascending=False)
