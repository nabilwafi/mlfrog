"""Concurrent-heat aware portfolio simulator on frozen Confidence path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from research.confidence_layer.services.research_ops import risk_for_confidence
from research.portfolio_backtest.services.engine import _lots_from_equity
from research.portfolio_backtest.services.metrics import compute_metrics
from research.portfolio_heat import BASE_RISK, CONF_SKIP, H1_HOURS, STARTING_EQUITY, HeatPolicy
from settings.strategy import CONTRACT_SIZE

# Production confidence schedule (frozen)
SKIP40_FLAT1: tuple[tuple[float, float, float], ...] = (
    (0.0, CONF_SKIP, 0.0),
    (CONF_SKIP, 100.01, BASE_RISK),
)


@dataclass
class _OpenPos:
    exit_ts: pd.Timestamp
    side: str
    risk: float
    r_unit: float  # equity_at_entry * risk for PnL-in-R


def _to_ts(v: Any) -> pd.Timestamp:
    t = pd.Timestamp(v)
    if t.tzinfo is None:
        return t.tz_localize("UTC")
    return t.tz_convert("UTC")


def _session_of(row: pd.Series) -> str:
    if float(row.get("session_london_ny_overlap", 0) or 0) >= 0.5:
        return "overlap"
    if float(row.get("session_london", 0) or 0) >= 0.5:
        return "london"
    if float(row.get("session_newyork", 0) or 0) >= 0.5:
        return "newyork"
    if float(row.get("session_asia", 0) or 0) >= 0.5:
        return "asia"
    return "other"


def _vol_mult(row: pd.Series) -> float:
    # High ATR / vol -> lower heat
    if "atr_percentile_252" in row.index and pd.notna(row["atr_percentile_252"]):
        p = float(row["atr_percentile_252"])
    elif "volatility_rank" in row.index and pd.notna(row["volatility_rank"]):
        p = float(row["volatility_rank"])
    else:
        return 1.0
    if p >= 0.66:
        return 0.5
    if p <= 0.33:
        return 1.25
    return 1.0


def _d1_mult(row: pd.Series) -> float:
    reg = str(row.get("d1_regime", "") or "")
    return {
        "Strong Bull": 1.25,
        "Weak Bull": 1.0,
        "Sideways": 0.75,
        "Compression": 0.5,
        "Weak Bear": 0.5,
        "Strong Bear": 0.5,
    }.get(reg, 1.0)


def _session_mult(row: pd.Series) -> float:
    return {"asia": 0.5, "london": 1.0, "overlap": 1.25, "newyork": 1.0, "other": 0.75}[_session_of(row)]


def _conf_mult(row: pd.Series) -> float:
    c = float(row.get("confidence", 50) or 50)
    if c >= 70:
        return 1.25
    if c < 50:
        return 0.75
    return 1.0


def prepare_universe(panel: pd.DataFrame) -> pd.DataFrame:
    """Confidence gate already encoded in schedule; keep full panel sorted."""
    t = panel.sort_values("timestamp").reset_index(drop=True).copy()
    if "atr_price" not in t.columns:
        if "atr_entry" in t.columns:
            t["atr_price"] = t["atr_entry"].astype(float)
        else:
            t["atr_price"] = t["entry_price"].astype(float) * t["distance_to_sl"].astype(float)
    hold = t["holding_bars"].astype(float).fillna(4.0) if "holding_bars" in t.columns else 4.0
    t["exit_ts"] = pd.to_datetime(t["timestamp"], utc=True) + pd.to_timedelta(hold * H1_HOURS, unit="h")
    return t


def _week_key(ts: pd.Timestamp) -> str:
    # ISO week label; avoids tz Period warnings
    iso = ts.isocalendar()
    return f"{int(iso.year)}-W{int(iso.week):02d}"


def run_heat_portfolio(
    panel: pd.DataFrame,
    policy: HeatPolicy,
    *,
    starting_equity: float = STARTING_EQUITY,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Baseline-compatible PnL settle at entry; heat uses open-risk via holding_bars.
    Confidence schedule frozen: skip40_flat1.
    """
    trades = prepare_universe(panel)
    eq = float(starting_equity)
    peak = eq
    opens: list[_OpenPos] = []
    logs: list[dict[str, Any]] = []
    curve = [{"timestamp": None, "equity": eq, "drawdown": 0.0, "trade_i": -1}]

    # rolling state
    day_pnl: dict[Any, float] = {}
    week_pnl: dict[Any, float] = {}
    day_trades: dict[Any, int] = {}
    day_london: dict[Any, int] = {}
    day_ny: dict[Any, int] = {}
    day_session_losses: dict[tuple[Any, str], int] = {}
    consec_losses = 0
    cooldown_until: pd.Timestamp | None = None

    for _, row in trades.iterrows():
        ts = _to_ts(row["timestamp"])
        # prune closed positions
        opens = [o for o in opens if o.exit_ts > ts]

        conf = float(row["confidence"])
        base_risk = risk_for_confidence(conf, SKIP40_FLAT1)
        skip_reason = None
        risk = base_risk

        if base_risk <= 0 or eq <= 0:
            skip_reason = "conf_gate_or_broke"
            risk = 0.0
        else:
            # scales
            mult = 1.0
            if policy.vol_scale:
                mult *= _vol_mult(row)
            if policy.d1_scale:
                mult *= _d1_mult(row)
            if policy.conf_scale:
                mult *= _conf_mult(row)
            if policy.session_scale:
                mult *= _session_mult(row)
            risk = base_risk * mult

            day = ts.floor("D")
            week = _week_key(ts)
            sess = _session_of(row)
            side = str(row["side"])
            open_risk = sum(o.risk for o in opens)
            n_open = len(opens)
            n_dir = sum(1 for o in opens if o.side == side)
            dd = (peak - eq) / peak if peak > 0 else 0.0
            r_unit = eq * BASE_RISK  # 1R reference

            if cooldown_until is not None and ts < cooldown_until:
                skip_reason = "cooldown"
            elif policy.max_open < float("inf") and n_open >= policy.max_open:
                skip_reason = "max_open"
            elif policy.max_same_dir < float("inf") and n_dir >= policy.max_same_dir:
                skip_reason = "max_same_dir"
            elif policy.max_heat is not None and (open_risk + risk) > policy.max_heat + 1e-12:
                skip_reason = "max_heat"
            elif policy.float_dd_stop is not None and dd >= policy.float_dd_stop:
                skip_reason = "float_dd"
            elif (
                policy.daily_loss_r is not None
                and day_pnl.get(day, 0.0) <= -policy.daily_loss_r * r_unit
            ):
                skip_reason = "daily_loss"
            elif (
                policy.weekly_loss_r is not None
                and week_pnl.get(week, 0.0) <= -policy.weekly_loss_r * r_unit
            ):
                skip_reason = "weekly_loss"
            elif (
                policy.daily_profit_r is not None
                and day_pnl.get(day, 0.0) >= policy.daily_profit_r * r_unit
            ):
                skip_reason = "daily_profit_lock"
            elif policy.max_daily_trades < float("inf") and day_trades.get(day, 0) >= policy.max_daily_trades:
                skip_reason = "max_daily_trades"
            elif (
                policy.max_london_trades < float("inf")
                and sess == "london"
                and day_london.get(day, 0) >= policy.max_london_trades
            ):
                skip_reason = "max_london"
            elif (
                policy.max_ny_trades < float("inf")
                and sess == "newyork"
                and day_ny.get(day, 0) >= policy.max_ny_trades
            ):
                skip_reason = "max_ny"
            elif (
                policy.stop_london_after_losses is not None
                and sess == "london"
                and day_session_losses.get((day, "london"), 0) >= policy.stop_london_after_losses
            ):
                skip_reason = "london_session_stop"
            elif (
                policy.stop_ny_after_losses is not None
                and sess == "newyork"
                and day_session_losses.get((day, "newyork"), 0) >= policy.stop_ny_after_losses
            ):
                skip_reason = "ny_session_stop"

        if skip_reason is not None or risk <= 0:
            logs.append(
                {
                    **{k: row[k] for k in row.index},
                    "lots": 0.0,
                    "pnl": 0.0,
                    "skipped": True,
                    "skip_reason": skip_reason or "zero_risk",
                    "risk_pct": float(risk),
                    "equity_before": eq,
                    "equity_after": eq,
                    "open_n": len(opens),
                    "open_heat": float(sum(o.risk for o in opens)),
                }
            )
            continue

        lots = _lots_from_equity(
            eq, float(row["atr_price"]), mode="risk", fixed_lots=None, risk_pct=risk, enforce_volume_min=False
        )
        pnl = lots * CONTRACT_SIZE * float(row["entry_price"]) * float(row["net_return"])
        eq_b = eq
        eq = eq + pnl
        peak = max(peak, eq)
        dd = (peak - eq) / peak if peak > 0 else 0.0

        day = ts.floor("D")
        week = _week_key(ts)
        sess = _session_of(row)
        side = str(row["side"])

        day_pnl[day] = day_pnl.get(day, 0.0) + pnl
        week_pnl[week] = week_pnl.get(week, 0.0) + pnl
        day_trades[day] = day_trades.get(day, 0) + 1
        if sess == "london":
            day_london[day] = day_london.get(day, 0) + 1
        if sess == "newyork":
            day_ny[day] = day_ny.get(day, 0) + 1

        r_unit = eq_b * BASE_RISK
        if pnl <= 0:
            consec_losses += 1
            if sess in ("london", "newyork"):
                day_session_losses[(day, sess)] = day_session_losses.get((day, sess), 0) + 1
            if (
                policy.cooldown_after_losses is not None
                and consec_losses >= policy.cooldown_after_losses
            ):
                if policy.cooldown_mode == "next_day":
                    cooldown_until = day + pd.Timedelta(days=1)
                    if cooldown_until.tzinfo is None:
                        cooldown_until = cooldown_until.tz_localize("UTC")
                elif policy.cooldown_mode == "next_session":
                    # ponytail: 6h ≈ next session block
                    cooldown_until = ts + pd.Timedelta(hours=6)
                else:
                    hrs = float(policy.cooldown_hours or 2.0)
                    cooldown_until = ts + pd.Timedelta(hours=hrs)
                consec_losses = 0
        else:
            consec_losses = 0
            if policy.cooldown_after_win_r is not None and r_unit > 0:
                if pnl >= policy.cooldown_after_win_r * r_unit:
                    hrs = float(policy.cooldown_after_win_hours or 2.0)
                    cooldown_until = ts + pd.Timedelta(hours=hrs)

        opens.append(
            _OpenPos(exit_ts=_to_ts(row["exit_ts"]), side=side, risk=float(risk), r_unit=r_unit)
        )

        logs.append(
            {
                **{k: row[k] for k in row.index},
                "lots": lots,
                "pnl": pnl,
                "skipped": False,
                "skip_reason": "",
                "risk_pct": float(risk),
                "equity_before": eq_b,
                "equity_after": eq,
                "open_n": len(opens),
                "open_heat": float(sum(o.risk for o in opens)),
            }
        )
        curve.append({"timestamp": row["timestamp"], "equity": eq, "drawdown": dd, "trade_i": len([x for x in logs if not x["skipped"]]) - 1})
        if eq <= 0:
            break

    log_df = pd.DataFrame(logs)
    curve_df = pd.DataFrame(curve)
    metrics = compute_metrics(log_df, curve_df, starting_equity=starting_equity)
    taken = log_df.loc[~log_df["skipped"].astype(bool)] if not log_df.empty else log_df
    trade_ret = (
        (taken["pnl"] / taken["equity_before"].replace(0, np.nan)).to_numpy(dtype=float)
        if len(taken)
        else np.array([])
    )
    metrics["return_per_dd"] = (
        float(metrics["cagr"] / metrics["max_drawdown"])
        if metrics.get("max_drawdown") and metrics["max_drawdown"] > 0 and metrics.get("cagr") == metrics.get("cagr")
        else float("nan")
    )
    metrics["expectancy_frac"] = float(np.nanmean(trade_ret)) if len(trade_ret) else float("nan")
    metrics["capital_efficiency"] = (
        float(metrics["cagr"] / max(metrics.get("trades", 1), 1))
        if metrics.get("cagr") == metrics.get("cagr")
        else float("nan")
    )
    metrics["policy"] = policy.name
    metrics["n_candidates"] = int(len(trades))
    return log_df, curve_df, metrics
