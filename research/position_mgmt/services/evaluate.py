"""Evaluate position-mgmt policies on fixed trade set."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from research.portfolio_backtest.services.engine import run_portfolio
from research.portfolio_backtest.services.metrics import compute_metrics, yearly_report
from research.portfolio_heat.services.stats_tests import compare_policies
from research.position_mgmt import BASE_RISK, COST, STARTING_EQUITY
from research.position_mgmt.services.paths import entry_indices, load_h1, load_production_trades, prepare_market
from research.position_mgmt.services.simulator import ManageConfig, build_experiments, simulate_panel


def daily_stop_flags(trades: pd.DataFrame, *, starting_equity: float = STARTING_EQUITY) -> np.ndarray:
    """Approx: day already ≤ -1R (of current equity) before this entry using baseline returns."""
    eq = float(starting_equity)
    day_pnl: dict[Any, float] = {}
    flags = np.zeros(len(trades), dtype=bool)
    for i in range(len(trades)):
        row = trades.iloc[i]
        ts = pd.Timestamp(row["timestamp"])
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        day = ts.floor("D")
        r_unit = eq * BASE_RISK
        flags[i] = day_pnl.get(day, 0.0) <= -1.0 * r_unit
        # update equity / day with baseline outcome (for gate state only)
        atr = float(row["atr_entry"]) if pd.notna(row.get("atr_entry")) else float(row.get("atr_price", 1.0))
        from research.portfolio_backtest.services.engine import _lots_from_equity
        from settings.strategy import CONTRACT_SIZE

        lots = _lots_from_equity(eq, atr, mode="risk", fixed_lots=None, risk_pct=BASE_RISK, enforce_volume_min=False)
        pnl = lots * CONTRACT_SIZE * float(row["entry_price"]) * float(row["net_return"])
        day_pnl[day] = day_pnl.get(day, 0.0) + pnl
        eq = eq + pnl
    return flags


def run_fixed_portfolio(
    trades: pd.DataFrame,
    *,
    starting_equity: float = STARTING_EQUITY,
    risk_pct: float = BASE_RISK,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    t = trades.copy()
    if "atr_price" not in t.columns:
        t["atr_price"] = t["atr_entry"].astype(float)
    if "valid_year" not in t.columns:
        t["valid_year"] = pd.to_datetime(t["timestamp"], utc=True).dt.year
    log, curve = run_portfolio(
        t,
        starting_equity=starting_equity,
        mode="risk",
        fixed_lots=None,
        risk_pct=risk_pct,
        enforce_volume_min=False,
    )
    # mark none skipped by heat
    if "skipped" not in log.columns:
        log["skipped"] = False
    m = compute_metrics(log, curve, starting_equity=starting_equity)
    m["return_per_dd"] = (
        float(m["cagr"] / m["max_drawdown"])
        if m.get("max_drawdown") and m["max_drawdown"] > 0 and m.get("cagr") == m.get("cagr")
        else float("nan")
    )
    return log, curve, m


def acceptance(base: dict[str, Any], cand: dict[str, Any], *, severe: float = 0.20) -> dict[str, Any]:
    b_c, c_c = float(base.get("cagr") or np.nan), float(cand.get("cagr") or np.nan)
    b_d, c_d = float(base.get("max_drawdown") or np.nan), float(cand.get("max_drawdown") or np.nan)
    ret_up = c_c == c_c and b_c == b_c and c_c > b_c
    dd_down = c_d == c_d and b_d == b_d and c_d < b_d
    cagr_rel = (c_c - b_c) / abs(b_c) if b_c == b_c and abs(b_c) > 1e-12 else np.nan
    dd_rel = (c_d - b_d) / abs(b_d) if b_d == b_d and abs(b_d) > 1e-12 else np.nan
    severe_ret = cagr_rel == cagr_rel and cagr_rel < -severe
    severe_dd = dd_rel == dd_rel and dd_rel > severe
    return {
        "accepted": bool((ret_up or dd_down) and not (severe_ret or severe_dd)),
        "ret_up": bool(ret_up),
        "dd_down": bool(dd_down),
        "cagr_rel": float(cagr_rel) if cagr_rel == cagr_rel else float("nan"),
        "dd_rel": float(dd_rel) if dd_rel == dd_rel else float("nan"),
    }


def evaluate_all(
    heat_log: pd.DataFrame,
    h1: pd.DataFrame | None = None,
    *,
    starting_equity: float = STARTING_EQUITY,
    experiments: list[ManageConfig] | None = None,
    n_boot: int = 1000,
    n_perm: int = 1000,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, pd.DataFrame]]:
    trades = load_production_trades(heat_log)
    h1 = h1 if h1 is not None else load_h1()
    mkt = prepare_market(h1)
    eis = entry_indices(trades, mkt["ts"])
    flags = daily_stop_flags(trades, starting_equity=starting_equity)
    experiments = experiments or build_experiments()

    # baseline portfolio on original net_return (fixed trade set)
    base_log, base_curve, base_m = run_fixed_portfolio(trades, starting_equity=starting_equity)
    base_m["policy"] = "baseline"
    base_m["family"] = "baseline"

    rows = []
    panels: dict[str, pd.DataFrame] = {"baseline": trades.copy()}
    logs: dict[str, pd.DataFrame] = {"baseline": base_log}
    curves: dict[str, pd.DataFrame] = {"baseline": base_curve}

    rows.append({**base_m, **acceptance(base_m, base_m), "accepted_sig": False, "stat_significant": False})

    for cfg in experiments:
        if cfg.name == "baseline":
            continue
        sim = simulate_panel(trades, h1, cfg, daily_stop_flags=flags, mkt=mkt, eis=eis)
        risk = BASE_RISK
        if cfg.family == "scale_in":
            risk = BASE_RISK * float(np.clip(sim["size_avg"].mean(), 0.5, 1.0))
        log, curve, m = run_fixed_portfolio(sim, starting_equity=starting_equity, risk_pct=risk)
        m["policy"] = cfg.name
        m["family"] = cfg.family
        m["expectancy_frac"] = float(sim["net_return"].mean())
        acc = acceptance(base_m, m)
        st = compare_policies(base_log, log, n_boot=n_boot, n_perm=n_perm)
        sig = bool(
            (st.get("boot_ci_low") == st.get("boot_ci_low") and float(st["boot_ci_low"]) > 0)
            or (
                st.get("mannwhitney_p") == st.get("mannwhitney_p")
                and st.get("perm_p") == st.get("perm_p")
                and float(st["mannwhitney_p"]) < 0.05
                and float(st["perm_p"]) < 0.05
            )
        )
        row = {**m, **acc, **{f"stat_{k}": v for k, v in st.items()}, "stat_significant": sig}
        row["accepted_sig"] = bool(row["accepted"] and sig)
        rows.append(row)
        panels[cfg.name] = sim
        logs[cfg.name] = log
        curves[cfg.name] = curve

    table = pd.DataFrame(rows)
    if "accepted_sig" not in table.columns:
        table["accepted_sig"] = False
    table["accepted_sig"] = table["accepted_sig"].fillna(False).astype(bool)
    table["_rank"] = np.where(table["accepted_sig"], 0, np.where(table["accepted"].fillna(False), 1, 2))
    table = table.sort_values(["_rank", "calmar", "cagr"], ascending=[True, False, False]).drop(columns=["_rank"])
    arts = {"panels": panels, "logs": logs, "curves": curves, "baseline_metrics": base_m}
    return table, trades, arts, logs
