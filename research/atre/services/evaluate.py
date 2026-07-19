"""Evaluate ATRE policies with significance gates."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from research.atre import BASE_RISK, STARTING_EQUITY
from research.atre.services.policies import AtrePolicy, build_atre_policies, simulate_policy_panel
from research.portfolio_backtest.services.engine import run_portfolio
from research.portfolio_backtest.services.metrics import compute_metrics
from research.portfolio_heat.services.stats_tests import compare_policies
from research.position_mgmt.services.monte_carlo import monte_carlo_fixed


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
    if "skipped" not in log.columns:
        log["skipped"] = False
    m = compute_metrics(log, curve, starting_equity=starting_equity)
    m["return_per_dd"] = (
        float(m["cagr"] / m["max_drawdown"])
        if m.get("max_drawdown") and m["max_drawdown"] > 0 and m.get("cagr") == m.get("cagr")
        else float("nan")
    )
    m["n_intervened"] = int(t["atre_intervened"].sum()) if "atre_intervened" in t.columns else 0
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


def significant(st: dict[str, Any]) -> bool:
    return bool(
        (st.get("boot_ci_low") == st.get("boot_ci_low") and float(st["boot_ci_low"]) > 0)
        or (
            st.get("mannwhitney_p") == st.get("mannwhitney_p")
            and st.get("perm_p") == st.get("perm_p")
            and float(st["mannwhitney_p"]) < 0.05
            and float(st["perm_p"]) < 0.05
        )
    )


def evaluate_atre(
    trades: pd.DataFrame,
    lib: list[dict],
    scores: dict[int, float],
    *,
    starting_equity: float = STARTING_EQUITY,
    policies: list[AtrePolicy] | None = None,
    n_boot: int = 1000,
    n_perm: int = 1000,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    policies = policies or build_atre_policies()
    base_panel = simulate_policy_panel(trades, lib, AtrePolicy(name="baseline", family="baseline"), scores)
    base_log, base_curve, base_m = run_fixed_portfolio(base_panel, starting_equity=starting_equity)
    base_m["policy"] = "baseline"
    base_m["family"] = "baseline"

    rows = [{**base_m, **acceptance(base_m, base_m), "accepted_sig": False, "stat_significant": False}]
    arts: dict[str, Any] = {
        "baseline_metrics": base_m,
        "logs": {"baseline": base_log},
        "curves": {"baseline": base_curve},
        "panels": {"baseline": base_panel},
    }

    for pol in policies:
        if pol.name == "baseline":
            continue
        panel = simulate_policy_panel(trades, lib, pol, scores)
        log, curve, m = run_fixed_portfolio(panel, starting_equity=starting_equity)
        m["policy"] = pol.name
        m["family"] = pol.family
        m["expectancy_frac"] = float(panel["net_return"].mean())
        acc = acceptance(base_m, m)
        st = compare_policies(base_log, log, n_boot=n_boot, n_perm=n_perm)
        sig = significant(st)
        row = {**m, **acc, **{f"stat_{k}": v for k, v in st.items()}, "stat_significant": sig}
        row["accepted_sig"] = bool(row["accepted"] and sig)
        rows.append(row)
        arts["logs"][pol.name] = log
        arts["curves"][pol.name] = curve
        arts["panels"][pol.name] = panel

    table = pd.DataFrame(rows)
    table["accepted_sig"] = table["accepted_sig"].fillna(False).astype(bool)
    table["_r"] = np.where(table["accepted_sig"], 0, np.where(table["accepted"].fillna(False), 1, 2))
    table = table.sort_values(["_r", "calmar", "cagr"], ascending=[True, False, False]).drop(columns=["_r"])
    return table, arts
