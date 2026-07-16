"""Markdown reports + final Q&A."""

from __future__ import annotations

from typing import Any

import pandas as pd

from research.portfolio_backtest import META_THRESHOLD, STARTING_EQUITY


def _fmt(x: Any, d: int = 4) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/a"
    if v != v:
        return "n/a"
    if abs(v) == float("inf"):
        return "inf"
    return f"{v:.{d}f}"


def build_answers(
    *,
    metrics_table: pd.DataFrame,
    yearly: pd.DataFrame,
    mc_rows: pd.DataFrame,
    preferred_scenario: str,
) -> dict[str, Any]:
    """Heuristic Q&A from backtest (no threshold search)."""
    meta = metrics_table.loc[
        (metrics_table["system"] == "primary_plus_meta")
        & (metrics_table["scenario"] == preferred_scenario)
    ]
    prim = metrics_table.loc[
        (metrics_table["system"] == "primary") & (metrics_table["scenario"] == preferred_scenario)
    ]
    risk_scen = metrics_table.loc[
        (metrics_table["system"] == "primary_plus_meta")
        & metrics_table["scenario"].isin(
            ["B_risk_0_5pct_frac", "C_risk_1pct_frac", "D_risk_2pct_frac"]
        )
    ]
    # Optimal risk: among frac scenarios, prefer Calmar with DD soft cap 25%
    opt = preferred_scenario
    if not risk_scen.empty:
        scored = risk_scen.loc[risk_scen["trades"] > 0].copy()
        if not scored.empty:
            scored["score"] = scored["calmar"].fillna(-1e9) - 2.0 * scored["max_drawdown"].fillna(1.0)
            # Prefer DD < 25% when available
            under = scored.loc[scored["max_drawdown"] <= 0.25]
            pick = under if not under.empty else scored
            opt = str(pick.sort_values("score", ascending=False).iloc[0]["scenario"])

    m = meta.iloc[0].to_dict() if not meta.empty else {}
    p = prim.iloc[0].to_dict() if not prim.empty else {}
    mc = mc_rows.loc[
        (mc_rows["system"] == "primary_plus_meta") & (mc_rows["scenario"] == preferred_scenario)
    ]
    mc_d = mc.iloc[0].to_dict() if not mc.empty else {}

    final_eq = float(m.get("final_equity", STARTING_EQUITY))
    cagr = float(m.get("cagr", float("nan")))
    expected = {}
    for y in (1, 2, 3, 4):
        if cagr == cagr:
            expected[y] = STARTING_EQUITY * ((1 + cagr) ** y)
        else:
            expected[y] = float("nan")

    mdd = float(m.get("max_drawdown", float("nan")))
    lose_prob = float(mc_d.get("prob_lose_money", float("nan")))
    ruin25 = float(mc_d.get("prob_ruin_25pct", float("nan")))

    # Broker-realistic: VOLUME_MIN blocks risk sizing at $80
    realistic_zero = metrics_table.loc[
        (metrics_table["system"] == "primary_plus_meta")
        & (metrics_table["scenario"] == "C_risk_1pct")
        & (metrics_table["trades"] == 0)
    ]
    can_size_at_80 = realistic_zero.empty

    edge_ok = bool(
        final_eq > STARTING_EQUITY
        and (mdd == mdd and mdd < 0.35)
        and int(m.get("trades", 0)) >= 30
        and float(m.get("profit_factor", 0) or 0) > 1.0
    )
    tradable = bool(edge_ok and can_size_at_80)
    robust = bool(
        edge_ok
        and (lose_prob != lose_prob or lose_prob < 0.4)
        and (ruin25 != ruin25 or ruin25 < 0.15)
    )

    weakness = []
    if not can_size_at_80:
        weakness.append(
            "$80 cannot meet broker VOLUME_MIN=0.01 under risk sizing on XAU (need ~$5k+)"
        )
    if mdd == mdd and mdd > 0.35:
        weakness.append("historical max drawdown too large for small capital")
    if lose_prob == lose_prob and lose_prob > 0.35:
        weakness.append("Monte Carlo shows material chance of ending below start")
    if float(m.get("trades", 0) or 0) < 50:
        weakness.append("sparse trade count after meta filter")
    # MC final equity invariant under pure fractional sizing (product of multipliers)
    weakness.append(
        "fractional risk MC finals are order-invariant; path DD varies — do not over-read final-equity MC percentiles"
    )
    if not weakness:
        weakness.append("monitor live slippage and regime drift")

    recommended_account = 5_000.0

    return {
        "q1_tradable_real_money": tradable,
        "q1_note": (
            "edge visible on fractional risk analysis; live $80 blocked by VOLUME_MIN"
            if edge_ok and not can_size_at_80
            else ""
        ),
        "q2_optimal_risk_scenario": opt,
        "q3_fixed_vs_risk": "Risk Based (never Fixed 1 lot on small capital)",
        "q4_expected_balance": expected,
        "q4_cagr_used": cagr,
        "q5_worst_historical_dd": mdd,
        "q6_longest_losing_streak": int(m.get("longest_losing_streak", 0) or 0),
        "q7_recommended_account": recommended_account,
        "q8_statistically_robust": robust,
        "q9_biggest_weakness": weakness[0],
        "weaknesses": weakness,
        "preferred_scenario": preferred_scenario,
        "meta_metrics": m,
        "primary_metrics": p,
        "mc": mc_d,
        "meta_threshold": META_THRESHOLD,
    }


def build_backtest_report(
    *,
    symbol: str,
    timeframe: str,
    metrics_table: pd.DataFrame,
    answers: dict[str, Any],
    yearly_by_key: dict[str, pd.DataFrame],
) -> str:
    lines = [
        f"# Portfolio Backtest Report - {symbol} {timeframe}",
        "",
        "Sprint 21 - realistic portfolio execution (no ML / no retrain / no threshold search).",
        f"Meta gate: `meta_proba >= {META_THRESHOLD}` (fixed).",
        f"Starting equity: `${STARTING_EQUITY:.0f}`. Contract size 100, VOLUME_MIN 0.01.",
        "Sizing: sequential compounding; trade `net_return` includes training transaction-cost haircut.",
        "",
        "## Answers",
        "",
        f"1. **Tradable with real money?** `{answers.get('q1_tradable_real_money')}`"
        + (f" — {answers.get('q1_note')}" if answers.get("q1_note") else ""),
        f"2. **Optimal risk % (among tested)?** `{answers.get('q2_optimal_risk_scenario')}`",
        f"3. **Fixed Lot vs Risk Based?** `{answers.get('q3_fixed_vs_risk')}`",
        f"4. **Expected balance from $80** (using hist CAGR=`{_fmt(answers.get('q4_cagr_used'))}`):",
    ]
    for y, v in (answers.get("q4_expected_balance") or {}).items():
        lines.append(f"   - {y}y: `${_fmt(v, 2)}`")
    lines.extend(
        [
            f"5. **Worst historical drawdown:** `{_fmt(answers.get('q5_worst_historical_dd'))}`",
            f"6. **Longest losing streak:** `{answers.get('q6_longest_losing_streak')}`",
            f"7. **Recommended live account size:** `${_fmt(answers.get('q7_recommended_account'), 0)}`",
            f"8. **Statistically robust enough?** `{answers.get('q8_statistically_robust')}`",
            f"9. **Biggest remaining weakness:** {answers.get('q9_biggest_weakness')}",
            "",
            "## Portfolio metrics (all scenarios)",
            "",
        ]
    )
    if metrics_table.empty:
        lines.append("_(empty)_")
    else:
        lines.append(metrics_table.to_string(index=False))

    lines.extend(["", "## Yearly (preferred Primary+Meta)", ""])
    key = f"primary_plus_meta__{answers.get('preferred_scenario')}"
    ydf = yearly_by_key.get(key, pd.DataFrame())
    if ydf.empty:
        lines.append("_(empty)_")
    else:
        lines.append(ydf.to_string(index=False))

    lines.extend(
        [
            "",
            "## Charts",
            "",
            "See `charts/` for equity, drawdown, heatmaps, rolling stats, distributions, Monte Carlo.",
            "",
        ]
    )
    return "\n".join(lines)


def build_risk_report(*, answers: dict[str, Any], mc_rows: pd.DataFrame) -> str:
    lines = [
        "# Risk Report - Portfolio Backtest",
        "",
        f"Meta threshold fixed at `{META_THRESHOLD}`.",
        f"Preferred scenario for risk narrative: `{answers.get('preferred_scenario')}`",
        "",
        "## Risk of ruin (Monte Carlo)",
        "",
    ]
    if mc_rows.empty:
        lines.append("_(empty)_")
    else:
        show = mc_rows[
            [
                c
                for c in (
                    "system",
                    "scenario",
                    "prob_ruin_50pct",
                    "prob_ruin_25pct",
                    "prob_ruin_10pct",
                    "prob_lose_money",
                    "prob_double",
                    "prob_below_50",
                    "prob_above_500",
                    "median_final_equity",
                    "median_max_dd",
                )
                if c in mc_rows.columns
            ]
        ]
        lines.append(show.to_string(index=False))

    lines.extend(
        [
            "",
            "## Narrative",
            "",
            f"- Tradable: `{answers.get('q1_tradable_real_money')}`",
            f"- Robust: `{answers.get('q8_statistically_robust')}`",
            f"- Weaknesses: `{answers.get('weaknesses')}`",
            f"- Recommended account: `${_fmt(answers.get('q7_recommended_account'), 0)}`",
            "",
            "Fixed 1 lot on an $80 gold account is not a realistic live setup "
            "(1 lot = 100 oz notionally).",
            "",
        ]
    )
    return "\n".join(lines)
