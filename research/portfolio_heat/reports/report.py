"""Answers + markdown report for Portfolio Heat."""

from __future__ import annotations

from typing import Any

import pandas as pd

from research.portfolio_heat import BASE_RISK, CONF_SKIP, META_GATE


def build_answers(
    *,
    scored: pd.DataFrame,
    baseline: dict[str, Any],
    best_row: pd.Series,
    wf: pd.DataFrame,
    mc_summary: dict[str, Any],
    q_lookup: dict[str, Any],
) -> dict[str, Any]:
    b_cagr = float(baseline.get("cagr") or float("nan"))
    b_dd = float(baseline.get("max_drawdown") or float("nan"))
    c_cagr = float(best_row.get("cagr") or float("nan"))
    c_dd = float(best_row.get("max_drawdown") or float("nan"))
    cagr_imp = (
        (c_cagr - b_cagr) / abs(b_cagr) * 100
        if b_cagr == b_cagr and abs(b_cagr) > 1e-12 and c_cagr == c_cagr
        else float("nan")
    )
    dd_red = (
        (b_dd - c_dd) / abs(b_dd) * 100
        if b_dd == b_dd and abs(b_dd) > 1e-12 and c_dd == c_dd
        else float("nan")
    )
    accepted = scored.loc[scored["accepted"] == True]  # noqa: E712
    heat_exists = bool(
        (("production_ok" in scored.columns) and scored["production_ok"].any())
        or (
            len(accepted) > 0
            and str(best_row.get("policy")) != "baseline_skip40_flat1"
            and int(best_row.get("trades") or 0) >= 100
        )
    )

    return {
        "q1_heat_should_exist": bool(heat_exists),
        "q2_best_policy": str(best_row.get("policy")),
        "q3_best_daily_stop": q_lookup.get("best_daily_stop"),
        "q4_best_weekly_stop": q_lookup.get("best_weekly_stop"),
        "q5_best_exposure_limit": q_lookup.get("best_exposure"),
        "q6_best_concurrent": q_lookup.get("best_concurrent"),
        "q7_best_session_policy": q_lookup.get("best_session"),
        "q8_best_risk_cap": q_lookup.get("best_heat_cap"),
        "q9_cagr_improvement_pct": cagr_imp,
        "q10_dd_reduction_pct": dd_red,
        "q11_architecture": (
            f"Primary(frozen) -> Meta>={META_GATE} -> Confidence skip{int(CONF_SKIP)}_flat{int(BASE_RISK*100)} "
            f"-> Portfolio Heat (`{best_row.get('policy')}`) -> Execution"
        ),
        "baseline_cagr": b_cagr,
        "baseline_dd": b_dd,
        "best_cagr": c_cagr,
        "best_dd": c_dd,
        "n_accepted": int(len(accepted)),
        "mc": mc_summary,
        "wf_years_positive": _wf_ok(wf),
        "research_flags": q_lookup.get("flags", {}),
    }


def _wf_ok(wf: pd.DataFrame) -> dict[str, Any]:
    if wf.empty:
        return {}
    stand = wf.loc[wf["mode"] == "standalone_year"] if "mode" in wf.columns else wf
    if stand.empty:
        return {}
    rets = stand["final_equity"].astype(float)  # > starting means positive year
    return {
        "years": stand["valid_year"].tolist(),
        "final_equities": stand["final_equity"].tolist(),
        "drawdowns": stand["max_drawdown"].tolist(),
        "all_years_not_ruin": bool((stand["final_equity"].astype(float) > 40).all()),
    }


def _best_prefix(scored: pd.DataFrame, prefix: str) -> str:
    sub = scored.loc[scored["policy"].astype(str).str.startswith(prefix)]
    if sub.empty:
        return "off"
    for col in ("production_ok", "accepted"):
        if col in sub.columns:
            use = sub.loc[sub[col] == True]  # noqa: E712
            if not use.empty:
                return str(use.iloc[0]["policy"])
    return "off"


def research_question_lookup(scored: pd.DataFrame) -> dict[str, Any]:
    """Map research Q1-18 to policy evidence."""
    def best(prefix: str) -> str:
        return _best_prefix(scored, prefix)

    def accepted_any(prefix: str) -> bool:
        sub = scored.loc[scored["policy"].astype(str).str.startswith(prefix)]
        if sub.empty:
            return False
        if "production_ok" in sub.columns and (sub["production_ok"] == True).any():  # noqa: E712
            return True
        return bool((sub["accepted"] == True).any())  # noqa: E712

    flags = {
        "limit_heat_helps": accepted_any("heat_cap_") or accepted_any("combo_"),
        "max_concurrent": accepted_any("max_open_"),
        "same_dir_limit": accepted_any("max_same_dir_"),
        "daily_loss_stop": accepted_any("daily_loss_"),
        "weekly_loss_stop": accepted_any("weekly_loss_"),
        "daily_profit_lock": accepted_any("daily_profit_"),
        "cooldown_losses": accepted_any("cd_L"),
        "cooldown_winner": accepted_any("cd_win"),
        "london_stop_3L": accepted_any("london_stop"),
        "ny_stop_3L": accepted_any("ny_stop"),
        "daily_risk_cap": accepted_any("heat_cap_") or accepted_any("daily_loss_"),
        "float_dd_stop": accepted_any("float_dd_"),
        "max_trades_day": accepted_any("max_day_trades_"),
        "max_trades_session": accepted_any("max_london_") or accepted_any("max_ny_"),
        "max_risk_direction": accepted_any("max_same_dir_"),
        "vol_scale": accepted_any("vol_scale") or accepted_any("combo_d1_vol"),
        "d1_scale": accepted_any("d1_scale") or accepted_any("combo_d1_vol") or accepted_any("combo_full"),
        "conf_scale": accepted_any("conf_scale"),
    }
    return {
        "best_daily_stop": best("daily_loss_"),
        "best_weekly_stop": best("weekly_loss_"),
        "best_exposure": best("max_same_dir_"),
        "best_concurrent": best("max_open_"),
        "best_session": best("max_london_") if accepted_any("max_london_") else (
            best("london_stop") if accepted_any("london_stop") else best("max_ny_")
        ),
        "best_heat_cap": best("heat_cap_"),
        "flags": flags,
    }


def build_report(
    *,
    symbol: str,
    timeframe: str,
    answers: dict[str, Any],
    scored: pd.DataFrame,
    wf: pd.DataFrame,
    stats_df: pd.DataFrame,
    mc_summary: dict[str, Any],
) -> str:
    lines = [
        f"# Portfolio Heat Management Report - {symbol} {timeframe}",
        "",
        "Sprint 24 - execution-layer portfolio management. Models frozen.",
        f"Baseline: Meta>={META_GATE}, Confidence skip{int(CONF_SKIP)}_flat1, risk {int(BASE_RISK*100)}%.",
        "",
        "## Deliverable Answers",
        "",
        f"1. **Should Portfolio Heat Management exist?** `{answers.get('q1_heat_should_exist')}`",
        f"2. **Best Portfolio Heat Policy** `{answers.get('q2_best_policy')}`",
        f"3. **Best Daily Stop** `{answers.get('q3_best_daily_stop')}`",
        f"4. **Best Weekly Stop** `{answers.get('q4_best_weekly_stop')}`",
        f"5. **Best Exposure Limit** `{answers.get('q5_best_exposure_limit')}`",
        f"6. **Best Concurrent Positions** `{answers.get('q6_best_concurrent')}`",
        f"7. **Best Session Policy** `{answers.get('q7_best_session_policy')}`",
        f"8. **Best Risk Cap** `{answers.get('q8_best_risk_cap')}`",
        f"9. **Expected CAGR improvement** `{answers.get('q9_cagr_improvement_pct')}`% "
        f"(base={answers.get('baseline_cagr')} -> best={answers.get('best_cagr')})",
        f"10. **Expected DD reduction** `{answers.get('q10_dd_reduction_pct')}`% "
        f"(base={answers.get('baseline_dd')} -> best={answers.get('best_dd')})",
        f"11. **Production-ready Portfolio Engine** {answers.get('q11_architecture')}",
        "",
        f"Accepted policies: `{answers.get('n_accepted')}` | WF: `{answers.get('wf_years_positive')}`",
        f"Research flags: `{answers.get('research_flags')}`",
        "",
        "## Monte Carlo (best policy)",
        "",
        f"`{mc_summary}`",
        "",
        "## Policy leaderboard (accepted first)",
        "",
    ]
    show = scored.head(25)
    lines.append(show.to_string(index=False) if not show.empty else "_(empty)_")
    lines.extend(["", "## Walk-forward", ""])
    lines.append(wf.to_string(index=False) if not wf.empty else "_(empty)_")
    lines.extend(["", "## Statistical validation (top accepted)", ""])
    lines.append(stats_df.to_string(index=False) if not stats_df.empty else "_(empty)_")
    lines.extend(
        [
            "",
            "## Charts",
            "",
            "- `charts/policy_scatter.png`",
            "- `charts/best_equity.png`",
            "- `charts/yearly_wf.png`",
            "- `charts/monte_carlo.png`",
            "- `charts/skip_reasons.png`",
            "",
        ]
    )
    return "\n".join(lines)
