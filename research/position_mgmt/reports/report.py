"""Reports for position management sprint."""

from __future__ import annotations

from typing import Any

import pandas as pd


def _best_family(table: pd.DataFrame, family: str) -> dict[str, Any]:
    sub = table.loc[table["family"] == family]
    if sub.empty:
        return {"exist": False, "best": "off", "accepted_sig": False}
    if "accepted_sig" in sub.columns:
        sig = sub.loc[sub["accepted_sig"] == True]  # noqa: E712
    else:
        sig = pd.DataFrame()
    if sig.empty:
        return {
            "exist": False,
            "best": "off",
            "accepted_sig": False,
            "note": "no statistically significant improve",
        }
    best = sig.iloc[0]
    return {
        "exist": True,
        "best": str(best["policy"]),
        "accepted_sig": True,
        "cagr": float(best.get("cagr", float("nan"))),
        "max_drawdown": float(best.get("max_drawdown", float("nan"))),
        "calmar": float(best.get("calmar", float("nan"))),
    }


def build_answers(
    *,
    table: pd.DataFrame,
    baseline: dict[str, Any],
    best_row: pd.Series,
    mc_base: dict[str, Any],
    mc_best: dict[str, Any],
    ranks: list[dict[str, Any]],
) -> dict[str, Any]:
    families = ["be", "partial", "trail", "time", "vol", "pyramid", "scale_in"]
    fam = {f: _best_family(table, f) for f in families}

    b_cagr = float(baseline.get("cagr") or float("nan"))
    b_dd = float(baseline.get("max_drawdown") or float("nan"))
    has_best = str(best_row.get("policy")) != "baseline" and bool(best_row.get("accepted_sig"))
    c_cagr = float(best_row.get("cagr") or b_cagr) if has_best else b_cagr
    c_dd = float(best_row.get("max_drawdown") or b_dd) if has_best else b_dd

    # Economic runner-up (accepted path rule but not significant) for transparency
    econ = table.loc[(table["policy"] != "baseline") & (table["accepted"] == True)]  # noqa: E712
    econ_best = econ.iloc[0].to_dict() if not econ.empty else None

    cagr_imp = (
        (c_cagr - b_cagr) / abs(b_cagr) * 100
        if has_best and b_cagr == b_cagr and abs(b_cagr) > 1e-12
        else 0.0
    )
    dd_imp = (
        (b_dd - c_dd) / abs(b_dd) * 100
        if has_best and b_dd == b_dd and abs(b_dd) > 1e-12
        else 0.0
    )
    mc_imp = 0.0
    if has_best and mc_base and mc_best:
        mb = float(mc_base.get("median_equity") or float("nan"))
        mt = float(mc_best.get("median_equity") or float("nan"))
        mc_imp = ((mt - mb) / abs(mb) * 100) if mb == mb and abs(mb) > 1e-12 and mt == mt else 0.0

    pipeline = (
        "Primary(frozen) -> Meta>=0.45 -> Confidence skip40_flat1 -> Heat daily_loss_-1R"
        + (
            f" -> PositionMgmt(`{best_row.get('policy')}`)"
            if has_best
            else " -> PositionMgmt(none — keep original TP/SL 2.0/1.5 ATR)"
        )
        + " -> Execution"
    )

    return {
        "q1_be_exist": fam["be"]["exist"],
        "q2_best_be": fam["be"]["best"],
        "q3_partial_exist": fam["partial"]["exist"],
        "q4_best_partial": fam["partial"]["best"],
        "q5_trail_exist": fam["trail"]["exist"],
        "q6_best_trail": fam["trail"]["best"],
        "q7_time_exist": fam["time"]["exist"],
        "q8_best_time": fam["time"]["best"],
        "q9_pyramid_exist": fam["pyramid"]["exist"],
        "q10_best_pyramid": fam["pyramid"]["best"],
        "q11_scale_exist": fam["scale_in"]["exist"],
        "q12_best_scale": fam["scale_in"]["best"],
        "q13_cagr_improvement_pct": cagr_imp,
        "q14_dd_improvement_pct": dd_imp,
        "q15_mc_improvement_pct": mc_imp,
        "q16_pipeline": pipeline,
        "best_overall": str(best_row.get("policy")) if has_best else "baseline",
        "economic_runner_up": econ_best,
        "baseline_cagr": b_cagr,
        "baseline_dd": b_dd,
        "best_cagr": c_cagr,
        "best_dd": c_dd,
        "family_results": fam,
        "ranking": ranks,
        "mc_baseline": mc_base,
        "mc_best": mc_best,
        "vol_note": _best_family(table, "vol"),
        "significance_rule": "boot_ci_low>0 OR (MW p<0.05 AND perm p<0.05); insignificant rejected",
    }


def build_report(
    *,
    symbol: str,
    timeframe: str,
    answers: dict[str, Any],
    table: pd.DataFrame,
    wf: pd.DataFrame,
) -> str:
    lines = [
        f"# Position Management Report - {symbol} {timeframe}",
        "",
        "Sprint 25 — execution only. Same production entries (Heat `daily_loss_-1R` accepted set).",
        "Primary / Meta / Confidence / Heat frozen. Management alters exits after open via H1 path sim.",
        "",
        "## Answers",
        "",
        f"1. **Should Break Even exist?** `{answers.get('q1_be_exist')}`",
        f"2. **Best Break Even** `{answers.get('q2_best_be')}`",
        f"3. **Should Partial TP exist?** `{answers.get('q3_partial_exist')}`",
        f"4. **Best Partial TP** `{answers.get('q4_best_partial')}`",
        f"5. **Should ATR Trail exist?** `{answers.get('q5_trail_exist')}`",
        f"6. **Best ATR Trail** `{answers.get('q6_best_trail')}`",
        f"7. **Should Time Exit exist?** `{answers.get('q7_time_exist')}`",
        f"8. **Best Time Exit** `{answers.get('q8_best_time')}`",
        f"9. **Should Pyramid exist?** `{answers.get('q9_pyramid_exist')}`",
        f"10. **Best Pyramid Rule** `{answers.get('q10_best_pyramid')}`",
        f"11. **Should Scale In exist?** `{answers.get('q11_scale_exist')}`",
        f"12. **Best Scale In Rule** `{answers.get('q12_best_scale')}`",
        f"13. **Expected CAGR improvement** `{answers.get('q13_cagr_improvement_pct')}`%",
        f"14. **Expected DD improvement** `{answers.get('q14_dd_improvement_pct')}`%",
        f"15. **Expected Monte Carlo improvement** `{answers.get('q15_mc_improvement_pct')}`%",
        f"16. **Production pipeline** {answers.get('q16_pipeline')}",
        "",
        f"Overall best (accepted + significant): `{answers.get('best_overall')}`",
        f"Economic runner-up (not significant): `{answers.get('economic_runner_up')}`",
        f"Significance rule: `{answers.get('significance_rule')}`",
        f"Ranking: `{answers.get('ranking')}`",
        f"Vol-exit: `{answers.get('vol_note')}`",
        "",
        "## Leaderboard",
        "",
    ]
    cols = [
        c
        for c in (
            "policy",
            "family",
            "cagr",
            "max_drawdown",
            "calmar",
            "profit_factor",
            "sharpe",
            "recovery_factor",
            "trades",
            "accepted",
            "accepted_sig",
            "stat_mannwhitney_p",
            "stat_perm_p",
            "stat_boot_ci_low",
            "stat_boot_ci_high",
        )
        if c in table.columns
    ]
    lines.append(table[cols].to_string(index=False) if not table.empty else "_(empty)_")
    lines.extend(["", "## Walk-forward (best)", ""])
    lines.append(wf.to_string(index=False) if not wf.empty else "_(empty)_")
    lines.extend(
        [
            "",
            "## Charts",
            "",
            "- `charts/family_scatter.png`",
            "- `charts/best_equity.png`",
            "- `charts/ranking.png`",
            "- `charts/monte_carlo.png`",
            "- `charts/sim_vs_base_returns.png`",
            "",
        ]
    )
    return "\n".join(lines)
