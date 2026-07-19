"""ATRE report answers."""

from __future__ import annotations

from typing import Any

import pandas as pd


def _best_fam(table: pd.DataFrame, family: str) -> dict[str, Any]:
    sub = table.loc[table["family"] == family]
    if sub.empty:
        return {"exist": False, "best": "off"}
    sig = sub.loc[sub["accepted_sig"] == True] if "accepted_sig" in sub.columns else pd.DataFrame()  # noqa: E712
    if sig.empty:
        return {"exist": False, "best": "off", "note": "not significant"}
    b = sig.iloc[0]
    return {"exist": True, "best": str(b["policy"]), "cagr": float(b["cagr"]), "dd": float(b["max_drawdown"])}


def build_answers(
    *,
    table: pd.DataFrame,
    baseline: dict[str, Any],
    best_row: pd.Series,
    diagnostics: dict[str, Any],
    importance: pd.DataFrame,
    mc_base: dict[str, Any],
    mc_best: dict[str, Any],
    wf_ok: bool,
) -> dict[str, Any]:
    families = {
        "early_exit": _best_fam(table, "early_exit"),
        "reduce": _best_fam(table, "reduce"),
        "compress": _best_fam(table, "compress"),
        "score_exit": _best_fam(table, "score_exit"),
        "score_reduce": _best_fam(table, "score_reduce"),
        "mom": _best_fam(table, "mom"),
        "struct": _best_fam(table, "struct"),
    }
    has = str(best_row.get("policy")) != "baseline" and bool(best_row.get("accepted_sig")) and wf_ok
    # also require MC better if both present
    if has and mc_base and mc_best:
        mb = float(mc_base.get("median_equity") or 0)
        mt = float(mc_best.get("median_equity") or 0)
        if mt < mb:
            has = False

    b_cagr = float(baseline.get("cagr") or float("nan"))
    b_dd = float(baseline.get("max_drawdown") or float("nan"))
    c_cagr = float(best_row.get("cagr") or b_cagr) if has else b_cagr
    c_dd = float(best_row.get("max_drawdown") or b_dd) if has else b_dd

    any_sig = bool(table["accepted_sig"].any()) if "accepted_sig" in table.columns else False
    engine_exist = bool(has and any_sig)

    top_feat = str(importance.iloc[0]["feature"]) if importance is not None and not importance.empty else "n/a"
    side = diagnostics.get("side_recovery")
    side_diff = False
    if isinstance(side, pd.DataFrame) and len(side) >= 2:
        pts = side["p_tp"].dropna()
        side_diff = bool(len(pts) >= 2 and abs(float(pts.iloc[0]) - float(pts.iloc[1])) > 0.1)

    best_signal = "none"
    for k in ("score_exit", "early_exit", "mom", "struct", "reduce", "compress"):
        if families[k]["exist"]:
            best_signal = families[k]["best"]
            break

    score_needed = families["score_exit"]["exist"] or families["score_reduce"]["exist"]

    # best threshold from recovery curve: lowest mae with p_tp < 0.35
    thr = "n/a"
    rc = diagnostics.get("recovery_prob")
    if isinstance(rc, pd.DataFrame) and not rc.empty:
        low = rc.loc[rc["p_tp"] < 0.35]
        if not low.empty:
            thr = float(low.iloc[0]["cond_mae_atr"])

    pipeline = (
        "Primary -> Meta>=0.45 -> Confidence skip40_flat1 -> Heat daily_loss_-1R"
        + (
            f" -> ATRE(`{best_row.get('policy')}`)"
            if engine_exist
            else " -> ATRE(none)"
        )
        + " -> Original TP/SL -> Execution"
    )

    verdict = (
        "Recovery Engine should exist."
        if engine_exist
        else "Recovery Engine should NOT exist."
    )

    return {
        "verdict": verdict,
        "q1_engine_exist": engine_exist,
        "q2_best_signal": best_signal,
        "q3_best_early_exit": families["early_exit"]["best"],
        "q4_best_reduce": families["reduce"]["best"] if families["reduce"]["exist"] else (
            families["score_reduce"]["best"] if families["score_reduce"]["exist"] else "off"
        ),
        "q5_best_dynamic_stop": families["compress"]["best"],
        "q6_score_needed": score_needed,
        "q7_top_feature": top_feat,
        "q8_best_threshold": thr,
        "q9_long_short_different": side_diff,
        "q10_cagr_improvement_pct": (
            (c_cagr - b_cagr) / abs(b_cagr) * 100 if engine_exist and abs(b_cagr) > 1e-12 else 0.0
        ),
        "q11_dd_improvement_pct": (
            (b_dd - c_dd) / abs(b_dd) * 100 if engine_exist and abs(b_dd) > 1e-12 else 0.0
        ),
        "q12_pipeline": pipeline,
        "baseline_cagr": b_cagr,
        "baseline_dd": b_dd,
        "best_policy": str(best_row.get("policy")) if engine_exist else "baseline",
        "families": families,
        "diagnostics_summary": {
            "momentum": diagnostics.get("momentum"),
            "structure": diagnostics.get("structure"),
        },
        "mc_baseline": mc_base,
        "mc_best": mc_best,
        "wf_ok": wf_ok,
    }


def build_report(*, symbol: str, timeframe: str, answers: dict[str, Any], table: pd.DataFrame, wf: pd.DataFrame, diag_tables: dict[str, pd.DataFrame]) -> str:
    lines = [
        f"# Adverse Trade Recovery Engine (ATRE) - {symbol} {timeframe}",
        "",
        "Sprint 26 — execution recovery only. Models/Heat/PositionMgmt frozen.",
        "Same production entries. Causal path decisions (no future leak).",
        "",
        f"## Verdict",
        "",
        f"**{answers.get('verdict')}**",
        "",
        "## Answers",
        "",
        f"1. **Should Recovery Engine exist?** `{answers.get('q1_engine_exist')}`",
        f"2. **Best recovery signal?** `{answers.get('q2_best_signal')}`",
        f"3. **Best early exit?** `{answers.get('q3_best_early_exit')}`",
        f"4. **Best reduce position?** `{answers.get('q4_best_reduce')}`",
        f"5. **Best dynamic stop?** `{answers.get('q5_best_dynamic_stop')}`",
        f"6. **Recovery score needed?** `{answers.get('q6_score_needed')}`",
        f"7. **Most important feature?** `{answers.get('q7_top_feature')}`",
        f"8. **Best recovery threshold?** `{answers.get('q8_best_threshold')}`",
        f"9. **Long/Short different?** `{answers.get('q9_long_short_different')}`",
        f"10. **CAGR improvement?** `{answers.get('q10_cagr_improvement_pct')}`%",
        f"11. **DD improvement?** `{answers.get('q11_dd_improvement_pct')}`%",
        f"12. **Production pipeline:** {answers.get('q12_pipeline')}",
        "",
        "## Leaderboard",
        "",
    ]
    cols = [c for c in (
        "policy", "family", "cagr", "max_drawdown", "profit_factor", "sharpe", "sortino", "calmar",
        "expectancy", "trades", "n_intervened", "accepted", "accepted_sig",
        "stat_mannwhitney_p", "stat_perm_p", "stat_boot_ci_low", "stat_boot_ci_high",
    ) if c in table.columns]
    lines.append(table[cols].head(30).to_string(index=False))
    for name, df in diag_tables.items():
        lines.extend(["", f"## {name}", ""])
        lines.append(df.to_string(index=False) if df is not None and not df.empty else "_(empty)_")
    lines.extend(["", "## Walk-forward", ""])
    lines.append(wf.to_string(index=False) if not wf.empty else "_(empty)_")
    lines.extend([
        "", "## Charts", "",
        "- `charts/mae_distribution.png`",
        "- `charts/time_underwater.png`",
        "- `charts/recovery_probability.png`",
        "- `charts/early_exit_compare.png`",
        "- `charts/recovery_score_dist.png`",
        "- `charts/feature_importance.png`",
        "- `charts/walk_forward.png`",
        "- `charts/monte_carlo.png`",
        "- `charts/equity_curve.png`",
        "",
    ])
    return "\n".join(lines)
