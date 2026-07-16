"""Reports and answers for Trade Quality Engine."""

from __future__ import annotations

from typing import Any

import pandas as pd

from research.trade_quality import META_GATE


def build_answers(
    *,
    stability: pd.DataFrame,
    tq_buckets: pd.DataFrame,
    conf_buckets: pd.DataFrame,
    risk_search: pd.DataFrame,
    interactions: pd.DataFrame,
    capital: pd.DataFrame,
    yearly: pd.DataFrame,
    shap_df: pd.DataFrame,
    mono: dict[str, Any],
    baseline_cagr: float,
    baseline_dd: float,
) -> dict[str, Any]:
    best_method = str(stability.iloc[0]["method"]) if not stability.empty else "n/a"
    tq_only = risk_search.loc[~risk_search["schedule"].astype(str).str.startswith("conf_")]
    conf_sched = risk_search.loc[risk_search["schedule"] == "conf_skip40_flat1"]
    overall_best = risk_search.iloc[0] if not risk_search.empty else None
    best_tq_sched = tq_only.iloc[0] if not tq_only.empty else None

    def _f(row: pd.Series | None, key: str) -> float:
        if row is None:
            return float("nan")
        v = row.get(key)
        return float(v) if v == v else float("nan")

    tq_best_cagr = _f(best_tq_sched, "cagr")
    tq_best_dd = _f(best_tq_sched, "max_drawdown")
    conf_cagr = _f(conf_sched.iloc[0] if not conf_sched.empty else None, "cagr")
    conf_dd = _f(conf_sched.iloc[0] if not conf_sched.empty else None, "max_drawdown")
    conf_calmar = _f(conf_sched.iloc[0] if not conf_sched.empty else None, "calmar")
    tq_calmar = _f(best_tq_sched, "calmar")

    # Compare TQ vs confidence via bucket expectancy on mid buckets
    def mid_e(b: pd.DataFrame) -> float:
        mid = b.loc[b["bucket"].isin(["40-60", "60-80", "80-100"])]
        if mid.empty or mid["trades"].sum() == 0:
            return float("nan")
        return float((mid["expectancy"] * mid["trades"]).sum() / mid["trades"].sum())

    tq_mid = mid_e(tq_buckets)
    conf_mid = mid_e(conf_buckets) if not conf_buckets.empty else float("nan")

    # Portfolio: does best TQ schedule beat baseline AND confidence?
    beats_base = tq_best_cagr == tq_best_cagr and tq_best_cagr > baseline_cagr
    beats_conf_port = (
        tq_calmar == tq_calmar
        and conf_calmar == conf_calmar
        and tq_calmar >= conf_calmar
    )
    beats_conf_bucket = tq_mid == tq_mid and conf_mid == conf_mid and tq_mid >= conf_mid
    tq_outperforms = bool(beats_base and (beats_conf_port or beats_conf_bucket))

    cagr_imp = (
        (tq_best_cagr - baseline_cagr) / abs(baseline_cagr) * 100
        if baseline_cagr == baseline_cagr and abs(baseline_cagr) > 1e-12 and tq_best_cagr == tq_best_cagr
        else float("nan")
    )
    dd_red = (
        (baseline_dd - tq_best_dd) / abs(baseline_dd) * 100
        if baseline_dd == baseline_dd and abs(baseline_dd) > 1e-12 and tq_best_dd == tq_best_dd
        else float("nan")
    )

    top_seg = str(interactions.iloc[0]["segment"]) if not interactions.empty else "n/a"
    top_feat = str(shap_df.iloc[0]["feature"]) if not shap_df.empty else "n/a"

    long_e = interactions.loc[interactions["segment"] == "side=long", "tq_edge"]
    short_e = interactions.loc[interactions["segment"] == "side=short", "tq_edge"]
    different_ls = False
    if len(long_e) and len(short_e):
        different_ls = abs(float(long_e.iloc[0]) - float(short_e.iloc[0])) > 0.001

    d1_edge = interactions.loc[interactions["segment"].astype(str).str.startswith("regime=")]
    use_d1 = (not d1_edge.empty) and float(d1_edge["tq_edge"].abs().max()) > 0.001
    use_m5 = top_feat == "m5_entry_quality_01" or (
        not shap_df.empty
        and "m5_entry_quality_01" in shap_df["feature"].tolist()
        and float(shap_df.loc[shap_df["feature"] == "m5_entry_quality_01", "mean_abs_shap"].iloc[0])
        >= float(shap_df["mean_abs_shap"].median())
    )

    # Retire confidence only if TQ clearly dominates both bucket mid-edge and Calmar
    remove_conf = bool(beats_conf_bucket and beats_conf_port)

    prod_sched = best_tq_sched if best_tq_sched is not None else overall_best
    if not remove_conf and not conf_sched.empty:
        # Keep confidence gate as production driver when it wins Calmar
        if conf_calmar == conf_calmar and (tq_calmar != tq_calmar or conf_calmar > tq_calmar):
            prod_sched = conf_sched.iloc[0]

    best_bucket_alloc = capital.loc[capital["trades"] > 0].sort_values("expected_return", ascending=False)
    bucket_alloc = (
        best_bucket_alloc[["bucket", "suggested_risk_pct", "expected_return"]].to_dict(orient="records")
        if not best_bucket_alloc.empty
        else []
    )

    sched_name = str(prod_sched["schedule"]) if prod_sched is not None else "n/a"
    return {
        "q1_tq_outperforms_confidence": tq_outperforms,
        "q1_detail": {
            "tq_mid_expectancy": tq_mid,
            "conf_mid_expectancy": conf_mid,
            "best_method": best_method,
            "best_tq_schedule": str(best_tq_sched["schedule"]) if best_tq_sched is not None else "n/a",
            "best_tq_calmar": tq_calmar,
            "conf_calmar": conf_calmar,
        },
        "q2_cagr_improvement_pct": cagr_imp,
        "q2_baseline_cagr": baseline_cagr,
        "q2_best_cagr": tq_best_cagr,
        "q2_vs_confidence_cagr_pct": (
            (tq_best_cagr - conf_cagr) / abs(conf_cagr) * 100
            if conf_cagr == conf_cagr and abs(conf_cagr) > 1e-12 and tq_best_cagr == tq_best_cagr
            else float("nan")
        ),
        "q3_dd_reduction_pct": dd_red,
        "q3_baseline_dd": baseline_dd,
        "q3_best_dd": tq_best_dd,
        "q4_best_risk_schedule": sched_name,
        "q4_best_tq_schedule": str(best_tq_sched["schedule"]) if best_tq_sched is not None else "n/a",
        "q4_overall_leader": str(overall_best["schedule"]) if overall_best is not None else "n/a",
        "q5_best_bucket_allocation": bucket_alloc,
        "q6_remove_confidence_entirely": remove_conf,
        "q7_d1_affect_risk": use_d1,
        "q8_m5_affect_risk": use_m5,
        "q9_long_short_different_rules": different_ls,
        "q10_architecture": (
            f"Primary (frozen) -> Meta>={META_GATE} (frozen) -> "
            + (
                f"Confidence (`skip40_flat1`) with TQ (`{best_method}`) as sizing overlay"
                if not remove_conf
                else f"Trade Quality Engine (`{best_method}`)"
            )
            + f" -> Dynamic size (`{sched_name}`) -> Execution"
        ),
        "mono": mono,
        "top_interaction": top_seg,
        "top_shap_feature": top_feat,
        "yearly_ranking_gap_mean": float(yearly["ranking_gap"].mean()) if not yearly.empty else float("nan"),
    }


def build_report(
    *,
    symbol: str,
    timeframe: str,
    answers: dict[str, Any],
    stability: pd.DataFrame,
    tq_buckets: pd.DataFrame,
    conf_buckets: pd.DataFrame,
    risk_search: pd.DataFrame,
    interactions: pd.DataFrame,
    capital: pd.DataFrame,
    yearly: pd.DataFrame,
    shap_df: pd.DataFrame,
) -> str:
    lines = [
        f"# Trade Quality Engine Report - {symbol} {timeframe}",
        "",
        "Sprint 23 - portfolio execution research. Models frozen. Meta thr fixed at 0.45.",
        "Trade Quality is a composite execution score (0-100), not a re-fitted classifier objective.",
        "",
        "## Answers",
        "",
        f"1. **Does TQ outperform Confidence?** `{answers.get('q1_tq_outperforms_confidence')}` "
        f"`{answers.get('q1_detail')}`",
        f"2. **CAGR improvement (best TQ sched vs always 1%)?** `{answers.get('q2_cagr_improvement_pct')}`% "
        f"(base={answers.get('q2_baseline_cagr')} -> best TQ={answers.get('q2_best_cagr')}; "
        f"vs conf={answers.get('q2_vs_confidence_cagr_pct')}%)",
        f"3. **DD reduction (best TQ sched vs always 1%)?** `{answers.get('q3_dd_reduction_pct')}`% "
        f"(base={answers.get('q3_baseline_dd')} -> best TQ={answers.get('q3_best_dd')})",
        f"4. **Best risk schedule (production)?** `{answers.get('q4_best_risk_schedule')}` "
        f"(best TQ=`{answers.get('q4_best_tq_schedule')}`, overall=`{answers.get('q4_overall_leader')}`)",
        f"5. **Best bucket allocation?** `{answers.get('q5_best_bucket_allocation')}`",
        f"6. **Remove confidence entirely?** `{answers.get('q6_remove_confidence_entirely')}`",
        f"7. **D1 affect risk sizing?** `{answers.get('q7_d1_affect_risk')}`",
        f"8. **M5 affect risk sizing?** `{answers.get('q8_m5_affect_risk')}`",
        f"9. **Long/Short different rules?** `{answers.get('q9_long_short_different_rules')}`",
        f"10. **Production pipeline:** {answers.get('q10_architecture')}",
        "",
        f"Monotonicity: `{answers.get('mono')}` | Top interaction: `{answers.get('top_interaction')}` | "
        f"Top SHAP: `{answers.get('top_shap_feature')}`",
        "",
        "## Method stability (LOO)",
        "",
    ]
    lines.append(stability.to_string(index=False) if not stability.empty else "_(empty)_")
    lines.extend(["", "## TQ bucket performance", ""])
    lines.append(tq_buckets.to_string(index=False) if not tq_buckets.empty else "_(empty)_")
    lines.extend(["", "## Confidence bucket performance (compare)", ""])
    lines.append(conf_buckets.to_string(index=False) if not conf_buckets.empty else "_(empty)_")
    lines.extend(["", "## Dynamic risk search", ""])
    lines.append(risk_search.to_string(index=False) if not risk_search.empty else "_(empty)_")
    lines.extend(["", "## Interactions (TQ edge)", ""])
    lines.append(interactions.head(20).to_string(index=False) if not interactions.empty else "_(empty)_")
    lines.extend(["", "## Capital allocation by bucket", ""])
    lines.append(capital.to_string(index=False) if not capital.empty else "_(empty)_")
    lines.extend(["", "## Yearly stability", ""])
    lines.append(yearly.to_string(index=False) if not yearly.empty else "_(empty)_")
    lines.extend(["", "## SHAP importance", ""])
    lines.append(shap_df.to_string(index=False) if not shap_df.empty else "_(empty)_")
    lines.extend(
        [
            "",
            "## Charts",
            "",
            "- `charts/tq_distribution.png`",
            "- `charts/bucket_performance.png`",
            "- `charts/yearly_stability.png`",
            "- `charts/risk_compare.png`",
            "- `charts/shap_importance.png`",
            "- `charts/interaction_heatmap.png`",
            "- `charts/tq_vs_expectancy.png`",
            "- `charts/tq_vs_drawdown.png`",
            "- `charts/calibration.png`",
            "- `charts/rolling_performance.png`",
            "- `charts/capital_allocation.png`",
            "",
        ]
    )
    return "\n".join(lines)
