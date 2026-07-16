"""Report + answers for confidence layer."""

from __future__ import annotations

from typing import Any

import pandas as pd

from research.confidence_layer import META_GATE


def build_answers(
    *,
    weights: pd.DataFrame,
    buckets: pd.DataFrame,
    regimes: pd.DataFrame,
    risk_search: pd.DataFrame,
    tp_research: pd.DataFrame,
    trail_research: pd.DataFrame,
    m5_research: dict[str, Any],
    calibration_summary: dict[str, Any],
    baseline_cagr: float,
    best_cagr: float,
    baseline_dd: float,
    best_dd: float,
) -> dict[str, Any]:
    # Edge buckets: expectancy > 0 and PF > 1
    edge_buckets = []
    if not buckets.empty and "expectancy" in buckets.columns:
        for _, r in buckets.iterrows():
            if int(r.get("trades", 0) or 0) < 20:
                continue
            if float(r.get("expectancy", 0) or 0) > 0 and float(r.get("profit_factor", 0) or 0) > 1:
                edge_buckets.append(str(r["bucket"]))

    best_sched = str(risk_search.iloc[0]["schedule"]) if not risk_search.empty else "n/a"
    # Confidence threshold = lower bound of best first active band in schedule name heuristic
    conf_thr = 40.0
    if "skip50" in best_sched:
        conf_thr = 50.0
    elif "skip60" in best_sched:
        conf_thr = 60.0
    elif "skip40" in best_sched:
        conf_thr = 40.0

    use_d1 = False
    if not regimes.empty and "expectancy" in regimes.columns:
        # dispersion of expectancy across regimes
        use_d1 = float(regimes["expectancy"].std()) > 1e-5 and regimes["trades"].sum() > 50

    use_m5 = bool(m5_research.get("improves_expectancy"))

    cagr_imp = (
        (best_cagr - baseline_cagr) / abs(baseline_cagr) * 100.0
        if baseline_cagr == baseline_cagr and abs(baseline_cagr) > 1e-12
        else float("nan")
    )
    dd_red = (
        (baseline_dd - best_dd) / abs(baseline_dd) * 100.0
        if baseline_dd == baseline_dd and abs(baseline_dd) > 1e-12
        else float("nan")
    )

    return {
        "q1_add_confidence": bool(edge_buckets),
        "q2_use_d1_regime": use_d1,
        "q3_use_m5_execution": use_m5,
        "q4_best_dynamic_risk": best_sched,
        "q5_best_confidence_threshold": conf_thr,
        "q5_edge_buckets": edge_buckets,
        "q6_cagr_improvement_pct": cagr_imp,
        "q6_baseline_cagr": baseline_cagr,
        "q6_best_cagr": best_cagr,
        "q7_dd_reduction_pct": dd_red,
        "q7_baseline_dd": baseline_dd,
        "q7_best_dd": best_dd,
        "q8_architecture": (
            "Primary (frozen) -> Meta thr=0.45 (frozen) -> Confidence LOO score -> "
            f"skip if conf < {conf_thr} -> dynamic risk schedule `{best_sched}`"
            + ("; optional D1 regime filter" if use_d1 else "")
            + ("; optional M5 quality gate" if use_m5 else "")
        ),
        "weights": weights.to_dict(orient="records") if not weights.empty else [],
        "calibration": calibration_summary,
        "meta_gate": META_GATE,
        "tp_best": str(tp_research.iloc[0]["policy"]) if not tp_research.empty else "n/a",
        "trail_best": str(trail_research.iloc[0]["policy"]) if not trail_research.empty else "n/a",
    }


def build_report(
    *,
    symbol: str,
    timeframe: str,
    answers: dict[str, Any],
    weights: pd.DataFrame,
    buckets: pd.DataFrame,
    regimes: pd.DataFrame,
    risk_search: pd.DataFrame,
    tp_research: pd.DataFrame,
    trail_research: pd.DataFrame,
    calibration_bins: pd.DataFrame,
    shap_importance: pd.DataFrame,
    m5_research: dict[str, Any],
) -> str:
    lines = [
        f"# Confidence Layer Report - {symbol} {timeframe}",
        "",
        "Sprint 22 - execution layer above frozen Primary + Meta. No retrain.",
        f"Candidate filter: Meta gate `meta_proba >= {META_GATE}` (fixed, not re-optimized).",
        "Confidence weights estimated leave-one-year-out logistic regression.",
        "",
        "## Answers",
        "",
        f"1. **Should confidence be added?** `{answers.get('q1_add_confidence')}` "
        f"edge buckets=`{answers.get('q5_edge_buckets')}`",
        f"2. **Should D1 regime be used?** `{answers.get('q2_use_d1_regime')}`",
        f"3. **Should M5 execution score be used?** `{answers.get('q3_use_m5_execution')}` "
        f"`{m5_research}`",
        f"4. **Best dynamic risk schedule?** `{answers.get('q4_best_dynamic_risk')}`",
        f"5. **Best confidence threshold?** `{answers.get('q5_best_confidence_threshold')}`",
        f"6. **Expected CAGR improvement?** `{answers.get('q6_cagr_improvement_pct')}`% "
        f"(base={answers.get('q6_baseline_cagr')} → best={answers.get('q6_best_cagr')})",
        f"7. **Expected DD reduction?** `{answers.get('q7_dd_reduction_pct')}`% "
        f"(base={answers.get('q7_baseline_dd')} → best={answers.get('q7_best_dd')})",
        f"8. **Recommended production architecture:** {answers.get('q8_architecture')}",
        "",
        f"TP research best=`{answers.get('tp_best')}` | Trail best=`{answers.get('trail_best')}`",
        f"Calibration: `{answers.get('calibration')}`",
        "",
        "## Confidence weights (LOO mean share)",
        "",
    ]
    lines.append(weights.to_string(index=False) if not weights.empty else "_(empty)_")
    lines.extend(["", "## Confidence buckets", ""])
    lines.append(buckets.to_string(index=False) if not buckets.empty else "_(empty)_")
    lines.extend(["", "## D1 regime performance", ""])
    lines.append(regimes.to_string(index=False) if not regimes.empty else "_(empty)_")
    lines.extend(["", "## Dynamic risk search", ""])
    lines.append(risk_search.to_string(index=False) if not risk_search.empty else "_(empty)_")
    lines.extend(["", "## Dynamic TP research", ""])
    lines.append(tp_research.to_string(index=False) if not tp_research.empty else "_(empty)_")
    lines.extend(["", "## Dynamic trail research", ""])
    lines.append(trail_research.to_string(index=False) if not trail_research.empty else "_(empty)_")
    lines.extend(["", "## Calibration bins", ""])
    lines.append(calibration_bins.to_string(index=False) if not calibration_bins.empty else "_(empty)_")
    lines.extend(["", "## SHAP importance (components)", ""])
    lines.append(shap_importance.to_string(index=False) if not shap_importance.empty else "_(empty)_")
    lines.extend(
        [
            "",
            "## Charts",
            "",
            "- `charts/confidence_distribution.png`",
            "- `charts/bucket_expectancy.png`",
            "- `charts/regime_performance.png`",
            "- `charts/risk_schedule_compare.png`",
            "- `charts/reliability_diagram.png`",
            "- `charts/shap_importance.png`",
            "",
        ]
    )
    return "\n".join(lines)
