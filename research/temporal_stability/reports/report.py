"""Markdown report + production recommendation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from research.analyzers._common import frame_to_markdown
from research.temporal_stability import META_GATE


def build_answers(arts: dict[str, Any]) -> dict[str, Any]:
    rolling = arts.get("rolling", pd.DataFrame())
    aging = arts.get("aging", pd.DataFrame())
    recency = arts.get("recency", pd.DataFrame())
    retrain = arts.get("retrain", pd.DataFrame())
    drift = arts.get("drift", pd.DataFrame())
    feat_stab = arts.get("feature_stability", pd.DataFrame())
    thr = arts.get("threshold", pd.DataFrame())
    year_f = arts.get("year_frozen", pd.DataFrame())
    mc = arts.get("monte_carlo", pd.DataFrame())
    prob = arts.get("probability", pd.DataFrame())

    answers: dict[str, Any] = {}

    # 1-2 aging from rolling ROC trend
    if isinstance(rolling, pd.DataFrame) and not rolling.empty and "roc_auc" in rolling.columns:
        r = rolling.sort_values("test_year")
        rocs = r["roc_auc"].astype(float).to_numpy()
        years = r["test_year"].astype(int).to_numpy()
        # linear slope
        if len(rocs) >= 3:
            slope = float(np.polyfit(np.arange(len(rocs)), rocs, 1)[0])
        else:
            slope = 0.0
        answers["q1_model_ages"] = bool(slope < -0.005) or (float(np.nanmin(rocs)) < float(np.nanmax(rocs)) - 0.05)
        # degradation start: first year where ROC < median - 0.03
        med = float(np.nanmedian(rocs))
        deg_year = None
        for y, v in zip(years, rocs):
            if v < med - 0.03:
                deg_year = int(y)
                break
        answers["q2_degradation_starts"] = deg_year
        answers["rolling_roc_slope"] = slope
        answers["rolling_roc_min"] = float(np.nanmin(rocs))
        answers["rolling_roc_max"] = float(np.nanmax(rocs))
    else:
        answers["q1_model_ages"] = "insufficient_data"
        answers["q2_degradation_starts"] = None

    # 3-4 history window
    if isinstance(aging, pd.DataFrame) and not aging.empty:
        g = aging.groupby("history_window", as_index=False)[["roc_auc", "profit_factor"]].mean()
        best = g.sort_values("roc_auc", ascending=False).iloc[0]
        answers["q3_best_history_window"] = str(best["history_window"])
        answers["q3_best_history_roc"] = float(best["roc_auc"])
        full = g.loc[g["history_window"] == "full"]
        best_lim = g.loc[g["history_window"] != "full"].sort_values("roc_auc", ascending=False)
        if not full.empty and not best_lim.empty:
            answers["q4_discard_old_data"] = bool(float(best_lim.iloc[0]["roc_auc"]) >= float(full.iloc[0]["roc_auc"]) - 0.005)
        else:
            answers["q4_discard_old_data"] = None
        answers["aging_summary"] = g.to_dict(orient="records")
    else:
        answers["q3_best_history_window"] = None
        answers["q4_discard_old_data"] = None

    # 5 retrain frequency
    if isinstance(retrain, pd.DataFrame) and not retrain.empty:
        g = retrain.groupby("freeze_months", as_index=False)[["roc_auc", "profit_factor", "ece"]].mean()
        base = g.loc[g["freeze_months"] == g["freeze_months"].min()]
        base_roc = float(base["roc_auc"].iloc[0]) if not base.empty else float("nan")
        pick = None
        for _, row in g.sort_values("freeze_months").iterrows():
            if row["roc_auc"] < base_roc - 0.02:
                pick = int(row["freeze_months"])
                break
        answers["q5_retrain_every_months"] = pick or int(g["freeze_months"].max())
        answers["retrain_curve"] = g.to_dict(orient="records")
    else:
        answers["q5_retrain_every_months"] = None

    # 6 recency
    if isinstance(recency, pd.DataFrame) and not recency.empty:
        g = recency.groupby("weight_mode", as_index=False)["roc_auc"].mean()
        uni = g.loc[g["weight_mode"] == "uniform"]
        best = g.sort_values("roc_auc", ascending=False).iloc[0]
        answers["q6_recency_beneficial"] = bool(
            str(best["weight_mode"]) != "uniform"
            and (uni.empty or float(best["roc_auc"]) > float(uni["roc_auc"].iloc[0]) + 0.005)
        )
        answers["q6_best_weight_mode"] = str(best["weight_mode"])
        answers["recency_summary"] = g.to_dict(orient="records")
    else:
        answers["q6_recency_beneficial"] = None
        answers["q6_best_weight_mode"] = None

    # 7 features
    if isinstance(feat_stab, pd.DataFrame) and not feat_stab.empty:
        stable = feat_stab.loc[feat_stab["cv_gain_share"] < 0.5].head(10)
        unstable = feat_stab.sort_values("cv_gain_share", ascending=False).head(5)
        answers["q7_stable_features"] = stable["feature"].tolist()
        answers["q7_unstable_features"] = unstable["feature"].tolist()
    else:
        answers["q7_stable_features"] = []
        answers["q7_unstable_features"] = []

    # 8 threshold
    if isinstance(thr, pd.DataFrame) and not thr.empty:
        bt = thr["best_threshold"].astype(float)
        answers["q8_threshold_drifts"] = bool(bt.std(ddof=1) > 0.03) if len(bt) > 1 else False
        answers["q8_mean_best_threshold"] = float(bt.mean())
        answers["q8_near_045"] = bool(abs(float(bt.mean()) - META_GATE) <= 0.05)
    else:
        answers["q8_threshold_drifts"] = None
        answers["q8_mean_best_threshold"] = None
        answers["q8_near_045"] = None

    # 9 calibration
    if isinstance(prob, pd.DataFrame) and not prob.empty:
        cal = prob.loc[prob["class"] == "calibration"] if "class" in prob.columns else prob
        if not cal.empty and "ece" in cal.columns:
            answers["q9_calibration_stable"] = bool(cal["ece"].astype(float).std(ddof=1) < 0.03) if len(cal) > 1 else True
            answers["q9_mean_ece"] = float(cal["ece"].astype(float).mean())
        else:
            answers["q9_calibration_stable"] = None
            answers["q9_mean_ece"] = None
    elif isinstance(rolling, pd.DataFrame) and not rolling.empty and "ece" in rolling.columns:
        answers["q9_calibration_stable"] = bool(rolling["ece"].astype(float).std(ddof=1) < 0.03) if len(rolling) > 1 else True
        answers["q9_mean_ece"] = float(rolling["ece"].astype(float).mean())
    else:
        answers["q9_calibration_stable"] = None
        answers["q9_mean_ece"] = None

    # 10 live robustness
    robust_bits = []
    if isinstance(year_f, pd.DataFrame) and not year_f.empty and "profit_factor" in year_f.columns:
        pf = year_f["profit_factor"].astype(float)
        robust_bits.append(bool((pf > 1.0).mean() >= 0.5))
    if isinstance(mc, pd.DataFrame) and not mc.empty:
        robust_bits.append(bool((mc["risk_of_ruin_50pct"].astype(float) < 0.2).all()))
    if isinstance(rolling, pd.DataFrame) and not rolling.empty and "roc_auc" in rolling.columns:
        robust_bits.append(bool(rolling["roc_auc"].astype(float).mean() > 0.52))
    answers["q10_robust_for_live"] = bool(all(robust_bits)) if robust_bits else None
    answers["q10_evidence"] = {
        "frozen_years_pf_gt1_share": None,
        "mc_ruin_ok": None,
        "rolling_mean_roc": float(rolling["roc_auc"].mean()) if isinstance(rolling, pd.DataFrame) and not rolling.empty and "roc_auc" in rolling.columns else None,
    }
    if isinstance(year_f, pd.DataFrame) and not year_f.empty and "profit_factor" in year_f.columns:
        answers["q10_evidence"]["frozen_years_pf_gt1_share"] = float((year_f["profit_factor"].astype(float) > 1.0).mean())
    if isinstance(mc, pd.DataFrame) and not mc.empty:
        answers["q10_evidence"]["mc_ruin_ok"] = bool((mc["risk_of_ruin_50pct"].astype(float) < 0.2).all())

    # production recommendation
    answers["recommendation"] = _recommend(answers)
    if isinstance(drift, pd.DataFrame) and not drift.empty and "feature_psi_mean" in drift.columns:
        answers["drift_mean_psi"] = float(drift["feature_psi_mean"].astype(float).mean())
    return answers


def _recommend(a: dict[str, Any]) -> str:
    parts = []
    if a.get("q10_robust_for_live") and not a.get("q1_model_ages"):
        parts.append("KEEP CURRENT MODEL")
    if a.get("q5_retrain_every_months"):
        parts.append(f"RETRAIN EVERY {a['q5_retrain_every_months']} MONTHS")
    win = a.get("q3_best_history_window")
    if win and win != "full" and a.get("q4_discard_old_data"):
        parts.append(f"USE LAST {win} ONLY")
    if a.get("q6_recency_beneficial"):
        parts.append(f"ENABLE RECENCY WEIGHTING ({a.get('q6_best_weight_mode')})")
    if not parts:
        if a.get("q1_model_ages"):
            parts.append("RETRAIN EVERY 12 MONTHS")
            if win and win != "full":
                parts.append(f"USE LAST {win} ONLY")
        else:
            parts.append("KEEP CURRENT MODEL")
    return " + ".join(parts)


def build_report(*, symbol: str, timeframe: str, answers: dict[str, Any], arts: dict[str, Any]) -> str:
    lines = [
        f"# Temporal Stability & Dataset Aging — {symbol} {timeframe}",
        "",
        "**Sprint 28 (research only).** Production stack frozen: Meta≥0.45, Confidence skip40, Heat daily_loss_-1R, Original TP/SL.",
        "",
        "This sprint diagnoses temporal robustness. It does **not** optimize strategy.",
        "",
        "## Production recommendation",
        "",
        f"**{answers.get('recommendation')}**",
        "",
        "## Answers",
        "",
        f"1. Does the model age? **{answers.get('q1_model_ages')}** (ROC slope={answers.get('rolling_roc_slope')})",
        f"2. Degradation starts? **{answers.get('q2_degradation_starts')}**",
        f"3. How much history? **{answers.get('q3_best_history_window')}** (ROC={answers.get('q3_best_history_roc')})",
        f"4. Discard old data? **{answers.get('q4_discard_old_data')}**",
        f"5. Retrain every? **{answers.get('q5_retrain_every_months')} months**",
        f"6. Recency weighting beneficial? **{answers.get('q6_recency_beneficial')}** best=`{answers.get('q6_best_weight_mode')}`",
        f"7. Stable features: `{answers.get('q7_stable_features')}`",
        f"   Unstable: `{answers.get('q7_unstable_features')}`",
        f"8. Threshold drifts? **{answers.get('q8_threshold_drifts')}** mean_best={answers.get('q8_mean_best_threshold')} near_0.45={answers.get('q8_near_045')}",
        f"9. Calibration stable? **{answers.get('q9_calibration_stable')}** mean_ECE={answers.get('q9_mean_ece')}",
        f"10. Robust for live? **{answers.get('q10_robust_for_live')}** evidence={answers.get('q10_evidence')}",
        "",
    ]
    for title, key in (
        ("Rolling performance", "rolling"),
        ("Year summary (frozen production)", "year_frozen"),
        ("Dataset aging", "aging"),
        ("Recency weights", "recency"),
        ("Retrain frequency", "retrain"),
        ("Drift summary", "drift"),
        ("Feature stability", "feature_stability"),
        ("Threshold by year", "threshold"),
        ("Monte Carlo by year", "monte_carlo"),
    ):
        df = arts.get(key)
        lines.append(f"## {title}")
        lines.append("")
        if isinstance(df, pd.DataFrame) and not df.empty:
            show = df.head(30)
            lines.append(frame_to_markdown(show))
        else:
            lines.append("_(empty)_")
        lines.append("")
    return "\n".join(lines)


def year_summary_md(year_frozen: pd.DataFrame, year_ml: pd.DataFrame) -> str:
    lines = ["# Year-by-year summary", ""]
    if isinstance(year_frozen, pd.DataFrame) and not year_frozen.empty:
        lines.append("## Frozen production path")
        lines.append("")
        lines.append(frame_to_markdown(year_frozen))
        lines.append("")
    if isinstance(year_ml, pd.DataFrame) and not year_ml.empty:
        lines.append("## ML aging panel (research LOO)")
        lines.append("")
        lines.append(frame_to_markdown(year_ml))
        lines.append("")
    return "\n".join(lines)
