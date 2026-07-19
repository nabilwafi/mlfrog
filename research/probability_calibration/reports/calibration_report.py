"""Markdown report for Sprint 16."""

from __future__ import annotations

from typing import Any

import pandas as pd


def _fmt(x: Any, digits: int = 4) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/a"
    if v != v:
        return "n/a"
    if v == float("inf"):
        return "inf"
    return f"{v:.{digits}f}"


def build_report(
    *,
    symbol: str,
    timeframe: str,
    cost: float,
    calibration: pd.DataFrame,
    thresholds: pd.DataFrame,
    context: pd.DataFrame,
    optimal: dict[str, dict],
    answers: dict[str, Any],
) -> str:
    lines = [
        f"# Probability Calibration & Percentile Thresholds — {symbol} {timeframe}",
        "",
        "Sprint 16 — research only. No live execution.",
        "",
        f"- Transaction cost assumption (per trade, return units): `{cost}`",
        "- Calibration fit: train split per WF window; scored on validation only.",
        "",
        "## 1. Calibration comparison",
        "",
    ]
    if calibration.empty:
        lines.append("_(empty)_")
    else:
        lines.append(calibration.to_string(index=False))

    lines.extend(["", "## 2. Percentile threshold analysis", ""])
    if thresholds.empty:
        lines.append("_(empty)_")
    else:
        show = thresholds[
            [
                c
                for c in (
                    "side",
                    "percentile",
                    "prob_col",
                    "n_trades",
                    "win_rate",
                    "avg_return",
                    "expectancy",
                    "profit_factor",
                    "max_drawdown",
                    "wf_windows",
                    "wf_win_rate_std",
                    "wf_positive_expectancy_windows",
                )
                if c in thresholds.columns
            ]
        ]
        lines.append(show.to_string(index=False))

    lines.extend(["", "## 3. Context filter analysis", ""])
    if context.empty:
        lines.append("_(empty)_")
    else:
        lines.append(context.to_string(index=False))

    lines.extend(["", "## Optimal thresholds (per side)", ""])
    for side in ("long", "short"):
        opt = optimal.get(side, {})
        lines.append(
            f"- **{side}:** top `{_fmt(opt.get('percentile'), 2)}` "
            f"via `{opt.get('prob_col', 'n/a')}` | "
            f"n=`{opt.get('n_trades', 'n/a')}` WR=`{_fmt(opt.get('win_rate'))}` "
            f"E=`{_fmt(opt.get('expectancy'))}` PF=`{_fmt(opt.get('profit_factor'))}` "
            f"DD=`{_fmt(opt.get('max_drawdown'))}` "
            f"WF+exp=`{opt.get('wf_positive_expectancy_windows', 'n/a')}`"
        )

    lines.extend(
        [
            "",
            "## Research questions",
            "",
            "### Does calibration improve decision quality?",
            "",
            f"**Answer:** `{answers.get('calibration_helps', 'n/a')}`",
            "",
            str(answers.get("calibration_rationale", "")),
            "",
            "### Optimal threshold strategy",
            "",
            str(answers.get("threshold_summary", "")),
            "",
            "### Is meta-labeling required?",
            "",
            f"**Answer:** `{answers.get('meta_labeling', 'n/a')}`",
            "",
            str(answers.get("meta_rationale", "")),
            "",
            "## Charts",
            "",
            "- `charts/reliability_curve.png`",
            "- `charts/threshold_expectancy.png`",
            "- `charts/context_filter.png`",
            "",
        ]
    )
    return "\n".join(lines)


def build_answers(
    *,
    calibration: pd.DataFrame,
    thresholds: pd.DataFrame,
    context: pd.DataFrame,
    optimal: dict[str, dict],
    best_methods: dict[str, str],
) -> dict[str, Any]:
    # Calibration helps if ECE/Brier drop vs raw AND threshold expectancy improves
    cal_helps_parts = []
    ece_improved = False
    for side in sorted(best_methods):
        method = best_methods[side]
        sub = calibration.loc[calibration["side"] == side]
        raw = sub.loc[sub["method"] == "raw"]
        best = sub.loc[sub["method"] == method]
        if raw.empty or best.empty:
            continue
        d_ece = float(best.iloc[0]["ece"] - raw.iloc[0]["ece"])
        d_brier = float(best.iloc[0]["brier"] - raw.iloc[0]["brier"])
        cal_helps_parts.append(
            f"{side}: best=`{method}` dECE={d_ece:+.4f} dBrier={d_brier:+.4f}"
        )
        if d_ece < -0.001 or d_brier < -0.001:
            ece_improved = True

    # Decision quality: compare raw vs calibrated threshold best expectancy
    thr_help = False
    thr_notes = []
    for side in ("long", "short"):
        side_thr = thresholds.loc[thresholds["side"] == side]
        if side_thr.empty:
            continue
        by_col = side_thr.groupby("prob_col")["expectancy"].max()
        if "y_prob_raw" in by_col.index:
            raw_e = float(by_col["y_prob_raw"])
            others = by_col.drop(labels=["y_prob_raw"], errors="ignore")
            if not others.empty:
                best_e = float(others.max())
                thr_notes.append(f"{side}: raw_best_E={raw_e:.5f} cal_best_E={best_e:.5f}")
                if best_e > raw_e + 1e-6:
                    thr_help = True

    if ece_improved and thr_help:
        cal_ans = "yes — metrics and threshold expectancy"
    elif ece_improved:
        cal_ans = "partial — reliability improves; trade expectancy similar"
    elif thr_help:
        cal_ans = "partial — trade expectancy lifts without clear ECE win"
    else:
        cal_ans = "no clear improvement"

    # Meta-labeling: if context filter beats prob-only expectancy materially
    meta = False
    meta_notes = []
    for side in ("long", "short"):
        opt = optimal.get(side, {})
        base_e = float(opt.get("expectancy") or float("nan"))
        ctx = context.loc[context["side"] == side]
        if ctx.empty or base_e != base_e:
            continue
        best_ctx = float(ctx["expectancy"].max()) if ctx["expectancy"].notna().any() else float("nan")
        meta_notes.append(f"{side}: prob_only_E={base_e:.5f} best_ctx_E={best_ctx:.5f}")
        if best_ctx == best_ctx and base_e == base_e and best_ctx > base_e + 1e-5:
            meta = True

    thr_summary = "; ".join(
        f"{s}: top {optimal.get(s, {}).get('percentile', 'n/a')} "
        f"({optimal.get(s, {}).get('prob_col', 'n/a')})"
        for s in ("long", "short")
    )

    return {
        "calibration_helps": cal_ans,
        "calibration_rationale": "; ".join(cal_helps_parts)
        + ("; " + "; ".join(thr_notes) if thr_notes else ""),
        "threshold_summary": thr_summary,
        "meta_labeling": "yes — recommend secondary filter / meta-label"
        if meta
        else "not required yet — percentile gate is primary",
        "meta_rationale": "; ".join(meta_notes)
        if meta_notes
        else "insufficient context lift over probability-only selection",
    }
