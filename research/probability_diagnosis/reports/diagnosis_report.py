"""Markdown report for probability collapse diagnosis."""

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
    return f"{v:.{digits}f}"


def build_report(
    *,
    symbol: str,
    timeframe: str,
    labels: pd.DataFrame,
    pred_dist: pd.DataFrame,
    wf_drift: pd.DataFrame,
    compare: pd.DataFrame,
    answers: dict[str, Any],
) -> str:
    lines = [
        f"# Probability Collapse Diagnosis — {symbol} {timeframe}",
        "",
        "Sprint 15 — research only. No calibration, no threshold tuning, no execution.",
        "",
        "Models: Long/Short v2 LightGBM (H1 + selected H4 structure) vs H1 baseline.",
        "",
        "## 1. Label distribution",
        "",
    ]
    if labels.empty:
        lines.append("_(empty)_")
    else:
        lines.append(labels.to_string(index=False))

    lines.extend(["", "## 2. Prediction distribution (train vs validation)", ""])
    if pred_dist.empty:
        lines.append("_(empty)_")
    else:
        cols = [
            "side",
            "experiment_id",
            "split",
            "n",
            "pos_rate",
            "mean",
            "std",
            "p50",
            "p90",
            "p95",
            "p99",
            "frac_above_050",
            "mean_minus_pos_rate",
            "roc_auc",
            "probability_separation",
        ]
        show = pred_dist[[c for c in cols if c in pred_dist.columns]]
        lines.append(show.to_string(index=False))

    lines.extend(["", "## 3. Walk-forward probability drift (validation)", ""])
    if wf_drift.empty:
        lines.append("_(empty)_")
    else:
        drift_cols = [
            "side",
            "experiment_id",
            "valid_year",
            "mean",
            "std",
            "p90",
            "p95",
            "p99",
            "frac_above_050",
            "pos_rate",
        ]
        v2 = wf_drift.loc[wf_drift["experiment_id"] == "B_v2_context"]
        show = v2[[c for c in drift_cols if c in v2.columns]].sort_values(
            ["side", "valid_year"]
        )
        lines.append(show.to_string(index=False))

    lines.extend(["", "## 4. Baseline vs v2 (validation)", ""])
    if compare.empty:
        lines.append("_(empty)_")
    else:
        lines.append(compare.to_string(index=False))

    lines.extend(
        [
            "",
            "## 5. Model confidence — does context help?",
            "",
            f"- Ranking vs confidence: "
            f"`{'ranking only' if answers.get('ranking_only') else 'ranking + some confidence lift'}`",
            f"- Detail: {answers.get('conf_notes', '')}",
            "",
            "## Research questions",
            "",
            "### 1. Is probability collapse caused by class imbalance?",
            "",
            f"**Answer:** `{'yes — primary contributor' if answers.get('imbalance_cause') else 'partial / not sole cause'}`",
            "",
            str(answers.get("imbalance_notes", "")),
            "",
            "_If mean predicted probability tracks the empirical positive rate, "
            "the model is prior-anchored (typical under mild imbalance + weak signal)._",
            "",
            "### 2. Is collapse caused by model regularization?",
            "",
            f"**Answer:** `{answers.get('reg_answer', 'n/a')}`",
            "",
            f"- Params: `{answers.get('reg_params', '')}`",
            f"- Train collapse check: {answers.get('train_notes', '')}",
            "",
            "### 3. Does v2 improve ranking but not confidence?",
            "",
            f"**Answer:** `{'yes' if answers.get('ranking_only') else 'no — some confidence expansion too'}`",
            "",
            str(answers.get("conf_notes", "")),
            "",
            "### 4. Should calibration proceed?",
            "",
            f"**Answer:** `{'yes — with percentile thresholds' if answers.get('proceed_calibration') else 'not yet'}`",
            "",
            "Calibration remaps levels; it does not widen a Dirac prior. "
            "Proceed only if ranking (ROC / separation) is useful, and plan "
            "thresholds on empirical percentiles — not fixed 0.50/0.60 cuts.",
            "",
            "## Charts",
            "",
            "- `charts/prediction_histogram.png`",
            "- `charts/wf_probability_shift.png`",
            "- `charts/baseline_vs_v2_probability.png`",
            "",
        ]
    )
    return "\n".join(lines)
