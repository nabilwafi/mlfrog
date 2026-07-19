"""Markdown report for meta-dataset suitability."""

from __future__ import annotations

from typing import Any

import pandas as pd

from research.meta_dataset_validation.services.analyzers import MIN_PER_WINDOW, MIN_TOTAL_SAMPLES


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
    summary: pd.DataFrame,
    wf: pd.DataFrame,
    stability: pd.DataFrame,
    answers: dict[str, Any],
) -> str:
    rec = _recommendations(answers, stability, summary)
    lines = [
        f"# Meta Dataset Validation Report — {symbol} {timeframe}",
        "",
        "Research only. No meta-model training. No threshold/hyperparameter optimization.",
        "",
        "Primary models: Long v2 (H1 + 5 H4 structure feats), Short v2 (H1 + swing_quality).",
        f"Transaction cost assumption: `{cost}` (return units, per trade).",
        "Candidate definition: within each walk-forward validation window, top-P% by raw probability rank.",
        f"Trainability heuristic: n≥`{MIN_TOTAL_SAMPLES}`, min/window≥`{MIN_PER_WINDOW}`, ≥30 pos & neg meta labels.",
        "",
        "## Executive Summary",
        "",
        rec["executive"],
        "",
        "## Research Answers",
        "",
        "### Q1. Which percentile produces the best trade quality?",
        "",
    ]
    for side in ("long", "short"):
        q = answers.get(side, {}).get("best_quality", {})
        lines.append(
            f"- **{side}:** top `{_fmt(q.get('percentile'), 2)}` | "
            f"n=`{q.get('n_trades', 'n/a')}` WR=`{_fmt(q.get('win_rate'))}` "
            f"E=`{_fmt(q.get('expectancy'))}` PF=`{_fmt(q.get('profit_factor'))}`"
        )

    lines.extend(["", "### Q2. Which percentile produces the most stable WF performance?", ""])
    for side in ("long", "short"):
        q = answers.get(side, {}).get("best_stable", {})
        lines.append(
            f"- **{side}:** top `{_fmt(q.get('percentile'), 2)}` | "
            f"WF+E frac=`{_fmt(q.get('wf_positive_frac'))}` "
            f"mean E=`{_fmt(q.get('wf_expectancy_mean'))}` "
            f"std E=`{_fmt(q.get('wf_expectancy_std'))}`"
        )

    lines.extend(["", "### Q3. Which percentile provides enough samples for a Meta Model?", ""])
    for side in ("long", "short"):
        enough = answers.get(side, {}).get("enough_percentiles", [])
        lines.append(
            f"- **{side}:** `{enough if enough else 'none at current floors'}`"
        )

    lines.extend(
        [
            "",
            "### Q4. Should Long and Short use different candidate thresholds?",
            "",
            f"**Answer:** `{rec['different_thresholds']}`",
            "",
            rec["different_rationale"],
            "",
            "### Q5. What candidate trade definition should be used to create the Meta dataset?",
            "",
            rec["candidate_definition"],
            "",
            "## Percentile quality (pooled OOF)",
            "",
        ]
    )
    if summary.empty:
        lines.append("_(empty)_")
    else:
        lines.append(summary.to_string(index=False))

    lines.extend(["", "## Walk-forward by percentile", ""])
    if wf.empty:
        lines.append("_(empty)_")
    else:
        show = wf[
            [
                c
                for c in (
                    "side",
                    "percentile",
                    "valid_year",
                    "n_trades",
                    "win_rate",
                    "expectancy",
                    "profit_factor",
                    "max_drawdown",
                    "positive_expectancy",
                )
                if c in wf.columns
            ]
        ]
        lines.append(show.to_string(index=False))

    lines.extend(["", "## Dataset stability / label balance", ""])
    if stability.empty:
        lines.append("_(empty)_")
    else:
        lines.append(stability.to_string(index=False))

    lines.extend(
        [
            "",
            "## Recommendations",
            "",
            rec["recommendations"],
            "",
            "## Charts",
            "",
            "- `charts/return_hist.png`",
            "- `charts/mae_hist.png`",
            "- `charts/mfe_hist.png`",
            "- `charts/holding_hist.png`",
            "- `charts/probability_hist.png`",
            "- `charts/percentile_quality.png`",
            "- `charts/wf_expectancy.png`",
            "",
        ]
    )
    return "\n".join(lines)


def _recommendations(
    answers: dict[str, Any],
    stability: pd.DataFrame,
    summary: pd.DataFrame,
) -> dict[str, str]:
    long_q = answers.get("long", {}).get("best_quality", {})
    short_q = answers.get("short", {}).get("best_quality", {})
    long_s = answers.get("long", {}).get("best_stable", {})
    short_s = answers.get("short", {}).get("best_stable", {})
    long_e = answers.get("long", {}).get("enough_percentiles", [])
    short_e = answers.get("short", {}).get("enough_percentiles", [])

    lp = long_q.get("percentile")
    sp = short_q.get("percentile")
    different = (
        lp == lp
        and sp == sp
        and abs(int(round(float(lp) * 100)) - int(round(float(sp) * 100))) >= 2
    )

    # Compromise candidate: prefer quality that is also in enough set; else widen
    def _pick(side: str, quality_pct: float, stable_pct: float, enough: list[float]) -> float:
        if quality_pct == quality_pct and enough and quality_pct in enough:
            return float(quality_pct)
        if stable_pct == stable_pct and enough and stable_pct in enough:
            return float(stable_pct)
        if enough:
            # closest enough to quality
            if quality_pct == quality_pct:
                return float(min(enough, key=lambda x: abs(x - quality_pct)))
            return float(enough[0])
        return float(quality_pct) if quality_pct == quality_pct else float("nan")

    long_def = _pick("long", float(lp) if lp == lp else float("nan"), float(long_s.get("percentile", float("nan"))), long_e)
    short_def = _pick("short", float(sp) if sp == sp else float("nan"), float(short_s.get("percentile", float("nan"))), short_e)

    trainable = bool(long_e or short_e)
    exec_bits = [
        f"Best quality: Long top-{_fmt(lp, 2)}, Short top-{_fmt(sp, 2)}.",
        f"Most stable WF: Long top-{_fmt(long_s.get('percentile'), 2)}, Short top-{_fmt(short_s.get('percentile'), 2)}.",
        (
            "Enough samples exist for a meta dataset at wider gates "
            f"(Long `{long_e}`, Short `{short_e}`)."
            if trainable
            else "Sample floors are tight — meta training is marginal without widening the gate."
        ),
        "WF consistency remains weak (often ≤2/4 positive-E windows); meta may help filter losers but is not justified as a silver bullet.",
    ]

    rec_lines = [
        f"1. Define Long meta candidates as **top {_fmt(long_def, 2)}** (window-wise rank on raw `y_prob`).",
        f"2. Define Short meta candidates as **top {_fmt(short_def, 2)}**.",
        "3. Meta label = `1` iff `realized_return - cost > 0` (costs already assumed in this report).",
        "4. Do **not** replace ranking with Platt/Isotonic for candidate selection (Sprint 16).",
        "5. Proceed to meta-model design only if willing to accept sparse regimes and side-specific gates; otherwise prioritize Long context filter from Sprint 16 over a full second-stage model.",
    ]

    return {
        "executive": " ".join(exec_bits),
        "different_thresholds": "yes" if different else "optional / similar",
        "different_rationale": (
            f"Quality optima differ (Long `{_fmt(lp, 2)}` vs Short `{_fmt(sp, 2)}`); "
            "feature packs and WF behaviour already differ by side."
            if different
            else "Quality optima are close; still prefer independent side gates for implementation clarity."
        ),
        "candidate_definition": (
            f"Per WF validation window, take top-{_fmt(long_def, 2)} Long and top-{_fmt(short_def, 2)} Short "
            "by raw model probability. Attach primary features already used by v2 plus trade diagnostics "
            "(return, holding, MAE/MFE) as optional meta inputs in a later sprint — this report does not add features."
        ),
        "recommendations": "\n".join(rec_lines),
    }
