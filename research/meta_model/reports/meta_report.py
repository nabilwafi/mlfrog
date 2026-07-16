"""Reports for production meta model."""

from __future__ import annotations

from typing import Any

import pandas as pd

from research.meta_model import FINAL_FEATURES, REFERENCE_THRESHOLD


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
    summary: pd.DataFrame,
    by_year: pd.DataFrame,
    equity: pd.DataFrame,
    answers: dict[str, Any],
    features: tuple[str, ...] = FINAL_FEATURES,
) -> str:
    thr = answers.get("threshold_ref", REFERENCE_THRESHOLD)
    lines = [
        f"# Production Meta Model Report - {symbol} {timeframe}",
        "",
        "Sprint 20 - production Meta Model (Trade vs Skip).",
        "Primary Long/Short models are frozen. No HPO. Thresholds reported only (not optimized).",
        f"Reference threshold for Q1-Q6: `{thr}` (not chosen as 'best').",
        "",
        f"Features ({len(features)}): " + ", ".join(f"`{f}`" for f in features),
        "",
        "## Answers",
        "",
        f"1. **Did Meta improve profitability?** `{answers.get('q1_profitability_improved')}`",
        f"   - detail: `{answers.get('q1_detail')}`",
        "",
        f"2. **Did Meta reduce drawdown?** `{answers.get('q2_drawdown_reduced')}`",
        f"   - detail: `{answers.get('q2_detail')}`",
        "",
        f"3. **Did Meta reduce trade count?** `{answers.get('q3_trade_count_reduced')}`",
        f"   - detail: `{answers.get('q3_detail')}`",
        "",
        f"4. **Annual return** before Meta=`{_fmt(answers.get('q4_annual_return_before'))}` "
        f"| after Meta=`{_fmt(answers.get('q4_annual_return_after'))}`",
        "",
        f"5. **Equity from ${answers.get('q5_start', 80):.0f}** (compounded yearly)",
        f"   - Primary final=`{_fmt(answers.get('q5_primary_final'), 2)}` "
        f"| Meta final=`{_fmt(answers.get('q5_meta_final'), 2)}`",
        "",
    ]
    for row in answers.get("q5_equity_curve") or []:
        lines.append(
            f"   - {row['year']}: primary=${row['primary']:.2f} (AR={row['primary_ar']:.4f}) | "
            f"meta=${row['meta']:.2f} (AR={row['meta_ar']:.4f})"
        )

    lines.extend(
        [
            "",
            f"6. **Statistically meaningful?** `{answers.get('q6_statistically_meaningful')}`",
            f"   - detail: `{answers.get('q6_detail')}`",
            "",
            "## Primary vs Primary+Meta (pooled)",
            "",
        ]
    )
    if summary.empty:
        lines.append("_(empty)_")
    else:
        show = summary.copy()
        lines.append(show.to_string(index=False))

    lines.extend(["", "## Walk-forward by year", ""])
    if by_year.empty:
        lines.append("_(empty)_")
    else:
        cols = [
            c
            for c in (
                "system",
                "threshold",
                "valid_year",
                "n_trades",
                "n_skipped",
                "trade_reduction_pct",
                "win_rate",
                "expectancy",
                "profit_factor",
                "annual_return",
                "sharpe",
                "max_drawdown",
                "clf_roc_auc",
                "clf_pr_auc",
                "clf_ece",
            )
            if c in by_year.columns
        ]
        lines.append(by_year[cols].to_string(index=False))

    lines.extend(["", f"## Equity curves (all thresholds, start $80)", ""])
    if equity.empty:
        lines.append("_(empty)_")
    else:
        lines.append(equity.to_string(index=False))

    lines.extend(
        [
            "",
            "## Charts",
            "",
            "- `charts/primary_vs_meta.png`",
            "- `charts/threshold_metrics.png`",
            "- `charts/equity_curve.png`",
            "- `charts/wf_comparison.png`",
            "",
            "Models: `models/loo_val_{year}.txt`, `models/meta_full.txt`",
            "",
        ]
    )
    return "\n".join(lines)
