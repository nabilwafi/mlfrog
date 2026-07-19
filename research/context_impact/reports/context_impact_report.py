"""Markdown report answering Sprint-10 research questions."""

from __future__ import annotations

from typing import Any

import pandas as pd

from research.context_impact.entities.experiment_result import ExperimentResult


def _fmt(v: Any, digits: int = 4) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        if v != v:
            return "n/a"
        return f"{v:.{digits}f}"
    return str(v)


def build_comparison(results: list[ExperimentResult]) -> pd.DataFrame:
    """Baseline vs all_context per side."""
    rows: list[dict[str, Any]] = []
    by_key = {(r.side, r.experiment_id): r for r in results}
    for side in sorted({r.side for r in results}):
        base = by_key.get((side, "baseline"))
        all_ctx = by_key.get((side, "all_context"))
        if base is None or all_ctx is None:
            continue
        # Paired window wins
        base_ok = {w.valid_year: w for w in base.windows if w.status == "ok"}
        all_ok = {w.valid_year: w for w in all_ctx.windows if w.status == "ok"}
        years = sorted(set(base_ok) & set(all_ok))
        wins = sum(1 for y in years if all_ok[y].roc_auc > base_ok[y].roc_auc)
        rows.append(
            {
                "side": side,
                "baseline_roc": base.mean_roc_auc,
                "all_context_roc": all_ctx.mean_roc_auc,
                "delta_roc": all_ctx.mean_roc_auc - base.mean_roc_auc,
                "baseline_pr": base.mean_pr_auc,
                "all_context_pr": all_ctx.mean_pr_auc,
                "delta_pr": all_ctx.mean_pr_auc - base.mean_pr_auc,
                "baseline_logloss": base.mean_log_loss,
                "all_context_logloss": all_ctx.mean_log_loss,
                "delta_logloss": all_ctx.mean_log_loss - base.mean_log_loss,
                "baseline_ece": base.mean_calibration_error,
                "all_context_ece": all_ctx.mean_calibration_error,
                "delta_ece": all_ctx.mean_calibration_error - base.mean_calibration_error,
                "baseline_f1": base.mean_f1,
                "all_context_f1": all_ctx.mean_f1,
                "delta_f1": all_ctx.mean_f1 - base.mean_f1,
                "baseline_sep": base.mean_probability_separation,
                "all_context_sep": all_ctx.mean_probability_separation,
                "delta_sep": all_ctx.mean_probability_separation - base.mean_probability_separation,
                "baseline_wf_stability": base.wf_stability,
                "all_context_wf_stability": all_ctx.wf_stability,
                "windows_all_beats_baseline": wins,
                "windows_compared": len(years),
            }
        )
    return pd.DataFrame(rows)


class ContextImpactReportBuilder:
    def build(
        self,
        *,
        results: list[ExperimentResult],
        comparison: pd.DataFrame,
        ablation: pd.DataFrame,
        symbol: str,
        timeframe: str,
    ) -> str:
        sides = sorted({r.side for r in results})
        by_key = {(r.side, r.experiment_id): r for r in results}

        # Q1: overall improvement?
        improves_any = False
        if not comparison.empty:
            improves_any = bool((comparison["delta_roc"] > 0).any())
        meaningful = False
        if not comparison.empty:
            # Meaningful if |delta_roc| >= 0.01 OR wins majority of WF windows on a side
            meaningful = bool(
                (comparison["delta_roc"].abs() >= 0.01).any()
                or (
                    (
                        comparison["windows_all_beats_baseline"]
                        > comparison["windows_compared"] / 2
                    )
                    & (comparison["delta_roc"] > 0)
                ).any()
            )

        # Q2: long vs short
        long_delta = short_delta = float("nan")
        if not comparison.empty:
            if "long" in set(comparison["side"]):
                long_delta = float(comparison.loc[comparison["side"] == "long", "delta_roc"].iloc[0])
            if "short" in set(comparison["side"]):
                short_delta = float(comparison.loc[comparison["side"] == "short", "delta_roc"].iloc[0])
        if long_delta == long_delta and short_delta == short_delta:
            if long_delta > short_delta + 0.002:
                side_answer = f"Long benefits more (delta ROC long=`{_fmt(long_delta)}` vs short=`{_fmt(short_delta)}`)"
            elif short_delta > long_delta + 0.002:
                side_answer = f"Short benefits more (delta ROC short=`{_fmt(short_delta)}` vs long=`{_fmt(long_delta)}`)"
            else:
                side_answer = f"Similar impact (long=`{_fmt(long_delta)}`, short=`{_fmt(short_delta)}`)"
        else:
            side_answer = "insufficient side coverage"

        # Q3: best category (avg delta vs baseline across sides)
        cat_deltas: dict[str, list[float]] = {}
        for side in sides:
            base = by_key.get((side, "baseline"))
            if base is None:
                continue
            for exp_id, cat in (
                ("trend_context_only", "trend"),
                ("volatility_context_only", "volatility"),
                ("structure_context_only", "structure"),
                ("all_context", "all"),
            ):
                r = by_key.get((side, exp_id))
                if r is None:
                    continue
                cat_deltas.setdefault(cat, []).append(r.mean_roc_auc - base.mean_roc_auc)
        cat_mean = {c: sum(v) / len(v) for c, v in cat_deltas.items() if v}
        best_cat = max(cat_mean, key=cat_mean.get) if cat_mean else None

        # Q5: production recommendation — require meaningful lift OR strong category signal
        prod_yes = bool(
            meaningful
            and not comparison.empty
            and (comparison["delta_roc"] > 0).any()
        )
        if not prod_yes and best_cat and cat_mean.get(best_cat, 0) >= 0.01:
            prod_yes = True

        lines = [
            f"# Context Impact Report — {symbol} {timeframe}",
            "",
            "Sprint 10 — Does H4 market context improve the H1 LightGBM baseline?",
            "",
            "Protocol: identical walk-forward windows, LightGBM config, threshold, and labels. "
            "Context columns come from Sprint-9 causal join (no re-engineering).",
            "",
            f"- Sides: `{', '.join(sides)}`",
            f"- Experiments per side: `{len({r.experiment_id for r in results})}`",
            "",
            "## 1. Does H4 context improve predictive performance?",
            "",
        ]
        if comparison.empty:
            lines.append("**Answer:** inconclusive (no comparison rows).")
        else:
            lines.append(
                f"**Answer:** `{'yes on at least one side' if improves_any else 'no clear improvement'}`"
            )
            lines.append("")
            for _, row in comparison.iterrows():
                lines.append(
                    f"- `{row['side']}`: baseline ROC=`{_fmt(row['baseline_roc'])}` → "
                    f"all_context=`{_fmt(row['all_context_roc'])}` "
                    f"(Δ=`{_fmt(row['delta_roc'])}`); "
                    f"WF wins `{int(row['windows_all_beats_baseline'])}/{int(row['windows_compared'])}`"
                )

        lines.extend(
            [
                "",
                "## 2. Does context help Long more than Short?",
                "",
                f"**Answer:** {side_answer}",
                "",
                "## 3. Which context category contributes most?",
                "",
            ]
        )
        if best_cat:
            lines.append(
                f"**Answer:** `{best_cat}` (mean ΔROC vs baseline across sides = `{_fmt(cat_mean[best_cat])}`)"
            )
            lines.append("")
            for cat, delta in sorted(cat_mean.items(), key=lambda kv: kv[1], reverse=True):
                lines.append(f"- `{cat}`: `{_fmt(delta)}`")
        else:
            lines.append("**Answer:** inconclusive.")

        lines.extend(
            [
                "",
                "## 4. Are improvements statistically meaningful?",
                "",
                f"**Answer:** `{'plausible / yes' if meaningful else 'no — deltas are small or inconsistent'}`",
                "",
                "Heuristic (not a formal test): |ΔROC| ≥ 0.01 **or** all_context beats baseline "
                "on a majority of walk-forward windows with positive mean ΔROC.",
                "",
                "## 5. Should context features become part of production dataset?",
                "",
                f"**Answer:** `{'yes — candidate for production v2' if prod_yes else 'not yet — keep as research context only'}`",
                "",
                "Rationale: require statistically meaningful lift (Q4) with positive ΔROC on at least "
                "one side, or a category mean ΔROC ≥ 0.01.",
                "",
                "## Comparison (baseline vs all context)",
                "",
                comparison.to_string(index=False) if not comparison.empty else "_(empty)_",
                "",
                "## Ablation results",
                "",
                ablation.to_string(index=False) if not ablation.empty else "_(empty)_",
                "",
                "## Charts",
                "",
                "- `charts/roc_comparison.png`",
                "- `charts/delta_roc_by_side.png`",
                "- `charts/category_ablation.png`",
                "- `charts/probability_separation.png`",
                "- `charts/walk_forward_stability.png`",
                "",
            ]
        )
        return "\n".join(lines)
