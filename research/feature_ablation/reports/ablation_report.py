"""Markdown report answering the 10 ablation research questions."""

from __future__ import annotations

from typing import Any

import pandas as pd

from research.feature_ablation.entities.experiment_result import ExperimentResult


def _fmt(v: Any, digits: int = 4) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        if v != v:
            return "n/a"
        return f"{v:.{digits}f}"
    return str(v)


class AblationReportBuilder:
    def build(
        self,
        *,
        results: list[ExperimentResult],
        summary: pd.DataFrame,
        rankings: pd.DataFrame,
        symbol: str,
        timeframe: str,
        side: str,
    ) -> str:
        by_id = {r.experiment_id: r for r in results}
        baseline = by_id.get("baseline")
        base_roc = baseline.mean_roc_auc if baseline else float("nan")

        # Category contribution = drop in mean ROC when category removed
        remove_map = {
            "trend": "remove_trend",
            "momentum": "remove_momentum",
            "volatility": "remove_volatility",
            "candle": "remove_candle",
            "session": "remove_session",
            "statistical": "remove_statistical",
        }
        contrib: dict[str, float] = {}
        for cat, exp_id in remove_map.items():
            r = by_id.get(exp_id)
            if r is None or baseline is None:
                continue
            contrib[cat] = float(base_roc - r.mean_roc_auc)

        best_cat = max(contrib, key=contrib.get) if contrib else None
        worst_cat = min(contrib, key=contrib.get) if contrib else None
        # removable if removing improves or does not hurt much (delta <= 0.002)
        removable = [c for c, d in contrib.items() if d <= 0.002]

        best_wf = summary.sort_values("mean_roc_auc", ascending=False).iloc[0] if not summary.empty else None
        best_cal = (
            summary.sort_values("mean_calibration_error", ascending=True).iloc[0]
            if not summary.empty
            else None
        )

        fewer = by_id.get("top10_shap") or by_id.get("top10_mi")
        fewer_helps = bool(
            fewer
            and baseline
            and fewer.mean_roc_auc == fewer.mean_roc_auc
            and base_roc == base_roc
            and fewer.mean_roc_auc >= base_roc - 0.005
            and fewer.n_features < baseline.n_features
        )

        remove_stat = by_id.get("remove_statistical")
        stat_hurts = bool(
            remove_stat
            and baseline
            and (base_roc - remove_stat.mean_roc_auc) > 0.01
        )

        session_delta = contrib.get("session", float("nan"))
        trend_delta = contrib.get("trend", float("nan"))

        # Production recommendation: best ROC among keep_only / top20_shap / stable / baseline
        candidates = [
            by_id[i]
            for i in ("keep_only", "top20_shap", "top10_shap", "stable_only", "baseline")
            if i in by_id
        ]
        prod = max(candidates, key=lambda r: r.mean_roc_auc) if candidates else None

        lines = [
            f"# Feature Ablation Report — {symbol} {timeframe} {side}",
            "",
            f"- Experiments: `{len(results)}`",
            f"- Baseline mean WF ROC: `{_fmt(base_roc)}`",
            "",
            "## 1. Which feature category contributes the most?",
            "",
            f"**Answer:** `{best_cat}` (ROC drop when removed = `{_fmt(contrib.get(best_cat) if best_cat else None)}`)",
            "",
            "Category contribution (baseline ROC − ROC after removal):",
            "",
        ]
        for cat, delta in sorted(contrib.items(), key=lambda kv: kv[1], reverse=True):
            lines.append(f"- `{cat}`: `{_fmt(delta)}`")

        lines.extend(
            [
                "",
                "## 2. Which category can be completely removed?",
                "",
                f"**Answer:** `{removable if removable else 'none clearly safe'}`",
                "",
                f"Least harmful removal: `{worst_cat}` (delta=`{_fmt(contrib.get(worst_cat) if worst_cat else None)}`).",
                "",
                "## 3. Does fewer features improve generalization?",
                "",
                f"**Answer:** `{'likely yes / not worse' if fewer_helps else 'not clearly'}`",
                "",
            ]
        )
        if fewer and baseline:
            lines.append(
                f"- Baseline feats=`{baseline.n_features}` ROC=`{_fmt(baseline.mean_roc_auc)}` vs "
                f"`{fewer.experiment_id}` feats=`{fewer.n_features}` ROC=`{_fmt(fewer.mean_roc_auc)}`"
            )

        lines.extend(
            [
                "",
                "## 4. Does removing statistical features destroy performance?",
                "",
                f"**Answer:** `{'yes — material ROC drop' if stat_hurts else 'no — drop is small or absent'}`",
                "",
            ]
        )
        if remove_stat and baseline:
            lines.append(
                f"- remove_statistical ROC=`{_fmt(remove_stat.mean_roc_auc)}` "
                f"(delta=`{_fmt(base_roc - remove_stat.mean_roc_auc)}`)"
            )

        lines.extend(
            [
                "",
                "## 5. Is Session useful?",
                "",
                f"**Answer:** `{'yes' if session_delta == session_delta and session_delta > 0.002 else 'marginal / weak'}` "
                f"(ROC contribution=`{_fmt(session_delta)}`)",
                "",
                "## 6. Is Trend useful?",
                "",
                f"**Answer:** `{'yes' if trend_delta == trend_delta and trend_delta > 0.002 else 'marginal / weak'}` "
                f"(ROC contribution=`{_fmt(trend_delta)}`)",
                "",
                "## 7. Which feature subset has the best Walk Forward ROC?",
                "",
            ]
        )
        if best_wf is not None:
            lines.append(
                f"**Answer:** `{best_wf['name']}` (`{best_wf['experiment_id']}`) "
                f"mean ROC=`{_fmt(best_wf['mean_roc_auc'])}` n_features=`{best_wf['n_features']}`"
            )

        lines.extend(["", "## 8. Which feature subset has the best calibration?", ""])
        if best_cal is not None:
            lines.append(
                f"**Answer:** `{best_cal['name']}` mean ECE=`{_fmt(best_cal['mean_calibration_error'])}`"
            )

        lines.extend(
            [
                "",
                "## 9. Which subset is recommended for production?",
                "",
            ]
        )
        if prod is not None:
            lines.append(
                f"**Answer:** `{prod.name}` (`{prod.experiment_id}`) — "
                f"feats=`{prod.n_features}` ROC=`{_fmt(prod.mean_roc_auc)}` "
                f"ECE=`{_fmt(prod.mean_calibration_error)}`"
            )
            lines.append("")
            lines.append("Rationale: best mean walk-forward ROC among keep/top-SHAP/stable/baseline.")

        lines.extend(
            [
                "",
                "## 10. How many features should Production V1 contain?",
                "",
            ]
        )
        if prod is not None:
            lines.append(f"**Answer:** `{prod.n_features}` features (from `{prod.experiment_id}`).")
        else:
            lines.append("**Answer:** inconclusive.")

        lines.extend(
            [
                "",
                "## Rankings",
                "",
                rankings.to_string(index=False) if not rankings.empty else "_(empty)_",
                "",
                "## Charts",
                "",
                "- `charts/category_contribution.png`",
                "- `charts/walk_forward_comparison.png`",
                "- `charts/roc_comparison.png`",
                "- `charts/feature_count_vs_roc.png`",
                "- `charts/feature_count_vs_calibration.png`",
                "- `charts/feature_contribution.png`",
                "",
            ]
        )
        return "\n".join(lines)


def build_rankings(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    rank_roc = summary["mean_roc_auc"].rank(ascending=False, method="min")
    rank_cal = summary["mean_calibration_error"].rank(ascending=True, method="min")
    rank_f1 = summary["mean_f1"].rank(ascending=False, method="min")
    out = summary[
        [
            "experiment_id",
            "name",
            "n_features",
            "mean_roc_auc",
            "mean_calibration_error",
            "mean_f1",
            "mean_pr_auc",
        ]
    ].copy()
    out["rank_roc"] = rank_roc
    out["rank_calibration"] = rank_cal
    out["rank_f1"] = rank_f1
    out["rank_score"] = (rank_roc + rank_cal) / 2.0
    return out.sort_values("rank_score").reset_index(drop=True)
