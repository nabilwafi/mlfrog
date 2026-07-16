"""Markdown report for H4 structure enhancement."""

from __future__ import annotations

from typing import Any

import pandas as pd

from research.h4_structure.entities.experiment_result import ExperimentResult


def _fmt(v: Any, digits: int = 4) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        if v != v:
            return "n/a"
        return f"{v:.{digits}f}"
    return str(v)


def build_feature_statistics(
    structure_on_h1: pd.DataFrame,
    structure_cols: list[str],
    importance_by_side: dict[str, dict[str, float]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for col in structure_cols:
        s = structure_on_h1[col].dropna()
        row: dict[str, Any] = {
            "feature": col,
            "count": int(len(s)),
            "mean": float(s.mean()) if len(s) else float("nan"),
            "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
            "min": float(s.min()) if len(s) else float("nan"),
            "max": float(s.max()) if len(s) else float("nan"),
            "pct_nonzero": float((s != 0).mean()) if len(s) else float("nan"),
        }
        for side, imp in importance_by_side.items():
            row[f"importance_{side}"] = float(imp.get(col, 0.0))
        rows.append(row)
    return pd.DataFrame(rows)


class StructureReportBuilder:
    def build(
        self,
        *,
        results: list[ExperimentResult],
        ablation: pd.DataFrame,
        feature_statistics: pd.DataFrame,
        symbol: str,
        timeframe: str,
    ) -> str:
        by_key = {(r.side, r.experiment_id): r for r in results}
        sides = sorted({r.side for r in results})

        # Q1: most contributing structure feature (avg importance across sides from all_structure)
        best_feat = None
        best_imp = -1.0
        if not feature_statistics.empty:
            imp_cols = [c for c in feature_statistics.columns if c.startswith("importance_")]
            if imp_cols:
                avg = feature_statistics[imp_cols].mean(axis=1)
                idx = int(avg.idxmax())
                best_feat = str(feature_statistics.loc[idx, "feature"])
                best_imp = float(avg.loc[idx])

        def delta(side: str, exp_id: str) -> float:
            base = by_key.get((side, "baseline"))
            exp = by_key.get((side, exp_id))
            if base is None or exp is None:
                return float("nan")
            return exp.mean_roc_auc - base.mean_roc_auc

        def sep_delta(side: str, exp_id: str) -> float:
            base = by_key.get((side, "baseline"))
            exp = by_key.get((side, exp_id))
            if base is None or exp is None:
                return float("nan")
            return exp.mean_probability_separation - base.mean_probability_separation

        long_help = delta("long", "all_structure")
        short_help = delta("short", "all_structure")
        long_new = delta("long", "new_structure")
        short_new = delta("short", "new_structure")

        sep_long = sep_delta("long", "all_structure")
        sep_short = sep_delta("short", "all_structure")
        sep_improves = (
            (sep_long == sep_long and sep_long > 0)
            or (sep_short == sep_short and sep_short > 0)
        )

        # Production: all_structure or new_structure beats baseline by >= 0.005 on a side
        # and not worse than -0.005 on the other, or clear importance + positive sep
        prod = False
        for side in sides:
            d_all = delta(side, "all_structure")
            d_new = delta(side, "new_structure")
            if (d_all == d_all and d_all >= 0.005) or (d_new == d_new and d_new >= 0.005):
                prod = True

        lines = [
            f"# H4 Structure Enhancement Report — {symbol} {timeframe}",
            "",
            "Sprint 11 — richer H4 structure context (causal join). No volatility features added. "
            "H1 model recipe unchanged (evaluation only).",
            "",
            f"- Sides: `{', '.join(sides)}`",
            f"- Experiments per side: `4` (baseline, swing_quality, new_structure, all_structure)",
            "",
            "## 1. Which structure feature contributes most?",
            "",
        ]
        if best_feat:
            lines.append(
                f"**Answer:** `{best_feat}` (mean LightGBM gain importance across sides = `{_fmt(best_imp, 2)}`)"
            )
            lines.append("")
            top = feature_statistics.copy()
            if "importance_long" in top.columns or "importance_short" in top.columns:
                imp_cols = [c for c in top.columns if c.startswith("importance_")]
                top["_avg"] = top[imp_cols].mean(axis=1)
                top = top.sort_values("_avg", ascending=False).head(8)
                for _, row in top.iterrows():
                    lines.append(f"- `{row['feature']}`: `{_fmt(row['_avg'], 2)}`")
        else:
            lines.append("**Answer:** inconclusive (no importance collected).")

        lines.extend(
            [
                "",
                "## 2. Does structure help Long?",
                "",
                f"**Answer:** `{'yes' if long_help == long_help and long_help > 0 else 'no / not clearly'}` "
                f"(all_structure delta ROC=`{_fmt(long_help)}`, new_structure delta ROC=`{_fmt(long_new)}`)",
                "",
                "## 3. Does structure help Short?",
                "",
                f"**Answer:** `{'yes' if short_help == short_help and short_help > 0 else 'no / not clearly'}` "
                f"(all_structure delta ROC=`{_fmt(short_help)}`, new_structure delta ROC=`{_fmt(short_new)}`)",
                "",
                "## 4. Does structure improve probability separation?",
                "",
                f"**Answer:** `{'yes on at least one side' if sep_improves else 'no clear improvement'}` "
                f"(long delta sep=`{_fmt(sep_long)}`, short delta sep=`{_fmt(sep_short)}`)",
                "",
                "## 5. Should structure become production feature?",
                "",
                f"**Answer:** `{'yes — candidate (prefer selective structure subset)' if prod else 'not yet — keep research-only'}`",
                "",
                "Heuristic: all_structure or new_structure mean WF ROC lift ≥ 0.005 on at least one side.",
                "",
                "## Ablation results",
                "",
                ablation.to_string(index=False) if not ablation.empty else "_(empty)_",
                "",
                "## Charts",
                "",
                "- `charts/roc_comparison.png`",
                "- `charts/delta_roc.png`",
                "- `charts/probability_separation.png`",
                "- `charts/feature_importance.png`",
                "",
            ]
        )
        return "\n".join(lines)
