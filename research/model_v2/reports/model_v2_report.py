"""Model v2 validation reports."""

from __future__ import annotations

from typing import Any

import pandas as pd

from research.structure_selection.entities.selection_result import ExperimentResult


def _fmt(v: Any, digits: int = 4) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        if v != v:
            return "n/a"
        return f"{v:.{digits}f}"
    return str(v)


def build_comparison(results: list[ExperimentResult]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    by_key = {(r.side, r.experiment_id): r for r in results}
    for side in sorted({r.side for r in results}):
        a = by_key.get((side, "A_baseline"))
        b = by_key.get((side, "B_v2_context"))
        if a is None or b is None:
            continue
        a_ok = {w.valid_year: w for w in a.windows if w.status == "ok"}
        b_ok = {w.valid_year: w for w in b.windows if w.status == "ok"}
        years = sorted(set(a_ok) & set(b_ok))
        wins = sum(1 for y in years if b_ok[y].roc_auc > a_ok[y].roc_auc)
        rows.append(
            {
                "side": side,
                "baseline_roc": a.mean_roc_auc,
                "v2_roc": b.mean_roc_auc,
                "delta_roc": b.mean_roc_auc - a.mean_roc_auc,
                "baseline_pr": a.mean_pr_auc,
                "v2_pr": b.mean_pr_auc,
                "delta_pr": b.mean_pr_auc - a.mean_pr_auc,
                "baseline_logloss": a.mean_log_loss,
                "v2_logloss": b.mean_log_loss,
                "delta_logloss": b.mean_log_loss - a.mean_log_loss,
                "baseline_ece": a.mean_calibration_error,
                "v2_ece": b.mean_calibration_error,
                "delta_ece": b.mean_calibration_error - a.mean_calibration_error,
                "baseline_sep": a.mean_probability_separation,
                "v2_sep": b.mean_probability_separation,
                "delta_sep": b.mean_probability_separation - a.mean_probability_separation,
                "baseline_wf_stability": a.wf_stability,
                "v2_wf_stability": b.wf_stability,
                "windows_v2_beats_baseline": wins,
                "windows_compared": len(years),
            }
        )
    return pd.DataFrame(rows)


def build_feature_importance(results: list[ExperimentResult]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for r in results:
        if r.experiment_id != "B_v2_context":
            continue
        # Aggregate gain/split/shap across ok windows
        from collections import defaultdict

        import numpy as np

        buckets: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: {"gain": [], "split": [], "shap": []}
        )
        for w in r.windows:
            if w.status != "ok":
                continue
            for k, v in w.gain_importance.items():
                buckets[k]["gain"].append(v)
            for k, v in w.split_importance.items():
                buckets[k]["split"].append(v)
            for k, v in w.shap_importance.items():
                buckets[k]["shap"].append(v)
        for feat, b in buckets.items():
            rows.append(
                {
                    "side": r.side,
                    "feature": feat,
                    "mean_gain": float(np.mean(b["gain"])) if b["gain"] else 0.0,
                    "mean_split": float(np.mean(b["split"])) if b["split"] else 0.0,
                    "mean_shap": float(np.mean(b["shap"])) if b["shap"] else 0.0,
                    "n_windows": max(len(b["gain"]), len(b["shap"]), 0),
                }
            )
    return pd.DataFrame(rows)


class ModelV2ReportBuilder:
    def build_side_report(
        self,
        *,
        side: str,
        results: list[ExperimentResult],
        comparison: pd.DataFrame,
        feature_importance: pd.DataFrame,
        feature_stability: pd.DataFrame,
        context_features: tuple[str, ...],
        symbol: str,
        timeframe: str,
    ) -> str:
        side_res = [r for r in results if r.side == side]
        by_id = {r.experiment_id: r for r in side_res}
        a = by_id.get("A_baseline")
        b = by_id.get("B_v2_context")
        row = None
        if not comparison.empty and side in set(comparison["side"]):
            row = comparison.loc[comparison["side"] == side].iloc[0]

        delta = float(row["delta_roc"]) if row is not None else float("nan")
        wins = int(row["windows_v2_beats_baseline"]) if row is not None else 0
        nwin = int(row["windows_compared"]) if row is not None else 0
        stable_improve = bool(
            delta == delta and delta > 0 and nwin > 0 and wins >= (nwin + 1) // 2
        )
        # Consistent across windows: majority wins AND positive mean
        consistent = bool(delta == delta and delta > 0 and wins >= max(nwin - 1, 1))

        # Dominating features
        imp = feature_importance.loc[feature_importance["side"] == side].copy()
        top_feats: list[str] = []
        if not imp.empty:
            imp = imp.sort_values("mean_gain", ascending=False)
            top_feats = [str(x) for x in imp["feature"].head(5).tolist()]

        # Calibration gate: lift without ECE blow-up
        ece_ok = True
        if row is not None:
            ece_ok = float(row["delta_ece"]) < 0.02
        proceed = bool(stable_improve and ece_ok and delta >= 0.002)

        q_improve = (
            "Does Long structure context provide stable improvement?"
            if side == "long"
            else "Does Short swing quality provide stable improvement?"
        )

        lines = [
            f"# {side.title()} Model v2 Report — {symbol} {timeframe}",
            "",
            "Sprint 13 — validate H1 + selected H4 structure context (no threshold tuning, no execution).",
            "",
            f"- Context features: `{', '.join(context_features)}`",
            "",
            f"## 1. {q_improve}" if side == "long" else "## 1. (see Long report for Q1)",
            "",
        ]
        if side == "long":
            lines.append(
                f"**Answer:** `{'yes' if stable_improve else 'no / not clearly'}` "
                f"(delta ROC=`{_fmt(delta)}`, WF wins `{wins}/{nwin}`)"
            )
        else:
            lines.append(
                f"Long-side question is answered in `long_model_v2_report.md`. "
                f"This short run delta ROC=`{_fmt(delta)}`, wins `{wins}/{nwin}`."
            )

        lines.extend(
            [
                "",
                f"## 2. {'Does Short swing quality provide stable improvement?' if side == 'short' else '(see Short report for Q2)'}",
                "",
            ]
        )
        if side == "short":
            lines.append(
                f"**Answer:** `{'yes' if stable_improve else 'no / not clearly'}` "
                f"(delta ROC=`{_fmt(delta)}`, WF wins `{wins}/{nwin}`)"
            )
        else:
            lines.append("Short-side question is answered in `short_model_v2_report.md`.")

        lines.extend(
            [
                "",
                "## 3. Which features dominate?",
                "",
                f"**Answer:** `{top_feats if top_feats else 'n/a'}`",
                "",
            ]
        )
        if not imp.empty:
            for _, r in imp.head(8).iterrows():
                lines.append(
                    f"- `{r['feature']}`: gain=`{_fmt(r['mean_gain'], 2)}` "
                    f"split=`{_fmt(r['mean_split'], 2)}` shap=`{_fmt(r['mean_shap'], 4)}`"
                )

        lines.extend(
            [
                "",
                "## 4. Is the improvement consistent across walk forward windows?",
                "",
                f"**Answer:** `{'yes' if consistent else 'mixed / no'}` "
                f"(wins `{wins}/{nwin}`)",
                "",
            ]
        )
        if a and b:
            for w in b.windows:
                if w.status != "ok":
                    continue
                aw = next((x for x in a.windows if x.valid_year == w.valid_year and x.status == "ok"), None)
                if aw is None:
                    continue
                d = w.roc_auc - aw.roc_auc
                lines.append(
                    f"- {w.valid_year}: baseline=`{_fmt(aw.roc_auc)}` v2=`{_fmt(w.roc_auc)}` "
                    f"delta=`{_fmt(d)}`"
                )

        lines.extend(
            [
                "",
                "## 5. Should this model proceed to calibration?",
                "",
                f"**Answer:** `{'yes — candidate for calibration' if proceed else 'not yet'}`",
                "",
                "Heuristic: positive mean ROC lift, majority WF wins, and ECE delta < 0.02.",
                "",
                "## Metrics summary",
                "",
            ]
        )
        if a and b:
            lines.append(
                f"| metric | baseline | v2 | delta |\n|---|---|---|---|\n"
                f"| ROC | {_fmt(a.mean_roc_auc)} | {_fmt(b.mean_roc_auc)} | {_fmt(delta)} |\n"
                f"| PR | {_fmt(a.mean_pr_auc)} | {_fmt(b.mean_pr_auc)} | {_fmt(b.mean_pr_auc - a.mean_pr_auc)} |\n"
                f"| LogLoss | {_fmt(a.mean_log_loss)} | {_fmt(b.mean_log_loss)} | {_fmt(b.mean_log_loss - a.mean_log_loss)} |\n"
                f"| ECE | {_fmt(a.mean_calibration_error)} | {_fmt(b.mean_calibration_error)} | {_fmt(b.mean_calibration_error - a.mean_calibration_error)} |\n"
                f"| ProbSep | {_fmt(a.mean_probability_separation)} | {_fmt(b.mean_probability_separation)} | {_fmt(b.mean_probability_separation - a.mean_probability_separation)} |\n"
                f"| WF stability | {_fmt(a.wf_stability)} | {_fmt(b.wf_stability)} | {_fmt(b.wf_stability - a.wf_stability)} |"
            )

        stab = feature_stability.loc[
            (feature_stability["side"] == side)
            & (feature_stability["experiment_id"] == "B_v2_context")
        ] if not feature_stability.empty else pd.DataFrame()
        lines.extend(["", "## Feature stability (B v2)", ""])
        if stab.empty:
            lines.append("_(empty)_")
        else:
            lines.append(stab.to_string(index=False))

        lines.extend(
            [
                "",
                "## Charts",
                "",
                f"- `charts/{side}_roc_windows.png`",
                f"- `charts/{side}_feature_importance.png`",
                "- `charts/comparison_delta.png`",
                "",
            ]
        )
        return "\n".join(lines)
