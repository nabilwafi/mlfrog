"""Selection report answering Sprint-12 research questions."""

from __future__ import annotations

from typing import Any

import pandas as pd

from research.structure_selection.entities.selection_result import ExperimentResult
from research.structure_selection.services.experiment_catalog import DISTANCE_EQ, SWING_QUALITY


def _fmt(v: Any, digits: int = 4) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        if v != v:
            return "n/a"
        return f"{v:.{digits}f}"
    return str(v)


class SelectionReportBuilder:
    def build(
        self,
        *,
        results: list[ExperimentResult],
        experiment_results: pd.DataFrame,
        feature_stability: pd.DataFrame,
        regime_results: pd.DataFrame,
        top_structure: tuple[str, ...],
        symbol: str,
        timeframe: str,
    ) -> str:
        by_key = {(r.side, r.experiment_id): r for r in results}
        sides = sorted({r.side for r in results})

        def delta(side: str, exp_id: str) -> float:
            base = by_key.get((side, "A_baseline"))
            exp = by_key.get((side, exp_id))
            if base is None or exp is None:
                return float("nan")
            return exp.mean_roc_auc - base.mean_roc_auc

        # Q1 robust: high stability_score + top3_count across D/E
        robust: list[str] = []
        if not feature_stability.empty:
            focus = feature_stability.loc[
                feature_stability["experiment_id"].isin(["D_swing_plus_eq", "E_top_structure"])
            ]
            if not focus.empty:
                agg = (
                    focus.groupby("feature")
                    .agg(
                        mean_stability=("stability_score", "mean"),
                        mean_top3=("top3_count", "mean"),
                        mean_gain=("mean_gain", "mean"),
                    )
                    .reset_index()
                )
                agg = agg.sort_values(
                    ["mean_stability", "mean_top3", "mean_gain"], ascending=False
                )
                robust = [
                    str(r.feature)
                    for r in agg.itertuples()
                    if r.mean_stability >= 0.5 and r.mean_top3 >= 1.0
                ][:5]
                if not robust:
                    robust = [str(x) for x in agg["feature"].head(3).tolist()]

        # Q2 distance_eq survives?
        dist_deltas = {s: delta(s, "C_distance_eq") for s in sides}
        dist_ok = any(d == d and d > 0 for d in dist_deltas.values())
        dist_stable = False
        if not feature_stability.empty:
            dist_rows = feature_stability.loc[
                feature_stability["feature"] == DISTANCE_EQ
            ]
            if not dist_rows.empty:
                dist_stable = bool(dist_rows["stability_score"].mean() >= 0.55)

        # Q3 swing_quality useful?
        swing_deltas = {s: delta(s, "B_swing_quality") for s in sides}
        swing_useful = any(d == d and d > 0 for d in swing_deltas.values())

        # Q4 long vs short different?
        long_best = short_best = None
        for side in sides:
            cands = [
                (eid, delta(side, eid))
                for eid in (
                    "B_swing_quality",
                    "C_distance_eq",
                    "D_swing_plus_eq",
                    "E_top_structure",
                )
            ]
            cands = [(e, d) for e, d in cands if d == d]
            if cands:
                best = max(cands, key=lambda x: x[1])
                if side == "long":
                    long_best = best
                else:
                    short_best = best
        different = bool(
            long_best and short_best and long_best[0] != short_best[0]
        )

        # Q5 minimal set
        # Prefer B if both sides positive; else D if better; else swing only
        minimal: list[str] = []
        if swing_useful and all(
            (swing_deltas.get(s, float("nan")) == swing_deltas.get(s, float("nan")))
            and swing_deltas.get(s, -1) >= -0.002
            for s in sides
        ):
            minimal = [SWING_QUALITY]
        # Upgrade to D if D beats B on average
        d_avg = sum(delta(s, "D_swing_plus_eq") for s in sides) / max(len(sides), 1)
        b_avg = sum(delta(s, "B_swing_quality") for s in sides) / max(len(sides), 1)
        if d_avg == d_avg and b_avg == b_avg and d_avg > b_avg + 0.002 and dist_ok:
            minimal = [SWING_QUALITY, DISTANCE_EQ]
        elif not minimal and swing_useful:
            minimal = [SWING_QUALITY]
        elif not minimal and dist_ok:
            minimal = [DISTANCE_EQ]

        # Q6 enter v2?
        enter_v2 = bool(minimal) and (
            (b_avg == b_avg and b_avg >= 0.002)
            or (d_avg == d_avg and d_avg >= 0.002)
        )

        lines = [
            f"# Structure Selection Report — {symbol} {timeframe}",
            "",
            "Sprint 12 — select robust H4 structure features (no new features).",
            "",
            f"- Sides: `{', '.join(sides)}`",
            f"- Experiment E top features: `{', '.join(top_structure)}`",
            "",
            "## 1. Which structure features are robust?",
            "",
            f"**Answer:** `{robust if robust else 'none clearly stable'}`",
            "",
            "Based on cross-window rank stability (gain) in experiments D/E.",
            "",
            "## 2. Does distance_from_equilibrium survive validation?",
            "",
            f"**Answer:** `{'partially / yes on lift' if dist_ok else 'no clear ROC lift'}`"
            f"{' and rank-stable' if dist_stable else ' (rank stability weak)' if dist_ok else ''}",
            "",
        ]
        for s, d in dist_deltas.items():
            lines.append(f"- `{s}` experiment C delta ROC=`{_fmt(d)}`")

        lines.extend(
            [
                "",
                "## 3. Does swing_quality remain useful?",
                "",
                f"**Answer:** `{'yes' if swing_useful else 'no'}`",
                "",
            ]
        )
        for s, d in swing_deltas.items():
            lines.append(f"- `{s}` experiment B delta ROC=`{_fmt(d)}`")

        lines.extend(
            [
                "",
                "## 4. Does Long and Short require different structure features?",
                "",
                f"**Answer:** `{'yes' if different else 'no — same preference'}`",
                "",
            ]
        )
        if long_best:
            lines.append(f"- long best: `{long_best[0]}` (delta=`{_fmt(long_best[1])}`)")
        if short_best:
            lines.append(f"- short best: `{short_best[0]}` (delta=`{_fmt(short_best[1])}`)")

        lines.extend(
            [
                "",
                "## 5. What is the minimal production context feature set?",
                "",
                f"**Answer:** `{minimal if minimal else 'empty — do not add structure yet'}`",
                "",
                "## 6. Should structure features enter H1 dataset v2?",
                "",
                f"**Answer:** `{'yes — add minimal set only' if enter_v2 else 'not yet / research-only'}`",
                "",
                "Heuristic: minimal set non-empty and mean WF ROC lift >= 0.002 vs baseline.",
                "",
                "## Experiment results",
                "",
                experiment_results.to_string(index=False)
                if not experiment_results.empty
                else "_(empty)_",
                "",
                "## Regime notes",
                "",
            ]
        )
        if regime_results.empty:
            lines.append("_(regime analysis unavailable or empty)_")
        else:
            summary = (
                regime_results.groupby(["side", "regime_family", "regime_bin"])["roc_auc"]
                .mean()
                .reset_index()
            )
            lines.append(summary.to_string(index=False))

        lines.extend(
            [
                "",
                "## Charts",
                "",
                "- `charts/experiment_roc.png`",
                "- `charts/delta_vs_baseline.png`",
                "- `charts/feature_stability.png`",
                "- `charts/regime_roc.png`",
                "",
            ]
        )
        return "\n".join(lines)
