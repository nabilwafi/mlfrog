"""Markdown report for meta feature ablation."""

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


def summarize_stages(window_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stage, g in window_metrics.groupby("stage_id", sort=False):
        rows.append(
            {
                "stage_id": stage,
                "n_features": int(g["n_features"].iloc[0]),
                "roc_mean": float(g["roc_auc"].mean()),
                "roc_std": float(g["roc_auc"].std(ddof=1)) if len(g) > 1 else 0.0,
                "roc_best": float(g["roc_auc"].max()),
                "roc_worst": float(g["roc_auc"].min()),
                "expectancy_mean": float(g["expectancy"].mean()),
                "expectancy_std": float(g["expectancy"].std(ddof=1)) if len(g) > 1 else 0.0,
                "pf_mean": float(g["profit_factor"].replace([float("inf")], pd.NA).dropna().mean())
                if g["profit_factor"].notna().any()
                else float("nan"),
                "n_trades_mean": float(g["n_trades"].mean()),
                "annual_return_sum": float(g["annual_return"].sum()),
                "f1_mean": float(g["f1"].mean()),
                "ece_mean": float(g["ece"].mean()),
            }
        )
    return pd.DataFrame(rows)


def build_answers(
    *,
    stage_summary: pd.DataFrame,
    removal: pd.DataFrame,
    stability: pd.DataFrame,
    groups_delta: dict[str, Any],
) -> dict[str, Any]:
    if stage_summary.empty:
        return {
            "group_most": "n/a",
            "session_helps": "n/a",
            "entry_helps": "n/a",
            "h1_reduce": "n/a",
            "h4_reduce": "n/a",
            "minimal_set": [],
            "minimal_stage": "n/a",
            "recommended": [],
        }

    # Robustness-first ranking
    best = stage_summary.sort_values(
        ["annual_return_sum", "expectancy_mean", "roc_mean"],
        ascending=[False, False, False],
    ).iloc[0]

    order = list(stage_summary["stage_id"])
    deltas: dict[str, float] = {}
    prev_e = None
    for sid in order:
        e = float(stage_summary.loc[stage_summary["stage_id"] == sid, "expectancy_mean"].iloc[0])
        deltas[sid] = e if prev_e is None else e - prev_e
        prev_e = e

    add_map = {
        "A_primary": "A_primary",
        "A_plus_H4": "C_h4",
        "A_H4_H1": "B_h1",
        "A_H4_H1_Session": "E_session",
        "A_H4_H1_Session_Entry": "F_entry_diag",
    }
    inc = {add_map.get(k, k): v for k, v in deltas.items() if k != "A_primary"}
    group_most = max(inc, key=inc.get) if inc else "A_primary"
    ar_best_stage = str(
        stage_summary.sort_values("annual_return_sum", ascending=False).iloc[0]["stage_id"]
    )

    def _stage_e(sid: str) -> float:
        if sid not in set(stage_summary["stage_id"]):
            return float("nan")
        return float(stage_summary.loc[stage_summary["stage_id"] == sid, "expectancy_mean"].iloc[0])

    def _stage_ar(sid: str) -> float:
        if sid not in set(stage_summary["stage_id"]):
            return float("nan")
        return float(stage_summary.loc[stage_summary["stage_id"] == sid, "annual_return_sum"].iloc[0])

    def _stage_roc(sid: str) -> float:
        if sid not in set(stage_summary["stage_id"]):
            return float("nan")
        return float(stage_summary.loc[stage_summary["stage_id"] == sid, "roc_mean"].iloc[0])

    e_a, e_h4, e_h1, e_sess, e_entry = (
        _stage_e("A_primary"),
        _stage_e("A_plus_H4"),
        _stage_e("A_H4_H1"),
        _stage_e("A_H4_H1_Session"),
        _stage_e("A_H4_H1_Session_Entry"),
    )
    ar_sess, ar_entry = _stage_ar("A_H4_H1_Session"), _stage_ar("A_H4_H1_Session_Entry")
    roc_sess, roc_entry = _stage_roc("A_H4_H1_Session"), _stage_roc("A_H4_H1_Session_Entry")

    session_helps = "yes" if e_sess == e_sess and e_h1 == e_h1 and e_sess > e_h1 else "no"
    if (
        e_entry == e_entry
        and e_sess == e_sess
        and e_entry > e_sess
        and ar_entry == ar_entry
        and ar_sess == ar_sess
        and ar_entry < ar_sess
    ):
        entry_helps = "mixed — lifts E/ROC but lowers annual_return_sum"
    elif e_entry == e_entry and e_sess == e_sess and e_entry > e_sess:
        entry_helps = "yes"
    else:
        entry_helps = "no / flat"

    metric = "annual_return_sum"
    target = float(best[metric]) * 0.95
    minimal_stage = str(best["stage_id"])
    for _, row in stage_summary.iterrows():
        if float(row[metric]) >= target:
            minimal_stage = str(row["stage_id"])
            break

    remove_set = set(removal["feature"].tolist()) if not removal.empty else set()
    return {
        "group_most": group_most,
        "ar_best_stage": ar_best_stage,
        "deltas": deltas,
        "session_helps": session_helps,
        "session_delta_E": float(e_sess - e_h1) if e_sess == e_sess and e_h1 == e_h1 else float("nan"),
        "entry_helps": entry_helps,
        "entry_delta_E": float(e_entry - e_sess) if e_entry == e_entry and e_sess == e_sess else float("nan"),
        "entry_delta_AR": float(ar_entry - ar_sess) if ar_entry == ar_entry and ar_sess == ar_sess else float("nan"),
        "entry_delta_ROC": float(roc_entry - roc_sess) if roc_entry == roc_entry and roc_sess == roc_sess else float("nan"),
        "h4_delta_E": float(e_h4 - e_a) if e_h4 == e_h4 and e_a == e_a else float("nan"),
        "h1_delta_E": float(e_h1 - e_h4) if e_h1 == e_h1 and e_h4 == e_h4 else float("nan"),
        "minimal_stage": minimal_stage,
        "best_stage": str(best["stage_id"]),
        "metric_used": metric,
        "target_95": target,
        "remove_set": sorted(remove_set),
        "e_by_stage": {
            "A": e_a,
            "H4": e_h4,
            "H1": e_h1,
            "Session": e_sess,
            "Entry": e_entry,
        },
    }


def build_report(
    *,
    symbol: str,
    timeframe: str,
    stage_summary: pd.DataFrame,
    window_metrics: pd.DataFrame,
    removal: pd.DataFrame,
    vif: pd.DataFrame,
    stability: pd.DataFrame,
    interactions: pd.DataFrame,
    answers: dict[str, Any],
    recommended: list[str],
    stage_features: dict[str, list[str]],
) -> str:
    lines = [
        f"# Meta Feature Ablation Report - {symbol} {timeframe}",
        "",
        "Sprint 19 - research only. Ablation LightGBM uses **fixed** params; not the final meta model.",
        "No threshold optimization (fixed 0.5). No hyperparameter search.",
        "",
        "Protocol: leave-one-year-out on top-3% meta candidates; train years != val year.",
        "",
        "## Answers",
        "",
        f"1. **Which feature group contributes most?** `{answers.get('group_most')}`",
        f"   - Stage expectancy path: `{answers.get('e_by_stage')}`",
        "",
        f"2. **Does Session improve trading?** `{answers.get('session_helps')}` "
        f"(dE=`{_fmt(answers.get('session_delta_E'))}`)",
        "",
        f"3. **Does Entry Diagnostic improve Meta?** `{answers.get('entry_helps')}` "
        f"(dE=`{_fmt(answers.get('entry_delta_E'))}`, dAR=`{_fmt(answers.get('entry_delta_AR'))}`, "
        f"dROC=`{_fmt(answers.get('entry_delta_ROC'))}`)",
        "",
        f"4. **Can H1 features be reduced?** `{answers.get('h1_reduce', 'see recommendations')}`",
        "",
        f"5. **Can H4 features be reduced?** `{answers.get('h4_reduce', 'see recommendations')}`",
        "",
        f"6. **Smallest set >=95% of best `{answers.get('metric_used')}`?** "
        f"stage=`{answers.get('minimal_stage')}` (best=`{answers.get('best_stage')}`)",
        "",
        "7. **FINAL recommended feature list**",
        "",
    ]
    for i, f in enumerate(recommended, 1):
        lines.append(f"   {i}. `{f}`")

    lines.extend(["", "## Ablation stage summary", ""])
    if stage_summary.empty:
        lines.append("_(empty)_")
    else:
        lines.append(stage_summary.to_string(index=False))

    lines.extend(["", "## Walk-forward metrics (all stages)", ""])
    if window_metrics.empty:
        lines.append("_(empty)_")
    else:
        show = window_metrics[
            [
                c
                for c in (
                    "stage_id",
                    "valid_year",
                    "roc_auc",
                    "pr_auc",
                    "f1",
                    "ece",
                    "n_trades",
                    "win_rate",
                    "expectancy",
                    "profit_factor",
                    "max_drawdown",
                    "sharpe",
                    "annual_return",
                )
                if c in window_metrics.columns
            ]
        ]
        lines.append(show.to_string(index=False))

    lines.extend(["", "## Correlation removal candidates (Pearson>0.95 or VIF>10)", ""])
    if removal.empty:
        lines.append("_(none)_")
    else:
        lines.append(removal.to_string(index=False))

    lines.extend(["", "## VIF (top)", ""])
    if vif.empty:
        lines.append("_(empty)_")
    else:
        lines.append(vif.head(25).to_string(index=False))

    lines.extend(["", "## Importance stability (full stage)", ""])
    if stability.empty:
        lines.append("_(empty)_")
    else:
        lines.append(stability.head(25).to_string(index=False))

    lines.extend(["", "## Top interactions", ""])
    if interactions.empty:
        lines.append("_(empty)_")
    else:
        lines.append(interactions.head(20).to_string(index=False))

    lines.extend(["", "## Stage feature counts", ""])
    for sid, feats in stage_features.items():
        lines.append(f"- `{sid}`: {len(feats)} features")

    lines.extend(
        [
            "",
            "## Charts",
            "",
            "- `charts/feature_correlation_heatmap.png`",
            "- `charts/ablation_curve.png`",
            "- `charts/metric_progression.png`",
            "- `charts/wf_metric_progression.png`",
            "- `charts/importance_stability.png`",
            "- `charts/interaction_matrix.png`",
            "- `charts/annual_return_progression.png`",
            "",
            "Also: `vif_table.csv`",
            "",
        ]
    )
    return "\n".join(lines)
