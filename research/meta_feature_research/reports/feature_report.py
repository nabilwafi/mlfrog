"""Markdown report for meta feature research."""

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
    groups: dict[str, list[str]],
    univariate: pd.DataFrame,
    group_importance: pd.DataFrame,
    stability: pd.DataFrame,
    interactions: pd.DataFrame,
    session_table: pd.DataFrame,
    answers: dict[str, Any],
) -> str:
    lines = [
        f"# Meta Feature Research Report — {symbol} {timeframe}",
        "",
        "Sprint 18 — research only. **No meta-model training. No LightGBM. No threshold tuning.**",
        "",
        "Candidate set: Long/Short **top 3%** OOF by raw probability (per WF window).",
        "Target: `meta_label = 1` iff net return after cost > 0.",
        "",
        "Leakage excluded: realized/net return, MAE/MFE, holding_bars, exit_reason, any future path stats.",
        "Entry-known diagnostics (ATR stop, RR, TP/SL distance) allowed.",
        "",
        "SHAP note: full SHAP needs a fitted model — report uses **shap_proxy = sign(ρ)·|ρ|·MI**.",
        "Permutation importance: model-free **ΔMI** vs shuffled feature.",
        "",
        "## Executive recommendation",
        "",
        f"**`{answers.get('proceed', 'n/a')}`**",
        "",
        str(answers.get("proceed_rationale", "")),
        "",
        "## 1. Highest individual predictive power",
        "",
    ]
    top = univariate.head(15)
    if top.empty:
        lines.append("_(empty)_")
    else:
        lines.append(
            top[
                [
                    c
                    for c in (
                        "feature",
                        "group",
                        "mutual_info",
                        "permutation_mi",
                        "spearman",
                        "information_value",
                        "ks_statistic",
                        "shap_proxy",
                    )
                    if c in top.columns
                ]
            ].to_string(index=False)
        )

    lines.extend(["", "## 2. Feature group contribution", ""])
    if group_importance.empty:
        lines.append("_(empty)_")
    else:
        lines.append(group_importance.to_string(index=False))

    lines.extend(
        [
            "",
            "## 3. Does M15 add incremental information beyond H1 + H4?",
            "",
            f"**Answer:** `{answers.get('m15_adds', 'n/a')}`",
            "",
            str(answers.get("m15_rationale", "")),
            "",
            "## 4. Does Session matter?",
            "",
            f"**Answer:** `{answers.get('session_matters', 'n/a')}`",
            "",
            str(answers.get("session_rationale", "")),
            "",
        ]
    )
    if not session_table.empty:
        lines.extend(["### Session win-rate / expectancy", "", session_table.to_string(index=False), ""])

    lines.extend(
        [
            "",
            "## 5. Features stable across Walk Forward",
            "",
        ]
    )
    if stability.empty:
        lines.append("_(empty)_")
    else:
        stab = stability.head(20)
        lines.append(stab.to_string(index=False))

    lines.extend(["", "## 6. Top 20 recommended Meta Features", ""])
    rec = answers.get("top20", [])
    if not rec:
        lines.append("_(empty)_")
    else:
        for i, name in enumerate(rec, 1):
            lines.append(f"{i}. `{name}`")

    lines.extend(["", "## 7. Top feature interactions", ""])
    if interactions.empty:
        lines.append("_(empty)_")
    else:
        lines.append(interactions.head(15).to_string(index=False))

    lines.extend(
        [
            "",
            "## 8. Should Meta Model proceed?",
            "",
            f"**Answer:** `{answers.get('proceed', 'n/a')}`",
            "",
            str(answers.get("proceed_rationale", "")),
            "",
            "## Feature inventory by group",
            "",
        ]
    )
    for g, cols in groups.items():
        lines.append(f"- **{g}:** {len(cols)} features")

    lines.extend(
        [
            "",
            "## Charts",
            "",
            "- `charts/feature_importance.png`",
            "- `charts/feature_group_importance.png`",
            "- `charts/feature_stability.png`",
            "- `charts/interaction_heatmap.png`",
            "- `charts/walkforward_feature_stability.png`",
            "- `charts/session_analysis.png`",
            "- `charts/m15_vs_h4_gain.png`",
            "",
        ]
    )
    return "\n".join(lines)


def build_answers(
    *,
    univariate: pd.DataFrame,
    group_importance: pd.DataFrame,
    stability: pd.DataFrame,
    interactions: pd.DataFrame,
    session_table: pd.DataFrame,
    groups: dict[str, list[str]],
) -> dict[str, Any]:
    u = univariate.copy()
    if u.empty:
        return {
            "proceed": "Collect more features first",
            "proceed_rationale": "No scorable features.",
            "top20": [],
            "m15_adds": "n/a",
            "session_matters": "n/a",
        }

    # Top 20: blend MI and stability
    if not stability.empty:
        merged = u.merge(
            stability[["feature", "mean_importance", "std_rank", "top5_count", "stability_score"]],
            on="feature",
            how="left",
            suffixes=("", "_wf"),
        )
        merged["rec_score"] = (
            0.5 * merged["mutual_info"].fillna(0)
            + 0.3 * merged["mean_importance"].fillna(0)
            + 0.2 * (merged["top5_count"].fillna(0) / 4.0)
        )
    else:
        merged = u.copy()
        merged["rec_score"] = merged["mutual_info"].fillna(0)
    top20 = merged.sort_values("rec_score", ascending=False)["feature"].head(20).tolist()

    # M15 vs H4
    m15_cols = set(groups.get("D_m15", []))
    h4_cols = set(groups.get("C_h4", []))
    h1_cols = set(groups.get("B_h1", []))
    mi_m15 = float(u.loc[u["feature"].isin(m15_cols), "mutual_info"].mean()) if m15_cols else float("nan")
    mi_h4 = float(u.loc[u["feature"].isin(h4_cols), "mutual_info"].mean()) if h4_cols else float("nan")
    mi_h1 = float(u.loc[u["feature"].isin(h1_cols), "mutual_info"].mean()) if h1_cols else float("nan")
    top_m15 = u.loc[u["feature"].isin(m15_cols)].head(3)["feature"].tolist()
    top_h4 = u.loc[u["feature"].isin(h4_cols)].head(3)["feature"].tolist()
    best_overall_mi = float(u["mutual_info"].max())
    # Incremental: any M15 in top20 or mean MI comparable to H4
    m15_in_top20 = [f for f in top20 if f in m15_cols]
    m15_adds = "yes" if m15_in_top20 or (mi_m15 == mi_m15 and mi_h4 == mi_h4 and mi_m15 >= 0.7 * mi_h4) else "weak / limited"
    m15_rationale = (
        f"mean MI M15=`{_fmt(mi_m15)}` H4=`{_fmt(mi_h4)}` H1=`{_fmt(mi_h1)}`; "
        f"M15 in top20=`{m15_in_top20}`; top M15=`{top_m15}`; top H4=`{top_h4}`."
    )

    # Session
    sess_feats = set(groups.get("E_session", []))
    sess_mi = u.loc[u["feature"].isin(sess_feats)]
    max_sess = float(sess_mi["mutual_info"].max()) if not sess_mi.empty else float("nan")
    # win rate spread across hour buckets if present
    wr_spread = float("nan")
    if not session_table.empty and "win_rate" in session_table.columns:
        wr_spread = float(session_table["win_rate"].max() - session_table["win_rate"].min())
    session_matters = (
        "yes"
        if (max_sess == max_sess and max_sess > 0.005)
        or (wr_spread == wr_spread and wr_spread > 0.08)
        else "marginal"
    )
    session_rationale = (
        f"max session feature MI=`{_fmt(max_sess)}` vs best overall MI=`{_fmt(best_overall_mi)}`; "
        f"categorical win-rate spread=`{_fmt(wr_spread)}`."
    )

    # Proceed decision
    n_strong = int((u["mutual_info"] > 0.01).sum()) if "mutual_info" in u.columns else 0
    n_stable = (
        int(((stability["top5_count"] >= 2) & (stability["mean_importance"] > 0.005)).sum())
        if not stability.empty
        else 0
    )
    if n_strong >= 5 and n_stable >= 3:
        proceed = "Proceed"
        proceed_rationale = (
            f"Found `{n_strong}` features with MI>0.01 and `{n_stable}` WF-stable signals. "
            f"Recommended starter set size=20. Caveat: meta target is still noisy; keep feature set lean."
        )
    elif n_strong >= 2:
        proceed = "Proceed"
        proceed_rationale = (
            f"Signal exists but modest (`{n_strong}` MI>0.01). Proceed with a small meta model using "
            f"top recommended features; do not expand feature engineering aggressively first."
        )
    else:
        proceed = "Collect more features first"
        proceed_rationale = (
            "Predictive power vs meta_label is weak under current candidate definition. "
            "Prefer refining candidate gate / labels before training a meta classifier."
        )

    return {
        "proceed": proceed,
        "proceed_rationale": proceed_rationale,
        "top20": top20,
        "m15_adds": m15_adds,
        "m15_rationale": m15_rationale,
        "session_matters": session_matters,
        "session_rationale": session_rationale,
        "mi_m15": mi_m15,
        "mi_h4": mi_h4,
        "mi_h1": mi_h1,
    }
