"""Interaction & walk-forward stability (model-free)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.meta_feature_research.services.univariate import mutual_info, score_all_features


def interaction_mi(
    panel: pd.DataFrame,
    pairs: list[tuple[str, str]],
    *,
    label_col: str = "meta_label",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Interaction strength via MI of z-scored product vs label, and lift over max single MI.
    """
    y = panel[label_col].to_numpy(dtype=int)
    rows = []
    for a, b in pairs:
        if a not in panel.columns or b not in panel.columns:
            continue
        xa = panel[a].astype(float).to_numpy()
        xb = panel[b].astype(float).to_numpy()
        mask = np.isfinite(xa) & np.isfinite(xb)
        if mask.sum() < 50:
            continue
        za = (xa[mask] - np.nanmean(xa[mask])) / (np.nanstd(xa[mask]) + 1e-9)
        zb = (xb[mask] - np.nanmean(xb[mask])) / (np.nanstd(xb[mask]) + 1e-9)
        prod = za * zb
        mi_ab = mutual_info(prod, y[mask], random_state=random_state)
        mi_a = mutual_info(xa[mask], y[mask], random_state=random_state)
        mi_b = mutual_info(xb[mask], y[mask], random_state=random_state)
        base = np.nanmax([mi_a, mi_b])
        rows.append(
            {
                "feature_a": a,
                "feature_b": b,
                "mi_interaction": mi_ab,
                "mi_a": mi_a,
                "mi_b": mi_b,
                "lift_vs_best_single": float(mi_ab - base) if mi_ab == mi_ab and base == base else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def walkforward_stability(
    panel: pd.DataFrame,
    feature_cols: list[str],
    *,
    label_col: str = "meta_label",
    score_col: str = "mutual_info",
) -> pd.DataFrame:
    """Per-window univariate scores → stability aggregates."""
    window_scores: list[pd.DataFrame] = []
    for window, g in panel.groupby("window", sort=True):
        if g[label_col].nunique() < 2 or len(g) < 40:
            continue
        scored = score_all_features(g, feature_cols, label_col=label_col)
        scored["window"] = window
        window_scores.append(scored)
    if not window_scores:
        return pd.DataFrame()

    all_w = pd.concat(window_scores, ignore_index=True)
    rows = []
    for feat, g in all_w.groupby("feature"):
        vals = g[score_col].astype(float)
        # ranks within each window (higher MI = rank 1)
        ranks = []
        top3 = top5 = 0
        for window, gw in g.groupby("window"):
            # need full window ranking
            pass
        # recompute ranks using full all_w
        for window in g["window"].unique():
            sub = all_w.loc[all_w["window"] == window].copy()
            sub = sub.sort_values(score_col, ascending=False)
            sub["rank"] = np.arange(1, len(sub) + 1)
            r = sub.loc[sub["feature"] == feat, "rank"]
            if r.empty:
                continue
            rk = int(r.iloc[0])
            ranks.append(rk)
            if rk <= 3:
                top3 += 1
            if rk <= 5:
                top5 += 1
        rows.append(
            {
                "feature": feat,
                "n_windows": int(len(vals)),
                "mean_importance": float(vals.mean()),
                "std_importance": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "mean_rank": float(np.mean(ranks)) if ranks else float("nan"),
                "std_rank": float(np.std(ranks, ddof=1)) if len(ranks) > 1 else 0.0,
                "top3_count": int(top3),
                "top5_count": int(top5),
                "stability_score": float(
                    (vals.mean() / (vals.std(ddof=1) + 1e-9)) if len(vals) > 1 else vals.mean()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_importance", ascending=False)
