"""Pair interaction study for meta ablation (research)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.meta_feature_research.services.univariate import mutual_info


def interaction_study(
    panel: pd.DataFrame,
    features: list[str],
    importance: pd.DataFrame,
    *,
    label_col: str = "meta_label",
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Top pairs by mean gain; report MI lift of product vs best single.
    Interaction gain proxy = mean(gain_a)+mean(gain_b) when co-important; SHAP proxy product of mean |shap|.
    """
    if importance.empty or len(features) < 2:
        return pd.DataFrame()

    mean_imp = (
        importance.groupby("feature", as_index=False)[["gain", "shap"]]
        .mean()
        .sort_values("gain", ascending=False)
    )
    top_feats = mean_imp["feature"].head(min(12, len(mean_imp))).tolist()
    top_feats = [f for f in top_feats if f in panel.columns and f in features]

    y = panel[label_col].to_numpy(dtype=int)
    rows = []
    for i, a in enumerate(top_feats):
        for b in top_feats[i + 1 :]:
            xa = panel[a].astype(float).to_numpy()
            xb = panel[b].astype(float).to_numpy()
            mask = np.isfinite(xa) & np.isfinite(xb)
            if mask.sum() < 50 or len(np.unique(y[mask])) < 2:
                continue
            za = (xa[mask] - np.nanmean(xa[mask])) / (np.nanstd(xa[mask]) + 1e-9)
            zb = (xb[mask] - np.nanmean(xb[mask])) / (np.nanstd(xb[mask]) + 1e-9)
            prod = za * zb
            mi_ab = mutual_info(prod, y[mask])
            mi_a = mutual_info(xa[mask], y[mask])
            mi_b = mutual_info(xb[mask], y[mask])
            ga = float(mean_imp.loc[mean_imp["feature"] == a, "gain"].iloc[0])
            gb = float(mean_imp.loc[mean_imp["feature"] == b, "gain"].iloc[0])
            sa = float(mean_imp.loc[mean_imp["feature"] == a, "shap"].iloc[0])
            sb = float(mean_imp.loc[mean_imp["feature"] == b, "shap"].iloc[0])
            base = np.nanmax([mi_a, mi_b])
            rows.append(
                {
                    "feature_a": a,
                    "feature_b": b,
                    "interaction_gain_proxy": ga + gb,
                    "interaction_shap_proxy": (sa if sa == sa else 0) * (sb if sb == sb else 0),
                    "mi_interaction": mi_ab,
                    "mi_lift": float(mi_ab - base) if mi_ab == mi_ab and base == base else float("nan"),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["mi_lift", "interaction_gain_proxy"], ascending=[False, False]
    ).head(top_n)
