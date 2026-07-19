"""Correlation / VIF analysis for meta features."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


def pearson_spearman(frame: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = frame[features].astype(float)
    return x.corr(method="pearson"), x.corr(method="spearman")


def variance_inflation_factors(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Classic VIF via OLS R^2; drop constant/near-constant cols."""
    from numpy.linalg import lstsq

    rows = []
    x = frame[features].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    feats = [f for f in features if x[f].std(ddof=0) > 1e-12]
    for i, target in enumerate(feats):
        others = [f for f in feats if f != target]
        if not others:
            rows.append({"feature": target, "vif": 1.0})
            continue
        y = x[target].to_numpy()
        X = np.column_stack([np.ones(len(x)), x[others].to_numpy()])
        try:
            beta, _, _, _ = lstsq(X, y, rcond=None)
            pred = X @ beta
            ss_res = float(np.sum((y - pred) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            r2 = min(max(r2, 0.0), 0.999999)
            vif = 1.0 / (1.0 - r2)
        except Exception:
            vif = float("nan")
        rows.append({"feature": target, "vif": float(vif)})
    return pd.DataFrame(rows).sort_values("vif", ascending=False)


def correlation_clusters(
    pearson: pd.DataFrame, *, threshold: float = 0.95
) -> pd.DataFrame:
    """Hierarchical clusters from |corr| distance; pairs above threshold flagged."""
    cols = list(pearson.columns)
    if len(cols) < 2:
        return pd.DataFrame(columns=["feature", "cluster", "pair_high_corr"])
    corr = np.array(pearson.abs().fillna(0.0).to_numpy(), dtype=float, copy=True)
    np.fill_diagonal(corr, 1.0)
    dist = np.clip(1.0 - corr, 0.0, 1.0)
    np.fill_diagonal(dist, 0.0)
    # Ensure symmetry
    dist = (dist + dist.T) / 2.0
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")
    labels = fcluster(Z, t=1.0 - threshold, criterion="distance")
    # High-corr partners
    rows = []
    for i, feat in enumerate(cols):
        partners = [
            cols[j]
            for j in range(len(cols))
            if i != j and abs(float(pearson.iloc[i, j])) > threshold
        ]
        rows.append(
            {
                "feature": feat,
                "cluster": int(labels[i]),
                "pair_high_corr": ",".join(partners) if partners else "",
            }
        )
    return pd.DataFrame(rows)


def removal_recommendations(
    pearson: pd.DataFrame,
    vif: pd.DataFrame,
    *,
    pearson_thr: float = 0.95,
    vif_thr: float = 10.0,
) -> pd.DataFrame:
    """Greedy: for each high-corr pair keep lower-VIF feature; also flag VIF>thr."""
    remove: dict[str, str] = {}
    vif_map = {r.feature: float(r.vif) for r in vif.itertuples()}
    cols = list(pearson.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = cols[i], cols[j]
            c = abs(float(pearson.iloc[i, j]))
            if c != c or c <= pearson_thr:  # skip NaN / below threshold
                continue
            va, vb = vif_map.get(a, 0.0), vif_map.get(b, 0.0)
            if va != va:
                va = 0.0
            if vb != vb:
                vb = 0.0
            # drop higher VIF; if tied drop b
            drop, keep = (a, b) if va >= vb else (b, a)
            if drop not in remove:
                remove[drop] = f"corr({keep})={c:.3f}; prefer keep lower VIF"
    for feat, v in vif_map.items():
        if v == v and v > vif_thr and feat not in remove:
            remove[feat] = f"VIF={v:.2f}>{vif_thr}"
    return pd.DataFrame(
        [{"feature": f, "reason": r} for f, r in sorted(remove.items())]
    )
