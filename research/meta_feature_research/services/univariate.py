"""Model-free univariate scores (no classifier / no LightGBM)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_classif


def _clean_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=int).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    x, y = _clean_xy(x, y)
    if len(x) < 10 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return float("nan")
    rho, _ = stats.spearmanr(x, y)
    return float(rho)


def mutual_info(x: np.ndarray, y: np.ndarray, *, random_state: int = 42) -> float:
    x, y = _clean_xy(x, y)
    if len(x) < 20 or np.unique(y).size < 2:
        return float("nan")
    mi = mutual_info_classif(
        x.reshape(-1, 1),
        y,
        discrete_features=False,
        random_state=random_state,
        n_neighbors=3,
    )
    return float(mi[0])


def permutation_mi(
    x: np.ndarray, y: np.ndarray, *, random_state: int = 42, n_perm: int = 1
) -> float:
    """Model-free: MI(x,y) - MI(shuffle(x), y)."""
    base = mutual_info(x, y, random_state=random_state)
    if base != base:
        return float("nan")
    rng = np.random.default_rng(random_state)
    x2, y2 = _clean_xy(x, y)
    drops = []
    for i in range(n_perm):
        shuf = x2.copy()
        rng.shuffle(shuf)
        drops.append(base - mutual_info(shuf, y2, random_state=random_state + 1 + i))
    return float(np.mean(drops))


def ks_statistic(x: np.ndarray, y: np.ndarray) -> float:
    x, y = _clean_xy(x, y)
    if len(x) < 20:
        return float("nan")
    pos = x[y == 1]
    neg = x[y == 0]
    if len(pos) < 5 or len(neg) < 5:
        return float("nan")
    return float(stats.ks_2samp(pos, neg).statistic)


def information_value(x: np.ndarray, y: np.ndarray, *, n_bins: int = 10) -> float:
    """IV via quantile bins + WoE (cap extremes)."""
    x, y = _clean_xy(x, y)
    if len(x) < 30 or np.unique(y).size < 2:
        return float("nan")
    try:
        bins = pd.qcut(x, q=min(n_bins, max(2, len(np.unique(x)) // 2)), duplicates="drop")
    except ValueError:
        return float("nan")
    tab = pd.crosstab(bins, y)
    if tab.shape[1] < 2:
        return float("nan")
    # ensure both classes
    if 0 not in tab.columns:
        tab[0] = 0
    if 1 not in tab.columns:
        tab[1] = 0
    tab = tab[[0, 1]].astype(float) + 0.5  # Laplace
    dist_neg = tab[0] / tab[0].sum()
    dist_pos = tab[1] / tab[1].sum()
    woe = np.log(dist_pos / dist_neg)
    iv = ((dist_pos - dist_neg) * woe).sum()
    return float(iv)


def shap_proxy(mi: float, spearman: float) -> float:
    """ponytail: no model SHAP — signed strength |rho| * MI."""
    if mi != mi or spearman != spearman:
        return float("nan")
    return float(np.sign(spearman) * abs(spearman) * mi)


def score_feature(x: np.ndarray, y: np.ndarray, *, random_state: int = 42) -> dict[str, float]:
    mi = mutual_info(x, y, random_state=random_state)
    sp = spearman_corr(x, y)
    return {
        "mutual_info": mi,
        "permutation_mi": permutation_mi(x, y, random_state=random_state),
        "spearman": sp,
        "information_value": information_value(x, y),
        "ks_statistic": ks_statistic(x, y),
        "shap_proxy": shap_proxy(mi, sp),
    }


def categorical_outcome_table(
    frame: pd.DataFrame, feature: str, *, label_col: str = "meta_label"
) -> pd.DataFrame:
    rows = []
    for val, g in frame.groupby(feature, dropna=False):
        n = len(g)
        if n == 0:
            continue
        y = g[label_col].astype(int)
        # expectancy needs net_return if present
        exp = float(g["net_return"].mean()) if "net_return" in g.columns else float("nan")
        rows.append(
            {
                "feature": feature,
                "value": val,
                "samples": n,
                "win_rate": float(y.mean()),
                "expectancy": exp,
            }
        )
    return pd.DataFrame(rows)


def score_all_features(
    panel: pd.DataFrame,
    feature_cols: list[str],
    *,
    label_col: str = "meta_label",
    random_state: int = 42,
) -> pd.DataFrame:
    y = panel[label_col].to_numpy(dtype=int)
    rows = []
    for feat in feature_cols:
        if feat not in panel.columns:
            continue
        scores = score_feature(panel[feat].to_numpy(), y, random_state=random_state)
        rows.append({"feature": feat, **scores})
    return pd.DataFrame(rows)
