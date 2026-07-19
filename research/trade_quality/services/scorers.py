"""Multiple Trade Quality scoring methods (LOO; not classifier retrain)."""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

FEATURE_COLS: tuple[str, ...] = (
    "meta_proba",
    "raw_probability",
    "h4_context",
    "d1_trend",
    "m5_entry_quality_01",
    "session_score",
    "hour_score",
    "vol_regime",
    "atr_pctile",
    "trend_duration_n",
    "swing_quality",
    "rejection",
)


def prepare_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Build TQ feature matrix from confidence panel / OOF columns. Causal only."""
    out = panel.copy()
    # H4 composite
    h4 = [
        c
        for c in (
            "ctx_h4_rejection_strength",
            "ctx_h4_swing_quality",
            "ctx_h4_swing_strength",
            "ctx_h4_distance_from_equilibrium",
        )
        if c in out.columns
    ]
    out["h4_context"] = out[h4].astype(float).rank(pct=True).mean(axis=1) if h4 else 0.5
    side_sign = np.where(out["side"].astype(str).to_numpy() == "long", 1.0, -1.0)
    dist = out["d1_trend_dist"].astype(float).to_numpy() if "d1_trend_dist" in out.columns else np.zeros(len(out))
    out["d1_trend"] = (side_sign * np.tanh(np.nan_to_num(dist, nan=0.0) / 2.0) + 1.0) / 2.0
    m5 = out["m5_entry_quality"].astype(float) if "m5_entry_quality" in out.columns else pd.Series(50.0, index=out.index)
    out["m5_entry_quality_01"] = (m5.fillna(50) / 100.0).clip(0, 1)
    # Session favorability: london/ny/overlap preferred
    sess = 0.0
    for col, w in (
        ("session_london_ny_overlap", 1.0),
        ("session_london", 0.7),
        ("session_newyork", 0.7),
        ("session_asia", 0.3),
    ):
        if col in out.columns:
            sess = sess + w * out[col].astype(float)
    out["session_score"] = (sess / 2.4).clip(0, 1) if isinstance(sess, pd.Series) else 0.5
    # Hour: mid-day liquidity preference (proxy)
    if "hour_of_day" in out.columns:
        h = out["hour_of_day"].astype(float)
        out["hour_score"] = ((h >= 7) & (h <= 16)).astype(float) * 0.7 + 0.3
    else:
        out["hour_score"] = 0.5
    if "volatility_regime_score" in out.columns:
        out["vol_regime"] = out["volatility_regime_score"].astype(float).rank(pct=True)
    elif "volatility_rank" in out.columns:
        out["vol_regime"] = out["volatility_rank"].astype(float).clip(0, 1)
    else:
        out["vol_regime"] = 0.5
    if "atr_percentile_252" in out.columns:
        out["atr_pctile"] = out["atr_percentile_252"].astype(float).clip(0, 1)
    elif "atr_percent" in out.columns:
        out["atr_pctile"] = out["atr_percent"].astype(float).rank(pct=True)
    else:
        out["atr_pctile"] = 0.5
    if "ema_trend_duration" in out.columns:
        out["trend_duration_n"] = out["ema_trend_duration"].astype(float).rank(pct=True)
    else:
        out["trend_duration_n"] = 0.5
    out["swing_quality"] = (
        out["ctx_h4_swing_quality"].astype(float).rank(pct=True)
        if "ctx_h4_swing_quality" in out.columns
        else 0.5
    )
    out["rejection"] = (
        out["ctx_h4_rejection_strength"].astype(float).rank(pct=True)
        if "ctx_h4_rejection_strength" in out.columns
        else 0.5
    )
    for c in FEATURE_COLS:
        if c not in out.columns:
            out[c] = 0.5
        out[c] = out[c].astype(float).fillna(0.5)
    return out


def _rank_to_100(x: np.ndarray) -> np.ndarray:
    s = pd.Series(x)
    return (s.rank(pct=True).to_numpy() * 100.0).clip(0, 100)


def method_weighted(train: pd.DataFrame, val: pd.DataFrame) -> np.ndarray:
    """Weights proportional to train Spearman with net_return (positive only)."""
    cols = list(FEATURE_COLS)
    w = []
    for c in cols:
        rho = train[c].corr(train["net_return"], method="spearman")
        w.append(max(float(rho) if rho == rho else 0.0, 0.0))
    w = np.asarray(w, dtype=float)
    if w.sum() <= 0:
        w = np.ones(len(cols))
    w = w / w.sum()
    x = val[cols].to_numpy(dtype=float)
    score = (x * w).sum(axis=1)
    return _rank_to_100(score)


def method_logistic(train: pd.DataFrame, val: pd.DataFrame) -> np.ndarray:
    cols = list(FEATURE_COLS)
    x_tr = train[cols].to_numpy(dtype=float)
    y = train["meta_label"].astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        return method_weighted(train, val)
    sc = StandardScaler()
    xs = sc.fit_transform(x_tr)
    clf = LogisticRegression(max_iter=400, random_state=42)
    clf.fit(xs, y)
    p = clf.predict_proba(sc.transform(val[cols].to_numpy(dtype=float)))[:, 1]
    return np.clip(p * 100.0, 0, 100)


def method_isotonic(train: pd.DataFrame, val: pd.DataFrame) -> np.ndarray:
    """Isotonic on weighted raw score -> empirical win rate."""
    raw_tr = method_weighted(train, train)
    raw_va = method_weighted(train, val)
    y = train["meta_label"].astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        return raw_va
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(raw_tr, y)
    return np.clip(iso.predict(raw_va) * 100.0, 0, 100)


def method_rank_agg(train: pd.DataFrame, val: pd.DataFrame) -> np.ndarray:
    """Average percentile ranks of features (train-centered ranks via val self-rank is OK for score)."""
    cols = list(FEATURE_COLS)
    # Direction: flip features with negative train spearman
    parts = []
    for c in cols:
        rho = train[c].corr(train["net_return"], method="spearman")
        r = val[c].rank(pct=True).to_numpy(dtype=float)
        if rho == rho and rho < 0:
            r = 1.0 - r
        parts.append(r)
    return _rank_to_100(np.mean(parts, axis=0))


def method_bayesian(train: pd.DataFrame, val: pd.DataFrame) -> np.ndarray:
    """Beta-binomial style: prior from train win rate, tilted by feature z."""
    cols = list(FEATURE_COLS)
    p0 = float(train["meta_label"].mean()) if len(train) else 0.5
    p0 = min(max(p0, 0.05), 0.95)
    # tilt: logistic of z-scored mean of directed features
    wscore = method_weighted(train, val) / 100.0
    # Bayes pull toward evidence
    evidence = 0.5 * wscore + 0.5 * p0
    return np.clip(evidence * 100.0, 0, 100)


def method_ensemble(train: pd.DataFrame, val: pd.DataFrame) -> np.ndarray:
    scores = np.vstack(
        [
            method_weighted(train, val),
            method_logistic(train, val),
            method_isotonic(train, val),
            method_rank_agg(train, val),
            method_bayesian(train, val),
        ]
    )
    return np.clip(np.mean(scores, axis=0), 0, 100)


METHODS: dict[str, Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]] = {
    "weighted": method_weighted,
    "logistic": method_logistic,
    "isotonic": method_isotonic,
    "rank_agg": method_rank_agg,
    "bayesian": method_bayesian,
    "ensemble": method_ensemble,
}


def loo_trade_quality(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Leave-one-year-out TQ for each method. Stability = mean Spearman of TQ vs expectancy proxy across years
    + monotonicity of bucket expectancy.
    """
    frame = prepare_features(panel)
    years = sorted(int(y) for y in frame["valid_year"].dropna().unique())
    stab_rows = []

    for method, fn in METHODS.items():
        parts = []
        year_spearman = []
        year_mono = []
        for year in years:
            tr = frame.loc[frame["valid_year"] != year]
            va = frame.loc[frame["valid_year"] == year].copy()
            if tr.empty or va.empty:
                continue
            tq = fn(tr, va)
            va = va.copy()
            va["trade_quality"] = tq
            va["tq_method"] = method
            parts.append(va)
            # stability vs net_return
            sp = pd.Series(tq).corr(va["net_return"], method="spearman")
            year_spearman.append(float(sp) if sp == sp else 0.0)
            # mono: mean net_return by quintile
            try:
                q = pd.qcut(tq, 5, labels=False, duplicates="drop")
                means = va.groupby(q)["net_return"].mean()
                mono = float(means.diff().dropna().gt(0).mean()) if len(means) > 1 else 0.0
            except Exception:
                mono = 0.0
            year_mono.append(mono)
        if not parts:
            continue
        stab_rows.append(
            {
                "method": method,
                "mean_spearman_net": float(np.mean(year_spearman)) if year_spearman else float("nan"),
                "std_spearman_net": float(np.std(year_spearman)) if len(year_spearman) > 1 else 0.0,
                "mean_monotonicity": float(np.mean(year_mono)) if year_mono else float("nan"),
                "std_monotonicity": float(np.std(year_mono)) if len(year_mono) > 1 else 0.0,
                "stability_score": float(
                    (np.mean(year_spearman) if year_spearman else 0)
                    + (np.mean(year_mono) if year_mono else 0)
                    - 0.5 * (np.std(year_spearman) if len(year_spearman) > 1 else 0)
                ),
            }
        )
        logger.info("TQ method=%s years=%s", method, len(year_spearman))

    stability = pd.DataFrame(stab_rows).sort_values("stability_score", ascending=False)
    best = str(stability.iloc[0]["method"]) if not stability.empty else "ensemble"
    # LOO scores for every method as columns; production score = best method
    base = prepare_features(panel).copy()
    for method, fn in METHODS.items():
        col = np.full(len(base), np.nan)
        for year in years:
            tr = base.loc[base["valid_year"] != year]
            va_idx = base.index[base["valid_year"] == year]
            if len(va_idx) == 0 or tr.empty:
                continue
            va = base.loc[va_idx]
            col[base.index.get_indexer(va_idx)] = fn(tr, va)
        base[f"tq_{method}"] = col
    base["trade_quality"] = base[f"tq_{best}"]
    base["tq_best_method"] = best
    if "confidence" in panel.columns:
        base["confidence"] = panel["confidence"].to_numpy()
    return base, stability
