"""Confidence score: LOO-estimated weights on evidence components."""

from __future__ import annotations

import logging
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

COMPONENT_COLS: tuple[str, ...] = (
    "meta_proba",
    "raw_probability",
    "h4_context",
    "d1_trend",
    "m5_entry_quality",
)


def _h4_context(frame: pd.DataFrame) -> np.ndarray:
    cols = [
        c
        for c in (
            "ctx_h4_rejection_strength",
            "ctx_h4_swing_quality",
            "ctx_h4_swing_strength",
            "ctx_h4_distance_from_equilibrium",
        )
        if c in frame.columns
    ]
    if not cols:
        return np.zeros(len(frame))
    x = frame[cols].astype(float)
    # Rank-average to 0-1
    ranks = x.rank(pct=True).mean(axis=1)
    return ranks.to_numpy(dtype=float)


def _d1_trend_score(frame: pd.DataFrame) -> np.ndarray:
    """Map regime + trend_dist to signed alignment with trade side."""
    side_sign = np.where(frame["side"].astype(str).to_numpy() == "long", 1.0, -1.0)
    dist = frame["d1_trend_dist"].astype(float).to_numpy() if "d1_trend_dist" in frame.columns else np.zeros(len(frame))
    # Align: long wants positive dist
    aligned = side_sign * np.tanh(np.nan_to_num(dist, nan=0.0) / 2.0)
    return (aligned + 1.0) / 2.0  # 0-1


def attach_components(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["h4_context"] = _h4_context(out)
    out["d1_trend"] = _d1_trend_score(out)
    if "m5_entry_quality" not in out.columns:
        out["m5_entry_quality"] = 50.0
    else:
        out["m5_entry_quality"] = out["m5_entry_quality"].astype(float).fillna(50.0)
    # scale m5 to 0-1 for logistic
    out["m5_entry_quality_01"] = (out["m5_entry_quality"].clip(0, 100) / 100.0)
    return out


def fit_loo_confidence(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Leave-one-year-out logistic weights; OOF confidence score 0-100.
    Weights reported as mean |coef| share across folds.
    """
    frame = attach_components(panel)
    # features for model: meta, primary, h4, d1, m5_01
    feat_map = {
        "meta_proba": "meta_proba",
        "raw_probability": "raw_probability",
        "h4_context": "h4_context",
        "d1_trend": "d1_trend",
        "m5_entry_quality": "m5_entry_quality_01",
    }
    feats = list(feat_map.values())
    years = sorted(int(y) for y in frame["valid_year"].dropna().unique())
    oof_rows = []
    weight_rows = []

    for year in years:
        tr = frame.loc[frame["valid_year"] != year].copy()
        va = frame.loc[frame["valid_year"] == year].copy()
        if tr["meta_label"].nunique() < 2 or va.empty:
            continue
        x_tr = tr[feats].astype(float).fillna(0.5)
        y_tr = tr["meta_label"].astype(int).to_numpy()
        x_va = va[feats].astype(float).fillna(0.5)
        scaler = StandardScaler()
        xs_tr = scaler.fit_transform(x_tr)
        xs_va = scaler.transform(x_va)
        clf = LogisticRegression(max_iter=500, random_state=42)
        clf.fit(xs_tr, y_tr)
        proba = clf.predict_proba(xs_va)[:, 1]
        confidence = np.clip(proba * 100.0, 0, 100)
        part = va.copy()
        part["confidence"] = confidence
        part["confidence_raw_logit_p"] = proba
        oof_rows.append(part)

        coef = np.abs(clf.coef_.ravel())
        share = coef / (coef.sum() + 1e-12)
        for name, col, s, c in zip(feat_map.keys(), feats, share, clf.coef_.ravel()):
            weight_rows.append(
                {
                    "valid_year": year,
                    "component": name,
                    "weight_share": float(s),
                    "coef": float(c),
                }
            )
        logger.info("Confidence LOO year=%s n=%s", year, len(va))

    oof = pd.concat(oof_rows, ignore_index=True) if oof_rows else pd.DataFrame()
    weights = pd.DataFrame(weight_rows)
    if not weights.empty:
        summary = (
            weights.groupby("component", as_index=False)
            .agg(weight_share_mean=("weight_share", "mean"), coef_mean=("coef", "mean"))
            .sort_values("weight_share_mean", ascending=False)
        )
    else:
        summary = pd.DataFrame()
    return oof, summary


def shap_components(panel: pd.DataFrame, *, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Small LGBM on components for SHAP-style pred_contrib (explanation only)."""
    frame = attach_components(panel)
    feats = ["meta_proba", "raw_probability", "h4_context", "d1_trend", "m5_entry_quality_01"]
    x = frame[feats].astype(float).fillna(0.5)
    y = frame["meta_label"].astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        return pd.DataFrame(), pd.DataFrame()
    dtrain = lgb.Dataset(x, label=y, feature_name=feats)
    booster = lgb.train(
        {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            "seed": seed,
            "learning_rate": 0.05,
            "num_leaves": 15,
        },
        dtrain,
        num_boost_round=80,
    )
    contrib = np.asarray(booster.predict(x, pred_contrib=True), dtype=float)
    shap_abs = np.abs(contrib[:, :-1]).mean(axis=0)
    importance = pd.DataFrame({"feature": feats, "mean_abs_shap": shap_abs}).sort_values(
        "mean_abs_shap", ascending=False
    )
    # Pair interactions proxy: product of mean abs shap
    rows = []
    for i, a in enumerate(feats):
        for j, b in enumerate(feats):
            if j <= i:
                continue
            rows.append(
                {
                    "feature_a": a,
                    "feature_b": b,
                    "interaction_proxy": float(shap_abs[i] * shap_abs[j]),
                }
            )
    interactions = pd.DataFrame(rows).sort_values("interaction_proxy", ascending=False)
    return importance, interactions
