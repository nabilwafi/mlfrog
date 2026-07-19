"""Concept drift + feature importance stability."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from diagnostics.analyzers._common import safe_ks, safe_psi
from research.temporal_stability.services import js_divergence, kl_divergence
from research.temporal_stability.services.data import fit_predict, ml_feature_names

logger = logging.getLogger(__name__)


def year_pair_drift(panel: pd.DataFrame, features: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = features or ml_feature_names(panel)
    years = sorted(int(y) for y in panel["year"].unique())
    summary_rows = []
    detail_rows = []
    for a, b in zip(years[:-1], years[1:]):
        fa = panel.loc[panel["year"] == a]
        fb = panel.loc[panel["year"] == b]
        ya = float((fa["label"].astype(int) == 1).mean()) if "label" in fa.columns else float("nan")
        yb = float((fb["label"].astype(int) == 1).mean()) if "label" in fb.columns else float("nan")
        p_psi = p_ks = float("nan")
        if "raw_probability" in fa.columns and "raw_probability" in fb.columns:
            p_psi = safe_psi(fa["raw_probability"].to_numpy(dtype=float), fb["raw_probability"].to_numpy(dtype=float))
            p_ks = safe_ks(fa["raw_probability"].to_numpy(dtype=float), fb["raw_probability"].to_numpy(dtype=float))
        feat_psis, feat_ks, feat_kl, feat_js = [], [], [], []
        for f in features:
            if f not in fa.columns or f not in fb.columns:
                continue
            va = fa[f].to_numpy(dtype=float)
            vb = fb[f].to_numpy(dtype=float)
            psi = safe_psi(va, vb)
            ks = safe_ks(va, vb)
            kl = kl_divergence(va, vb)
            js = js_divergence(va, vb)
            feat_psis.append(psi)
            feat_ks.append(ks)
            feat_kl.append(kl)
            feat_js.append(js)
            detail_rows.append(
                {"year_from": a, "year_to": b, "feature": f, "psi": psi, "ks": ks, "kl": kl, "js": js}
            )
        summary_rows.append(
            {
                "year_from": a,
                "year_to": b,
                "label_pos_rate_from": ya,
                "label_pos_rate_to": yb,
                "label_pos_rate_delta": (yb - ya) if ya == ya and yb == yb else float("nan"),
                "prob_psi": p_psi,
                "prob_ks": p_ks,
                "feature_psi_mean": float(np.nanmean(feat_psis)) if feat_psis else float("nan"),
                "feature_psi_max": float(np.nanmax(feat_psis)) if feat_psis else float("nan"),
                "feature_ks_mean": float(np.nanmean(feat_ks)) if feat_ks else float("nan"),
                "feature_kl_mean": float(np.nanmean(feat_kl)) if feat_kl else float("nan"),
                "feature_js_mean": float(np.nanmean(feat_js)) if feat_js else float("nan"),
                "n_features": len(feat_psis),
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(detail_rows)


def feature_importance_over_time(panel: pd.DataFrame) -> pd.DataFrame:
    years = sorted(int(y) for y in panel["year"].unique())
    features = ml_feature_names(panel)
    rows = []
    for year in years:
        train = panel.loc[panel["year"] < year]
        test = panel.loc[panel["year"] == year]
        if train.empty or test.empty:
            continue
        try:
            _, _, booster = fit_predict(train, test, features)
        except ValueError:
            continue
        gain = booster.feature_importance(importance_type="gain")
        split = booster.feature_importance(importance_type="split")
        names = booster.feature_name()
        total_g = float(np.sum(gain)) + 1e-12
        for n, g, s in zip(names, gain, split):
            rows.append(
                {
                    "year": year,
                    "feature": n,
                    "gain": float(g),
                    "gain_share": float(g) / total_g,
                    "split": float(s),
                }
            )
    return pd.DataFrame(rows)


def feature_stability_summary(importance: pd.DataFrame) -> pd.DataFrame:
    if importance.empty:
        return pd.DataFrame()
    rows = []
    for feat, g in importance.groupby("feature"):
        shares = g["gain_share"].astype(float)
        ranks = g["gain_share"].rank(ascending=False)
        rows.append(
            {
                "feature": feat,
                "mean_gain_share": float(shares.mean()),
                "std_gain_share": float(shares.std(ddof=1)) if len(shares) > 1 else 0.0,
                "cv_gain_share": float(shares.std(ddof=1) / (shares.mean() + 1e-12)) if len(shares) > 1 else 0.0,
                "rank_variance": float(ranks.var(ddof=1)) if len(ranks) > 1 else 0.0,
                "years_present": int(len(g)),
                "mean_rank": float(ranks.mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_gain_share", ascending=False).reset_index(drop=True)


def monte_carlo_year(
    trades: pd.DataFrame, *, n_sims: int = 500, seed: int = 42, starting: float = 80.0
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for year, g in trades.groupby("year"):
        if "net_return" not in g.columns:
            continue
        r = g["net_return"].astype(float).to_numpy()
        r = r[np.isfinite(r)]
        if len(r) < 5:
            continue
        finals, dds = [], []
        for _ in range(n_sims):
            rr = rng.permutation(r)
            eq = starting * np.cumprod(1.0 + rr)
            finals.append(float(eq[-1]))
            peak = np.maximum.accumulate(eq)
            dd = (peak - eq) / np.where(peak > 0, peak, np.nan)
            dds.append(float(np.nanmax(dd)))
        fa = np.asarray(finals)
        da = np.asarray(dds)
        cagr = fa / starting - 1.0
        rows.append(
            {
                "year": int(year),
                "n_sims": n_sims,
                "trades": len(r),
                "median_equity": float(np.median(fa)),
                "median_dd": float(np.median(da)),
                "median_cagr": float(np.median(cagr)),
                "prob_double": float(np.mean(fa >= 2 * starting)),
                "risk_of_ruin_50pct": float(np.mean(fa <= 0.5 * starting)),
                "prob_lose": float(np.mean(fa < starting)),
            }
        )
    return pd.DataFrame(rows)
