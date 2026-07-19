"""Yearly PSI/KS feature stability + regime PSI."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from feature_diagnostics.analyzers._common import safe_ks, safe_psi


class StabilityAnalyzer:
    def analyze_yearly(
        self,
        x: pd.DataFrame,
        timestamps: pd.Series,
    ) -> tuple[dict[str, Any], pd.DataFrame]:
        years = timestamps.dt.year
        year_list = sorted(int(y) for y in years.unique())
        base_year = year_list[0]
        rows = []
        for feat in x.columns:
            base = x.loc[years == base_year, feat].to_numpy(dtype=float)
            psi_by_year = {}
            ks_by_year = {}
            for y in year_list:
                vals = x.loc[years == y, feat].to_numpy(dtype=float)
                psi_by_year[y] = safe_psi(base, vals)
                ks_by_year[y] = safe_ks(base, vals)
            max_psi = float(np.nanmax(list(psi_by_year.values())))
            rows.append(
                {
                    "feature": feat,
                    "base_year": base_year,
                    "max_psi": max_psi,
                    "max_ks": float(np.nanmax(list(ks_by_year.values()))),
                    "mean_psi": float(np.nanmean(list(psi_by_year.values()))),
                    "psi_by_year": str(psi_by_year),
                    "drift_flag": bool(max_psi >= 0.25),
                    "stable_flag": bool(max_psi < 0.1),
                }
            )
        frame = (
            pd.DataFrame(rows)
            .sort_values("max_psi", ascending=False, na_position="last")
            .reset_index(drop=True)
        )
        summary = {
            "years": year_list,
            "n_drift": int(frame["drift_flag"].sum()),
            "n_stable": int(frame["stable_flag"].sum()),
            "top_unstable": frame.head(10).to_dict(orient="records"),
        }
        return summary, frame

    def analyze_regimes(
        self,
        x: pd.DataFrame,
        *,
        regime: pd.Series,
    ) -> pd.DataFrame:
        """PSI of each feature vs global distribution, per regime."""
        regimes = sorted(str(r) for r in regime.dropna().unique())
        rows = []
        for feat in x.columns:
            global_vals = x[feat].to_numpy(dtype=float)
            for reg in regimes:
                part = x.loc[regime.astype(str) == reg, feat].to_numpy(dtype=float)
                if len(part) < 30:
                    continue
                psi = safe_psi(global_vals, part)
                rows.append(
                    {
                        "feature": feat,
                        "regime": reg,
                        "n": int(len(part)),
                        "psi_vs_global": psi,
                        "ks_vs_global": safe_ks(global_vals, part),
                        "unstable_in_regime": bool(psi == psi and psi >= 0.25),
                    }
                )
        frame = (
            pd.DataFrame(rows)
            .sort_values("psi_vs_global", ascending=False, na_position="last")
            .reset_index(drop=True)
        )
        return frame


def assign_regimes(x: pd.DataFrame) -> pd.Series:
    """Lightweight regimes from engineered features when available."""
    if "volatility_regime_score" in x.columns:
        vol = x["volatility_regime_score"]
    else:
        vol = pd.Series(0.0, index=x.index)
    if "ema_alignment_score" in x.columns:
        trend = x["ema_alignment_score"]
    elif "ema_cross_distance" in x.columns:
        trend = x["ema_cross_distance"]
    else:
        trend = pd.Series(0.0, index=x.index)

    labels = []
    for i in range(len(x)):
        v = float(vol.iloc[i]) if np.isfinite(vol.iloc[i]) else 0.0
        t = float(trend.iloc[i]) if np.isfinite(trend.iloc[i]) else 0.0
        if v >= 0.5:
            labels.append("High Volatility")
        elif v <= -0.5:
            labels.append("Low Volatility")
        elif t >= 0.25:
            labels.append("Bull")
        elif t <= -0.25:
            labels.append("Bear")
        else:
            labels.append("Sideways")
    return pd.Series(labels, index=x.index)
