"""Year-over-year feature stability (mean/std/var, PSI/KS, ranking drift)."""

from __future__ import annotations

import ast
from typing import Any

import numpy as np
import pandas as pd

from research.analyzers._common import feature_cols, safe_ks, safe_psi


class FeatureStabilityAnalyzer:
    def analyze(self, frame: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
        df = frame.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["year"] = df["timestamp"].dt.year
        feats = feature_cols(df)
        years = sorted(int(y) for y in df["year"].unique())
        rows: list[dict[str, Any]] = []

        for feat in feats:
            series_by_year: dict[int, np.ndarray] = {}
            mean_by_year: dict[int, float] = {}
            std_by_year: dict[int, float] = {}
            var_by_year: dict[int, float] = {}
            for y in years:
                vals = df.loc[df["year"] == y, feat].to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                series_by_year[y] = vals
                mean_by_year[y] = float(np.mean(vals)) if len(vals) else float("nan")
                std_by_year[y] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                var_by_year[y] = float(np.var(vals, ddof=1)) if len(vals) > 1 else 0.0

            base_year = next((y for y in years if len(series_by_year[y]) >= 50), years[0])
            base = series_by_year[base_year]
            psi_by_year = {y: safe_psi(base, series_by_year[y]) for y in years}
            ks_by_year = {y: safe_ks(base, series_by_year[y]) for y in years}

            means = np.array([mean_by_year[y] for y in years], dtype=float)
            finite_means = means[np.isfinite(means)]
            slope = (
                float(np.polyfit(np.arange(len(finite_means), dtype=float), finite_means, 1)[0])
                if len(finite_means) >= 2
                else float("nan")
            )

            rows.append(
                {
                    "feature": feat,
                    "base_year": base_year,
                    "mean_by_year_json": str({y: mean_by_year[y] for y in years}),
                    "std_by_year_json": str({y: std_by_year[y] for y in years}),
                    "var_by_year_json": str({y: var_by_year[y] for y in years}),
                    "psi_by_year_json": str({y: psi_by_year[y] for y in years}),
                    "ks_by_year_json": str({y: ks_by_year[y] for y in years}),
                    "mean_first": mean_by_year[years[0]],
                    "mean_last": mean_by_year[years[-1]],
                    "mean_trend_slope": slope,
                    "max_psi": float(np.nanmax(list(psi_by_year.values()))),
                    "max_ks": float(np.nanmax(list(ks_by_year.values()))),
                }
            )

        stability = pd.DataFrame(rows)
        if years and not stability.empty:
            first_year, last_year = years[0], years[-1]
            var_first: dict[str, float] = {}
            var_last: dict[str, float] = {}
            for _, r in stability.iterrows():
                vf = self._parse_map(str(r["var_by_year_json"]))
                var_first[str(r["feature"])] = float(vf.get(first_year, float("nan")))
                var_last[str(r["feature"])] = float(vf.get(last_year, float("nan")))
            rank_first = self._ranks(var_first)
            rank_last = self._ranks(var_last)
            stability["variance_rank_first"] = stability["feature"].map(rank_first)
            stability["variance_rank_last"] = stability["feature"].map(rank_last)
            stability["ranking_change"] = (
                stability["variance_rank_last"] - stability["variance_rank_first"]
            )

        corr_drift = self._mean_corr_drift(df, feats, years)
        stability = stability.sort_values(
            "max_psi", ascending=False, na_position="last"
        ).reset_index(drop=True)
        summary = {
            "n_features": int(len(feats)),
            "years": years,
            "top_unstable_features": stability.head(10)[
                ["feature", "max_psi", "max_ks", "mean_trend_slope"]
            ].to_dict(orient="records"),
            "mean_corr_drift": corr_drift,
        }
        return summary, stability

    @staticmethod
    def _parse_map(raw: str) -> dict[int, float]:
        data = ast.literal_eval(raw)
        return {int(k): float(v) for k, v in data.items()}

    @staticmethod
    def _ranks(values: dict[str, float]) -> dict[str, int]:
        ordered = sorted(
            values.items(),
            key=lambda kv: (kv[1] != kv[1], -(kv[1] if kv[1] == kv[1] else -1e18)),
        )
        return {name: i + 1 for i, (name, _) in enumerate(ordered)}

    @staticmethod
    def _mean_corr_drift(
        df: pd.DataFrame, feats: list[str], years: list[int]
    ) -> float:
        if len(years) < 2 or len(feats) < 2:
            return float("nan")
        first = df.loc[df["year"] == years[0], feats].astype(float)
        last = df.loc[df["year"] == years[-1], feats].astype(float)
        if len(first) < 30 or len(last) < 30:
            return float("nan")
        c1 = first.corr().to_numpy(dtype=float)
        c2 = last.corr().to_numpy(dtype=float)
        mask = ~np.eye(len(feats), dtype=bool)
        d = np.abs(c1[mask] - c2[mask])
        d = d[np.isfinite(d)]
        return float(np.mean(d)) if len(d) else float("nan")
