"""Per-feature statistics and correlation diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from datasets.entities.dataset import Dataset
from diagnostics.analyzers._common import feature_matrix


class FeatureAnalyzer:
    def analyze(
        self,
        train: Dataset,
        *,
        near_constant_std: float = 1e-8,
        corr_threshold: float = 0.95,
    ) -> tuple[dict[str, Any], pd.DataFrame]:
        x = feature_matrix(train)
        rows: list[dict[str, Any]] = []
        constant: list[str] = []
        near_constant: list[str] = []

        for col in x.columns:
            s = x[col]
            finite = s[np.isfinite(s.to_numpy(dtype=float))]
            n = len(s)
            missing_pct = float(s.isna().mean() * 100.0)
            std = float(finite.std(ddof=1)) if len(finite) > 1 else 0.0
            var = float(finite.var(ddof=1)) if len(finite) > 1 else 0.0
            nunique = int(finite.nunique())
            is_const = nunique <= 1
            is_near = (not is_const) and std <= near_constant_std
            if is_const:
                constant.append(col)
            if is_near:
                near_constant.append(col)
            skew = float(finite.skew()) if len(finite) > 2 else float("nan")
            kurt = float(finite.kurtosis()) if len(finite) > 3 else float("nan")
            rows.append(
                {
                    "feature": col,
                    "mean": float(finite.mean()) if len(finite) else float("nan"),
                    "median": float(finite.median()) if len(finite) else float("nan"),
                    "std": std,
                    "variance": var,
                    "min": float(finite.min()) if len(finite) else float("nan"),
                    "max": float(finite.max()) if len(finite) else float("nan"),
                    "missing_pct": missing_pct,
                    "n_unique": nunique,
                    "is_constant": is_const,
                    "is_near_constant": is_near,
                    "skewness": skew,
                    "kurtosis": kurt,
                }
            )

        stats = pd.DataFrame(rows).sort_values("variance", ascending=False).reset_index(drop=True)

        corr = x.corr(method="pearson")
        high_pairs: list[dict[str, Any]] = []
        cols = list(corr.columns)
        for i, a in enumerate(cols):
            for b in cols[i + 1 :]:
                v = corr.loc[a, b]
                if pd.isna(v):
                    continue
                if abs(float(v)) >= corr_threshold:
                    high_pairs.append(
                        {
                            "feature_a": a,
                            "feature_b": b,
                            "correlation": float(v),
                        }
                    )
        high_pairs.sort(key=lambda d: abs(d["correlation"]), reverse=True)

        summary = {
            "n_features": int(len(x.columns)),
            "constant_features": constant,
            "near_constant_features": near_constant,
            "high_correlation_pairs": high_pairs,
            "corr_threshold": corr_threshold,
            "variance_ranking": stats["feature"].tolist(),
        }
        return summary, stats
