"""Mutual information vs binary target."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif


class MutualInformationAnalyzer:
    def analyze(
        self,
        x: pd.DataFrame,
        y: np.ndarray,
        *,
        random_state: int = 42,
        metadata: dict[str, dict[str, Any]] | None = None,
    ) -> tuple[dict[str, Any], pd.DataFrame]:
        meta = metadata or {}
        # fill NaN with column median for MI estimator
        x_fill = x.copy()
        for c in x_fill.columns:
            med = float(x_fill[c].median()) if x_fill[c].notna().any() else 0.0
            x_fill[c] = x_fill[c].fillna(med)

        mi = mutual_info_classif(
            x_fill.to_numpy(dtype=float),
            y,
            discrete_features=False,
            random_state=random_state,
        )
        rows = []
        for name, score in zip(x.columns, mi, strict=True):
            rows.append(
                {
                    "feature": name,
                    "mutual_information": float(score),
                    "category": (meta.get(name) or {}).get("category", ""),
                }
            )
        frame = (
            pd.DataFrame(rows)
            .sort_values("mutual_information", ascending=False)
            .reset_index(drop=True)
        )
        summary = {
            "top_features": frame.head(15).to_dict(orient="records"),
            "category_mean_mi": (
                frame.groupby("category")["mutual_information"].mean().sort_values(ascending=False).to_dict()
                if frame["category"].astype(bool).any()
                else {}
            ),
        }
        return summary, frame
