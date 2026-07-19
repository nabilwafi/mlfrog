"""Pearson & Spearman feature correlation."""

from __future__ import annotations

from typing import Any

import pandas as pd


class CorrelationAnalyzer:
    def analyze(self, x: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
        pearson = x.corr(method="pearson")
        spearman = x.corr(method="spearman")
        rows = []
        cols = list(x.columns)
        for i, a in enumerate(cols):
            for b in cols[i + 1 :]:
                rows.append(
                    {
                        "feature_a": a,
                        "feature_b": b,
                        "pearson": float(pearson.loc[a, b]),
                        "spearman": float(spearman.loc[a, b]),
                        "abs_pearson": float(abs(pearson.loc[a, b])),
                        "abs_spearman": float(abs(spearman.loc[a, b])),
                    }
                )
        frame = (
            pd.DataFrame(rows)
            .sort_values("abs_pearson", ascending=False, na_position="last")
            .reset_index(drop=True)
        )
        summary = {
            "n_pairs": int(len(frame)),
            "n_high_pearson_095": int((frame["abs_pearson"] >= 0.95).sum()),
            "n_high_spearman_095": int((frame["abs_spearman"] >= 0.95).sum()),
            "top_pairs": frame.head(15).to_dict(orient="records"),
        }
        return summary, frame
