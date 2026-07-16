"""Redundant feature groups from high absolute correlation."""

from __future__ import annotations

from typing import Any

import pandas as pd


class RedundancyAnalyzer:
    def analyze(
        self,
        correlation_pairs: pd.DataFrame,
        *,
        threshold: float = 0.95,
        method: str = "pearson",
    ) -> tuple[dict[str, Any], pd.DataFrame]:
        col = "abs_pearson" if method == "pearson" else "abs_spearman"
        high = correlation_pairs.loc[correlation_pairs[col] >= threshold].copy()
        parent: dict[str, str] = {}

        def find(x: str) -> str:
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for _, row in high.iterrows():
            union(str(row["feature_a"]), str(row["feature_b"]))

        groups: dict[str, list[str]] = {}
        members = set(high["feature_a"].astype(str)) | set(high["feature_b"].astype(str))
        for feat in members:
            root = find(feat)
            groups.setdefault(root, []).append(feat)

        group_list = [sorted(v) for v in groups.values() if len(v) >= 2]
        group_list.sort(key=len, reverse=True)

        rows = []
        for i, group in enumerate(group_list, start=1):
            for feat in group:
                rows.append(
                    {
                        "group_id": i,
                        "feature": feat,
                        "group_size": len(group),
                        "group_members": "|".join(group),
                        "threshold": threshold,
                        "method": method,
                    }
                )
        frame = pd.DataFrame(rows)
        summary = {
            "threshold": threshold,
            "method": method,
            "n_groups": len(group_list),
            "groups": group_list,
            "n_redundant_features": int(sum(len(g) for g in group_list)),
        }
        return summary, frame
