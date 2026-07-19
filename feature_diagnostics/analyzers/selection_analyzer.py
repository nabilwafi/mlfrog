"""Combine diagnostics into keep/remove/candidate library v3."""

from __future__ import annotations

from typing import Any

import pandas as pd


class SelectionAnalyzer:
    def analyze(
        self,
        *,
        features: list[str],
        mi: pd.DataFrame,
        permutation: pd.DataFrame,
        shap: pd.DataFrame,
        stability: pd.DataFrame,
        redundancy_groups: list[list[str]],
        metadata: dict[str, dict[str, Any]] | None = None,
        corr_threshold: float = 0.95,
    ) -> tuple[dict[str, Any], pd.DataFrame]:
        meta = metadata or {}
        mi_map = dict(zip(mi["feature"], mi["mutual_information"], strict=False)) if not mi.empty else {}
        perm_map = (
            dict(zip(permutation["feature"], permutation["permutation_importance_mean"], strict=False))
            if not permutation.empty
            else {}
        )
        shap_map = (
            dict(zip(shap["feature"], shap["mean_abs_shap"], strict=False)) if not shap.empty else {}
        )
        stab_map = (
            {str(r["feature"]): dict(r) for _, r in stability.iterrows()}
            if not stability.empty
            else {}
        )

        # ranks (1 = best)
        def ranks(scores: dict[str, float], *, higher_better: bool = True) -> dict[str, float]:
            def key(f: str) -> tuple[int, float]:
                v = scores.get(f, float("nan"))
                if v != v:
                    return (1, 0.0)
                return (0, -v if higher_better else v)

            ordered = sorted(features, key=key)
            return {f: float(i + 1) for i, f in enumerate(ordered)}

        mi_rank = ranks(mi_map, higher_better=True)
        perm_rank = ranks(perm_map, higher_better=True)
        shap_rank = ranks(shap_map, higher_better=True)

        # within each redundancy group keep the best mean rank
        drop_for_redundancy: set[str] = set()
        for group in redundancy_groups:
            scored = []
            for f in group:
                mean_r = (mi_rank.get(f, 999) + perm_rank.get(f, 999) + shap_rank.get(f, 999)) / 3.0
                scored.append((mean_r, f))
            scored.sort()
            for _, f in scored[1:]:
                drop_for_redundancy.add(f)

        rows = []
        keep: list[str] = []
        remove: list[str] = []
        drift_sensitive: list[str] = []
        stable: list[str] = []

        for feat in features:
            st = stab_map.get(feat, {})
            max_psi = float(st.get("max_psi", float("nan")))
            is_drift = bool(st.get("drift_flag", False))
            is_stable = bool(st.get("stable_flag", False))
            if is_drift:
                drift_sensitive.append(feat)
            if is_stable:
                stable.append(feat)

            mean_rank = (
                mi_rank.get(feat, 999) + perm_rank.get(feat, 999) + shap_rank.get(feat, 999)
            ) / 3.0
            reasons: list[str] = []
            decision = "keep"

            if feat in drop_for_redundancy:
                decision = "remove"
                reasons.append(f"redundant_corr>={corr_threshold}")
            if is_drift and mean_rank > len(features) * 0.6:
                decision = "remove"
                reasons.append("high_psi_and_weak_signal")
            if (
                mi_map.get(feat, 0.0) <= 1e-6
                and perm_map.get(feat, 0.0) <= 0.0
                and shap_map.get(feat, 0.0) <= 1e-9
            ):
                decision = "remove"
                reasons.append("near_zero_predictive_signal")

            if decision == "keep":
                keep.append(feat)
            else:
                remove.append(feat)

            rows.append(
                {
                    "feature": feat,
                    "category": (meta.get(feat) or {}).get("category", ""),
                    "decision": decision,
                    "reasons": "|".join(reasons) if reasons else "",
                    "mi": float(mi_map.get(feat, float("nan"))),
                    "permutation_importance": float(perm_map.get(feat, float("nan"))),
                    "mean_abs_shap": float(shap_map.get(feat, float("nan"))),
                    "max_psi": max_psi,
                    "mean_signal_rank": mean_rank,
                    "candidate_v3": decision == "keep" and not is_drift,
                }
            )

        candidates = (
            pd.DataFrame(rows)
            .sort_values(["decision", "mean_signal_rank"])
            .reset_index(drop=True)
        )
        library_v3 = [
            r["feature"]
            for _, r in candidates.iterrows()
            if bool(r["candidate_v3"])
        ]

        # category power from MI among keep set
        cat_power: dict[str, float] = {}
        if not mi.empty and "category" in mi.columns:
            keep_set = set(keep)
            sub = mi.loc[mi["feature"].isin(keep_set)]
            if not sub.empty:
                cat_power = (
                    sub.groupby("category")["mutual_information"]
                    .sum()
                    .sort_values(ascending=False)
                    .to_dict()
                )

        summary = {
            "keep": keep,
            "remove": remove,
            "drift_sensitive": drift_sensitive,
            "stable": stable,
            "candidate_library_v3": library_v3,
            "correlated_groups": redundancy_groups,
            "category_power": cat_power,
        }
        return summary, candidates
