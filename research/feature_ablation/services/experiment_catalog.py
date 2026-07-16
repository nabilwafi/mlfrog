"""Build the 13 ablation experiment definitions and resolve feature lists."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from research.feature_ablation.entities.ablation_experiment import AblationExperiment


def default_experiment_catalog() -> list[AblationExperiment]:
    return [
        AblationExperiment(
            "baseline",
            "Baseline",
            "baseline",
            "All engineered features.",
        ),
        AblationExperiment(
            "remove_trend",
            "Remove Trend",
            "remove_category",
            "Drop every trend feature.",
            remove_category="trend",
        ),
        AblationExperiment(
            "remove_momentum",
            "Remove Momentum",
            "remove_category",
            "Drop every momentum feature.",
            remove_category="momentum",
        ),
        AblationExperiment(
            "remove_volatility",
            "Remove Volatility",
            "remove_category",
            "Drop every volatility feature.",
            remove_category="volatility",
        ),
        AblationExperiment(
            "remove_candle",
            "Remove Candle",
            "remove_category",
            "Drop every candle feature.",
            remove_category="candle",
        ),
        AblationExperiment(
            "remove_session",
            "Remove Session",
            "remove_category",
            "Drop every session feature.",
            remove_category="session",
        ),
        AblationExperiment(
            "remove_statistical",
            "Remove Statistical",
            "remove_category",
            "Drop every statistical feature.",
            remove_category="statistical",
        ),
        AblationExperiment(
            "stable_only",
            "Stable Features Only",
            "stable_only",
            "Only features flagged stable in diagnostics.",
        ),
        AblationExperiment(
            "keep_only",
            "Keep Features Only",
            "keep_only",
            "Only features marked KEEP in diagnostics.",
        ),
        AblationExperiment(
            "top20_shap",
            "Top 20 SHAP",
            "top_shap",
            "Top 20 features by mean absolute SHAP.",
            top_k=20,
        ),
        AblationExperiment(
            "top10_shap",
            "Top 10 SHAP",
            "top_shap",
            "Top 10 features by mean absolute SHAP.",
            top_k=10,
        ),
        AblationExperiment(
            "top20_mi",
            "Top 20 Mutual Information",
            "top_mi",
            "Top 20 features by mutual information.",
            top_k=20,
        ),
        AblationExperiment(
            "top10_mi",
            "Top 10 Mutual Information",
            "top_mi",
            "Top 10 features by mutual information.",
            top_k=10,
        ),
    ]


class ExperimentResolver:
    """Resolve concrete feature lists from metadata + diagnostics artifacts."""

    def __init__(
        self,
        *,
        all_features: list[str],
        category_by_feature: dict[str, str],
        keep_features: list[str],
        stable_features: list[str],
        shap_ranked: list[str],
        mi_ranked: list[str],
    ) -> None:
        self.all_features = list(all_features)
        self.category_by_feature = dict(category_by_feature)
        self.keep_features = [f for f in keep_features if f in self.all_features]
        self.stable_features = [f for f in stable_features if f in self.all_features]
        self.shap_ranked = [f for f in shap_ranked if f in self.all_features]
        self.mi_ranked = [f for f in mi_ranked if f in self.all_features]

    def resolve(self, experiment: AblationExperiment) -> AblationExperiment:
        if experiment.kind == "baseline":
            names = self.all_features
        elif experiment.kind == "remove_category":
            cat = (experiment.remove_category or "").lower()
            names = [f for f in self.all_features if self.category_by_feature.get(f, "") != cat]
        elif experiment.kind == "stable_only":
            names = self.stable_features or self.all_features
        elif experiment.kind == "keep_only":
            names = self.keep_features or self.all_features
        elif experiment.kind == "top_shap":
            k = int(experiment.top_k or 20)
            names = self.shap_ranked[:k] or self.all_features[:k]
        elif experiment.kind == "top_mi":
            k = int(experiment.top_k or 20)
            names = self.mi_ranked[:k] or self.all_features[:k]
        else:
            raise ValueError(f"unknown experiment kind {experiment.kind!r}")
        if not names:
            raise ValueError(f"experiment {experiment.experiment_id} resolved to zero features")
        return experiment.with_features(names)

    def category_counts(self, features: list[str] | tuple[str, ...]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for f in features:
            cat = self.category_by_feature.get(f, "unknown")
            counts[cat] = counts.get(cat, 0) + 1
        return counts


def load_diagnostics_lists(diagnostics_dir: Path) -> dict[str, Any]:
    """Load KEEP / stable / SHAP / MI rankings from Sprint 6.5 artifacts."""
    candidates = diagnostics_dir / "feature_selection_candidates.csv"
    shap_path = diagnostics_dir / "feature_shap_importance.csv"
    mi_path = diagnostics_dir / "feature_mutual_information.csv"
    stability_path = diagnostics_dir / "feature_stability.csv"

    keep: list[str] = []
    stable: list[str] = []
    if candidates.is_file():
        cdf = pd.read_csv(candidates)
        keep = cdf.loc[cdf["decision"] == "keep", "feature"].astype(str).tolist()
    if stability_path.is_file():
        sdf = pd.read_csv(stability_path)
        if "stable_flag" in sdf.columns:
            stable = sdf.loc[sdf["stable_flag"].astype(bool), "feature"].astype(str).tolist()

    shap_ranked: list[str] = []
    if shap_path.is_file():
        sh = pd.read_csv(shap_path).sort_values("mean_abs_shap", ascending=False)
        shap_ranked = sh["feature"].astype(str).tolist()

    mi_ranked: list[str] = []
    if mi_path.is_file():
        mi = pd.read_csv(mi_path).sort_values("mutual_information", ascending=False)
        mi_ranked = mi["feature"].astype(str).tolist()

    return {
        "keep": keep,
        "stable": stable,
        "shap_ranked": shap_ranked,
        "mi_ranked": mi_ranked,
    }
