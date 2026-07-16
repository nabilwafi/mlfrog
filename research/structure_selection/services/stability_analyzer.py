"""Feature stability across walk-forward windows."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from research.structure_selection.entities.selection_result import ExperimentResult


def build_feature_stability(results: list[ExperimentResult]) -> pd.DataFrame:
    """Rank consistency of structure features across windows (gain + split + SHAP)."""
    rows: list[dict] = []
    # Focus on experiments that include structure features
    for result in results:
        if result.experiment_id == "A_baseline":
            continue
        per_feat: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: {"gain": [], "split": [], "shap": [], "rank_gain": []}
        )
        for w in result.windows:
            if w.status != "ok":
                continue
            gain = w.gain_importance
            if not gain:
                continue
            ranked = sorted(gain.items(), key=lambda kv: kv[1], reverse=True)
            rank_map = {name: i + 1 for i, (name, _) in enumerate(ranked)}
            for name, g in gain.items():
                per_feat[name]["gain"].append(g)
                per_feat[name]["rank_gain"].append(float(rank_map[name]))
                per_feat[name]["split"].append(float(w.split_importance.get(name, 0.0)))
                per_feat[name]["shap"].append(float(w.shap_importance.get(name, 0.0)))

        for name, buckets in per_feat.items():
            gains = buckets["gain"]
            ranks = buckets["rank_gain"]
            if not gains:
                continue
            rows.append(
                {
                    "side": result.side,
                    "experiment_id": result.experiment_id,
                    "feature": name,
                    "n_windows": len(gains),
                    "mean_gain": float(np.mean(gains)),
                    "std_gain": float(np.std(gains, ddof=1)) if len(gains) > 1 else 0.0,
                    "mean_split": float(np.mean(buckets["split"])),
                    "mean_shap": float(np.mean(buckets["shap"])),
                    "mean_rank_gain": float(np.mean(ranks)),
                    "std_rank_gain": float(np.std(ranks, ddof=1)) if len(ranks) > 1 else 0.0,
                    "top1_count": int(sum(1 for r in ranks if r == 1.0)),
                    "top3_count": int(sum(1 for r in ranks if r <= 3.0)),
                    "stability_score": float(
                        1.0 / (1.0 + (np.std(ranks, ddof=1) if len(ranks) > 1 else 0.0))
                    ),
                }
            )
    return pd.DataFrame(rows)
