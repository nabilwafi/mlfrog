"""Per-feature train vs validation drift with severity labels."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from datasets.entities.dataset import Dataset
from research.analyzers._common import drift_severity, feature_cols, safe_ks, safe_psi


class FeatureDriftAnalyzer:
    def analyze(
        self,
        train: Dataset | pd.DataFrame,
        validation: Dataset | pd.DataFrame,
        *,
        mild: float = 0.1,
        severe: float = 0.25,
    ) -> tuple[dict[str, Any], pd.DataFrame]:
        train_df = train.frame if isinstance(train, Dataset) else train
        val_df = validation.frame if isinstance(validation, Dataset) else validation
        feats = [c for c in feature_cols(train_df) if c in val_df.columns]
        rows = []
        for feat in feats:
            a = train_df[feat].to_numpy(dtype=float)
            b = val_df[feat].to_numpy(dtype=float)
            a_f = a[np.isfinite(a)]
            b_f = b[np.isfinite(b)]
            psi = safe_psi(a_f, b_f)
            ks = safe_ks(a_f, b_f)
            rows.append(
                {
                    "feature": feat,
                    "train_mean": float(np.mean(a_f)) if len(a_f) else float("nan"),
                    "validation_mean": float(np.mean(b_f)) if len(b_f) else float("nan"),
                    "train_std": float(np.std(a_f, ddof=1)) if len(a_f) > 1 else 0.0,
                    "validation_std": float(np.std(b_f, ddof=1)) if len(b_f) > 1 else 0.0,
                    "psi": psi,
                    "ks": ks,
                    "drift_severity": drift_severity(psi, mild=mild, severe=severe),
                }
            )
        drift = (
            pd.DataFrame(rows)
            .sort_values("psi", ascending=False, na_position="last")
            .reset_index(drop=True)
        )
        summary = {
            "n_features": int(len(drift)),
            "n_severe": int((drift["drift_severity"] == "severe").sum()) if not drift.empty else 0,
            "n_moderate": int((drift["drift_severity"] == "moderate").sum()) if not drift.empty else 0,
            "n_stable": int((drift["drift_severity"] == "stable").sum()) if not drift.empty else 0,
            "max_psi": float(drift["psi"].max()) if not drift.empty else float("nan"),
            "top_drift_features": drift.head(15).to_dict(orient="records"),
        }
        return summary, drift
