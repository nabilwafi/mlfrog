"""Train vs validation drift diagnostics (PSI / KS)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from datasets.entities.dataset import Dataset
from diagnostics.analyzers._common import feature_matrix, map_binary_target, safe_ks, safe_psi


class DriftAnalyzer:
    def analyze(
        self,
        train: Dataset,
        validation: Dataset,
        *,
        y_train: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        p_train: np.ndarray | None = None,
        p_val: np.ndarray | None = None,
        target_mode: str = "exclude_timeout",
        psi_severe: float = 0.25,
        ks_severe: float = 0.2,
    ) -> tuple[dict[str, Any], pd.DataFrame]:
        x_train = feature_matrix(train)
        x_val = feature_matrix(validation)
        rows = []
        for col in x_train.columns:
            a = x_train[col].to_numpy(dtype=float)
            b = x_val[col].to_numpy(dtype=float)
            psi = safe_psi(a, b)
            ks = safe_ks(a, b)
            rows.append(
                {
                    "feature": col,
                    "psi": psi,
                    "ks": ks,
                    "train_mean": float(np.nanmean(a)),
                    "val_mean": float(np.nanmean(b)),
                    "mean_shift": float(np.nanmean(b) - np.nanmean(a)),
                    "severe_psi": bool(psi == psi and psi >= psi_severe),
                    "severe_ks": bool(ks == ks and ks >= ks_severe),
                }
            )
        drift_df = (
            pd.DataFrame(rows)
            .sort_values("psi", ascending=False, na_position="last")
            .reset_index(drop=True)
        )

        if y_train is None or y_val is None:
            _, y_train, _ = map_binary_target(train, target_mode=target_mode)
            _, y_val, _ = map_binary_target(validation, target_mode=target_mode)

        target_drift = {
            "train_positive_rate": float(np.mean(y_train)),
            "val_positive_rate": float(np.mean(y_val)),
            "abs_rate_diff": float(abs(float(np.mean(y_train)) - float(np.mean(y_val)))),
            "psi": safe_psi(y_train.astype(float), y_val.astype(float), bins=2),
            "ks": safe_ks(y_train.astype(float), y_val.astype(float)),
        }

        probability_drift: dict[str, Any] = {}
        if p_train is not None and p_val is not None and len(p_train) and len(p_val):
            probability_drift = {
                "train_mean": float(np.mean(p_train)),
                "val_mean": float(np.mean(p_val)),
                "psi": safe_psi(p_train, p_val),
                "ks": safe_ks(p_train, p_val),
            }

        max_psi = float(drift_df["psi"].max()) if not drift_df.empty else float("nan")
        max_ks = float(drift_df["ks"].max()) if not drift_df.empty else float("nan")
        summary = {
            "max_psi": max_psi,
            "max_ks": max_ks,
            "n_severe_psi_features": int(drift_df["severe_psi"].sum()) if not drift_df.empty else 0,
            "n_severe_ks_features": int(drift_df["severe_ks"].sum()) if not drift_df.empty else 0,
            "psi_severe_threshold": psi_severe,
            "ks_severe_threshold": ks_severe,
            "target_drift": target_drift,
            "probability_drift": probability_drift,
            "top_psi_features": drift_df.head(10)[["feature", "psi", "ks"]].to_dict(
                orient="records"
            ),
        }
        return summary, drift_df
