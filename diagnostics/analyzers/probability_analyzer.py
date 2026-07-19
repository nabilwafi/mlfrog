"""Probability distribution + threshold sweep diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import f1_score, precision_score, recall_score


class ProbabilityAnalyzer:
    def analyze(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        *,
        n_bins: int = 20,
    ) -> tuple[dict[str, Any], pd.DataFrame]:
        y_true = np.asarray(y_true, dtype=int).ravel()
        y_proba = np.asarray(y_proba, dtype=float).ravel()
        percentiles = {
            f"p{p}": float(np.percentile(y_proba, p))
            for p in (1, 5, 10, 25, 50, 75, 90, 95, 99)
        }
        hist_counts, hist_edges = np.histogram(y_proba, bins=n_bins, range=(0.0, 1.0))
        hist_rows = []
        for i in range(len(hist_counts)):
            lo = float(hist_edges[i])
            hi = float(hist_edges[i + 1])
            hist_rows.append(
                {
                    "bin_left": lo,
                    "bin_right": hi,
                    "bin_center": (lo + hi) / 2.0,
                    "count": int(hist_counts[i]),
                    "density": float(hist_counts[i] / max(len(y_proba), 1)),
                }
            )
        hist_df = pd.DataFrame(hist_rows)

        pos = y_proba[y_true == 1]
        neg = y_proba[y_true == 0]
        by_class = {
            "positive_mean": float(pos.mean()) if len(pos) else float("nan"),
            "negative_mean": float(neg.mean()) if len(neg) else float("nan"),
            "positive_std": float(pos.std(ddof=1)) if len(pos) > 1 else float("nan"),
            "negative_std": float(neg.std(ddof=1)) if len(neg) > 1 else float("nan"),
            "positive_median": float(np.median(pos)) if len(pos) else float("nan"),
            "negative_median": float(np.median(neg)) if len(neg) else float("nan"),
        }

        # Overlap: fraction of mass in shared support via histogram intersection
        overlap = float("nan")
        if len(pos) and len(neg):
            c_pos, edges = np.histogram(pos, bins=n_bins, range=(0.0, 1.0), density=True)
            c_neg, _ = np.histogram(neg, bins=edges, density=True)
            widths = np.diff(edges)
            overlap = float(np.minimum(c_pos, c_neg).dot(widths))

        calibration = {}
        if len(np.unique(y_true)) >= 2 and len(y_true) >= 10:
            frac_pos, mean_pred = calibration_curve(
                y_true, y_proba, n_bins=min(10, max(2, len(y_true) // 50)), strategy="quantile"
            )
            calibration = {
                "fraction_positives": [float(x) for x in frac_pos.tolist()],
                "mean_predicted": [float(x) for x in mean_pred.tolist()],
            }

        std = float(y_proba.std(ddof=1)) if len(y_proba) > 1 else 0.0
        pmin = float(y_proba.min()) if len(y_proba) else float("nan")
        pmax = float(y_proba.max()) if len(y_proba) else float("nan")
        pmean = float(y_proba.mean()) if len(y_proba) else float("nan")
        pmedian = float(np.median(y_proba)) if len(y_proba) else float("nan")

        flags = {
            "collapsed_probability_distribution": bool(std < 0.02 or (pmax - pmin) < 0.05),
            "collapsed_predictions": bool(std < 0.01),
            "overconfident_model": bool(
                (len(pos) and float(np.mean(pos > 0.9)) > 0.3)
                or (len(neg) and float(np.mean(neg < 0.1)) > 0.3)
            ),
            "underconfident_model": bool(std < 0.05 and 0.4 <= pmean <= 0.6),
        }

        summary: dict[str, Any] = {
            "min": pmin,
            "max": pmax,
            "mean": pmean,
            "median": pmedian,
            "std": std,
            "percentiles": percentiles,
            "histogram_bins": n_bins,
            "probability_by_class": by_class,
            "probability_overlap": overlap,
            "calibration_curve": calibration,
            "flags": flags,
        }
        return summary, hist_df

    def threshold_sweep(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        *,
        start: float = 0.05,
        stop: float = 0.95,
        step: float = 0.05,
    ) -> tuple[dict[str, Any], pd.DataFrame]:
        y_true = np.asarray(y_true, dtype=int).ravel()
        y_proba = np.asarray(y_proba, dtype=float).ravel()
        rows = []
        thresholds = np.round(np.arange(start, stop + 1e-9, step), 2)
        for thr in thresholds:
            pred = (y_proba >= thr).astype(int)
            rows.append(
                {
                    "threshold": float(thr),
                    "precision": float(precision_score(y_true, pred, zero_division=0)),
                    "recall": float(recall_score(y_true, pred, zero_division=0)),
                    "f1": float(f1_score(y_true, pred, zero_division=0)),
                    "predicted_positives": int((pred == 1).sum()),
                    "predicted_negatives": int((pred == 0).sum()),
                }
            )
        frame = pd.DataFrame(rows)
        best = frame.loc[frame["f1"].idxmax()].to_dict() if not frame.empty else {}
        return {"best_f1_threshold": best, "n_thresholds": int(len(frame))}, frame
