"""Walk-forward-safe Platt / Isotonic calibration."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from models.calibration.engine import IsotonicCalibrator, PlattCalibrator
from research.probability_calibration.services.metrics import calibration_scores

logger = logging.getLogger(__name__)


def _fit_platt(raw: np.ndarray, y: np.ndarray) -> PlattCalibrator | None:
    if len(y) < 20 or len(np.unique(y)) < 2:
        return None
    lr = LogisticRegression(max_iter=1000, solver="lbfgs")
    lr.fit(np.asarray(raw, dtype=float).reshape(-1, 1), np.asarray(y, dtype=int))
    return PlattCalibrator(lr)


def _fit_isotonic(raw: np.ndarray, y: np.ndarray) -> IsotonicCalibrator | None:
    if len(y) < 20 or len(np.unique(y)) < 2:
        return None
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(np.asarray(raw, dtype=float), np.asarray(y, dtype=int))
    return IsotonicCalibrator(iso)


def calibrate_walk_forward(predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Fit calibrators on train split per (side, window); apply to validation.

    Input columns: side, window, split, y_true, y_prob, ...
    Output: validation rows with y_prob_raw, y_prob_platt, y_prob_isotonic.
    """
    rows: list[pd.DataFrame] = []
    keys = predictions.groupby(["side", "window"], sort=True)
    for (side, window), g in keys:
        train = g.loc[g["split"] == "train"]
        val = g.loc[g["split"] == "validation"].copy()
        if train.empty or val.empty:
            continue
        raw_tr = train["y_prob"].to_numpy(dtype=float)
        y_tr = train["y_true"].to_numpy(dtype=int)
        raw_va = val["y_prob"].to_numpy(dtype=float)

        platt = _fit_platt(raw_tr, y_tr)
        iso = _fit_isotonic(raw_tr, y_tr)

        val["y_prob_raw"] = raw_va
        val["y_prob_platt"] = (
            platt.predict(raw_va) if platt is not None else raw_va.copy()
        )
        val["y_prob_isotonic"] = (
            iso.predict(raw_va) if iso is not None else raw_va.copy()
        )
        logger.info(
            "Calibrated %s/%s val_n=%s",
            side,
            window,
            len(val),
        )
        rows.append(val)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def build_calibration_comparison(calibrated: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for side in sorted(calibrated["side"].unique()):
        sub = calibrated.loc[calibrated["side"] == side]
        y = sub["y_true"].to_numpy(dtype=int)
        for method, col in (
            ("raw", "y_prob_raw"),
            ("platt", "y_prob_platt"),
            ("isotonic", "y_prob_isotonic"),
        ):
            scores = calibration_scores(y, sub[col].to_numpy())
            rows.append(
                {
                    "side": side,
                    "method": method,
                    "n": int(len(sub)),
                    "mean_prob": float(sub[col].mean()),
                    "std_prob": float(sub[col].std(ddof=1)) if len(sub) > 1 else 0.0,
                    **scores,
                }
            )
    return pd.DataFrame(rows)


def pick_best_method(comparison: pd.DataFrame, side: str) -> str:
    """Lowest ECE, then Brier."""
    sub = comparison.loc[comparison["side"] == side].copy()
    if sub.empty:
        return "raw"
    sub = sub.sort_values(["ece", "brier"], ascending=True)
    return str(sub.iloc[0]["method"])
