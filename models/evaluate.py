"""Evaluation helpers for baseline binary classifiers."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from models.data_prep import BREAKEVEN_WINRATE

# Reference threshold (no tuning).
DEFAULT_REF_THRESHOLD = 0.5

# Operational threshold candidates — selected ONLY on validation.
SELECTION_THRESHOLDS = [round(0.50 + 0.01 * i, 2) for i in range(11)]  # 0.50..0.60
MIN_SIGNALS_FOR_SELECTION = 200


def binary_metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> dict:
    y_pred = (y_prob >= thr).astype(int)
    n_pos = int(y_pred.sum())
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    edge = precision - BREAKEVEN_WINRATE
    return {
        "threshold": thr,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_signals": n_pos,
        "breakeven_winrate": BREAKEVEN_WINRATE,
        "edge_vs_breakeven": edge,
        "breakeven_flag": "ABOVE breakeven" if edge >= 0 else "BELOW breakeven",
    }


def threshold_table(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Iterable[float],
) -> pd.DataFrame:
    return pd.DataFrame(
        [binary_metrics_at_threshold(y_true, y_prob, float(t)) for t in thresholds]
    )


def select_threshold_from_validation(
    y_val: np.ndarray,
    p_val: np.ndarray,
    *,
    thresholds: Iterable[float] = SELECTION_THRESHOLDS,
    min_signals: int = MIN_SIGNALS_FOR_SELECTION,
) -> dict:
    """Pick operational threshold using VALIDATION ONLY.

    Rules:
    1. Candidate must have n_signals >= min_signals on validation.
    2. Among those, keep only ABOVE breakeven.
    3. Pick highest validation precision.
    4. If none qualify: selected_thr=None and note explicitly.
    """
    table = threshold_table(y_val, p_val, thresholds)
    eligible = table[
        (table["n_signals"] >= min_signals)
        & (table["breakeven_flag"] == "ABOVE breakeven")
    ].copy()

    if eligible.empty:
        return {
            "selected_thr": None,
            "selection_note": (
                "no threshold beats breakeven with sufficient sample size on validation "
                f"(min_signals={min_signals}, grid={list(thresholds)})"
            ),
            "val_table": table,
            "val_at_selected": None,
        }

    best = eligible.sort_values(
        ["precision", "n_signals"], ascending=[False, False]
    ).iloc[0]
    return {
        "selected_thr": float(best["threshold"]),
        "selection_note": (
            f"selected_thr={float(best['threshold']):.2f} from validation "
            f"(precision={float(best['precision']):.4f}, n={int(best['n_signals'])}, "
            f"min_signals={min_signals})"
        ),
        "val_table": table,
        "val_at_selected": best.to_dict(),
    }


def consistency_flag(val_flag: str | None, test_flag: str | None) -> str:
    if val_flag is None or test_flag is None:
        return "N/A"
    if val_flag == test_flag:
        return "CONSISTENT"
    return "INCONSISTENT"


def evaluate_probs(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    thresholds: Iterable[float] = SELECTION_THRESHOLDS,
) -> dict:
    """Compute AUC, @0.5 reference metrics, threshold table, calibration."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    out: dict = {}
    if len(np.unique(y_true)) < 2:
        out["auc_roc"] = float("nan")
        out["auc_pr"] = float("nan")
    else:
        out["auc_roc"] = float(roc_auc_score(y_true, y_prob))
        out["auc_pr"] = float(average_precision_score(y_true, y_prob))

    out["at_0_5"] = binary_metrics_at_threshold(y_true, y_prob, DEFAULT_REF_THRESHOLD)
    out["threshold_table"] = threshold_table(y_true, y_prob, thresholds)
    out["calibration"] = calibration_table(y_true, y_prob, n_bins=10)
    return out


def calibration_table(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Decile calibration: mean predicted prob vs realized win-rate."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    try:
        bins = pd.qcut(y_prob, q=n_bins, duplicates="drop")
    except ValueError:
        ranks = pd.Series(y_prob).rank(method="first")
        bins = pd.qcut(ranks, q=n_bins, duplicates="drop")

    df = pd.DataFrame({"y": y_true, "p": y_prob, "bin": bins})
    rows = []
    for i, (bin_label, g) in enumerate(df.groupby("bin", observed=True), start=1):
        rows.append(
            {
                "bin": i,
                "bin_range": str(bin_label),
                "n": len(g),
                "mean_pred_prob": float(g["p"].mean()),
                "realized_winrate": float(g["y"].mean()),
                "gap": float(g["p"].mean() - g["y"].mean()),
            }
        )
    return pd.DataFrame(rows)


def format_threshold_table(df: pd.DataFrame) -> str:
    lines = [
        "| threshold | precision | recall | f1 | n_signals | breakeven | flag |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['threshold']:.2f} | {r['precision']:.4f} | {r['recall']:.4f} | "
            f"{r['f1']:.4f} | {int(r['n_signals'])} | {r['breakeven_winrate']:.4f} | "
            f"{r['breakeven_flag']} |"
        )
    return "\n".join(lines)


def format_calibration_table(df: pd.DataFrame) -> str:
    lines = [
        "| bin | n | mean_pred_prob | realized_winrate | gap (pred-real) |",
        "|---:|---:|---:|---:|---:|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {int(r['bin'])} | {int(r['n'])} | {r['mean_pred_prob']:.4f} | "
            f"{r['realized_winrate']:.4f} | {r['gap']:.4f} |"
        )
    return "\n".join(lines)
