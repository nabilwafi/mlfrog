"""Analyzers for probability collapse diagnosis."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from research.context_impact.services.walk_forward_runner import probability_separation


def _pct(arr: np.ndarray, q: float) -> float:
    if len(arr) == 0:
        return float("nan")
    return float(np.percentile(arr, q))


def _dist_row(
    *,
    side: str,
    experiment_id: str,
    split: str,
    window: str | None,
    valid_year: int | None,
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, Any]:
    p = np.asarray(y_prob, dtype=float)
    y = np.asarray(y_true, dtype=int)
    n = int(len(p))
    pos = int((y == 1).sum()) if n else 0
    neg = int((y == 0).sum()) if n else 0
    pos_rate = float(pos / n) if n else float("nan")
    roc = float("nan")
    sep = float("nan")
    if n and len(np.unique(y)) >= 2:
        roc = float(roc_auc_score(y, p))
        sep = float(probability_separation(y, p))
    return {
        "side": side,
        "experiment_id": experiment_id,
        "split": split,
        "window": window if window is not None else "all",
        "valid_year": valid_year if valid_year is not None else -1,
        "n": n,
        "pos_count": pos,
        "neg_count": neg,
        "pos_rate": pos_rate,
        "neg_rate": float(neg / n) if n else float("nan"),
        "imbalance_ratio": float(neg / pos) if pos > 0 else float("nan"),
        "mean": float(np.mean(p)) if n else float("nan"),
        "std": float(np.std(p, ddof=1)) if n > 1 else 0.0,
        "min": float(np.min(p)) if n else float("nan"),
        "max": float(np.max(p)) if n else float("nan"),
        "range": float(np.max(p) - np.min(p)) if n else float("nan"),
        "p10": _pct(p, 10),
        "p25": _pct(p, 25),
        "p50": _pct(p, 50),
        "p75": _pct(p, 75),
        "p90": _pct(p, 90),
        "p95": _pct(p, 95),
        "p99": _pct(p, 99),
        "frac_above_050": float(np.mean(p >= 0.5)) if n else float("nan"),
        "mean_minus_pos_rate": float(np.mean(p) - pos_rate) if n else float("nan"),
        "probability_separation": sep,
        "roc_auc": roc,
    }


def build_label_distribution(predictions: pd.DataFrame) -> pd.DataFrame:
    """Label imbalance from validation rows (v2), pooled across windows."""
    rows = []
    sub = predictions.loc[
        (predictions["split"] == "validation")
        & (predictions["experiment_id"] == "B_v2_context")
    ]
    for side in sorted(sub["side"].unique()):
        y = sub.loc[sub["side"] == side, "y_true"].astype(int).to_numpy()
        n = len(y)
        pos = int((y == 1).sum())
        neg = int((y == 0).sum())
        rows.append(
            {
                "side": side,
                "n": n,
                "pos_count": pos,
                "neg_count": neg,
                "pos_rate": float(pos / n) if n else float("nan"),
                "neg_rate": float(neg / n) if n else float("nan"),
                "imbalance_ratio_neg_over_pos": float(neg / pos) if pos else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def build_prediction_distribution(predictions: pd.DataFrame) -> pd.DataFrame:
    """Train/val pooled distributions per side × experiment."""
    rows = []
    for (side, exp, split), g in predictions.groupby(
        ["side", "experiment_id", "split"], sort=True
    ):
        rows.append(
            _dist_row(
                side=side,
                experiment_id=exp,
                split=split,
                window=None,
                valid_year=None,
                y_true=g["y_true"].to_numpy(),
                y_prob=g["y_prob"].to_numpy(),
            )
        )
    return pd.DataFrame(rows)


def build_wf_drift(predictions: pd.DataFrame) -> pd.DataFrame:
    """Per-window validation probability stats (baseline + v2)."""
    rows = []
    sub = predictions.loc[predictions["split"] == "validation"]
    for (side, exp, window, year), g in sub.groupby(
        ["side", "experiment_id", "window", "valid_year"], sort=True
    ):
        rows.append(
            _dist_row(
                side=side,
                experiment_id=exp,
                split="validation",
                window=window,
                valid_year=int(year),
                y_true=g["y_true"].to_numpy(),
                y_prob=g["y_prob"].to_numpy(),
            )
        )
    return pd.DataFrame(rows)


def build_baseline_vs_v2(prediction_distribution: pd.DataFrame) -> pd.DataFrame:
    """Side-level baseline vs v2 on validation."""
    rows = []
    val = prediction_distribution.loc[prediction_distribution["split"] == "validation"]
    for side in sorted(val["side"].unique()):
        base = val.loc[
            (val["side"] == side) & (val["experiment_id"] == "A_baseline")
        ]
        v2 = val.loc[
            (val["side"] == side) & (val["experiment_id"] == "B_v2_context")
        ]
        if base.empty or v2.empty:
            continue
        b, v = base.iloc[0], v2.iloc[0]
        rows.append(
            {
                "side": side,
                "baseline_std": b["std"],
                "v2_std": v["std"],
                "delta_std": float(v["std"] - b["std"]),
                "baseline_range": b["range"],
                "v2_range": v["range"],
                "delta_range": float(v["range"] - b["range"]),
                "baseline_sep": b["probability_separation"],
                "v2_sep": v["probability_separation"],
                "delta_sep": float(v["probability_separation"] - b["probability_separation"]),
                "baseline_roc": b["roc_auc"],
                "v2_roc": v["roc_auc"],
                "delta_roc": float(v["roc_auc"] - b["roc_auc"]),
                "baseline_mean": b["mean"],
                "v2_mean": v["mean"],
                "baseline_frac_above_050": b["frac_above_050"],
                "v2_frac_above_050": v["frac_above_050"],
                "baseline_mean_minus_pos_rate": b["mean_minus_pos_rate"],
                "v2_mean_minus_pos_rate": v["mean_minus_pos_rate"],
            }
        )
    return pd.DataFrame(rows)


def diagnose_answers(
    *,
    labels: pd.DataFrame,
    pred_dist: pd.DataFrame,
    compare: pd.DataFrame,
    trainer_params: dict[str, Any],
) -> dict[str, Any]:
    """Heuristic answers to Sprint-15 research questions."""
    # Q1 imbalance: mean close to pos_rate + mild imbalance
    imbalance_cause = False
    imbalance_notes = []
    for _, row in labels.iterrows():
        side = row["side"]
        pos_rate = float(row["pos_rate"])
        imb = float(row["imbalance_ratio_neg_over_pos"])
        v2_val = pred_dist.loc[
            (pred_dist["side"] == side)
            & (pred_dist["experiment_id"] == "B_v2_context")
            & (pred_dist["split"] == "validation")
        ]
        if v2_val.empty:
            continue
        mean_p = float(v2_val.iloc[0]["mean"])
        gap = abs(mean_p - pos_rate)
        imbalance_notes.append(
            f"{side}: pos_rate={pos_rate:.3f} mean_p={mean_p:.3f} |gap|={gap:.3f} neg/pos={imb:.2f}"
        )
        if gap < 0.03 and imb > 1.1:
            imbalance_cause = True

    # Q2 regularization: train also collapsed, high min_child_samples / early stopping
    min_child = int(trainer_params.get("min_child_samples", 40))
    early = int(trainer_params.get("early_stopping_rounds", 50))
    reg_alpha = float(trainer_params.get("reg_alpha", 0.0))
    reg_lambda = float(trainer_params.get("reg_lambda", 0.0))
    train_collapsed = False
    train_notes = []
    for side in sorted(pred_dist["side"].unique()):
        tr = pred_dist.loc[
            (pred_dist["side"] == side)
            & (pred_dist["experiment_id"] == "B_v2_context")
            & (pred_dist["split"] == "train")
        ]
        if tr.empty:
            continue
        std = float(tr.iloc[0]["std"])
        frac = float(tr.iloc[0]["frac_above_050"])
        train_notes.append(f"{side} train std={std:.4f} frac>=0.5={frac:.4f}")
        if std < 0.05 and frac < 0.05:
            train_collapsed = True
    reg_likely = train_collapsed and (min_child >= 20 or early >= 20 or reg_alpha > 0 or reg_lambda > 0)
    # Stronger: if train AND val collapsed similarly → prior-pulling / weak signal + regularization
    # not val-only shrinkage
    reg_answer = (
        "contributing (train also collapsed; conservative LGBM settings)"
        if reg_likely
        else (
            "partial — train less collapsed than val"
            if not train_collapsed
            else "unlikely primary cause"
        )
    )

    # Q3 ranking vs confidence
    ranking_only = True
    conf_notes = []
    for _, row in compare.iterrows():
        side = row["side"]
        d_roc = float(row["delta_roc"])
        d_std = float(row["delta_std"])
        d_range = float(row["delta_range"])
        d_frac = float(row["v2_frac_above_050"] - row["baseline_frac_above_050"])
        conf_notes.append(
            f"{side}: dROC={d_roc:+.4f} dStd={d_std:+.4f} dRange={d_range:+.4f} dFrac>=0.5={d_frac:+.4f}"
        )
        # context increases confidence if std/range/frac_above_050 rise meaningfully
        if d_std > 0.005 or d_range > 0.02 or d_frac > 0.01:
            ranking_only = False

    # Q4 calibration
    # Proceed if ranking improved (positive dROC) even if collapsed — with percentile thresholds
    proceed = False
    if not compare.empty and float(compare["delta_roc"].max()) > 0.002:
        proceed = True

    return {
        "imbalance_cause": imbalance_cause,
        "imbalance_notes": "; ".join(imbalance_notes),
        "reg_answer": reg_answer,
        "reg_params": (
            f"min_child_samples={min_child}, early_stopping_rounds={early}, "
            f"reg_alpha={reg_alpha}, reg_lambda={reg_lambda}"
        ),
        "train_notes": "; ".join(train_notes),
        "ranking_only": ranking_only,
        "conf_notes": "; ".join(conf_notes),
        "proceed_calibration": proceed,
    }
