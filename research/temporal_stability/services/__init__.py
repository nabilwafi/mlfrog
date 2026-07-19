"""Classification + portfolio metrics for temporal stability."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)


def _finite(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return a[np.isfinite(a)]


def ece(y_true: np.ndarray, y_prob: np.ndarray, *, n_bins: int = 10) -> float:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    mask = np.isfinite(p)
    y, p = y[mask], p[mask]
    if len(y) < 5:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    n = len(y)
    for i in range(n_bins):
        m = (p >= bins[i]) & (p < bins[i + 1] if i < n_bins - 1 else p <= bins[i + 1])
        if not np.any(m):
            continue
        total += (m.sum() / n) * abs(y[m].mean() - p[m].mean())
    return float(total)


def calibration_slope_intercept(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """Logistic calibration slope/intercept via simple OLS on logit (ponytail)."""
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)
    mask = np.isfinite(p) & np.isfinite(y)
    y, p = y[mask], p[mask]
    if len(y) < 10 or y.min() == y.max():
        return float("nan"), float("nan")
    logit = np.log(p / (1 - p))
    x = np.column_stack([np.ones(len(logit)), logit])
    try:
        coef, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
        return float(coef[1]), float(coef[0])
    except Exception:
        return float("nan"), float("nan")


def class_metrics(y_true: np.ndarray, y_prob: np.ndarray, *, thr: float = 0.5) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    mask = np.isfinite(p)
    y, p = y[mask], p[mask]
    out = {
        "n": float(len(y)),
        "roc_auc": float("nan"),
        "pr_auc": float("nan"),
        "accuracy": float("nan"),
        "brier": float("nan"),
        "ece": float("nan"),
        "cal_slope": float("nan"),
        "cal_intercept": float("nan"),
    }
    if len(y) < 5 or len(np.unique(y)) < 2:
        return out
    pred = (p >= thr).astype(int)
    out["roc_auc"] = float(roc_auc_score(y, p))
    out["pr_auc"] = float(average_precision_score(y, p))
    out["accuracy"] = float(accuracy_score(y, pred))
    out["brier"] = float(brier_score_loss(y, p))
    out["ece"] = ece(y, p)
    slope, intercept = calibration_slope_intercept(y, p)
    out["cal_slope"] = slope
    out["cal_intercept"] = intercept
    return out


def trade_metrics_from_returns(rets: np.ndarray, *, starting: float = 80.0) -> dict[str, float]:
    r = _finite(np.asarray(rets, dtype=float))
    if len(r) == 0:
        return {
            "trades": 0.0,
            "win_rate": float("nan"),
            "profit_factor": float("nan"),
            "expectancy": float("nan"),
            "total_return": 0.0,
            "final_equity": float(starting),
            "max_drawdown": float("nan"),
        }
    wins = r[r > 0]
    losses = r[r < 0]
    pf = float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 else float("inf")
    eq = starting * np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / np.where(peak > 0, peak, np.nan)
    return {
        "trades": float(len(r)),
        "win_rate": float(np.mean(r > 0)),
        "profit_factor": pf,
        "expectancy": float(np.mean(r)),
        "total_return": float(eq[-1] / starting - 1.0),
        "final_equity": float(eq[-1]),
        "max_drawdown": float(np.nanmax(dd)),
    }


def js_divergence(a: np.ndarray, b: np.ndarray, *, bins: int = 20) -> float:
    a = _finite(a)
    b = _finite(b)
    if len(a) < 5 or len(b) < 5:
        return float("nan")
    lo = min(np.min(a), np.min(b))
    hi = max(np.max(a), np.max(b))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return 0.0
    pa, _ = np.histogram(a, bins=bins, range=(lo, hi), density=True)
    pb, _ = np.histogram(b, bins=bins, range=(lo, hi), density=True)
    pa = pa + 1e-12
    pb = pb + 1e-12
    pa = pa / pa.sum()
    pb = pb / pb.sum()
    m = 0.5 * (pa + pb)
    kl_pm = float(np.sum(pa * np.log(pa / m)))
    kl_qm = float(np.sum(pb * np.log(pb / m)))
    return 0.5 * (kl_pm + kl_qm)


def kl_divergence(a: np.ndarray, b: np.ndarray, *, bins: int = 20) -> float:
    a = _finite(a)
    b = _finite(b)
    if len(a) < 5 or len(b) < 5:
        return float("nan")
    lo = min(np.min(a), np.min(b))
    hi = max(np.max(a), np.max(b))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return 0.0
    pa, _ = np.histogram(a, bins=bins, range=(lo, hi), density=True)
    pb, _ = np.histogram(b, bins=bins, range=(lo, hi), density=True)
    pa = pa + 1e-12
    pb = pb + 1e-12
    pa = pa / pa.sum()
    pb = pb / pb.sum()
    return float(np.sum(pa * np.log(pa / pb)))


def optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray, candidates: tuple[float, ...]) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    best_thr = 0.5
    best_f1 = -1.0
    rows = []
    for thr in candidates:
        pred = (p >= thr).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        rows.append({"threshold": thr, "f1": f1, "precision": prec, "recall": rec})
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    return {"best_threshold": best_thr, "best_f1": best_f1, "grid": rows}
