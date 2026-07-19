"""Bucket / confidence / monotonicity analyzers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score

from research.probability_quality.entities import (
    CONFIDENCE_TIERS,
    PROB_BUCKETS,
    DistSummary,
)


def assign_bucket(prob: float) -> str | None:
    for lo, hi, name in PROB_BUCKETS:
        if hi is None:
            if prob >= lo:
                return name
        elif lo <= prob < hi:
            return name
    return None


def assign_confidence(prob: float) -> str | None:
    for lo, hi, name in CONFIDENCE_TIERS:
        if hi is None:
            if prob >= lo:
                return name
        elif lo <= prob < hi:
            return name
    return None


def distribution_summary(predictions: pd.DataFrame, side: str) -> DistSummary:
    p = predictions.loc[predictions["side"] == side, "y_prob"].astype(float).to_numpy()
    if len(p) == 0:
        nan = float("nan")
        return DistSummary(side=side, n=0, mean=nan, std=nan, p10=nan, p25=nan, p50=nan, p75=nan, p90=nan, p95=nan, p99=nan)
    return DistSummary(
        side=side,
        n=int(len(p)),
        mean=float(np.mean(p)),
        std=float(np.std(p, ddof=1)) if len(p) > 1 else 0.0,
        p10=float(np.percentile(p, 10)),
        p25=float(np.percentile(p, 25)),
        p50=float(np.percentile(p, 50)),
        p75=float(np.percentile(p, 75)),
        p90=float(np.percentile(p, 90)),
        p95=float(np.percentile(p, 95)),
        p99=float(np.percentile(p, 99)),
    )


def _bucket_stats(
    frame: pd.DataFrame,
    *,
    side: str,
    bucket: str,
    lo: float,
    hi: float | None,
    all_positives: int,
) -> dict:
    if hi is None:
        mask = frame["y_prob"] >= lo
    else:
        mask = (frame["y_prob"] >= lo) & (frame["y_prob"] < hi)
    sub = frame.loc[mask]
    n = int(len(sub))
    if n == 0:
        nan = float("nan")
        return {
            "side": side,
            "bucket": bucket,
            "bucket_lo": lo,
            "bucket_hi": hi if hi is not None else 1.0,
            "samples": 0,
            "win_rate": nan,
            "avg_label_return": nan,
            "expectancy": nan,
            "precision": nan,
            "recall": nan,
        }
    y = sub["y_true"].astype(int).to_numpy()
    rets = sub["realized_return"].astype(float).to_numpy()
    win_rate = float(np.mean(y))
    avg_ret = float(np.nanmean(rets))
    # ponytail: expectancy = mean realized return in bucket (unit size)
    expectancy = avg_ret
    # Treat bucket membership as a positive prediction
    precision = float(precision_score(y, np.ones(n, dtype=int), zero_division=0))
    recall = float(np.sum(y) / all_positives) if all_positives > 0 else float("nan")
    return {
        "side": side,
        "bucket": bucket,
        "bucket_lo": lo,
        "bucket_hi": hi if hi is not None else 1.0,
        "samples": n,
        "win_rate": win_rate,
        "avg_label_return": avg_ret,
        "expectancy": expectancy,
        "precision": precision,
        "recall": recall,
    }


def build_bucket_analysis(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for side in sorted(predictions["side"].unique()):
        frame = predictions.loc[predictions["side"] == side].copy()
        all_pos = int((frame["y_true"] == 1).sum())
        for lo, hi, name in PROB_BUCKETS:
            rows.append(
                _bucket_stats(
                    frame, side=side, bucket=name, lo=lo, hi=hi, all_positives=all_pos
                )
            )
    return pd.DataFrame(rows)


def build_confidence_analysis(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for side in sorted(predictions["side"].unique()):
        frame = predictions.loc[predictions["side"] == side].copy()
        frame["confidence"] = frame["y_prob"].map(assign_confidence)
        for _, _, tier in CONFIDENCE_TIERS:
            sub = frame.loc[frame["confidence"] == tier]
            n = int(len(sub))
            if n == 0:
                rows.append(
                    {
                        "side": side,
                        "confidence": tier,
                        "samples": 0,
                        "win_rate": float("nan"),
                        "avg_label_return": float("nan"),
                        "expectancy": float("nan"),
                        "wf_win_rate_mean": float("nan"),
                        "wf_win_rate_std": float("nan"),
                        "wf_windows": 0,
                    }
                )
                continue
            win_rate = float(sub["y_true"].mean())
            avg_ret = float(sub["realized_return"].mean())
            by_win = (
                sub.groupby("valid_year", sort=True)["y_true"]
                .mean()
                .astype(float)
            )
            rows.append(
                {
                    "side": side,
                    "confidence": tier,
                    "samples": n,
                    "win_rate": win_rate,
                    "avg_label_return": avg_ret,
                    "expectancy": avg_ret,
                    "wf_win_rate_mean": float(by_win.mean()) if len(by_win) else float("nan"),
                    "wf_win_rate_std": float(by_win.std(ddof=1)) if len(by_win) > 1 else 0.0,
                    "wf_windows": int(len(by_win)),
                }
            )
    return pd.DataFrame(rows)


def monotonicity_metrics(bucket_df: pd.DataFrame, side: str) -> dict[str, float | bool | str]:
    """Spearman-style rank check: higher bucket mid -> higher win_rate / expectancy."""
    sub = bucket_df.loc[(bucket_df["side"] == side) & (bucket_df["samples"] > 0)].copy()
    if len(sub) < 2:
        return {
            "side": side,
            "n_buckets": int(len(sub)),
            "win_rate_spearman": float("nan"),
            "expectancy_spearman": float("nan"),
            "win_rate_monotonic": False,
            "expectancy_monotonic": False,
        }
    sub = sub.sort_values("bucket_lo")
    mid = ((sub["bucket_lo"] + sub["bucket_hi"]) / 2.0).to_numpy()
    wr = sub["win_rate"].to_numpy(dtype=float)
    exp = sub["expectancy"].to_numpy(dtype=float)

    def _spearman(x: np.ndarray, y: np.ndarray) -> float:
        if len(x) < 2 or np.all(y == y[0]):
            return float("nan")
        rx = pd.Series(x).rank().to_numpy()
        ry = pd.Series(y).rank().to_numpy()
        if np.std(rx) == 0 or np.std(ry) == 0:
            return float("nan")
        return float(np.corrcoef(rx, ry)[0, 1])

    def _nondecreasing(y: np.ndarray) -> bool:
        return bool(np.all(np.diff(y) >= -1e-12))

    return {
        "side": side,
        "n_buckets": int(len(sub)),
        "win_rate_spearman": _spearman(mid, wr),
        "expectancy_spearman": _spearman(mid, exp),
        "win_rate_monotonic": _nondecreasing(wr),
        "expectancy_monotonic": _nondecreasing(exp),
    }


def build_decile_analysis(predictions: pd.DataFrame, *, n_bins: int = 10) -> pd.DataFrame:
    """Equal-count probability ranks — usable when mass is collapsed below 0.50."""
    rows: list[dict] = []
    for side in sorted(predictions["side"].unique()):
        frame = predictions.loc[predictions["side"] == side].copy()
        if frame.empty:
            continue
        try:
            frame["decile"] = pd.qcut(
                frame["y_prob"],
                q=min(n_bins, max(2, frame["y_prob"].nunique())),
                labels=False,
                duplicates="drop",
            )
        except ValueError:
            continue
        all_pos = int((frame["y_true"] == 1).sum())
        for d, sub in frame.groupby("decile", sort=True):
            n = int(len(sub))
            y = sub["y_true"].astype(int).to_numpy()
            rets = sub["realized_return"].astype(float).to_numpy()
            lo = float(sub["y_prob"].min())
            hi = float(sub["y_prob"].max())
            win_rate = float(np.mean(y))
            avg_ret = float(np.nanmean(rets))
            rows.append(
                {
                    "side": side,
                    "decile": int(d) + 1,
                    "bucket": f"D{int(d) + 1}",
                    "bucket_lo": lo,
                    "bucket_hi": hi,
                    "samples": n,
                    "win_rate": win_rate,
                    "avg_label_return": avg_ret,
                    "expectancy": avg_ret,
                    "precision": float(precision_score(y, np.ones(n, dtype=int), zero_division=0)),
                    "recall": float(np.sum(y) / all_pos) if all_pos > 0 else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def build_relative_confidence(predictions: pd.DataFrame) -> pd.DataFrame:
    """Low/med/high by within-side probability terciles (for collapsed distributions)."""
    rows: list[dict] = []
    for side in sorted(predictions["side"].unique()):
        frame = predictions.loc[predictions["side"] == side].copy()
        if frame.empty:
            continue
        try:
            frame["confidence"] = pd.qcut(
                frame["y_prob"],
                q=3,
                labels=["low", "medium", "high"],
                duplicates="drop",
            )
        except ValueError:
            frame["confidence"] = "low"
        for tier in ("low", "medium", "high"):
            sub = frame.loc[frame["confidence"].astype(str) == tier]
            n = int(len(sub))
            if n == 0:
                rows.append(
                    {
                        "side": side,
                        "confidence": tier,
                        "scheme": "relative_tercile",
                        "samples": 0,
                        "win_rate": float("nan"),
                        "avg_label_return": float("nan"),
                        "expectancy": float("nan"),
                        "wf_win_rate_mean": float("nan"),
                        "wf_win_rate_std": float("nan"),
                        "wf_windows": 0,
                        "prob_lo": float("nan"),
                        "prob_hi": float("nan"),
                    }
                )
                continue
            by_win = sub.groupby("valid_year", sort=True)["y_true"].mean().astype(float)
            rows.append(
                {
                    "side": side,
                    "confidence": tier,
                    "scheme": "relative_tercile",
                    "samples": n,
                    "win_rate": float(sub["y_true"].mean()),
                    "avg_label_return": float(sub["realized_return"].mean()),
                    "expectancy": float(sub["realized_return"].mean()),
                    "wf_win_rate_mean": float(by_win.mean()) if len(by_win) else float("nan"),
                    "wf_win_rate_std": float(by_win.std(ddof=1)) if len(by_win) > 1 else 0.0,
                    "wf_windows": int(len(by_win)),
                    "prob_lo": float(sub["y_prob"].min()),
                    "prob_hi": float(sub["y_prob"].max()),
                }
            )
    return pd.DataFrame(rows)


def collapsed_flag(dist: DistSummary) -> bool:
    return bool(dist.n > 0 and (dist.std < 0.05 or (dist.p99 - dist.p10) < 0.08))


def tradable_range(
    bucket_df: pd.DataFrame,
    side: str,
    *,
    min_win_rate: float | None = 0.5,
) -> str:
    """Contiguous high-prob range with positive expectancy and elevated win_rate."""
    sub = bucket_df.loc[bucket_df["side"] == side].copy()
    if sub.empty:
        return "n/a"
    # For collapsed models, default 0.5 is unreachable — use mean + epsilon
    if min_win_rate is None:
        filled = sub.loc[sub["samples"] > 0, "win_rate"]
        base = float(filled.mean()) if not filled.empty else 0.5
        min_win_rate = base
    good = sub.loc[
        (sub["samples"] >= 30)
        & (sub["win_rate"] > min_win_rate)
        & (sub["expectancy"] > 0)
    ].sort_values("bucket_lo")
    if good.empty:
        good = sub.loc[
            (sub["samples"] > 0)
            & (sub["win_rate"] > min_win_rate)
            & (sub["expectancy"] > 0)
        ].sort_values("bucket_lo")
    if good.empty:
        return f"none (no bucket with win_rate>{min_win_rate:.3f} and positive expectancy)"
    return f"{good['bucket'].iloc[0]} .. {good['bucket'].iloc[-1]} (n={int(good['samples'].sum())})"
