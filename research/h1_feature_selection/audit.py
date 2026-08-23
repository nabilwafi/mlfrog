"""Phase 2–4: definition notes, quality audit, redundancy on H1 feature matrix."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research.h1_feature_selection import FEATURE_GROUP, H1_ENGINE_38

# Exact formulas (causal) — documentation + audit helpers.
FEATURE_DEFINITIONS: dict[str, str] = {
    "ema20_distance_atr": "(close - ema20) / atr14",
    "ema50_distance_atr": "(close - ema50) / atr14",
    "ema20_distance_percent": "(close - ema20) / close * 100",
    "ema50_distance_percent": "(close - ema50) / close * 100",
    "ema_cross_distance": "(ema20 - ema50) / atr14",
    "ema20_slope": "(ema20_t - ema20_{t-3}) / 3 / |ema20|",
    "ema50_slope": "(ema50_t - ema50_{t-3}) / 3 / |ema50|",
    "ema_trend_duration": "streak length of (ema20 > ema50) regime",
    "ema_alignment_score": "mean(close>ema20, close>ema50, ema20>ema50) mapped to [-1,1]",
    "atr_percent": "atr14 / close * 100",
    "atr_percentile_252": "rolling percentile rank of atr14 over 252",
    "rolling_volatility": "rolling std of log returns (window 20)",
    "volatility_rank": "rolling rank of rolling_volatility",
    "volatility_regime_score": "(atr_percentile_252 - 0.5) * 2  [deterministic of CORE]",
    "macd_normalized": "macd / atr14",
    "macd_histogram_zscore": "rolling z-score of macd hist (100)",
    "macd_signal_distance": "(macd - signal) / atr14",
    "momentum_rank": "rolling rank of pct_change(10)",
    "rsi_percentile": "rolling percentile of rsi14",
    "roc_3": "close.pct_change(3)",
    "roc_12": "close.pct_change(12)",
    "momentum_acceleration": "diff(pct_change(6), 3)",
    "body_percent": "|close-open| / (high-low)",
    "upper_wick_percent": "upper wick / range",
    "lower_wick_percent": "lower wick / range",
    "close_position": "(close-low)/(high-low)",
    "range_percent": "(high-low)/close * 100",
    "body_rank": "rolling rank of body_percent",
    "hour_sin": "sin(2π hour/24)",
    "hour_cos": "cos(2π hour/24)",
    "day_sin": "sin(2π weekday/7)",
    "day_cos": "cos(2π weekday/7)",
    "rolling_zscore": "(ret - mean20) / std20",
    "rolling_percentile": "fraction of window returns ≤ current ret",
    "rolling_rank": "average-rank of current ret in window / n → [0,1]",
    "rolling_quantile": "window quantile(q=0.75) of returns (LEVEL, not rank)",
    "rolling_std": "std20 of returns",
    "rolling_mean_distance": "(ret - mean20) / std20  [≈ rolling_zscore]",
}


def load_h1_matrix(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    missing = [c for c in H1_ENGINE_38 if c not in df.columns]
    if missing:
        raise RuntimeError(f"feature matrix missing: {missing}")
    return df[["timestamp", *H1_ENGINE_38]].sort_values("timestamp").reset_index(drop=True)


def quality_audit(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    n = len(df)
    for col in H1_ENGINE_38:
        s = df[col].astype(float)
        finite = np.isfinite(s.to_numpy())
        vals = s.to_numpy()[finite]
        var = float(np.var(vals)) if len(vals) else 0.0
        rows.append(
            {
                "feature": col,
                "group": FEATURE_GROUP[col],
                "n": n,
                "n_finite": int(finite.sum()),
                "pct_missing": float(1.0 - finite.mean()),
                "n_inf": int(np.isinf(s.to_numpy()).sum()),
                "n_unique": int(pd.Series(vals).nunique()) if len(vals) else 0,
                "variance": var,
                "near_zero_var": bool(var < 1e-12),
                "mean": float(np.mean(vals)) if len(vals) else np.nan,
                "std": float(np.std(vals)) if len(vals) else np.nan,
                "p01": float(np.percentile(vals, 1)) if len(vals) else np.nan,
                "p99": float(np.percentile(vals, 99)) if len(vals) else np.nan,
                "definition": FEATURE_DEFINITIONS.get(col, ""),
                "causal_ok": True,  # builders audited: no center=True / future shift
                "exclude_reason": "",
            }
        )
    out = pd.DataFrame(rows)
    # Flag near-identical by construction (not auto-remove)
    notes = {
        "rolling_mean_distance": "NEAR_IDENTICAL_TO:rolling_zscore",
        "volatility_regime_score": "DETERMINISTIC_OF:atr_percentile_252 (CORE)",
    }
    for feat, note in notes.items():
        out.loc[out["feature"] == feat, "exclude_reason"] = note
    return out


def redundancy_matrices(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x = df[list(H1_ENGINE_38)].astype(float)
    pearson = x.corr(method="pearson")
    spearman = x.corr(method="spearman")
    # High-|ρ| pairs among candidates vs all
    pairs: list[dict[str, Any]] = []
    cols = list(H1_ENGINE_38)
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            p = float(pearson.loc[a, b])
            s = float(spearman.loc[a, b])
            if abs(p) >= 0.90 or abs(s) >= 0.90:
                pairs.append(
                    {
                        "feature_a": a,
                        "feature_b": b,
                        "group_a": FEATURE_GROUP[a],
                        "group_b": FEATURE_GROUP[b],
                        "pearson": p,
                        "spearman": s,
                    }
                )
    pairs_df = pd.DataFrame(pairs).sort_values("pearson", key=lambda s: s.abs(), ascending=False)
    return pearson, spearman, pairs_df


def statistical_relationship_note() -> str:
    return """
### Rolling statistical family (verified in code)

| Feature | Formula | Role |
|---------|---------|------|
| `rolling_zscore` | (ret − mean20) / std20 | standardized current return |
| `rolling_mean_distance` | (ret − mean20) / std20 | **identical formula** to zscore |
| `rolling_percentile` | P(window ret ≤ current) | rank of current ret ∈ [0,1] |
| `rolling_rank` | midrank(current)/n | near-equivalent to percentile |
| `rolling_quantile` | quantile(0.75) of window rets | **distribution LEVEL**, not current rank |
| `rolling_std` | std20(ret) | vol of returns |

Conclusion: `rolling_quantile` (in CORE) is **not** interchangeable with percentile/rank.
`rolling_zscore` ≈ `rolling_mean_distance` → treat as redundant pair.
"""
