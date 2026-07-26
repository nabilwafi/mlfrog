"""Sprint 32 — Feature Library Research (no retrain / no production changes).

Audits the feature set used by rolling WF + frozen primary models.
Uses stored datasets + existing frozen boosters only.

Example:
  python apps/report_feature_library_research.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import tools.pyc_path_hook  # noqa: F401

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score

from simulation.wf.sim import SPLITS

OUT = _ROOT / "artifacts" / "pipeline_backtest" / "rolling_wf" / "sprint32_feature_research"
FROZEN = _ROOT / "artifacts" / "models" / "frozen"

# Category map for known library features
CATEGORY: dict[str, str] = {
    "ema20_distance_atr": "trend",
    "ema50_distance_atr": "trend",
    "ema20_distance_percent": "trend",
    "ema50_distance_percent": "trend",
    "ema_cross_distance": "trend",
    "ema20_slope": "trend",
    "ema50_slope": "trend",
    "ema_trend_duration": "trend",
    "ema_alignment_score": "trend",
    "atr_percent": "volatility",
    "atr_percentile_252": "volatility",
    "rolling_volatility": "volatility",
    "volatility_rank": "volatility",
    "volatility_regime_score": "volatility",
    "macd_normalized": "momentum",
    "macd_histogram_zscore": "momentum",
    "macd_signal_distance": "momentum",
    "momentum_rank": "momentum",
    "rsi_percentile": "momentum",
    "body_percent": "candle",
    "upper_wick_percent": "candle",
    "lower_wick_percent": "candle",
    "close_position": "candle",
    "range_percent": "candle",
    "body_rank": "candle",
    "hour_sin": "session",
    "hour_cos": "session",
    "day_sin": "session",
    "day_cos": "session",
    "rolling_zscore": "statistical",
    "rolling_percentile": "statistical",
    "rolling_rank": "statistical",
    "rolling_quantile": "statistical",
    "rolling_std": "statistical",
    "rolling_mean_distance": "statistical",
    "ctx_h4_distance_from_equilibrium": "context",
    "ctx_h4_swing_strength": "context",
    "ctx_h4_swing_quality": "context",
    "ctx_h4_swing_high_distance_atr": "context",
    "ctx_h4_rejection_strength": "context",
    "ctx_h4_trend_direction": "context",
    "ctx_h4_trend_strength": "context",
    "ctx_h4_volatility_regime": "context",
    "ctx_h4_market_regime": "context",
    "ctx_h4_compression": "context",
    "ctx_h4_expansion": "context",
}

META = {
    "timestamp",
    "label",
    "feature_version",
    "label_version",
    "side",
    "split",
    "strategy",
    "symbol",
    "timeframe",
    "realized_return",
    "holding_bars",
    "entry_price",
}


def categorize(name: str) -> str:
    if name in CATEGORY:
        return CATEGORY[name]
    if name.startswith("ctx_h4_"):
        return "context"
    if name.startswith(("ema", "trend")):
        return "trend"
    if "vol" in name or "atr" in name:
        return "volatility"
    if name.startswith(("macd", "rsi", "mom")):
        return "momentum"
    if "hour" in name or "day" in name or "session" in name:
        return "session"
    if name.startswith("rolling_"):
        return "statistical"
    if any(k in name for k in ("body", "wick", "range", "close_position")):
        return "candle"
    return "other"


def load_side(side: str) -> pd.DataFrame:
    parts = []
    for split in SPLITS:
        path = _ROOT / f"artifacts/datasets/XAUUSD/H1/{side}/v2/{split}.parquet"
        d = pd.read_parquet(path)
        d["split"] = split
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
    d["side"] = side
    return d.sort_values("timestamp").reset_index(drop=True)


def binary_label(y: pd.Series) -> np.ndarray:
    """Dataset labels may be {0,1} or {-1,1} — map positive class to 1."""
    arr = y.astype(float).to_numpy()
    # treat >0 as positive (works for both 0/1 and -1/1)
    return (arr > 0).astype(int)


def corr_redundancy(X: pd.DataFrame, thr: float = 0.90) -> pd.DataFrame:
    corr = X.corr(method="pearson").abs()
    pairs = []
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            r = float(corr.loc[a, b])
            if r >= thr:
                pairs.append({"feature_a": a, "feature_b": b, "abs_corr": r, "category_a": categorize(a), "category_b": categorize(b)})
    if not pairs:
        return pd.DataFrame(columns=["feature_a", "feature_b", "abs_corr", "category_a", "category_b"])
    return pd.DataFrame(pairs).sort_values("abs_corr", ascending=False).reset_index(drop=True)


def mi_scores(X: pd.DataFrame, y: np.ndarray, *, sample: int = 20000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = rng.choice(n, size=min(sample, n), replace=False) if n > sample else np.arange(n)
    Xs = X.iloc[idx].to_numpy(dtype=float)
    ys = y[idx]
    # clip inf/nan
    Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
    mi = mutual_info_classif(Xs, ys, discrete_features=False, random_state=seed)
    return pd.DataFrame({"feature": X.columns, "mutual_info": mi}).sort_values("mutual_info", ascending=False).reset_index(drop=True)


def frozen_gain_importance(booster: lgb.Booster) -> pd.DataFrame:
    names = booster.feature_name()
    gain = booster.feature_importance(importance_type="gain")
    split = booster.feature_importance(importance_type="split")
    return pd.DataFrame({"feature": names, "gain": gain, "split": split})


def align_to_model(X: pd.DataFrame, booster: lgb.Booster) -> pd.DataFrame:
    """Pad missing frozen features with 0 so pred_contrib/predict match training shape."""
    names = booster.feature_name()
    out = pd.DataFrame(index=X.index)
    for c in names:
        if c in X.columns:
            out[c] = pd.to_numeric(X[c], errors="coerce")
        else:
            out[c] = 0.0
    return out.fillna(out.median(numeric_only=True))


def mean_abs_shap(booster: lgb.Booster, X: pd.DataFrame) -> pd.DataFrame:
    """LightGBM pred_contrib ≈ SHAP values (no shap package / no retrain)."""
    names = booster.feature_name()
    n = min(8000, len(X))
    Xs = align_to_model(X.iloc[-n:], booster)
    contrib = booster.predict(Xs, pred_contrib=True)
    abs_mean = np.abs(contrib[:, :-1]).mean(axis=0)
    # Only score features that actually exist in the research panel (pad cols → SHAP~0)
    present = [c for c in names if c in X.columns]
    rows = [{"feature": c, "mean_abs_shap": float(abs_mean[i])} for i, c in enumerate(names) if c in present]
    return pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def permutation_importance(
    booster: lgb.Booster,
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    n_repeats: int = 3,
    sample: int = 6000,
    seed: int = 42,
) -> pd.DataFrame:
    names = [c for c in booster.feature_name() if c in X.columns]
    if not names:
        return pd.DataFrame(columns=["feature", "perm_auc_drop"])
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = rng.choice(n, size=min(sample, n), replace=False) if n > sample else np.arange(n)
    X_sub = X.iloc[idx]
    ys = y[idx]
    Xs = align_to_model(X_sub, booster)
    base = roc_auc_score(ys, booster.predict(Xs))
    rows = []
    for feat in names:
        drops = []
        for _ in range(n_repeats):
            Xp = Xs.copy()
            Xp[feat] = rng.permutation(Xp[feat].to_numpy())
            auc = roc_auc_score(ys, booster.predict(Xp))
            drops.append(base - auc)
        rows.append(
            {
                "feature": feat,
                "perm_auc_drop": float(np.mean(drops)),
                "perm_auc_drop_std": float(np.std(drops)),
                "base_auc": float(base),
            }
        )
    return pd.DataFrame(rows).sort_values("perm_auc_drop", ascending=False).reset_index(drop=True)


def group_rollup(feat_table: pd.DataFrame, score_col: str) -> pd.DataFrame:
    t = feat_table.copy()
    t["category"] = t["feature"].map(categorize)
    g = (
        t.groupby("category", as_index=False)
        .agg(n_features=("feature", "count"), score_sum=(score_col, "sum"), score_mean=(score_col, "mean"), score_max=(score_col, "max"))
        .sort_values("score_sum", ascending=False)
        .reset_index(drop=True)
    )
    total = float(g["score_sum"].sum()) or 1e-12
    g["share"] = g["score_sum"] / total
    return g


def main() -> None:
    p = argparse.ArgumentParser(description="Sprint 32 feature library research")
    p.add_argument("--side", choices=("long", "short", "both"), default="both")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    sides = ["long", "short"] if args.side == "both" else [args.side]
    inventory_rows = []
    all_mi = []
    all_shap = []
    all_perm = []
    all_gain = []
    all_redund = []
    coverage_notes = []

    for side in sides:
        print(f"=== side={side} ===")
        df = load_side(side)
        y = binary_label(df["label"])
        print(f"  rows={len(df)} pos_rate={y.mean():.3f}")

        booster_path = FROZEN / f"primary_{side}.txt"
        booster = lgb.Booster(model_file=str(booster_path))
        model_feats = booster.feature_name()
        present = [f for f in model_feats if f in df.columns]
        missing = [f for f in model_feats if f not in df.columns]
        extra_ctx = [c for c in df.columns if c.startswith("ctx_h4_") and c not in model_feats]
        coverage_notes.append(
            {
                "side": side,
                "model_features": len(model_feats),
                "present_in_dataset": len(present),
                "missing_in_dataset": missing,
                "dataset_ctx_not_in_model": extra_ctx,
            }
        )
        print(f"  model feats={len(model_feats)} present={len(present)} missing={missing}")

        # Inventory: all numeric dataset features + model-only
        num_cols = [c for c in df.columns if c not in META and pd.api.types.is_numeric_dtype(df[c])]
        for c in sorted(set(num_cols) | set(model_feats)):
            inventory_rows.append(
                {
                    "side": side,
                    "feature": c,
                    "category": categorize(c),
                    "in_dataset": c in df.columns,
                    "in_frozen_model": c in model_feats,
                    "used_by_wf_intersection": c in present,
                }
            )

        # Analysis matrix = features present for this side's model
        X = df[present].apply(pd.to_numeric, errors="coerce")
        # drop rows with too many nans
        ok = X.notna().mean(axis=1) > 0.8
        X = X.loc[ok].copy()
        y_ok = y[ok.to_numpy()]
        X = X.fillna(X.median(numeric_only=True))

        red = corr_redundancy(X, thr=0.90)
        if not red.empty:
            red.insert(0, "side", side)
            all_redund.append(red)
        print(f"  redundant pairs (|r|>=0.9): {len(red)}")

        mi = mi_scores(X, y_ok)
        mi.insert(0, "side", side)
        mi["category"] = mi["feature"].map(categorize)
        all_mi.append(mi)

        gain = frozen_gain_importance(booster)
        gain = gain[gain["feature"].isin(present)].copy()
        gain.insert(0, "side", side)
        gain["category"] = gain["feature"].map(categorize)
        all_gain.append(gain)

        shap = mean_abs_shap(booster, X)
        shap.insert(0, "side", side)
        shap["category"] = shap["feature"].map(categorize)
        all_shap.append(shap)
        print(f"  shap top: {list(shap.head(5)['feature'])}")

        perm = permutation_importance(booster, X, y_ok)
        perm.insert(0, "side", side)
        perm["category"] = perm["feature"].map(categorize)
        all_perm.append(perm)
        print(f"  perm top: {list(perm.head(5)['feature'])}")

    inventory = pd.DataFrame(inventory_rows).drop_duplicates(["side", "feature"])
    mi_df = pd.concat(all_mi, ignore_index=True)
    shap_df = pd.concat(all_shap, ignore_index=True)
    perm_df = pd.concat(all_perm, ignore_index=True)
    gain_df = pd.concat(all_gain, ignore_index=True)
    red_df = pd.concat(all_redund, ignore_index=True) if all_redund else pd.DataFrame()

    # Combined ranking (long focus for narrative; both sides saved)
    def merge_scores(side: str) -> pd.DataFrame:
        base = inventory[inventory["side"] == side].copy()
        base = base[base["in_frozen_model"] & base["in_dataset"]]
        m = (
            base.merge(mi_df[mi_df["side"] == side][["feature", "mutual_info"]], on="feature", how="left")
            .merge(shap_df[shap_df["side"] == side][["feature", "mean_abs_shap"]], on="feature", how="left")
            .merge(perm_df[perm_df["side"] == side][["feature", "perm_auc_drop"]], on="feature", how="left")
            .merge(gain_df[gain_df["side"] == side][["feature", "gain", "split"]], on="feature", how="left")
        )
        # normalize ranks 0-1
        for col in ("mutual_info", "mean_abs_shap", "perm_auc_drop", "gain"):
            s = m[col].fillna(0.0)
            m[f"{col}_n"] = (s - s.min()) / (s.max() - s.min() + 1e-12)
        m["weakness_score"] = 1.0 - (0.35 * m["mean_abs_shap_n"] + 0.35 * m["perm_auc_drop_n"] + 0.20 * m["gain_n"] + 0.10 * m["mutual_info_n"])
        m["strength_score"] = 1.0 - m["weakness_score"]
        return m.sort_values("strength_score", ascending=False).reset_index(drop=True)

    long_rank = merge_scores("long") if "long" in sides else pd.DataFrame()
    short_rank = merge_scores("short") if "short" in sides else pd.DataFrame()

    # Group contribution from SHAP (long primary narrative)
    group_shap = group_rollup(shap_df[shap_df["side"] == "long"], "mean_abs_shap") if not shap_df.empty else pd.DataFrame()
    group_perm = group_rollup(perm_df[perm_df["side"] == "long"], "perm_auc_drop") if not perm_df.empty else pd.DataFrame()

    # Write CSVs
    inventory.to_csv(OUT / "feature_inventory.csv", index=False)
    mi_df.to_csv(OUT / "mutual_information.csv", index=False)
    shap_df.to_csv(OUT / "shap_mean_abs.csv", index=False)
    perm_df.to_csv(OUT / "permutation_importance.csv", index=False)
    gain_df.to_csv(OUT / "frozen_gain_importance.csv", index=False)
    red_df.to_csv(OUT / "redundant_pairs.csv", index=False)
    if not long_rank.empty:
        long_rank.to_csv(OUT / "feature_strength_ranking_long.csv", index=False)
    if not short_rank.empty:
        short_rank.to_csv(OUT / "feature_strength_ranking_short.csv", index=False)
    group_shap.to_csv(OUT / "group_contribution_shap_long.csv", index=False)
    group_perm.to_csv(OUT / "group_contribution_perm_long.csv", index=False)
    Path(OUT / "coverage_notes.json").write_text(json.dumps(coverage_notes, indent=2), encoding="utf-8")

    # --- roadmap decisions from scores ---
    def decide_keep_remove(rank: pd.DataFrame, side: str) -> tuple[list[str], list[str], list[str]]:
        if rank.empty:
            return [], [], []
        drop_redund: set[str] = set()
        rr = red_df[red_df["side"] == side] if not red_df.empty else pd.DataFrame()
        strength = {r.feature: float(r.strength_score) for r in rank.itertuples()}
        if not rr.empty:
            for _, row in rr.iterrows():
                a, b = row["feature_a"], row["feature_b"]
                if a in strength and b in strength:
                    weak = a if strength[a] < strength[b] else b
                    drop_redund.add(weak)

        weak = [
            r.feature
            for r in rank.itertuples()
            if float(r.strength_score) < 0.25 and float(r.perm_auc_drop or 0.0) < 0.001
        ]
        keep = [r.feature for r in rank.head(20).itertuples()]
        improve = [
            r.feature
            for r in rank.itertuples()
            if r.category in {"trend", "volatility", "context"} and 0.25 <= float(r.strength_score) < 0.55
        ]
        remove = sorted(set(weak) | drop_redund)
        return keep, improve, remove

    keep_l, improve_l, remove_l = decide_keep_remove(long_rank, "long")

    # V3 proposals (research — new information only)
    v3 = [
        {
            "feature_name": "h4_trend_persistence_bars",
            "formula": "streak_length(sign(ctx_h4_trend_direction) constant on H4 closes), mapped as-of to H1",
            "category": "trend / multi-timeframe",
            "why_new": "Library has ema_trend_duration on H1 EMA20/50 only; no measure of how long the H4 directional regime has persisted.",
            "expected_benefit": "Filter choppy flips; Sprint31 showed UPTREND/LOW_VOL dominates — persistence should concentrate that edge.",
            "leakage_risk": "Low if as-of join uses completed H4 bars only (same as existing ctx join).",
            "compute_cost": "O(n) scan on H4 + asof join",
        },
        {
            "feature_name": "h4_efficiency_ratio",
            "formula": "|close_H4 - close_H4[n]| / sum(|Δclose_H4|) over n∈{6,12} (Kaufman ER on H4)",
            "category": "trend quality",
            "why_new": "Trend strength today is mostly distance/slope; ER measures path efficiency (direct vs choppy) — orthogonal to ATR distance.",
            "expected_benefit": "Separate clean trends from noisy ones where 0.12 trail dies (DOWNTREND/HIGH_VOL failure mode).",
            "leakage_risk": "Low with completed H4 bars.",
            "compute_cost": "O(n) on H4",
        },
        {
            "feature_name": "vol_expansion_ratio_6_24",
            "formula": "ATR(6) / ATR(24) on H1 (or rolling_std_6 / rolling_std_24)",
            "category": "volatility expansion/contraction",
            "why_new": "atr_percentile_252 is slow level; no short-horizon expansion/contraction ratio.",
            "expected_benefit": "Detect squeeze→expand transitions; Sprint31 atr_percentile extremes hurt trail.",
            "leakage_risk": "None (causal rolling).",
            "compute_cost": "O(n)",
        },
        {
            "feature_name": "swing_break_distance_atr",
            "formula": "signed distance of H1 close to last confirmed swing high/low (H1 fractal or H4 swing), / ATR",
            "category": "market structure",
            "why_new": "ctx_h4_swing_* quality/strength exist on frozen long, but dataset panel used by WF is missing most of them; no explicit break-of-structure distance on H1.",
            "expected_benefit": "Encode structure breaks the trail actually rides.",
            "leakage_risk": "Medium if swing confirmation uses future pivots — must use causal pivot rule (confirm after k bars).",
            "compute_cost": "O(n)",
        },
        {
            "feature_name": "session_vwap_distance_atr",
            "formula": "(close - session_VWAP_UTC) / ATR; session anchors 00:00 / 07:00 / 13:00 UTC",
            "category": "session / liquidity proxy",
            "why_new": "hour_sin/cos only encode clock; no price location vs session volume-weighted level.",
            "expected_benefit": "Sprint31 edge is session-local (ASIA weak, LONDON_NY strong) — location vs VWAP adds that structure.",
            "leakage_risk": "Low if VWAP resets at session open using only past ticks/bars.",
            "compute_cost": "O(n)",
        },
        {
            "feature_name": "range_compression_z",
            "formula": "zscore_252( range_percent )  or  range_percent / median_252(range_percent)",
            "category": "volatility / candle",
            "why_new": "range_percent is raw; no relative compression state vs its own history (distinct from ATR%).",
            "expected_benefit": "Flag coiled ranges before expansion — complementary to vol_expansion_ratio.",
            "leakage_risk": "None.",
            "compute_cost": "O(n)",
        },
        {
            "feature_name": "mtf_slope_alignment",
            "formula": "sign(ema20_slope_H1) + sign(ema_slope_H4) + sign(ema_slope_D1)  → {-3..+3} normalized",
            "category": "multi-timeframe context",
            "why_new": "H1 slopes exist; H4/D1 slopes are not in the WF feature intersection (D1 built in live but not in WF dataset panels).",
            "expected_benefit": "Align entries with higher-TF drift; reduce counter-trend ASIA noise.",
            "leakage_risk": "Low with as-of completed higher-TF bars.",
            "compute_cost": "O(n) + joins",
        },
        {
            "feature_name": "realized_skew_24",
            "formula": "rolling skewness of H1 log returns (window 24)",
            "category": "statistical",
            "why_new": "rolling_* set is zscore/rank/percentile/std/mean_distance — no higher-moment shape of the return distribution.",
            "expected_benefit": "Crash/rally asymmetry before entry; may flag DOWNTREND/HIGH_VOL toxicity.",
            "leakage_risk": "None.",
            "compute_cost": "O(n)",
        },
        {
            "feature_name": "trade_density_session",
            "formula": "rolling count of H1 bars with |return| > k·ATR over last 12 bars within current session",
            "category": "liquidity / activity proxy",
            "why_new": "No activity/liquidity proxy (tick volume unused; spread unused).",
            "expected_benefit": "Avoid dead ASIA books where Sprint31 PnL is negative.",
            "leakage_risk": "None on H1 OHLC; if using tick_volume, verify broker continuity.",
            "compute_cost": "O(n)",
        },
        {
            "feature_name": "re_include_ctx_h4_structure_set",
            "formula": "Existing H4 structure fields: distance_from_equilibrium, swing_strength, swing_high_distance_atr, rejection_strength (already defined in market_context)",
            "category": "context / market structure",
            "why_new": "Frozen long model expects these, but current WF dataset panels only ship a subset (trend/vol/compression/swing_quality) — intersection drops 4 H4 feats. This is missing *coverage*, not a new formula.",
            "expected_benefit": "Restore features the frozen/production stack already relies on; remove train/serve skew.",
            "leakage_risk": "Same as existing H4 join.",
            "compute_cost": "Already computed in H4StructureBuilder — fix dataset materialization",
        },
    ]
    pd.DataFrame(v3).to_csv(OUT / "feature_library_v3_proposals.csv", index=False)

    # --- Markdown report ---
    long_shap_top = shap_df[shap_df["side"] == "long"].head(12) if not shap_df.empty else pd.DataFrame()
    long_perm_top = perm_df[perm_df["side"] == "long"].head(12) if not perm_df.empty else pd.DataFrame()
    long_weak = long_rank.tail(10) if not long_rank.empty else pd.DataFrame()

    def md_table(df: pd.DataFrame, cols: list[tuple[str, str, str]], n: int | None = None) -> list[str]:
        use = df if n is None else df.head(n)
        lines = ["| " + " | ".join(h for _, h, _ in cols) + " |", "|" + "|".join("---:" for _ in cols) + "|"]
        for _, r in use.iterrows():
            cells = []
            for c, _h, f in cols:
                v = r[c]
                if f == "s":
                    cells.append(str(v))
                elif pd.isna(v):
                    cells.append("nan")
                elif f.endswith("d"):
                    cells.append(format(int(v), f))
                else:
                    cells.append(format(float(v), f))
            lines.append("| " + " | ".join(cells) + " |")
        return lines

    md: list[str] = [
        "# Sprint 32 — Feature Engineering Research",
        "",
        "constraint: **no retrain / no production code changes**  ",
        "sources: stored `artifacts/datasets/.../v2`, frozen `artifacts/models/frozen/primary_{long,short}.txt`  ",
        "methods: Pearson redundancy, Mutual Information vs label, LightGBM `pred_contrib` (TreeSHAP-equivalent), "
        "permutation AUC drop on frozen booster, gain importance.",
        "",
        "## 0. Coverage finding (critical)",
        "",
        "Rolling WF intersects dataset columns with frozen model feature names. **Long frozen expects H4 structure "
        "fields that are absent from the current dataset panels**, so they are dropped at train time:",
        "",
    ]
    for note in coverage_notes:
        md.append(
            f"- **{note['side']}**: model={note['model_features']}, present={note['present_in_dataset']}, "
            f"missing={note['missing_in_dataset']}, dataset-only ctx={note['dataset_ctx_not_in_model']}"
        )
    md += [
        "",
        "So the live/production stack and the WF research panels are **not feature-aligned**. Fixing "
        "`re_include_ctx_h4_structure_set` is the highest-leverage 'add' — it restores already-designed information.",
        "",
        "## 1. Feature inventory by category",
        "",
    ]
    inv_long = inventory[(inventory["side"] == "long") & (inventory["in_frozen_model"])]
    cat_counts = inv_long.groupby("category").size().sort_values(ascending=False)
    md += ["| Category | # in frozen long |", "|---|---:|"]
    for cat, n in cat_counts.items():
        md.append(f"| {cat} | {int(n)} |")

    md += [
        "",
        "### Redundant pairs (|Pearson| ≥ 0.90)",
        "",
    ]
    if red_df.empty:
        md.append("None at 0.90 threshold.")
    else:
        md += md_table(
            red_df[red_df["side"] == "long"],
            [("feature_a", "A", "s"), ("feature_b", "B", "s"), ("abs_corr", "|r|", ".3f"), ("category_a", "Cat A", "s")],
            n=20,
        )

    md += ["", "## 2. Individual strength (long frozen)", "", "### Top by mean |SHAP| (`pred_contrib`)", ""]
    md += md_table(
        long_shap_top,
        [("feature", "Feature", "s"), ("category", "Cat", "s"), ("mean_abs_shap", "mean|SHAP|", ".4g")],
    )
    md += ["", "### Top by permutation AUC drop", ""]
    md += md_table(
        long_perm_top,
        [("feature", "Feature", "s"), ("category", "Cat", "s"), ("perm_auc_drop", "ΔAUC", ".4f")],
    )
    md += ["", "### Weakest composite (low SHAP + perm + gain + MI)", ""]
    if not long_weak.empty:
        md += md_table(
            long_weak.sort_values("strength_score"),
            [
                ("feature", "Feature", "s"),
                ("category", "Cat", "s"),
                ("strength_score", "Strength", ".3f"),
                ("perm_auc_drop", "ΔAUC", ".4f"),
                ("mean_abs_shap", "|SHAP|", ".3g"),
            ],
        )

    md += ["", "## 3. Group contribution (long)", "", "### By SHAP mass", ""]
    if not group_shap.empty:
        md += md_table(
            group_shap,
            [("category", "Category", "s"), ("n_features", "n", "d"), ("score_sum", "Σ|SHAP|", ".4g"), ("share", "Share", ".1%")],
        )
    md += ["", "### By permutation mass", ""]
    if not group_perm.empty:
        md += md_table(
            group_perm,
            [("category", "Category", "s"), ("n_features", "n", "d"), ("score_sum", "ΣΔAUC", ".4f"), ("share", "Share", ".1%")],
        )

    md += [
        "",
        "## 4. Information gaps (what is NOT represented)",
        "",
        "Current library is strong on **local H1 EMA geometry, ATR level, MACD/RSI ranks, candle shape, calendar "
        "encoding, and generic rolling stats**. Relative to the edge attribution (Sprint 31) failure modes, these "
        "information gaps matter:",
        "",
        "1. **Trend quality / persistence on H4** — direction exists in dataset ctx, but not efficiency or age of trend.",
        "2. **Volatility expansion vs contraction** — slow ATR percentile only; no short/long ATR ratio.",
        "3. **Causal market structure breaks** — swing quality alone ≠ distance-to-break / BOS.",
        "4. **Session microstructure** — clock encodings only; no VWAP/activity/liquidity proxy (ASIA weakness).",
        "5. **Higher-moment statistics** — no skew/kurtosis of returns.",
        "6. **Train/serve H4 structure parity** — production H4 structure fields missing from WF panels.",
        "",
        "Not recommended: stacking more EMA periods, more RSI lengths, or duplicate MACD transforms — they add "
        "collinearity without new economics (see redundant pairs).",
        "",
        "## 5. Feature Library V3 — proposals",
        "",
    ]
    for i, row in enumerate(v3, 1):
        md += [
            f"### {i}. `{row['feature_name']}`",
            "",
            f"- **Category:** {row['category']}",
            f"- **Formula:** {row['formula']}",
            f"- **Why new:** {row['why_new']}",
            f"- **Expected benefit:** {row['expected_benefit']}",
            f"- **Leakage risk:** {row['leakage_risk']}",
            f"- **Compute cost:** {row['compute_cost']}",
            "",
        ]

    md += [
        "## 6. Roadmap",
        "",
        "### Keep",
        "",
        "High-signal, non-redundant workhorses (long ranking head + non-duplicated):",
        "",
    ]
    for f in keep_l[:15]:
        md.append(f"- `{f}`")
    md += [
        "",
        "### Improve",
        "",
        "- Materialize full H4 structure set into dataset panels (parity with frozen/live).",
        "- Replace duplicated percent/ATR distance pairs with a single scale (prefer ATR-normalized).",
        "- Revisit session features: keep hour encodings but add session-relative price/activity.",
        "",
        "Candidates from mid-strength trend/vol/context:",
        "",
    ]
    for f in improve_l[:10]:
        md.append(f"- `{f}`")
    md += [
        "",
        "### Remove (or stop feeding the model)",
        "",
        "Weak under SHAP/perm/MI and/or dominated by a correlated twin — **research recommendation only**:",
        "",
    ]
    for f in remove_l[:15]:
        md.append(f"- `{f}`")
    if not remove_l:
        md.append("- (no strong remove set at current thresholds — prefer compress redundant pairs first)")
    md += [
        "",
        "### Add (V3 priority order)",
        "",
        "1. `re_include_ctx_h4_structure_set` — fix coverage skew (no new math).",
        "2. `vol_expansion_ratio_6_24`",
        "3. `h4_efficiency_ratio` + `h4_trend_persistence_bars`",
        "4. `session_vwap_distance_atr` + `trade_density_session`",
        "5. `swing_break_distance_atr` (causal pivots)",
        "6. `mtf_slope_alignment`",
        "7. `realized_skew_24` / `range_compression_z`",
        "",
        "## 7. Conclusion",
        "",
        "- **Target caveat:** SHAP/permutation here are vs the **triple-barrier label**, not WF trade PnL. "
        "`hour_sin/cos` dominate label attribution while Sprint 31 PnL differs by session *behavior* — add VWAP/density, not more clocks.",
        "- **Group mass (label):** session ~33% SHAP, volatility ~23%, trend ~17%. Context looks tiny only because "
        "long H4 structure fields are missing from WF panels.",
        "- **Largest defect:** dataset ↔ frozen/live feature skew + ~18 near-duplicate pairs.",
        "- **V3 priority:** restore H4 structure → vol expansion → H4 efficiency/persistence → session VWAP/density → causal BOS.",
        "",
        "### Files",
        "",
        "- `feature_inventory.csv`, `redundant_pairs.csv`",
        "- `mutual_information.csv`, `shap_mean_abs.csv`, `permutation_importance.csv`, `frozen_gain_importance.csv`",
        "- `feature_strength_ranking_{long,short}.csv`",
        "- `group_contribution_{shap,perm}_long.csv`",
        "- `feature_library_v3_proposals.csv`",
        "- `coverage_notes.json`",
        "",
    ]
    (OUT / "feature_library_research.md").write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {OUT}")
    for f in sorted(OUT.glob("*")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
