"""
SHORT v4: classic+SMC (v2) + D1/H4 standalone SHORT probabilities as meta-features.

Option A leakage mitigation:
  D1/H4 LGBM is fit on train<2023 (val 2023). Historical proba on that train window
  is in-sample. Walk-forward / sealed evaluation for H1-SHORT therefore emphasizes
  dates >= 2024-01-01 (+embargo), where D1/H4 proba is out-of-sample w.r.t. those models.

Usage:
    python -m models.train_short_v4_stacking
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from features.build_features_v2 import SMC_FEATURE_COLS
from models.data_prep import (
    BREAKEVEN_WINRATE,
    CAT_COLS,
    FEATURE_COLS,
    HORIZON_BARS,
    drop_timeouts,
)
from models.evaluate import (
    MIN_SIGNALS_FOR_SELECTION,
    binary_metrics_at_threshold,
    consistency_flag,
    evaluate_probs,
    select_threshold_from_validation,
)
from models.train_walk_forward_v2 import (
    FOLDS,
    TEST_START,
    _fmt_fold_table,
    _md_table,
    majority_threshold,
    split_embargo,
)

warnings.filterwarnings("ignore", category=UserWarning)

META_COLS = ["d1_short_proba", "h4_short_proba"]
FEATURE_COLS_V4 = FEATURE_COLS + SMC_FEATURE_COLS + META_COLS
V2_MEAN_FOLD_AUC_SHORT = 0.5394
V3_MEAN_FOLD_AUC_SHORT = 0.5177
# Option A: D1 model train ends before this (matches standalone split)
D1_OOS_START = pd.Timestamp("2024-01-01", tz="UTC")


def _fit_htf_short_and_score(
    *,
    feat_path: Path,
    lab_path: Path,
    feat_cols: list[str],
    embargo: pd.Timedelta,
    model_out: Path,
) -> pd.DataFrame:
    """Fit LGBM SHORT on HTF native labels; return Date + short_proba for ALL rows."""
    feat = pd.read_parquet(feat_path)
    lab = pd.read_parquet(lab_path)
    df = feat.merge(
        lab[["Date", "label_short"]],
        on="Date",
        how="inner",
    ).sort_values("Date").reset_index(drop=True)
    kept = df[df["label_short"] != -1].copy()

    te = pd.Timestamp("2023-01-01", tz="UTC")
    ve = pd.Timestamp("2024-01-01", tz="UTC")
    train = kept[kept["Date"] < (te - embargo)]
    val = kept[(kept["Date"] >= (te + embargo)) & (kept["Date"] < (ve - embargo))]

    xtr = train[feat_cols]
    ytr = train["label_short"].astype(int).to_numpy()
    xva = val[feat_cols]
    yva = val["label_short"].astype(int).to_numpy()

    model = lgb.LGBMClassifier(
        objective="binary",
        num_leaves=31,
        min_child_samples=50,
        learning_rate=0.05,
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        xtr,
        ytr,
        eval_set=[(xva, yva)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    # Score ALL feature rows (including timeouts / pre-train) for broadcast
    x_all = feat[feat_cols]
    proba = model.predict_proba(x_all)[:, 1]
    out = pd.DataFrame({"Date": feat["Date"].to_numpy(), "short_proba": proba})

    model_out.parent.mkdir(parents=True, exist_ok=True)
    with open(model_out, "wb") as f:
        pickle.dump({"model": model, "feature_cols": feat_cols, "embargo": str(embargo)}, f)
    print(f"Saved HTF short model -> {model_out} | scored rows={len(out)}")
    return out


def write_lookahead_check(
    h1: pd.DataFrame,
    d1_proba: pd.DataFrame,
    merged: pd.DataFrame,
    out_path: Path,
    *,
    n: int = 5,
    seed: int = 0,
) -> None:
    rng = np.random.default_rng(seed)
    # Sample H1 rows in OOS region
    cand = merged.index[merged["Date"] >= D1_OOS_START].to_numpy()
    picks = rng.choice(cand, size=min(n, len(cand)), replace=False)
    lines = [
        "# D1 Stacking Lookahead Check",
        "",
        "Rule: H1 row at close-time T may only use D1 bars whose close-time <= T "
        "(merge_asof direction=backward on close-time indices).",
        "",
        "D1 Date in native features is already OPEN+1d (close time).",
        "H1 Date in feature tables is OPEN+1h (close time).",
        "",
    ]
    d1 = d1_proba.sort_values("Date").reset_index(drop=True)
    for k, i in enumerate(picks, 1):
        row = merged.iloc[int(i)]
        t = row["Date"]
        # last D1 with Date <= t
        usable = d1[d1["Date"] <= t]
        last = usable.iloc[-1]
        nxt = d1[d1["Date"] > t]
        lines.append(f"## Sample {k}")
        lines.append(f"- H1 Date (close)={t}")
        lines.append(f"- Attached d1_short_proba={row['d1_short_proba']:.6f}")
        lines.append(f"- Last D1 close <= T: {last['Date']} proba={last['short_proba']:.6f}")
        if len(nxt):
            lines.append(f"- Next D1 close > T (MUST NOT be used): {nxt.iloc[0]['Date']}")
            ok = abs(float(row["d1_short_proba"]) - float(last["short_proba"])) < 1e-9
            lines.append(f"- Match last usable: **{'PASS' if ok else 'FAIL'}**")
        lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def build_v4_features(base: Path) -> pd.DataFrame:
    d1_cols = [
        "adx_d1", "atr_d1", "rsi_d1", "ema_fast_d1", "ema_slow_d1",
        "ema_cross_signal_d1", "return_1d", "return_5d",
    ]
    h4_cols = [
        "adx_h4", "atr_h4", "rsi_h4", "ema_fast_h4", "ema_slow_h4",
        "ema_cross_signal_h4", "return_1h4", "return_6h4",
    ]

    d1_scored = _fit_htf_short_and_score(
        feat_path=base / "features/d1_native_features.parquet",
        lab_path=base / "labels/d1_labeled.parquet",
        feat_cols=d1_cols,
        embargo=pd.Timedelta(days=10),
        model_out=base / "models_v4/htf/lightgbm_d1_short_stack_source.pkl",
    ).rename(columns={"short_proba": "d1_short_proba"})

    h4_scored = _fit_htf_short_and_score(
        feat_path=base / "features/h4_native_features.parquet",
        lab_path=base / "labels/h4_labeled.parquet",
        feat_cols=h4_cols,
        embargo=pd.Timedelta(hours=4 * 12),
        model_out=base / "models_v4/htf/lightgbm_h4_short_stack_source.pkl",
    ).rename(columns={"short_proba": "h4_short_proba"})

    v2 = pd.read_parquet(base / "features/xauusd_h1_h4_d1_features_v2.parquet")
    v2["Date"] = pd.to_datetime(v2["Date"], utc=True)
    d1_scored["Date"] = pd.to_datetime(d1_scored["Date"], utc=True)
    h4_scored["Date"] = pd.to_datetime(h4_scored["Date"], utc=True)

    merged = pd.merge_asof(
        v2.sort_values("Date"),
        d1_scored.sort_values("Date"),
        on="Date",
        direction="backward",
    )
    merged = pd.merge_asof(
        merged.sort_values("Date"),
        h4_scored.sort_values("Date"),
        on="Date",
        direction="backward",
    )
    before = len(merged)
    merged = merged.dropna(subset=META_COLS).reset_index(drop=True)
    out = base / "features/xauusd_h1_short_features_v4.parquet"
    merged.to_parquet(out, index=False)
    print(f"Saved v4 features -> {out} rows={len(merged)} (from {before})")

    write_lookahead_check(
        merged, d1_scored.rename(columns={"d1_short_proba": "short_proba"}), merged,
        base / "reports/d1_stacking_lookahead_check.md",
    )
    return merged


def _xy(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    x = df[FEATURE_COLS_V4].copy()
    for c in CAT_COLS:
        x[c] = x[c].astype("category")
    y = df["label_short"].astype(int).to_numpy()
    return x, y


def fit_lgbm(x_tr, y_tr, x_va, y_va) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        objective="binary",
        num_leaves=31,
        min_child_samples=100,
        learning_rate=0.05,
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        x_tr,
        y_tr,
        eval_set=[(x_va, y_va)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(30, verbose=False)],
        categorical_feature=CAT_COLS,
    )
    return model


def run_short_wf_fixed(df_all: pd.DataFrame, out_dir: Path) -> dict:
    kept, stats = drop_timeouts(df_all, "short")
    fold_rows = []
    fold_tables = []

    for fold in FOLDS:
        train, val = split_embargo(kept, train_end=fold["train_end"], val_end=fold["val_end"])
        x_tr, y_tr = _xy(train)
        x_va, y_va = _xy(val)
        model = fit_lgbm(x_tr, y_tr, x_va, y_va)
        p_va = model.predict_proba(x_va)[:, 1]
        ev = evaluate_probs(y_va, p_va)
        sel = select_threshold_from_validation(y_va, p_va)
        fold_tables.append(ev["threshold_table"])
        # Option A note: folds with val before 2024 use in-sample D1 proba
        oos_vs_d1 = val["Date"].min() >= D1_OOS_START
        fold_rows.append(
            {
                "fold": fold["name"],
                "train_end": fold["train_end"],
                "val_end": fold["val_end"],
                "n_train": len(train),
                "n_val": len(val),
                "auc_roc": ev["auc_roc"],
                "auc_pr": ev["auc_pr"],
                "fold_selected_thr": sel["selected_thr"],
                "fold_val_precision": (
                    None if sel["val_at_selected"] is None else float(sel["val_at_selected"]["precision"])
                ),
                "fold_val_n": (
                    None if sel["val_at_selected"] is None else int(sel["val_at_selected"]["n_signals"])
                ),
                "val_oos_vs_d1_model": bool(oos_vs_d1),
            }
        )
        print(
            f"  short {fold['name']}: auc={ev['auc_roc']:.4f} thr={sel['selected_thr']} "
            f"oos_vs_d1={oos_vs_d1}"
        )

    maj = majority_threshold(fold_tables)
    fold_df = pd.DataFrame(fold_rows)

    # Final model + sealed 2026 (always OOS vs D1 train)
    te = pd.Timestamp(TEST_START, tz="UTC")
    emb = pd.Timedelta(hours=HORIZON_BARS)
    train_full = kept[kept["Date"] < (te - emb)].copy()
    test = kept[kept["Date"] >= (te + emb)].copy()
    cut = train_full["Date"].quantile(0.85)
    tr_f = train_full[train_full["Date"] < cut]
    va_f = train_full[train_full["Date"] >= cut]
    x_tr, y_tr = _xy(tr_f)
    x_va, y_va = _xy(va_f)
    x_te, y_te = _xy(test)
    final = fit_lgbm(x_tr, y_tr, x_va, y_va)
    p_te = final.predict_proba(x_te)[:, 1]
    test_ev = evaluate_probs(y_te, p_te)
    thr = maj["selected_thr"]
    test_at = binary_metrics_at_threshold(y_te, p_te, thr) if thr is not None else None

    # Extra: evaluate only on val folds that are OOS vs D1 (fold4=2024, fold5=2025)
    oos_folds = fold_df[fold_df["val_oos_vs_d1_model"]]
    mean_oos_auc = float(oos_folds["auc_roc"].mean()) if len(oos_folds) else float("nan")

    imp = (
        pd.DataFrame(
            {
                "feature": FEATURE_COLS_V4,
                "importance_gain": final.booster_.feature_importance(importance_type="gain"),
            }
        )
        .sort_values("importance_gain", ascending=False)
        .reset_index(drop=True)
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "lightgbm_short_v4.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(
            {"model": final, "feature_cols": FEATURE_COLS_V4, "selected_thr": thr, "direction": "short"},
            f,
        )
    preds = test[["Date"]].copy()
    preds["y_true"] = y_te
    preds["y_prob"] = p_te
    preds_path = out_dir / "lightgbm_short_v4_test_preds.parquet"
    preds.to_parquet(preds_path, index=False)
    imp.to_csv(out_dir / "lightgbm_short_v4_importance.csv", index=False)

    meta_rank = {
        c: (int(imp.index[imp["feature"] == c][0]) + 1 if c in set(imp["feature"]) else None)
        for c in META_COLS
    }

    return {
        "stats": stats,
        "fold_df": fold_df,
        "majority": maj,
        "mean_fold_auc": float(fold_df["auc_roc"].mean()),
        "std_fold_auc": float(fold_df["auc_roc"].std()),
        "mean_oos_vs_d1_auc": mean_oos_auc,
        "test_ev": test_ev,
        "test_at_sel": test_at,
        "selected_thr": thr,
        "n_test": len(test),
        "importance": imp,
        "meta_rank": meta_rank,
        "model_path": model_path,
        "preds_path": preds_path,
        "consistency": (
            consistency_flag(
                "ABOVE breakeven" if thr is not None else None,
                None if test_at is None else test_at["breakeven_flag"],
            )
            if thr is not None and test_at is not None
            else "N/A"
        ),
    }


def build_report(r: dict) -> str:
    imp = r["importance"]
    top15 = list(imp.head(15)["feature"])
    meta_in_top15 = [c for c in META_COLS if c in top15]
    meta_in_top5 = [c for c in META_COLS if c in list(imp.head(5)["feature"])]
    lines = [
        "# Walk-Forward Validation Report — SHORT v4 (v2 features + D1/H4 short proba)",
        "",
        "## Leakage / Option A",
        "",
        "D1 and H4 SHORT LGBMs were trained on native labels with train < 2023-01-01 "
        "(val=2023). Scoring those models on their own train window produces **in-sample** "
        "probabilities. If H1-SHORT trains on that window, it can overfit to memorized D1 "
        "scores rather than transferable HTF signal.",
        "",
        "**Mitigation (Option A):** keep full-history `d1_short_proba` / `h4_short_proba` for "
        "feature engineering, but treat folds with val year < 2024 as contaminated for "
        "stacking claims. Prefer metrics on folds with `val_oos_vs_d1_model=True` (2024, 2025) "
        "and sealed test 2026 (always OOS vs D1/H4 train). Option B (out-of-fold HTF proba) "
        "is the rigorous follow-up if results look promising.",
        "",
        f"- Features: {len(FEATURE_COLS_V4)} (= 28 v2 + {META_COLS})",
        f"- Breakeven: {BREAKEVEN_WINRATE:.4f}; min_signals={MIN_SIGNALS_FOR_SELECTION}",
        "",
        "## Walk-forward folds",
        _fmt_fold_table(r["fold_df"]),
        "",
        f"- Mean fold AUC (all folds): **{r['mean_fold_auc']:.4f}** (std={r['std_fold_auc']:.4f})",
        f"- Mean fold AUC (OOS vs D1 only, 2024–2025): **{r['mean_oos_vs_d1_auc']:.4f}**",
        f"- v2 mean fold AUC: {V2_MEAN_FOLD_AUC_SHORT:.4f} | delta all-fold: "
        f"{r['mean_fold_auc'] - V2_MEAN_FOLD_AUC_SHORT:+.4f}",
        f"- v3 mean fold AUC: {V3_MEAN_FOLD_AUC_SHORT:.4f} | delta all-fold: "
        f"{r['mean_fold_auc'] - V3_MEAN_FOLD_AUC_SHORT:+.4f}",
        f"- Majority threshold: {r['majority']['note']}",
        "",
        "### Threshold majority summary",
        _md_table(r["majority"]["summary"]),
        "",
        "### Sealed test 2026",
        f"- n_test={r['n_test']}; AUC={r['test_ev']['auc_roc']:.4f}; AUPR={r['test_ev']['auc_pr']:.4f}",
    ]
    if r["test_at_sel"] is not None:
        t = r["test_at_sel"]
        lines.append(
            f"- @thr={r['selected_thr']:.2f}: precision={t['precision']:.4f}, "
            f"n={t['n_signals']}, {t['breakeven_flag']}"
        )
    else:
        lines.append("- No majority-valid threshold — sealed not scored at operational thr.")

    lines += [
        "",
        "### Feature importance top 15",
        _md_table(imp.head(15), floatfmt=".2f"),
        f"- Meta ranks: {r['meta_rank']}",
        f"- Meta in top 5: {meta_in_top5 or '(none)'}",
        f"- Meta in top 15: {meta_in_top15 or '(none)'}",
        "",
        "## Required conclusions",
        "",
        f"1. **Is d1_short_proba important?** "
        f"rank={r['meta_rank'].get('d1_short_proba')}; "
        f"in top15={('d1_short_proba' in meta_in_top15)}. "
        f"h4_short_proba rank={r['meta_rank'].get('h4_short_proba')}.",
        "",
        f"2. **Mean fold AUC vs v2/v3?** all-fold {r['mean_fold_auc']:.4f} vs v2 "
        f"{V2_MEAN_FOLD_AUC_SHORT:.4f} / v3 {V3_MEAN_FOLD_AUC_SHORT:.4f}; "
        f"OOS-vs-D1 folds {r['mean_oos_vs_d1_auc']:.4f}.",
        "",
    ]
    if r["selected_thr"] is not None:
        lines.append(
            f"3. **Majority threshold for SHORT?** **YES** @ {r['selected_thr']:.2f} "
            "(first pass if prior v1/v2/v3 failed)."
        )
    else:
        lines.append(
            "3. **Majority threshold for SHORT?** **NO** — still fails majority-of-folds rule."
        )

    # Recommendation
    improved = r["mean_fold_auc"] >= V2_MEAN_FOLD_AUC_SHORT + 0.01 or (
        not np.isnan(r["mean_oos_vs_d1_auc"]) and r["mean_oos_vs_d1_auc"] >= V2_MEAN_FOLD_AUC_SHORT + 0.01
    )
    meta_useful = "d1_short_proba" in meta_in_top15 or "h4_short_proba" in meta_in_top15
    if r["selected_thr"] is not None and (improved or meta_useful):
        rec = (
            "Continue stacking — next step Option B (OOF D1/H4 proba) before any SHORT backtest."
        )
    elif meta_useful and not improved:
        rec = (
            "Meta-feature is used by the tree but does not yet clear validation bar — "
            "try Option B once; if still flat, park SHORT and focus on LONG."
        )
    else:
        rec = (
            "Park SHORT for now; focus on LONG (v1/v3). Stacking did not unlock a valid "
            "SHORT threshold under Option A."
        )
    lines += [
        "",
        f"4. **Recommendation:** {rec}",
        "",
        "## Artifacts",
        f"- `{r['model_path']}`",
        f"- `{r['preds_path']}`",
        "- `data/features/xauusd_h1_short_features_v4.parquet`",
        "- `data/reports/d1_stacking_lookahead_check.md`",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    base = Path("data")
    print("Building v4 SHORT features (D1+H4 stacking)...")
    feat = build_v4_features(base)
    lab = pd.read_parquet(base / "labels/xauusd_triple_barrier_labels.parquet")
    df = feat.merge(
        lab[["Date", "label_short", "label_long", "bars_to_resolution_short"]],
        on="Date",
        how="inner",
    ).sort_values("Date").reset_index(drop=True)
    print(f"Joined rows={len(df)}")

    print("\n=== SHORT v4 walk-forward ===")
    r = run_short_wf_fixed(df, base / "models_v4/short")
    report = build_report(r)
    path = base / "reports/walk_forward_validation_report_v4_short.md"
    path.write_text(report, encoding="utf-8")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
