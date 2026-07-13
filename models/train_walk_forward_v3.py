"""
Walk-forward purged validation + LightGBM v3 (classic + SMC + reversal).

Same folds / embargo / majority-threshold rules as v2.

Usage:
    python -m models.train_walk_forward_v3
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from features.build_features_v2 import SMC_FEATURE_COLS
from features.reversal import REVERSAL_FEATURE_COLS
from models.data_prep import (
    BREAKEVEN_WINRATE,
    CAT_COLS,
    FEATURE_COLS,
    HORIZON_BARS,
    Direction,
    drop_timeouts,
)
from models.evaluate import (
    MIN_SIGNALS_FOR_SELECTION,
    SELECTION_THRESHOLDS,
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
    fit_lgbm,
    majority_threshold,
    split_embargo,
)

warnings.filterwarnings("ignore", category=UserWarning)

FEATURE_COLS_V3 = FEATURE_COLS + SMC_FEATURE_COLS + REVERSAL_FEATURE_COLS
# v2 mean fold AUC from walk_forward_validation_report.md
V2_MEAN_FOLD_AUC = {"long": 0.5385, "short": 0.5394}


def load_v3_dataset(features_path: Path, labels_path: Path) -> pd.DataFrame:
    feat = pd.read_parquet(features_path)
    lab = pd.read_parquet(labels_path)
    label_keep = [
        "Date",
        "label_long",
        "label_short",
        "bars_to_resolution_long",
        "bars_to_resolution_short",
    ]
    missing = [c for c in FEATURE_COLS_V3 + ["Date"] if c not in feat.columns]
    if missing:
        raise ValueError(f"v3 features missing: {missing}")
    merged = feat.merge(lab[label_keep], on="Date", how="inner")
    return merged.sort_values("Date").reset_index(drop=True)


def _xy(df: pd.DataFrame, direction: Direction) -> tuple[pd.DataFrame, np.ndarray]:
    x = df[FEATURE_COLS_V3].copy()
    for c in CAT_COLS:
        x[c] = x[c].astype("category")
    y = df[f"label_{direction}"].astype(int).to_numpy()
    return x, y


def run_direction(df_all: pd.DataFrame, direction: Direction, out_dir: Path) -> dict:
    kept, stats = drop_timeouts(df_all, direction)
    fold_rows = []
    fold_tables: list[pd.DataFrame] = []

    for fold in FOLDS:
        train, val = split_embargo(
            kept, train_end=fold["train_end"], val_end=fold["val_end"]
        )
        x_tr, y_tr = _xy(train, direction)
        x_va, y_va = _xy(val, direction)
        model = fit_lgbm(x_tr, y_tr, x_va, y_va)
        # fit_lgbm uses FEATURE_COLS_V2 internally for importance — we only need probs here
        p_va = model.predict_proba(x_va)[:, 1]
        ev = evaluate_probs(y_va, p_va)
        sel = select_threshold_from_validation(y_va, p_va)
        fold_tables.append(ev["threshold_table"])
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
                    None
                    if sel["val_at_selected"] is None
                    else float(sel["val_at_selected"]["precision"])
                ),
                "fold_val_n": (
                    None
                    if sel["val_at_selected"] is None
                    else int(sel["val_at_selected"]["n_signals"])
                ),
            }
        )
        print(
            f"  {direction} {fold['name']}: auc={ev['auc_roc']:.4f} "
            f"fold_thr={sel['selected_thr']} n_train={len(train)} n_val={len(val)}"
        )

    maj = majority_threshold(fold_tables)
    fold_df = pd.DataFrame(fold_rows)

    te = pd.Timestamp(TEST_START, tz="UTC")
    emb = pd.Timedelta(hours=HORIZON_BARS)
    train_full = kept[kept["Date"] < (te - emb)].copy()
    test = kept[kept["Date"] >= (te + emb)].copy()
    cut = train_full["Date"].quantile(0.85)
    tr_f = train_full[train_full["Date"] < cut]
    va_f = train_full[train_full["Date"] >= cut]
    x_tr, y_tr = _xy(tr_f, direction)
    x_va, y_va = _xy(va_f, direction)
    x_te, y_te = _xy(test, direction)

    # Fit with v3 feature frame (fit_lgbm categorical_feature=CAT_COLS works)
    final_model = lgb.LGBMClassifier(
        objective="binary",
        num_leaves=31,
        min_child_samples=100,
        learning_rate=0.05,
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    final_model.fit(
        x_tr,
        y_tr,
        eval_set=[(x_va, y_va)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(30, verbose=False)],
        categorical_feature=CAT_COLS,
    )
    p_te = final_model.predict_proba(x_te)[:, 1]
    test_ev = evaluate_probs(y_te, p_te)
    selected_thr = maj["selected_thr"]
    test_at_sel = (
        binary_metrics_at_threshold(y_te, p_te, selected_thr)
        if selected_thr is not None
        else None
    )

    imp = (
        pd.DataFrame(
            {
                "feature": FEATURE_COLS_V3,
                "importance_gain": final_model.booster_.feature_importance(importance_type="gain"),
            }
        )
        .sort_values("importance_gain", ascending=False)
        .reset_index(drop=True)
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"lightgbm_{direction}_v3.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "model": final_model,
                "feature_cols": FEATURE_COLS_V3,
                "selected_thr": selected_thr,
                "direction": direction,
            },
            f,
        )
    preds = test[["Date", f"label_{direction}"]].copy()
    preds = preds.rename(columns={f"label_{direction}": "y_true"})
    preds["y_prob"] = p_te
    preds_path = out_dir / f"lightgbm_{direction}_v3_test_preds.parquet"
    preds.to_parquet(preds_path, index=False)
    imp.to_csv(out_dir / f"lightgbm_{direction}_v3_importance.csv", index=False)

    rev_in_top15 = [c for c in imp.head(15)["feature"] if c in REVERSAL_FEATURE_COLS]
    smc_in_top15 = [c for c in imp.head(15)["feature"] if c in SMC_FEATURE_COLS]

    return {
        "direction": direction,
        "timeout_stats": stats,
        "fold_df": fold_df,
        "majority": maj,
        "mean_fold_auc": float(fold_df["auc_roc"].mean()),
        "std_fold_auc": float(fold_df["auc_roc"].std()),
        "v2_mean_fold_auc": V2_MEAN_FOLD_AUC[direction],
        "test_ev": test_ev,
        "test_at_sel": test_at_sel,
        "selected_thr": selected_thr,
        "n_test": len(test),
        "importance": imp,
        "rev_in_top15": rev_in_top15,
        "smc_in_top15": smc_in_top15,
        "model_path": model_path,
        "preds_path": preds_path,
        "consistency": (
            consistency_flag(
                "ABOVE breakeven" if selected_thr is not None else None,
                None if test_at_sel is None else test_at_sel["breakeven_flag"],
            )
            if selected_thr is not None and test_at_sel is not None
            else "N/A"
        ),
    }


def build_report(long_r: dict, short_r: dict) -> str:
    lines = [
        "# Walk-Forward Validation Report — LightGBM v3 (classic + SMC + reversal)",
        "",
        "## Setup",
        (
            f"- Features: {len(FEATURE_COLS_V3)} "
            f"(= {len(FEATURE_COLS)} classic + {len(SMC_FEATURE_COLS)} SMC + "
            f"{len(REVERSAL_FEATURE_COLS)} reversal)"
        ),
        f"- Embargo: +/-{HORIZON_BARS}h; expanding train; val 2021-2025; sealed test >=2026",
        f"- Breakeven: {BREAKEVEN_WINRATE:.4f}; min_signals={MIN_SIGNALS_FOR_SELECTION}",
        "",
        "## SHORT (primary focus)",
        _fmt_fold_table(short_r["fold_df"]),
        "",
        f"- Mean fold AUC-ROC: **{short_r['mean_fold_auc']:.4f}** (std={short_r['std_fold_auc']:.4f})",
        f"- v2 mean fold AUC-ROC: {short_r['v2_mean_fold_auc']:.4f}",
        f"- Delta vs v2: {short_r['mean_fold_auc'] - short_r['v2_mean_fold_auc']:+.4f}",
        f"- Majority threshold: {short_r['majority']['note']}",
        "",
        "### Threshold majority summary (SHORT)",
        _md_table(short_r["majority"]["summary"]),
        "",
        "### SHORT validity vs v1/v2",
    ]
    if short_r["selected_thr"] is not None:
        lines.append(
            f"- **YES** — majority-valid thr={short_r['selected_thr']:.2f} "
            "(previously failed on v1 and v2)."
        )
        if short_r["test_at_sel"] is not None:
            t = short_r["test_at_sel"]
            lines.append(
                f"- Sealed test: precision={t['precision']:.4f}, n={t['n_signals']}, "
                f"{t['breakeven_flag']}"
            )
    else:
        lines.append(
            "- **NO** — still no majority-valid threshold (same failure mode as v1/v2 SHORT)."
        )

    lines += [
        "",
        "### Feature importance top 15 (SHORT)",
        _md_table(short_r["importance"].head(15), floatfmt=".2f"),
        f"- Reversal features in top 15: {short_r['rev_in_top15'] or '(none)'}",
        f"- SMC features in top 15: {short_r['smc_in_top15'] or '(none)'}",
        "",
        "## LONG (sanity check — should not collapse)",
        _fmt_fold_table(long_r["fold_df"]),
        "",
        f"- Mean fold AUC-ROC: **{long_r['mean_fold_auc']:.4f}** (std={long_r['std_fold_auc']:.4f})",
        f"- v2 mean fold AUC-ROC: {long_r['v2_mean_fold_auc']:.4f}",
        f"- Delta vs v2: {long_r['mean_fold_auc'] - long_r['v2_mean_fold_auc']:+.4f}",
        f"- Majority threshold: {long_r['majority']['note']}",
        "",
        "### Threshold majority summary (LONG)",
        _md_table(long_r["majority"]["summary"]),
        "",
        "### Sealed test 2026 (LONG)",
        f"- n_test={long_r['n_test']}; AUC={long_r['test_ev']['auc_roc']:.4f}",
    ]
    if long_r["test_at_sel"] is not None:
        t = long_r["test_at_sel"]
        lines.append(
            f"- @thr={long_r['selected_thr']:.2f}: precision={t['precision']:.4f}, "
            f"n={t['n_signals']}, {t['breakeven_flag']}"
        )
    lines += [
        "",
        "### Feature importance top 15 (LONG)",
        _md_table(long_r["importance"].head(15), floatfmt=".2f"),
        f"- Reversal features in top 15: {long_r['rev_in_top15'] or '(none)'}",
        "",
        "## Artifacts",
        f"- `{long_r['model_path']}`",
        f"- `{long_r['preds_path']}`",
        f"- `{short_r['model_path']}`",
        f"- `{short_r['preds_path']}`",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    base = Path("data")
    feat_path = base / "features/xauusd_h1_h4_d1_features_v3.parquet"
    lab_path = base / "labels/xauusd_triple_barrier_labels.parquet"
    if not feat_path.exists():
        raise SystemExit(f"Missing {feat_path} — run features.build_features_v3 first")

    print("Loading v3 dataset...")
    df = load_v3_dataset(feat_path, lab_path)
    print(f"rows={len(df)} | {df['Date'].min()} -> {df['Date'].max()}")

    print("\n=== SHORT walk-forward (primary) ===")
    short_r = run_direction(df, "short", base / "models_v3/short")
    print("\n=== LONG walk-forward (sanity) ===")
    long_r = run_direction(df, "long", base / "models_v3/long")

    report = build_report(long_r, short_r)
    report_path = base / "reports/walk_forward_validation_report_v3.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"\nSaved report -> {report_path}")


if __name__ == "__main__":
    main()
