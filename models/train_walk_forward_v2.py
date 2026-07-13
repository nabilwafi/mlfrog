"""
Walk-forward purged validation + LightGBM v2 (classic + SMC features).

Folds (expanding train):
  1: train <2021, val 2021
  2: train <2022, val 2022
  ...
  5: train <2025, val 2025
Sealed test: >=2026 (embargo ±8h at each boundary).

Usage:
    python -m models.train_walk_forward_v2
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

warnings.filterwarnings("ignore", category=UserWarning)

FEATURE_COLS_V2 = FEATURE_COLS + SMC_FEATURE_COLS
# v1 AUC-ROC from modeling_report (LightGBM single-split val)
V1_LGBM_VAL_AUC = {"long": 0.539043, "short": 0.494149}

FOLDS = [
    {"name": "fold1", "train_end": "2021-01-01", "val_end": "2022-01-01"},
    {"name": "fold2", "train_end": "2022-01-01", "val_end": "2023-01-01"},
    {"name": "fold3", "train_end": "2023-01-01", "val_end": "2024-01-01"},
    {"name": "fold4", "train_end": "2024-01-01", "val_end": "2025-01-01"},
    {"name": "fold5", "train_end": "2025-01-01", "val_end": "2026-01-01"},
]
TEST_START = "2026-01-01"


def load_v2_dataset(features_path: Path, labels_path: Path) -> pd.DataFrame:
    feat = pd.read_parquet(features_path)
    lab = pd.read_parquet(labels_path)
    label_keep = [
        "Date",
        "label_long",
        "label_short",
        "bars_to_resolution_long",
        "bars_to_resolution_short",
    ]
    missing = [c for c in FEATURE_COLS_V2 + ["Date"] if c not in feat.columns]
    if missing:
        raise ValueError(f"v2 features missing: {missing}")
    merged = feat.merge(lab[label_keep], on="Date", how="inner")
    return merged.sort_values("Date").reset_index(drop=True)


def split_embargo(
    df: pd.DataFrame,
    *,
    train_end: str,
    val_end: str | None,
    embargo_hours: int = HORIZON_BARS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train, val) with ±embargo around train_end; val ends at val_end."""
    te = pd.Timestamp(train_end, tz="UTC")
    emb = pd.Timedelta(hours=embargo_hours)
    train = df[df["Date"] < (te - emb)].copy()
    if val_end is None:
        val = df[df["Date"] >= (te + emb)].copy()
    else:
        ve = pd.Timestamp(val_end, tz="UTC")
        val = df[(df["Date"] >= (te + emb)) & (df["Date"] < (ve - emb))].copy()
    return train.reset_index(drop=True), val.reset_index(drop=True)


def _xy(df: pd.DataFrame, direction: Direction) -> tuple[pd.DataFrame, np.ndarray]:
    x = df[FEATURE_COLS_V2].copy()
    for c in CAT_COLS:
        x[c] = x[c].astype("category")
    y = df[f"label_{direction}"].astype(int).to_numpy()
    return x, y


def fit_lgbm(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        objective="binary",
        num_leaves=31,
        min_child_samples=100,
        learning_rate=0.05,
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(30, verbose=False)],
        categorical_feature=CAT_COLS,
    )
    return model


def majority_threshold(fold_tables: list[pd.DataFrame]) -> dict:
    """Pick thr above BE with min_signals in majority of folds; max mean precision."""
    n_folds = len(fold_tables)
    need = (n_folds // 2) + 1  # majority
    rows = []
    for thr in SELECTION_THRESHOLDS:
        flags = []
        precs = []
        ns = []
        for tab in fold_tables:
            r = tab.loc[tab["threshold"] == thr].iloc[0]
            ok = (int(r["n_signals"]) >= MIN_SIGNALS_FOR_SELECTION) and (
                r["breakeven_flag"] == "ABOVE breakeven"
            )
            flags.append(ok)
            precs.append(float(r["precision"]) if ok else np.nan)
            ns.append(int(r["n_signals"]))
        n_ok = int(sum(flags))
        rows.append(
            {
                "threshold": thr,
                "n_folds_above_be": n_ok,
                "majority": n_ok >= need,
                "mean_precision_when_ok": float(np.nanmean(precs)) if n_ok else np.nan,
                "mean_n_signals": float(np.mean(ns)),
            }
        )
    summary = pd.DataFrame(rows)
    eligible = summary[summary["majority"]].copy()
    if eligible.empty:
        return {
            "selected_thr": None,
            "need_majority": need,
            "summary": summary,
            "note": (
                f"no threshold above BE with min_signals>={MIN_SIGNALS_FOR_SELECTION} "
                f"in majority of folds ({need}/{n_folds})"
            ),
        }
    best = eligible.sort_values(
        ["mean_precision_when_ok", "n_folds_above_be"], ascending=[False, False]
    ).iloc[0]
    return {
        "selected_thr": float(best["threshold"]),
        "need_majority": need,
        "summary": summary,
        "note": (
            f"selected_thr={float(best['threshold']):.2f} "
            f"(above BE in {int(best['n_folds_above_be'])}/{n_folds} folds, "
            f"mean_prec={float(best['mean_precision_when_ok']):.4f})"
        ),
    }


def run_direction(
    df_all: pd.DataFrame,
    direction: Direction,
    out_dir: Path,
) -> dict:
    kept, stats = drop_timeouts(df_all, direction)
    fold_rows = []
    fold_tables: list[pd.DataFrame] = []
    last_model = None
    last_imp = None

    for fold in FOLDS:
        train, val = split_embargo(
            kept, train_end=fold["train_end"], val_end=fold["val_end"]
        )
        if len(train) < 500 or len(val) < 100:
            raise RuntimeError(f"{direction} {fold['name']}: too few rows train={len(train)} val={len(val)}")

        x_tr, y_tr = _xy(train, direction)
        x_va, y_va = _xy(val, direction)
        model = fit_lgbm(x_tr, y_tr, x_va, y_va)
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
        last_model = model
        last_imp = (
            pd.DataFrame(
                {
                    "feature": FEATURE_COLS_V2,
                    "importance_gain": model.booster_.feature_importance(importance_type="gain"),
                }
            )
            .sort_values("importance_gain", ascending=False)
            .reset_index(drop=True)
        )
        print(
            f"  {direction} {fold['name']}: auc={ev['auc_roc']:.4f} "
            f"fold_thr={sel['selected_thr']} n_train={len(train)} n_val={len(val)}"
        )

    maj = majority_threshold(fold_tables)
    fold_df = pd.DataFrame(fold_rows)

    # Final model: train <2026-embargo, early-stop on 2025 val, sealed test 2026+
    train_final, val_es = split_embargo(
        kept, train_end="2025-01-01", val_end="2026-01-01"
    )
    # Expand train to include 2025 for final fit, keep small ES holdout from late 2025? 
    # Lazy: train through end-2025 embargo before test, ES on fold5-style 2025 window.
    te = pd.Timestamp(TEST_START, tz="UTC")
    emb = pd.Timedelta(hours=HORIZON_BARS)
    train_full = kept[kept["Date"] < (te - emb)].copy()
    test = kept[kept["Date"] >= (te + emb)].copy()

    # Use 2025 window as ES val (already have val_es); train_full for final without ES would overfit.
    # Train on pre-2025 + use 2025 for ES, then predict 2026 — same as fold5 train then extend.
    # Better: fit on train_full with ES split = last 20% of train_full by time.
    cut = train_full["Date"].quantile(0.85)
    tr_f = train_full[train_full["Date"] < cut]
    va_f = train_full[train_full["Date"] >= cut]
    x_tr, y_tr = _xy(tr_f, direction)
    x_va, y_va = _xy(va_f, direction)
    x_te, y_te = _xy(test, direction)
    final_model = fit_lgbm(x_tr, y_tr, x_va, y_va)
    p_te = final_model.predict_proba(x_te)[:, 1]
    test_ev = evaluate_probs(y_te, p_te)

    selected_thr = maj["selected_thr"]
    test_at_sel = None
    if selected_thr is not None:
        test_at_sel = binary_metrics_at_threshold(y_te, p_te, selected_thr)

    imp = (
        pd.DataFrame(
            {
                "feature": FEATURE_COLS_V2,
                "importance_gain": final_model.booster_.feature_importance(importance_type="gain"),
            }
        )
        .sort_values("importance_gain", ascending=False)
        .reset_index(drop=True)
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"lightgbm_{direction}_v2.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "model": final_model,
                "feature_cols": FEATURE_COLS_V2,
                "selected_thr": selected_thr,
                "direction": direction,
            },
            f,
        )

    preds = test[["Date", f"label_{direction}"]].copy()
    preds = preds.rename(columns={f"label_{direction}": "y_true"})
    preds["y_prob"] = p_te
    preds_path = out_dir / f"lightgbm_{direction}_v2_test_preds.parquet"
    preds.to_parquet(preds_path, index=False)
    imp.to_csv(out_dir / f"lightgbm_{direction}_v2_importance.csv", index=False)

    smc_in_top15 = [c for c in imp.head(15)["feature"] if c in SMC_FEATURE_COLS]

    return {
        "direction": direction,
        "timeout_stats": stats,
        "fold_df": fold_df,
        "majority": maj,
        "mean_fold_auc": float(fold_df["auc_roc"].mean()),
        "std_fold_auc": float(fold_df["auc_roc"].std()),
        "v1_val_auc": V1_LGBM_VAL_AUC[direction],
        "test_ev": test_ev,
        "test_at_sel": test_at_sel,
        "selected_thr": selected_thr,
        "n_test": len(test),
        "importance": imp,
        "smc_in_top15": smc_in_top15,
        "model_path": model_path,
        "preds_path": preds_path,
        "consistency": (
            consistency_flag(
                "ABOVE breakeven" if selected_thr is not None else None,
                None
                if test_at_sel is None
                else test_at_sel["breakeven_flag"],
            )
            if selected_thr is not None and test_at_sel is not None
            else "N/A"
        ),
    }


def _md_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    """Minimal markdown table (no tabulate dependency)."""
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                cells.append("nan" if pd.isna(v) else format(float(v), floatfmt))
            elif pd.isna(v):
                cells.append("nan")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _fmt_fold_table(fold_df: pd.DataFrame) -> str:
    cols = [
        "fold",
        "n_train",
        "n_val",
        "auc_roc",
        "auc_pr",
        "fold_selected_thr",
        "fold_val_precision",
        "fold_val_n",
    ]
    return _md_table(fold_df[cols])


def build_report(long_r: dict, short_r: dict) -> str:
    lines = [
        "# Walk-Forward Validation Report — LightGBM v2 (classic + SMC)",
        "",
        "## Setup",
        f"- Features: {len(FEATURE_COLS_V2)} (= {len(FEATURE_COLS)} classic + {len(SMC_FEATURE_COLS)} SMC)",
        f"- Embargo: ±{HORIZON_BARS}h at each train/val/test boundary",
        "- Expanding train; val years 2021-2025; sealed test >=2026",
        f"- Breakeven winrate: {BREAKEVEN_WINRATE:.4f}",
        f"- Threshold grid: {SELECTION_THRESHOLDS}; min_signals={MIN_SIGNALS_FOR_SELECTION}",
        "- Final thr = majority-of-folds above BE (not best single fold)",
        "",
        "## LONG",
        _fmt_fold_table(long_r["fold_df"]),
        "",
        f"- Mean fold AUC-ROC: **{long_r['mean_fold_auc']:.4f}** (std={long_r['std_fold_auc']:.4f})",
        f"- v1 single-split val AUC-ROC (LGBM): {long_r['v1_val_auc']:.4f}",
        f"- Delta mean-fold vs v1: {long_r['mean_fold_auc'] - long_r['v1_val_auc']:+.4f}",
        f"- Majority threshold: {long_r['majority']['note']}",
        "",
        "### Threshold majority summary (LONG)",
        _md_table(long_r["majority"]["summary"]),
        "",
        "### Sealed test 2026 (LONG)",
        f"- n_test={long_r['n_test']}",
        f"- test AUC-ROC={long_r['test_ev']['auc_roc']:.4f}, AUC-PR={long_r['test_ev']['auc_pr']:.4f}",
    ]
    if long_r["test_at_sel"] is not None:
        t = long_r["test_at_sel"]
        lines += [
            f"- @selected_thr={long_r['selected_thr']:.2f}: precision={t['precision']:.4f}, "
            f"n={t['n_signals']}, {t['breakeven_flag']}",
            f"- val/test consistency (thr exists + test flag): {long_r['consistency']}",
        ]
    else:
        lines.append("- No valid majority threshold — sealed test not scored at operational thr.")

    lines += [
        "",
        "### Feature importance top 15 (LONG final model)",
        _md_table(long_r["importance"].head(15), floatfmt=".2f"),
        f"- SMC features in top 15: {long_r['smc_in_top15'] or '(none)'}",
        "",
        "## SHORT",
        _fmt_fold_table(short_r["fold_df"]),
        "",
        f"- Mean fold AUC-ROC: **{short_r['mean_fold_auc']:.4f}** (std={short_r['std_fold_auc']:.4f})",
        f"- v1 single-split val AUC-ROC (LGBM): {short_r['v1_val_auc']:.4f}",
        f"- Delta mean-fold vs v1: {short_r['mean_fold_auc'] - short_r['v1_val_auc']:+.4f}",
        f"- Majority threshold: {short_r['majority']['note']}",
        "",
        "### Threshold majority summary (SHORT)",
        _md_table(short_r["majority"]["summary"]),
        "",
        "### SHORT validity (key question vs v1)",
    ]
    if short_r["selected_thr"] is not None:
        lines.append(
            f"- **YES** — SMC+v2 produced a majority-valid threshold "
            f"({short_r['selected_thr']:.2f})."
        )
        if short_r["test_at_sel"] is not None:
            t = short_r["test_at_sel"]
            lines.append(
                f"- Sealed test @thr: precision={t['precision']:.4f}, n={t['n_signals']}, "
                f"{t['breakeven_flag']}"
            )
    else:
        lines.append(
            "- **NO** — still no threshold above BE with min_signals in majority of folds "
            "(same failure mode as v1 SHORT)."
        )

    lines += [
        "",
        "### Feature importance top 15 (SHORT final model)",
        _md_table(short_r["importance"].head(15), floatfmt=".2f"),
        f"- SMC features in top 15: {short_r['smc_in_top15'] or '(none)'}",
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
    feat_path = base / "features/xauusd_h1_h4_d1_features_v2.parquet"
    lab_path = base / "labels/xauusd_triple_barrier_labels.parquet"
    if not feat_path.exists():
        raise SystemExit(f"Missing {feat_path} — run features.build_features_v2 first")

    print("Loading v2 dataset...")
    df = load_v2_dataset(feat_path, lab_path)
    print(f"rows={len(df)} | {df['Date'].min()} -> {df['Date'].max()}")

    print("\n=== LONG walk-forward ===")
    long_r = run_direction(df, "long", base / "models_v2/long")
    print("\n=== SHORT walk-forward ===")
    short_r = run_direction(df, "short", base / "models_v2/short")

    report = build_report(long_r, short_r)
    report_path = base / "reports/walk_forward_validation_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"\nSaved report -> {report_path}")
    print(report)


if __name__ == "__main__":
    main()
