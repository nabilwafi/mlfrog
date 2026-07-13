"""
Baseline classifiers for XAUUSD triple-barrier labels.

Trains SEPARATE models for LONG and SHORT (not multi-class):
  - Logistic Regression (scaled + one-hot)
  - LightGBM (native categorical)
  - XGBoost (enable_categorical if available, else one-hot)

Usage (from project root):
    python -m models.train_baseline
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from packaging.version import Version
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from models.data_prep import (
    BREAKEVEN_WINRATE,
    CAT_COLS,
    FEATURE_COLS,
    HORIZON_BARS,
    NUM_COLS,
    SL_MULT,
    TP_MULT,
    Direction,
    describe_split,
    drop_timeouts,
    load_joined_dataset,
    time_aware_split,
)
from models.evaluate import (
    DEFAULT_REF_THRESHOLD,
    MIN_SIGNALS_FOR_SELECTION,
    SELECTION_THRESHOLDS,
    binary_metrics_at_threshold,
    consistency_flag,
    evaluate_probs,
    format_calibration_table,
    format_threshold_table,
    select_threshold_from_validation,
)

warnings.filterwarnings("ignore", category=UserWarning)


def _xy(df: pd.DataFrame, direction: Direction) -> tuple[pd.DataFrame, np.ndarray]:
    x = df[FEATURE_COLS].copy()
    y = df[f"label_{direction}"].astype(int).to_numpy()
    return x, y


def _onehot_encoder() -> OneHotEncoder:
    # sklearn API differs across versions for sparse output name.
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_logreg_pipeline() -> Pipeline:
    """Logistic Regression: StandardScaler on numerics + one-hot categoricals."""
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUM_COLS),
            ("cat", _onehot_encoder(), CAT_COLS),
        ],
        remainder="drop",
    )
    clf = LogisticRegression(
        class_weight="balanced",
        penalty="l2",
        max_iter=2000,
        solver="lbfgs",
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def _to_categorical_frame(x: pd.DataFrame) -> pd.DataFrame:
    out = x.copy()
    for c in CAT_COLS:
        out[c] = out[c].astype("category")
    return out


def _xgb_supports_native_categorical() -> bool:
    try:
        import xgboost as xgb

        return Version(xgb.__version__) >= Version("1.6.0")
    except Exception:
        return False


def train_logreg(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame,
    x_test: pd.DataFrame,
) -> tuple[Any, np.ndarray, np.ndarray, pd.DataFrame]:
    pipe = build_logreg_pipeline()
    pipe.fit(x_train, y_train)
    p_val = pipe.predict_proba(x_val)[:, 1]
    p_test = pipe.predict_proba(x_test)[:, 1]

    # Absolute coefficients on transformed feature names.
    pre: ColumnTransformer = pipe.named_steps["pre"]
    clf: LogisticRegression = pipe.named_steps["clf"]
    try:
        feat_names = list(pre.get_feature_names_out())
    except Exception:
        feat_names = [f"f{i}" for i in range(len(clf.coef_[0]))]
    coef = clf.coef_[0]
    imp = (
        pd.DataFrame({"feature": feat_names, "coefficient": coef})
        .assign(abs_coef=lambda d: d["coefficient"].abs())
        .sort_values("abs_coef", ascending=False)
        .head(15)
        .reset_index(drop=True)
    )
    return pipe, p_val, p_test, imp


def train_lightgbm(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    x_test: pd.DataFrame,
) -> tuple[Any, np.ndarray, np.ndarray, pd.DataFrame]:
    import lightgbm as lgb

    xt = _to_categorical_frame(x_train)
    xv = _to_categorical_frame(x_val)
    xte = _to_categorical_frame(x_test)

    model = lgb.LGBMClassifier(
        objective="binary",
        num_leaves=31,
        min_child_samples=100,  # alias of min_data_in_leaf in sklearn API
        learning_rate=0.05,
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        xt,
        y_train,
        eval_set=[(xv, y_val)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(30, verbose=False)],
        categorical_feature=CAT_COLS,
    )
    p_val = model.predict_proba(xv)[:, 1]
    p_test = model.predict_proba(xte)[:, 1]

    imp = (
        pd.DataFrame(
            {
                "feature": FEATURE_COLS,
                "importance_gain": model.booster_.feature_importance(importance_type="gain"),
            }
        )
        .sort_values("importance_gain", ascending=False)
        .head(15)
        .reset_index(drop=True)
    )
    return model, p_val, p_test, imp


def train_xgboost(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    x_test: pd.DataFrame,
) -> tuple[Any, np.ndarray, np.ndarray, pd.DataFrame]:
    import xgboost as xgb

    n_pos = max(int((y_train == 1).sum()), 1)
    n_neg = max(int((y_train == 0).sum()), 1)
    scale_pos_weight = n_neg / n_pos

    use_native_cat = _xgb_supports_native_categorical()

    if use_native_cat:
        xt = _to_categorical_frame(x_train)
        xv = _to_categorical_frame(x_val)
        xte = _to_categorical_frame(x_test)
        model = xgb.XGBClassifier(
            objective="binary:logistic",
            max_depth=5,
            learning_rate=0.05,
            n_estimators=300,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            enable_categorical=True,
            tree_method="hist",
            eval_metric="auc",
            random_state=42,
            n_jobs=-1,
        )
        # early_stopping_rounds: constructor (xgb>=2) or fit kw (older)
        try:
            model.set_params(early_stopping_rounds=30)
            model.fit(xt, y_train, eval_set=[(xv, y_val)], verbose=False)
        except TypeError:
            model.fit(
                xt,
                y_train,
                eval_set=[(xv, y_val)],
                early_stopping_rounds=30,
                verbose=False,
            )
        p_val = model.predict_proba(xv)[:, 1]
        p_test = model.predict_proba(xte)[:, 1]
        names = FEATURE_COLS
        gains = model.feature_importances_
    else:
        # Fallback: one-hot encode categoricals for older XGBoost.
        enc = _onehot_encoder()
        xt_num = x_train[NUM_COLS].to_numpy()
        xv_num = x_val[NUM_COLS].to_numpy()
        xte_num = x_test[NUM_COLS].to_numpy()
        xt_cat = enc.fit_transform(x_train[CAT_COLS])
        xv_cat = enc.transform(x_val[CAT_COLS])
        xte_cat = enc.transform(x_test[CAT_COLS])
        xt = np.hstack([xt_num, xt_cat])
        xv = np.hstack([xv_num, xv_cat])
        xte = np.hstack([xte_num, xte_cat])
        cat_names = list(enc.get_feature_names_out(CAT_COLS))
        names = NUM_COLS + cat_names

        model = xgb.XGBClassifier(
            objective="binary:logistic",
            max_depth=5,
            learning_rate=0.05,
            n_estimators=300,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="auc",
            random_state=42,
            n_jobs=-1,
        )
        try:
            model.set_params(early_stopping_rounds=30)
            model.fit(xt, y_train, eval_set=[(xv, y_val)], verbose=False)
        except TypeError:
            model.fit(
                xt,
                y_train,
                eval_set=[(xv, y_val)],
                early_stopping_rounds=30,
                verbose=False,
            )
        p_val = model.predict_proba(xv)[:, 1]
        p_test = model.predict_proba(xte)[:, 1]
        gains = model.feature_importances_

    imp = (
        pd.DataFrame({"feature": names, "importance_gain": gains})
        .sort_values("importance_gain", ascending=False)
        .head(15)
        .reset_index(drop=True)
    )
    return model, p_val, p_test, imp


def _section_model(
    *,
    model_name: str,
    direction: Direction,
    y_val: np.ndarray,
    p_val: np.ndarray,
    y_test: np.ndarray,
    p_test: np.ndarray,
    importance: pd.DataFrame,
) -> tuple[str, dict]:
    """Build report section + summary row dict.

    Threshold is selected ONLY on validation, then frozen and applied to test.
    """
    val_m = evaluate_probs(y_val, p_val, thresholds=SELECTION_THRESHOLDS)
    test_m = evaluate_probs(y_test, p_test, thresholds=SELECTION_THRESHOLDS)

    selection = select_threshold_from_validation(
        y_val,
        p_val,
        thresholds=SELECTION_THRESHOLDS,
        min_signals=MIN_SIGNALS_FOR_SELECTION,
    )
    selected_thr = selection["selected_thr"]

    if selected_thr is None:
        val_at_sel = None
        test_at_sel = None
        consistency = "N/A"
    else:
        val_at_sel = binary_metrics_at_threshold(y_val, p_val, selected_thr)
        test_at_sel = binary_metrics_at_threshold(y_test, p_test, selected_thr)
        consistency = consistency_flag(
            val_at_sel["breakeven_flag"],
            test_at_sel["breakeven_flag"],
        )

    lines = [
        f"## {model_name} — {direction.upper()}",
        "",
        "### Validation metrics",
        f"- AUC-ROC: {val_m['auc_roc']:.4f}",
        f"- AUC-PR: {val_m['auc_pr']:.4f}",
        (
            f"- @{DEFAULT_REF_THRESHOLD:.1f} (reference, no tuning) "
            f"precision={val_m['at_0_5']['precision']:.4f} "
            f"recall={val_m['at_0_5']['recall']:.4f} "
            f"f1={val_m['at_0_5']['f1']:.4f} "
            f"n_signals={val_m['at_0_5']['n_signals']}"
        ),
        "",
        (
            f"#### Threshold grid on VALIDATION only "
            f"(candidates={SELECTION_THRESHOLDS}, min_signals={MIN_SIGNALS_FOR_SELECTION})"
        ),
        format_threshold_table(val_m["threshold_table"]),
        "",
        f"**Threshold selection (validation-only):** {selection['selection_note']}",
        "",
    ]

    if selected_thr is not None and val_at_sel is not None:
        lines.extend(
            [
                (
                    f"- selected_thr={selected_thr:.2f} | "
                    f"val precision={val_at_sel['precision']:.4f} | "
                    f"val recall={val_at_sel['recall']:.4f} | "
                    f"val f1={val_at_sel['f1']:.4f} | "
                    f"val n={val_at_sel['n_signals']} | "
                    f"{val_at_sel['breakeven_flag']}"
                ),
                "",
            ]
        )

    lines.extend(
        [
            "#### Calibration (validation, 10 bins)",
            format_calibration_table(val_m["calibration"]),
            "",
            "### Test metrics (selected_thr frozen from validation — no re-search)",
            f"- AUC-ROC: {test_m['auc_roc']:.4f}",
            f"- AUC-PR: {test_m['auc_pr']:.4f}",
            (
                f"- @{DEFAULT_REF_THRESHOLD:.1f} (reference, no tuning) "
                f"precision={test_m['at_0_5']['precision']:.4f} "
                f"recall={test_m['at_0_5']['recall']:.4f} "
                f"f1={test_m['at_0_5']['f1']:.4f} "
                f"n_signals={test_m['at_0_5']['n_signals']}"
            ),
            "",
        ]
    )

    if selected_thr is not None and test_at_sel is not None:
        lines.extend(
            [
                (
                    f"- **At selected_thr={selected_thr:.2f} (from val):** "
                    f"test precision={test_at_sel['precision']:.4f} | "
                    f"test recall={test_at_sel['recall']:.4f} | "
                    f"test f1={test_at_sel['f1']:.4f} | "
                    f"test n={test_at_sel['n_signals']} | "
                    f"{test_at_sel['breakeven_flag']}"
                ),
                f"- **val_test_consistency_flag:** **{consistency}**",
                "",
            ]
        )
        if consistency == "INCONSISTENT":
            lines.extend(
                [
                    (
                        "> **INCONSISTENT highlighted:** validation and test disagree on "
                        "whether precision is above/below breakeven at the same selected_thr. "
                        "Treat as likely noise, not a reliable edge."
                    ),
                    "",
                ]
            )
    else:
        lines.extend(
            [
                "- No operational selected_thr — test metrics at a tuned threshold are not reported.",
                "",
            ]
        )

    lines.extend(
        [
            "#### Threshold grid on TEST (diagnostic only — NOT used for selection)",
            format_threshold_table(test_m["threshold_table"]),
            "",
            "#### Calibration (test, 10 bins)",
            format_calibration_table(test_m["calibration"]),
            "",
            "### Feature importance / coefficients (top 15)",
            importance.to_markdown(index=False),
            "",
        ]
    )

    summary_row = {
        "direction": direction,
        "model": model_name,
        "val_auc_roc": val_m["auc_roc"],
        "val_auc_pr": val_m["auc_pr"],
        "test_auc_roc": test_m["auc_roc"],
        "test_auc_pr": test_m["auc_pr"],
        "selected_thr": selected_thr,
        "selection_note": selection["selection_note"],
        "val_precision_at_selected_thr": None
        if val_at_sel is None
        else float(val_at_sel["precision"]),
        "val_n_at_selected_thr": None if val_at_sel is None else int(val_at_sel["n_signals"]),
        "val_flag_at_selected_thr": None
        if val_at_sel is None
        else val_at_sel["breakeven_flag"],
        "test_precision_at_selected_thr": None
        if test_at_sel is None
        else float(test_at_sel["precision"]),
        "test_n_at_selected_thr": None if test_at_sel is None else int(test_at_sel["n_signals"]),
        "test_flag_at_selected_thr": None
        if test_at_sel is None
        else test_at_sel["breakeven_flag"],
        "val_test_consistency_flag": consistency,
        "val_precision_0_5": val_m["at_0_5"]["precision"],
        "val_n_0_5": val_m["at_0_5"]["n_signals"],
        "test_precision_0_5": test_m["at_0_5"]["precision"],
        "test_n_0_5": test_m["at_0_5"]["n_signals"],
    }
    return "\n".join(lines), summary_row


def run_direction(
    direction: Direction,
    joined: pd.DataFrame,
    report_parts: list[str],
    summary_rows: list[dict],
) -> None:
    kept, stats = drop_timeouts(joined, direction)
    report_parts.append(f"# Data prep — {direction.upper()}")
    report_parts.append(
        f"- timeout dropped: {stats['n_timeout']} ({stats['timeout_pct']:.2f}%)"
    )
    report_parts.append(
        f"- kept win/loss: {stats['n_kept']} ({stats['kept_pct']:.2f}%) "
        f"| win={stats['n_win']} loss={stats['n_loss']}"
    )

    splits = time_aware_split(kept, embargo_hours=HORIZON_BARS)
    report_parts.append("")
    report_parts.append(
        f"## Time-aware split ({direction.upper()}) after embargo ±{HORIZON_BARS}h"
    )
    report_parts.append(
        "Random split is NOT used: overlapping forward windows of length "
        f"{HORIZON_BARS} would leak label information across train/test."
    )
    report_parts.append(describe_split("train", splits.train))
    report_parts.append(describe_split("val", splits.val))
    report_parts.append(describe_split("test", splits.test))
    report_parts.append("")

    print(f"\n=== {direction.upper()} ===")
    print(
        f"timeout dropped={stats['n_timeout']} ({stats['timeout_pct']:.2f}%), "
        f"kept={stats['n_kept']}"
    )
    for name, frame in [
        ("train", splits.train),
        ("val", splits.val),
        ("test", splits.test),
    ]:
        print(describe_split(name, frame))

    x_tr, y_tr = _xy(splits.train, direction)
    x_va, y_va = _xy(splits.val, direction)
    x_te, y_te = _xy(splits.test, direction)

    trainers = [
        ("LogisticRegression", lambda: train_logreg(x_tr, y_tr, x_va, x_te)),
        ("LightGBM", lambda: train_lightgbm(x_tr, y_tr, x_va, y_va, x_te)),
        ("XGBoost", lambda: train_xgboost(x_tr, y_tr, x_va, y_va, x_te)),
    ]

    for model_name, trainer in trainers:
        print(f"Training {model_name} ({direction})...")
        _model, p_va, p_te, imp = trainer()
        section, summary_row = _section_model(
            model_name=model_name,
            direction=direction,
            y_val=y_va,
            p_val=p_va,
            y_test=y_te,
            p_test=p_te,
            importance=imp,
        )
        report_parts.append(section)
        summary_rows.append(summary_row)
        print(
            f"  selected_thr={summary_row['selected_thr']} | "
            f"consistency={summary_row['val_test_consistency_flag']}"
        )


def _recommendation(summary: pd.DataFrame) -> str:
    lines = [
        "# Final comparison & recommendation",
        "",
        f"Breakeven win-rate from SL/TP = {SL_MULT}/({SL_MULT}+{TP_MULT}) = **{BREAKEVEN_WINRATE:.4f}**.",
        "",
        "### Threshold methodology (fixed)",
        (
            f"- Select `selected_thr` **only on validation** from grid {SELECTION_THRESHOLDS} "
            f"with `n_signals >= {MIN_SIGNALS_FOR_SELECTION}` and precision ABOVE breakeven; "
            "pick highest validation precision."
        ),
        "- Apply that **same** threshold to test (no re-search on test).",
        "- If no validation candidate qualifies: do **not** force a near-miss threshold.",
        "- `val_test_consistency_flag`: CONSISTENT if val/test agree on above/below breakeven at selected_thr; else INCONSISTENT.",
        "",
        "## Summary table",
        summary.to_markdown(index=False),
        "",
    ]

    inconsistent = summary[summary["val_test_consistency_flag"] == "INCONSISTENT"]
    if not inconsistent.empty:
        lines.append("## INCONSISTENT rows (highlighted)")
        for _, r in inconsistent.iterrows():
            lines.append(
                f"- **{r['direction'].upper()} / {r['model']}**: "
                f"selected_thr={r['selected_thr']}, "
                f"val_prec={r['val_precision_at_selected_thr']}, "
                f"test_prec={r['test_precision_at_selected_thr']} "
                f"({r['val_flag_at_selected_thr']} vs {r['test_flag_at_selected_thr']})"
            )
        lines.append("")

    no_thr = summary[summary["selected_thr"].isna()]
    if not no_thr.empty:
        lines.append("## No valid selected_thr on validation")
        for _, r in no_thr.iterrows():
            lines.append(f"- **{r['direction'].upper()} / {r['model']}**: {r['selection_note']}")
        lines.append("")

    lines.append("## Recommendation")
    recs = []
    for direction in ["long", "short"]:
        sub = summary[summary["direction"] == direction].copy()
        # Only consider rows with a valid selected_thr AND CONSISTENT above-breakeven on both.
        valid = sub[
            sub["selected_thr"].notna()
            & (sub["val_test_consistency_flag"] == "CONSISTENT")
            & (sub["val_flag_at_selected_thr"] == "ABOVE breakeven")
            & (sub["test_flag_at_selected_thr"] == "ABOVE breakeven")
        ].copy()
        if valid.empty:
            recs.append(
                f"- **{direction.upper()}**: no model has a validation-selected threshold "
                "that stays ABOVE breakeven with CONSISTENT val/test behavior. "
                "**Do not force a recommendation.**"
            )
            continue
        best = valid.sort_values(
            ["test_auc_pr", "test_n_at_selected_thr"],
            ascending=[False, False],
        ).iloc[0]
        recs.append(
            f"- **{direction.upper()}**: **{best['model']}** "
            f"(selected_thr={best['selected_thr']}, "
            f"val_prec={best['val_precision_at_selected_thr']:.4f}, "
            f"test_prec={best['test_precision_at_selected_thr']:.4f}, "
            f"test_n={int(best['test_n_at_selected_thr'])}, "
            f"test AUC-PR={best['test_auc_pr']:.4f}, CONSISTENT)."
        )

    lines.extend(recs)
    lines.append("")
    lines.append(
        "Honesty check: if all directions lack a valid CONSISTENT above-breakeven "
        "selected_thr, the baseline does not yet justify advancing to backtest on "
        "thresholded signals alone. Ranking metrics (AUC) may still be informative, "
        "but operational edge is not established."
    )
    return "\n".join(lines)


def main() -> None:
    base = Path("data")
    features_path = base / "features/xauusd_h1_h4_d1_features.parquet"
    labels_path = base / "labels/xauusd_triple_barrier_labels.parquet"
    report_path = base / "reports/modeling_report.md"
    summary_path = base / "reports/modeling_summary.csv"

    print("Loading joined dataset...")
    joined = load_joined_dataset(features_path, labels_path)
    print(f"Joined rows={len(joined)} | {joined['Date'].min()} -> {joined['Date'].max()}")

    report_parts: list[str] = [
        "# XAUUSD Baseline Modeling Report",
        "",
        f"- Features: `{features_path}`",
        f"- Labels: `{labels_path}`",
        f"- Params: sl_mult={SL_MULT}, tp_mult={TP_MULT}, horizon_bars={HORIZON_BARS}",
        f"- Breakeven win-rate: {BREAKEVEN_WINRATE:.4f}",
        f"- Feature columns: {FEATURE_COLS}",
        "",
        "Models: Logistic Regression, LightGBM, XGBoost — trained separately for LONG and SHORT.",
        "Accuracy is NOT used as primary metric (class imbalance).",
        (
            f"Threshold selection: validation-only grid {SELECTION_THRESHOLDS}, "
            f"min_signals={MIN_SIGNALS_FOR_SELECTION}; frozen for test evaluation."
        ),
        "",
    ]
    summary_rows: list[dict] = []

    for direction in ("long", "short"):
        run_direction(direction, joined, report_parts, summary_rows)

    summary = pd.DataFrame(summary_rows)
    report_parts.append(_recommendation(summary))

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_parts), encoding="utf-8")
    summary.to_csv(summary_path, index=False)

    print(f"\nSaved report -> {report_path}")
    print(f"Saved summary -> {summary_path}")


if __name__ == "__main__":
    main()
