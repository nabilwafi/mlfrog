"""
Robustness check for LONG LightGBM @ selected_thr=0.52.

Because no model artifact was persisted from the previous modeling run, this
script re-fits LightGBM LONG once with the SAME split/params as train_baseline,
saves the model + test predictions, then evaluates:

1) Sub-period (semester) breakdown with frozen threshold 0.52
2) Bootstrap 95% CI on precision of thresholded test signals

Usage:
    python -m models.robustness_long_lgbm
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score

from models.data_prep import (
    BREAKEVEN_WINRATE,
    FEATURE_COLS,
    HORIZON_BARS,
    drop_timeouts,
    load_joined_dataset,
    time_aware_split,
)
from models.train_baseline import _to_categorical_frame, train_lightgbm

SELECTED_THR = 0.52
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 42

# Expected aggregate from modeling_report (sanity target; may differ slightly
# after re-fit due to LightGBM non-determinism / early stopping).
EXPECTED_TEST_N = 878
EXPECTED_TEST_PRECISION = 0.5228


def _period_label(ts: pd.Timestamp) -> str:
    y = int(ts.year)
    m = int(ts.month)
    if y == 2024 and m <= 6:
        return "2024 H1"
    if y == 2024 and m >= 7:
        return "2024 H2"
    if y == 2025 and m <= 6:
        return "2025 H1"
    if y == 2025 and m >= 7:
        return "2025 H2"
    if y == 2026 and m <= 6:
        return "2026 H1"
    if y == 2026 and m >= 7:
        return "2026 remainder"
    return f"{y} other"


PERIOD_ORDER = [
    "2024 H1",
    "2024 H2",
    "2025 H1",
    "2025 H2",
    "2026 H1",
    "2026 remainder",
]


def _metrics_at_thr(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    mask = p >= thr
    n_signals = int(mask.sum())
    if n_signals == 0:
        return {
            "n_signals": 0,
            "precision": float("nan"),
            "recall": 0.0,
            "breakeven_flag": "N/A (0 signals)",
        }
    y_hat = mask.astype(int)
    precision = float(precision_score(y, y_hat, zero_division=0))
    recall = float(recall_score(y, y_hat, zero_division=0))
    flag = "ABOVE breakeven" if precision >= BREAKEVEN_WINRATE else "BELOW breakeven"
    return {
        "n_signals": n_signals,
        "precision": precision,
        "recall": recall,
        "breakeven_flag": flag,
    }


def ensure_artifacts(
    *,
    features_path: Path,
    labels_path: Path,
    model_path: Path,
    preds_path: Path,
    force_refit: bool = False,
) -> tuple[object, pd.DataFrame]:
    """Load saved model+preds, or re-fit once and persist them."""
    if model_path.exists() and preds_path.exists() and not force_refit:
        model = joblib.load(model_path)
        preds = pd.read_parquet(preds_path)
        print(f"Loaded existing artifacts: {model_path} | {preds_path}")
        return model, preds

    print("No saved artifacts found - re-fitting LightGBM LONG once (same split/params)...")
    joined = load_joined_dataset(features_path, labels_path)
    kept, stats = drop_timeouts(joined, "long")
    print(
        f"LONG kept={stats['n_kept']} | timeout dropped={stats['n_timeout']} "
        f"({stats['timeout_pct']:.2f}%)"
    )
    splits = time_aware_split(kept, embargo_hours=HORIZON_BARS)
    print(
        f"train={len(splits.train)} val={len(splits.val)} test={len(splits.test)} | "
        f"test range={splits.test['Date'].min()} -> {splits.test['Date'].max()}"
    )

    x_tr = splits.train[FEATURE_COLS].copy()
    y_tr = splits.train["label_long"].astype(int).to_numpy()
    x_va = splits.val[FEATURE_COLS].copy()
    y_va = splits.val["label_long"].astype(int).to_numpy()
    x_te = splits.test[FEATURE_COLS].copy()
    y_te = splits.test["label_long"].astype(int).to_numpy()

    model, _p_val, p_test, _imp = train_lightgbm(x_tr, y_tr, x_va, y_va, x_te)

    preds = pd.DataFrame(
        {
            "Date": splits.test["Date"].to_numpy(),
            "y_true": y_te,
            "y_prob": p_test,
        }
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    preds.to_parquet(preds_path, index=False)
    print(f"Saved model -> {model_path}")
    print(f"Saved test preds -> {preds_path}")
    return model, preds


def subperiod_breakdown(preds: pd.DataFrame, thr: float) -> pd.DataFrame:
    df = preds.copy()
    df["period"] = df["Date"].map(_period_label)

    rows = []
    for period in PERIOD_ORDER:
        g = df[df["period"] == period]
        if g.empty:
            rows.append(
                {
                    "period": period,
                    "n_rows": 0,
                    "n_signals": 0,
                    "precision": float("nan"),
                    "recall": float("nan"),
                    "breakeven_flag": "N/A (empty)",
                }
            )
            continue
        m = _metrics_at_thr(g["y_true"].to_numpy(), g["y_prob"].to_numpy(), thr)
        rows.append(
            {
                "period": period,
                "n_rows": len(g),
                "n_signals": m["n_signals"],
                "precision": m["precision"],
                "recall": m["recall"],
                "breakeven_flag": m["breakeven_flag"],
            }
        )

    # Aggregate sanity row
    m_all = _metrics_at_thr(df["y_true"].to_numpy(), df["y_prob"].to_numpy(), thr)
    rows.append(
        {
            "period": "AGGREGATE (all test)",
            "n_rows": len(df),
            "n_signals": m_all["n_signals"],
            "precision": m_all["precision"],
            "recall": m_all["recall"],
            "breakeven_flag": m_all["breakeven_flag"],
        }
    )
    return pd.DataFrame(rows)


def evaluate_robustness_label(table: pd.DataFrame) -> dict:
    sub = table[table["period"] != "AGGREGATE (all test)"].copy()
    # Only periods with signals contribute to above/below counts meaningfully.
    usable = sub[sub["n_signals"] > 0].copy()
    above = usable[usable["breakeven_flag"] == "ABOVE breakeven"]
    below = usable[usable["breakeven_flag"] == "BELOW breakeven"]
    tiny_extreme = usable[
        (usable["n_signals"] < 20)
        & ((usable["precision"] <= 0.0) | (usable["precision"] >= 1.0))
    ]

    prec_vals = usable["precision"].dropna().to_numpy(dtype=float)
    prec_std = float(np.std(prec_vals, ddof=1)) if len(prec_vals) >= 2 else float("nan")

    n_periods = len(PERIOD_ORDER)
    n_above = int(len(above))
    n_below = int(len(below))

    if n_above >= 4 and tiny_extreme.empty:
        label = "ROBUST"
        note = (
            f"Precision ABOVE breakeven in {n_above}/{n_periods} sub-periods "
            f"(std precision={prec_std:.4f})."
        )
    elif n_above <= 2 or not tiny_extreme.empty:
        label = "INCONSISTENT/FRAGILE"
        note = (
            f"Above breakeven in only {n_above}/{n_periods} sub-periods "
            f"(below={n_below}); tiny/extreme periods={len(tiny_extreme)}; "
            f"std precision={prec_std:.4f}."
        )
    else:
        # e.g. 3 above / 3 below -> inconclusive
        label = "BELUM KONKLUSIF"
        note = (
            f"Mixed result: ABOVE={n_above}, BELOW={n_below} of {n_periods} "
            f"(std precision={prec_std:.4f}). Not rounded to either side."
        )

    return {
        "label": label,
        "note": note,
        "n_above": n_above,
        "n_below": n_below,
        "precision_std": prec_std,
        "tiny_extreme_periods": tiny_extreme["period"].tolist(),
    }


def bootstrap_precision_ci(
    y_signal: np.ndarray,
    *,
    n_boot: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """IID bootstrap over thresholded signals (does NOT account for autocorrelation)."""
    y = np.asarray(y_signal).astype(int)
    n = len(y)
    if n == 0:
        return {
            "n_signals": 0,
            "point_precision": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "ci_low_above_breakeven": False,
        }

    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = y[rng.integers(0, n, size=n)]
        boots[i] = float(sample.mean())

    ci_low = float(np.quantile(boots, 0.025))
    ci_high = float(np.quantile(boots, 0.975))
    point = float(y.mean())
    return {
        "n_signals": n,
        "point_precision": point,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_low_above_breakeven": ci_low >= BREAKEVEN_WINRATE,
        "boot_mean": float(boots.mean()),
        "boot_std": float(boots.std(ddof=1)),
    }


def final_recommendation(rob: dict, boot: dict, agg_precision: float, agg_n: int) -> str:
    lines = []
    lines.append("## Rekomendasi akhir")
    lines.append("")

    if rob["label"] == "ROBUST" and boot["ci_low_above_breakeven"]:
        lines.append(
            f"**Layak dianggap kandidat kuat untuk lanjut ke backtest (Layer 5)** "
            f"untuk LONG-LightGBM @ thr={SELECTED_THR:.2f}: "
            f"sub-period={rob['label']}, bootstrap 95% CI lower={boot['ci_low']:.4f} "
            f"masih >= breakeven {BREAKEVEN_WINRATE:.4f}, "
            f"agregat precision={agg_precision:.4f} (n={agg_n})."
        )
    elif rob["label"] == "BELUM KONKLUSIF" or (
        rob["label"] == "ROBUST" and not boot["ci_low_above_breakeven"]
    ):
        lines.append(
            f"**Belum konklusif / sinyal lemah-menengah** untuk lanjut backtest tanpa "
            f"catatan: sub-period={rob['label']} ({rob['note']}); "
            f"bootstrap 95% CI=[{boot['ci_low']:.4f}, {boot['ci_high']:.4f}] "
            f"(lower {'>=' if boot['ci_low_above_breakeven'] else '<'} breakeven). "
            "Jangan dibulatkan sebagai edge kuat."
        )
    else:
        lines.append(
            f"**Tidak cukup meyakinkan untuk lanjut ke backtest sebagai kandidat kuat.** "
            f"Sub-period={rob['label']} ({rob['note']}); "
            f"bootstrap 95% CI=[{boot['ci_low']:.4f}, {boot['ci_high']:.4f}]. "
            "Pertimbangkan perbaikan feature engineering / labeling sebelum Layer 5."
        )
    return "\n".join(lines)


def render_report(
    *,
    table: pd.DataFrame,
    rob: dict,
    boot: dict,
    model_path: Path,
    preds_path: Path,
) -> str:
    agg = table[table["period"] == "AGGREGATE (all test)"].iloc[0]
    lines = [
        "# Robustness Check — LONG LightGBM @ threshold 0.52",
        "",
        "## Setup",
        f"- Model artifact: `{model_path}`",
        f"- Test predictions: `{preds_path}`",
        f"- Frozen threshold: **{SELECTED_THR:.2f}** (validation-selected; NOT re-tuned per sub-period)",
        f"- Breakeven win-rate: **{BREAKEVEN_WINRATE:.4f}**",
        f"- Test range target: 2024-01-02 -> 2026-07-08 (LONG kept rows before thr filter)",
        (
            f"- Sanity target from prior report: precision~={EXPECTED_TEST_PRECISION}, "
            f"n~={EXPECTED_TEST_N} (re-fit may differ slightly)"
        ),
        "",
        "## Bagian 1 — Sub-period breakdown",
        "",
        table.to_markdown(index=False, floatfmt=".4f"),
        "",
        "### Robustness verdict",
        f"- **Label:** **{rob['label']}**",
        f"- Detail: {rob['note']}",
        f"- n_above={rob['n_above']}, n_below={rob['n_below']}, "
        f"precision_std={rob['precision_std']:.4f}",
    ]
    if rob["tiny_extreme_periods"]:
        lines.append(
            f"- Tiny/extreme periods (n_signals<20 & precision 0% or 100%): "
            f"{rob['tiny_extreme_periods']}"
        )
    lines.extend(
        [
            "",
            "## Bagian 2 — Bootstrap 95% CI on thresholded test precision",
            "",
            f"- n_signals (p>= {SELECTED_THR:.2f}): **{boot['n_signals']}**",
            f"- Point precision: **{boot['point_precision']:.4f}**",
            f"- Bootstrap iterations: {N_BOOTSTRAP} (with replacement, seed={BOOTSTRAP_SEED})",
            f"- 95% CI: **[{boot['ci_low']:.4f}, {boot['ci_high']:.4f}]**",
            (
                f"- CI lower bound vs breakeven {BREAKEVEN_WINRATE:.4f}: "
                f"**{'STILL ABOVE' if boot['ci_low_above_breakeven'] else 'BELOW / overlaps breakeven'}**"
            ),
            "",
            "### Keterbatasan metode",
            (
                "Bootstrap di atas mengasumsikan sample IID. Label triple-barrier punya "
                f"overlapping forward windows (horizon={HORIZON_BARS}), jadi antar baris "
                "saling berkorelasi. CI ini kemungkinan **under-estimate** ketidakpastian "
                "sebenarnya (terlalu sempit). Interpretasikan sebagai batas bawah optimis "
                "untuk ketidakpastian, bukan CI yang sepenuhnya time-series-aware."
            ),
            "",
            final_recommendation(
                rob,
                boot,
                float(agg["precision"]),
                int(agg["n_signals"]),
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    base = Path("data")
    features_path = base / "features/xauusd_h1_h4_d1_features.parquet"
    labels_path = base / "labels/xauusd_triple_barrier_labels.parquet"
    model_path = base / "models/lightgbm_long.pkl"
    preds_path = base / "models/lightgbm_long_test_preds.parquet"
    report_path = base / "reports/robustness_check_long_lightgbm.md"

    _model, preds = ensure_artifacts(
        features_path=features_path,
        labels_path=labels_path,
        model_path=model_path,
        preds_path=preds_path,
        force_refit=False,
    )

    # Ensure Date is datetime
    preds = preds.copy()
    preds["Date"] = pd.to_datetime(preds["Date"], utc=True)

    table = subperiod_breakdown(preds, SELECTED_THR)
    rob = evaluate_robustness_label(table)

    signal_mask = preds["y_prob"].to_numpy() >= SELECTED_THR
    y_signal = preds.loc[signal_mask, "y_true"].to_numpy()
    boot = bootstrap_precision_ci(y_signal)

    report = render_report(
        table=table,
        rob=rob,
        boot=boot,
        model_path=model_path,
        preds_path=preds_path,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    print("\n" + report)
    print(f"\nSaved report -> {report_path}")


if __name__ == "__main__":
    main()
