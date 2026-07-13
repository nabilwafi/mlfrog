"""
Standalone native D1 / H4 classification experiment.

Tests whether classic indicators are more informative on their native TF
than when broadcast onto H1 (dilution hypothesis).

Usage:
    python -m experiments.standalone_htf
"""

from __future__ import annotations

import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator, EMAIndicator
from ta.volatility import AverageTrueRange

from features.feature_engineering import load_ohlc_csv
from labels.triple_barrier_labeling import implied_breakeven_winrate
from models.evaluate import (
    MIN_SIGNALS_FOR_SELECTION,
    SELECTION_THRESHOLDS,
    evaluate_probs,
    select_threshold_from_validation,
)

warnings.filterwarnings("ignore", category=UserWarning)

H1_BASELINE_VAL_AUC = {"long": 0.539043, "short": 0.494149}
BE = implied_breakeven_winrate(1.5, 2.0)


def _onehot() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_native_features(df_raw: pd.DataFrame, *, tf: str, bar_delta: pd.Timedelta) -> pd.DataFrame:
    """Native technicals on close-time index (open + bar_delta)."""
    df = df_raw.copy()
    df["Date"] = df["Date"] + bar_delta
    df = df.sort_values("Date").reset_index(drop=True)

    adx = ADXIndicator(df["High"], df["Low"], df["Close"], window=14, fillna=False).adx()
    atr = AverageTrueRange(df["High"], df["Low"], df["Close"], window=14, fillna=False).average_true_range()
    rsi = RSIIndicator(df["Close"], window=14, fillna=False).rsi()
    ema_f = EMAIndicator(df["Close"], window=12, fillna=False).ema_indicator()
    ema_s = EMAIndicator(df["Close"], window=26, fillna=False).ema_indicator()

    bull = (ema_f.shift(1) <= ema_s.shift(1)) & (ema_f > ema_s)
    bear = (ema_f.shift(1) >= ema_s.shift(1)) & (ema_f < ema_s)
    cross = pd.Series(0, index=df.index, dtype=int)
    cross[bull] = 1
    cross[bear] = -1

    def _last_nz(arr: np.ndarray) -> float:
        nz = arr[arr != 0]
        return float(nz[-1]) if nz.size else 0.0

    cross_sig = cross.rolling(3, min_periods=3).apply(_last_nz, raw=True).fillna(0).astype(int)

    if tf == "d1":
        r1 = np.log(df["Close"] / df["Close"].shift(1))
        r5 = np.log(df["Close"] / df["Close"].shift(5))
        ret_cols = {"return_1d": r1, "return_5d": r5}
        feat_names = [
            "adx_d1", "atr_d1", "rsi_d1", "ema_fast_d1", "ema_slow_d1",
            "ema_cross_signal_d1", "return_1d", "return_5d",
        ]
        values = {
            "adx_d1": adx, "atr_d1": atr, "rsi_d1": rsi,
            "ema_fast_d1": ema_f, "ema_slow_d1": ema_s,
            "ema_cross_signal_d1": cross_sig, **ret_cols,
        }
    else:
        r1 = np.log(df["Close"] / df["Close"].shift(1))
        r5 = np.log(df["Close"] / df["Close"].shift(6))  # ~1 day of H4
        feat_names = [
            "adx_h4", "atr_h4", "rsi_h4", "ema_fast_h4", "ema_slow_h4",
            "ema_cross_signal_h4", "return_1h4", "return_6h4",
        ]
        values = {
            "adx_h4": adx, "atr_h4": atr, "rsi_h4": rsi,
            "ema_fast_h4": ema_f, "ema_slow_h4": ema_s,
            "ema_cross_signal_h4": cross_sig,
            "return_1h4": r1, "return_6h4": r5,
        }

    out = pd.DataFrame({"Date": df["Date"], **values})
    out["Open"] = df["Open"]
    out["High"] = df["High"]
    out["Low"] = df["Low"]
    out["Close"] = df["Close"]
    out = out.dropna(subset=feat_names).reset_index(drop=True)
    return out, feat_names


def label_native(
    df: pd.DataFrame,
    *,
    atr_col: str,
    sl_mult: float = 1.5,
    tp_mult: float = 2.0,
    horizon: int = 8,
) -> pd.DataFrame:
    high = df["High"].to_numpy(float)
    low = df["Low"].to_numpy(float)
    close = df["Close"].to_numpy(float)
    atr = df[atr_col].to_numpy(float)
    n = len(df)
    lab_l = np.full(n, -1, dtype=int)
    lab_s = np.full(n, -1, dtype=int)
    bars_l = np.full(n, horizon, dtype=int)
    bars_s = np.full(n, horizon, dtype=int)

    for i in range(n):
        if not np.isfinite(atr[i]):
            continue
        entry = close[i]
        tp_l = entry + tp_mult * atr[i]
        sl_l = entry - sl_mult * atr[i]
        tp_s = entry - tp_mult * atr[i]
        sl_s = entry + sl_mult * atr[i]
        end_j = min(n - 1, i + horizon)

        for j in range(i + 1, end_j + 1):
            if low[j] <= sl_l:
                lab_l[i], bars_l[i] = 0, j - i
                break
            if high[j] >= tp_l:
                lab_l[i], bars_l[i] = 1, j - i
                break
        else:
            lab_l[i] = -1
            bars_l[i] = min(horizon, n - 1 - i)

        for j in range(i + 1, end_j + 1):
            if high[j] >= sl_s:
                lab_s[i], bars_s[i] = 0, j - i
                break
            if low[j] <= tp_s:
                lab_s[i], bars_s[i] = 1, j - i
                break
        else:
            lab_s[i] = -1
            bars_s[i] = min(horizon, n - 1 - i)

    out = df.copy()
    out["label_long"] = lab_l
    out["label_short"] = lab_s
    out["bars_to_resolution_long"] = bars_l
    out["bars_to_resolution_short"] = bars_s
    return out


def _dist(labels: pd.Series) -> dict:
    n = len(labels)
    w = int((labels == 1).sum())
    l = int((labels == 0).sum())
    t = int((labels == -1).sum())
    return {
        "n": n,
        "pct_win": 100 * w / n if n else 0,
        "pct_loss": 100 * l / n if n else 0,
        "pct_timeout": 100 * t / n if n else 0,
        "wr_ex_to": w / (w + l) if (w + l) else 0,
        "n_kept": w + l,
    }


def pick_horizon(df_feat: pd.DataFrame, atr_col: str, horizons: list[int]) -> tuple[int, pd.DataFrame, list[dict]]:
    """Pick horizon with healthiest label mix (prefer timeout~30-45%, wr near BE, enough kept)."""
    rows = []
    best_h, best_score, best_df = horizons[0], -1e9, None
    for h in horizons:
        lab = label_native(df_feat, atr_col=atr_col, horizon=h)
        for d in ("long", "short"):
            st = _dist(lab[f"label_{d}"])
            rows.append({"horizon": h, "direction": d, **st})
        # score: both sides timeout not extreme, kept large, timeout in [25,50]
        sl = _dist(lab["label_long"])
        ss = _dist(lab["label_short"])
        to = 0.5 * (sl["pct_timeout"] + ss["pct_timeout"])
        kept = 0.5 * (sl["n_kept"] + ss["n_kept"])
        score = kept - abs(to - 37.5) * 30  # prefer mid timeout
        if to < 15 or to > 60:
            score -= 5000
        if score > best_score:
            best_score, best_h, best_df = score, h, lab
    assert best_df is not None
    return best_h, best_df, rows


def time_split(df: pd.DataFrame, *, embargo: pd.Timedelta) -> dict[str, pd.DataFrame]:
    te = pd.Timestamp("2023-01-01", tz="UTC")
    ve = pd.Timestamp("2024-01-01", tz="UTC")
    train = df[df["Date"] < (te - embargo)]
    val = df[(df["Date"] >= (te + embargo)) & (df["Date"] < (ve - embargo))]
    test = df[df["Date"] >= (ve + embargo)]
    return {"train": train.reset_index(drop=True), "val": val.reset_index(drop=True), "test": test.reset_index(drop=True)}


def run_models(df_lab: pd.DataFrame, feat_cols: list[str], direction: str, embargo: pd.Timedelta) -> dict:
    kept = df_lab[df_lab[f"label_{direction}"] != -1].copy()
    splits = time_split(kept, embargo=embargo)
    ycol = f"label_{direction}"

    def xy(frame: pd.DataFrame):
        return frame[feat_cols], frame[ycol].astype(int).to_numpy()

    xtr, ytr = xy(splits["train"])
    xva, yva = xy(splits["val"])
    xte, yte = xy(splits["test"])

    # LogReg
    pre = ColumnTransformer(
        [("num", StandardScaler(), feat_cols)],
        remainder="drop",
    )
    # ema_cross is int categorical-ish but treat numeric OK for quick experiment
    pipe = Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=2000, solver="lbfgs")),
    ])
    pipe.fit(xtr, ytr)
    p_va_lr = pipe.predict_proba(xva)[:, 1]
    p_te_lr = pipe.predict_proba(xte)[:, 1]
    ev_va_lr = evaluate_probs(yva, p_va_lr)
    ev_te_lr = evaluate_probs(yte, p_te_lr)
    sel_lr = select_threshold_from_validation(yva, p_va_lr)

    # LightGBM
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
        xtr, ytr,
        eval_set=[(xva, yva)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    p_va = model.predict_proba(xva)[:, 1]
    p_te = model.predict_proba(xte)[:, 1]
    ev_va = evaluate_probs(yva, p_va)
    ev_te = evaluate_probs(yte, p_te)
    sel = select_threshold_from_validation(yva, p_va)

    return {
        "direction": direction,
        "n_train": len(splits["train"]),
        "n_val": len(splits["val"]),
        "n_test": len(splits["test"]),
        "lgbm_val_auc": ev_va["auc_roc"],
        "lgbm_val_aupr": ev_va["auc_pr"],
        "lgbm_test_auc": ev_te["auc_roc"],
        "lgbm_test_aupr": ev_te["auc_pr"],
        "lgbm_selected_thr": sel["selected_thr"],
        "lgbm_selection_note": sel["selection_note"],
        "lr_val_auc": ev_va_lr["auc_roc"],
        "lr_test_auc": ev_te_lr["auc_roc"],
        "lr_selected_thr": sel_lr["selected_thr"],
    }


def experiment_tf(
    *,
    tf: str,
    raw_path: Path,
    bar_delta: pd.Timedelta,
    atr_col: str,
    horizons: list[int],
    embargo: pd.Timedelta,
    feat_out: Path,
    lab_out: Path,
    report_out: Path,
) -> None:
    print(f"\n=== Standalone {tf.upper()} ===")
    raw = load_ohlc_csv(raw_path)
    print(f"raw rows={len(raw)}")
    feat, feat_cols = build_native_features(raw, tf=tf, bar_delta=bar_delta)
    print(f"features after warmup={len(feat)} cols={feat_cols}")

    best_h, labeled, grid = pick_horizon(feat, atr_col, horizons)
    # Embargo matches chosen horizon in bar units.
    if tf == "d1":
        embargo = pd.Timedelta(days=best_h)
    else:
        embargo = pd.Timedelta(hours=4 * best_h)
    print(f"chosen horizon={best_h} embargo={embargo}")
    for r in grid:
        if r["horizon"] == best_h:
            print(
                f"  {r['direction']}: win={r['pct_win']:.1f}% loss={r['pct_loss']:.1f}% "
                f"to={r['pct_timeout']:.1f}% wr={r['wr_ex_to']:.3f} kept={r['n_kept']}"
            )

    feat_save = feat.drop(columns=["Open", "High", "Low", "Close"], errors="ignore")
    feat_out.parent.mkdir(parents=True, exist_ok=True)
    feat_save.to_parquet(feat_out, index=False)
    lab_out.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_parquet(lab_out, index=False)

    results = []
    for d in ("long", "short"):
        r = run_models(labeled, feat_cols, d, embargo)
        results.append(r)
        print(
            f"  {d}: LGBM valAUC={r['lgbm_val_auc']:.4f} testAUC={r['lgbm_test_auc']:.4f} "
            f"thr={r['lgbm_selected_thr']} | LR valAUC={r['lr_val_auc']:.4f}"
        )

    # report
    lines = [
        f"# Standalone {tf.upper()} Experiment",
        "",
        "## Data",
        f"- Raw: `{raw_path}`",
        f"- Native feature rows (post warm-up): {len(feat)}",
        f"- Close-time shift: {bar_delta}",
        f"- Features: {feat_cols}",
        "",
        "## Label grid (SL=1.5 ATR, TP=2.0 ATR)",
        f"- Chosen horizon: **{best_h}** bars",
        f"- Breakeven WR: {BE:.4f}",
        "",
        "| horizon | direction | win% | loss% | timeout% | wr_ex_to | n_kept |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for r in grid:
        lines.append(
            f"| {r['horizon']} | {r['direction']} | {r['pct_win']:.2f} | {r['pct_loss']:.2f} | "
            f"{r['pct_timeout']:.2f} | {r['wr_ex_to']:.4f} | {r['n_kept']} |"
        )

    lines += [
        "",
        "## Models (time split train<2023 / val 2023 / test≥2024, embargo = horizon)",
        "",
        "| direction | n_train | n_val | n_test | LGBM val AUC | LGBM test AUC | LGBM thr | LR val AUC | LR test AUC |",
        "|---|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    by_dir = {r["direction"]: r for r in results}
    for d in ("long", "short"):
        r = by_dir[d]
        lines.append(
            f"| {d} | {r['n_train']} | {r['n_val']} | {r['n_test']} | "
            f"{r['lgbm_val_auc']:.4f} | {r['lgbm_test_auc']:.4f} | {r['lgbm_selected_thr']} | "
            f"{r['lr_val_auc']:.4f} | {r['lr_test_auc']:.4f} |"
        )
        lines.append(f"- {d} LGBM selection: {r['lgbm_selection_note']}")

    if tf == "d1":
        lines += [
            "",
            "## Comparison vs H1 broadcast baseline (LGBM val AUC)",
            "",
            "| | AUC (broadcast di H1, existing) | AUC (standalone D1, native LGBM val) | Delta |",
            "|---|---:|---:|---:|",
        ]
        for d in ("long", "short"):
            base = H1_BASELINE_VAL_AUC[d]
            native = by_dir[d]["lgbm_val_auc"]
            delta = native - base
            lines.append(f"| {d.upper()} | {base:.3f} | {native:.3f} | {delta:+.3f} |")

        d_long = by_dir["long"]["lgbm_val_auc"] - H1_BASELINE_VAL_AUC["long"]
        d_short = by_dir["short"]["lgbm_val_auc"] - H1_BASELINE_VAL_AUC["short"]
        avg_delta = 0.5 * (d_long + d_short)
        lines += ["", "## Conclusion"]
        if avg_delta >= 0.03 or max(d_long, d_short) >= 0.05:
            lines += [
                "",
                f"**Hipotesis 'fitur encer waktu broadcast' TERKONFIRMASI** "
                f"(avg delta val AUC = {avg_delta:+.3f}; LONG {d_long:+.3f}, SHORT {d_short:+.3f}).",
                "",
                "Rekomendasi: lanjutkan hierarchical/stacking — latih model D1 (dan H4) standalone, "
                "ambil `predict_proba` pada close D1/H4, lalu `merge_asof(direction='backward')` ke row H1 "
                "sebagai meta-feature (ganti/lengkapi slope & trend_direction broadcast mentah). "
                "H1 model tetap pakai label/horizon H1; meta-feature hanya bias context yang sudah "
                "terkompresi jadi skor prediktif native-TF.",
            ]
        else:
            lines += [
                "",
                f"**Hipotesis broadcast-dilution TIDAK terkonfirmasi** "
                f"(avg delta val AUC = {avg_delta:+.3f}; LONG {d_long:+.3f}, SHORT {d_short:+.3f}).",
                "",
                "Sinyal indikator klasik lemah di skala native D1 juga. Effort berikutnya sebaiknya "
                "ke jenis fitur berbeda (reversal/exhaustion/liquidity), bukan arsitektur multi-TF stacking.",
            ]
    else:
        lines += [
            "",
            "## Note vs H1 baseline",
            f"- H1 broadcast LGBM val AUC LONG={H1_BASELINE_VAL_AUC['long']:.3f} SHORT={H1_BASELINE_VAL_AUC['short']:.3f}",
            f"- Standalone H4 LGBM val AUC LONG={by_dir['long']['lgbm_val_auc']:.3f} "
            f"SHORT={by_dir['short']['lgbm_val_auc']:.3f}",
            "",
            "## Conclusion",
            "Lihat juga `standalone_d1_experiment.md` untuk keputusan stacking utama; "
            "H4 adalah cek tambahan di skala intermediate.",
        ]

    lines += [
        "",
        "## Artifacts",
        f"- `{feat_out}`",
        f"- `{lab_out}`",
        "",
    ]
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {report_out}")


def main() -> None:
    base = Path("data")
    experiment_tf(
        tf="d1",
        raw_path=base / "raw/XAUUSD_D1.csv",
        bar_delta=pd.Timedelta(days=1),
        atr_col="atr_d1",
        horizons=[5, 8, 10],
        embargo=pd.Timedelta(days=8),  # matched to chosen horizon later loosely; use 8d
        feat_out=base / "features/d1_native_features.parquet",
        lab_out=base / "labels/d1_labeled.parquet",
        report_out=base / "reports/standalone_d1_experiment.md",
    )
    # re-run D1 embargo with chosen horizon from file? For simplicity embargo=8d is fine for all D1 grids.

    experiment_tf(
        tf="h4",
        raw_path=base / "raw/XAUUSD_H4.csv",
        bar_delta=pd.Timedelta(hours=4),
        atr_col="atr_h4",
        horizons=[6, 8, 12],
        embargo=pd.Timedelta(hours=4 * 8),
        feat_out=base / "features/h4_native_features.parquet",
        lab_out=base / "labels/h4_labeled.parquet",
        report_out=base / "reports/standalone_h4_experiment.md",
    )


if __name__ == "__main__":
    main()
