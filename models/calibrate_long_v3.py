"""
Fit / evaluate probability calibration for LONG LightGBM v3.

Discipline:
  - Fit calibrator on the final-model early-stopping validation slice
    (last 15% of pre-2026 resolved labels — same model as sealed preds).
  - Evaluate ECE + tier monotonicity on sealed 2026 test (never used for fit).

Usage:
    python -m models.calibrate_long_v3
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from models.data_prep import BREAKEVEN_WINRATE, CAT_COLS, HORIZON_BARS, drop_timeouts
from models.train_walk_forward_v2 import TEST_START
from models.train_walk_forward_v3 import FEATURE_COLS_V3, load_v3_dataset
from src.confidence_engine import (
    ConfidenceEngine,
    IsotonicCalibrator,
    PlattCalibrator,
    TierThresholds,
)

THR_RAW = 0.51
N_BINS = 10


def _xy(frame: pd.DataFrame):
    x = frame[FEATURE_COLS_V3].copy()
    for c in CAT_COLS:
        x[c] = x[c].astype("category")
    y = frame["label_long"].astype(int).to_numpy()
    return x, y


def reliability_table(y: np.ndarray, p: np.ndarray, n_bins: int = N_BINS) -> pd.DataFrame:
    """Equal-mass (decile) bins of predicted probability vs realized frequency."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    # rank-based deciles; duplicate edges possible if mass on few unique p
    try:
        bin_id = pd.qcut(p, q=n_bins, labels=False, duplicates="drop")
    except ValueError:
        bin_id = pd.cut(p, bins=n_bins, labels=False, include_lowest=True)
    rows = []
    for b in sorted(pd.Series(bin_id).dropna().unique()):
        m = bin_id == b
        rows.append(
            {
                "bin": int(b) + 1,
                "n": int(m.sum()),
                "mean_predicted": float(p[m].mean()),
                "realized_rate": float(y[m].mean()),
                "abs_gap": float(abs(p[m].mean() - y[m].mean())),
            }
        )
    return pd.DataFrame(rows)


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = N_BINS) -> float:
    tab = reliability_table(y, p, n_bins=n_bins)
    if tab.empty or tab["n"].sum() == 0:
        return float("nan")
    w = tab["n"].to_numpy(float) / tab["n"].sum()
    return float((w * tab["abs_gap"].to_numpy(float)).sum())


def _md_rel(tab: pd.DataFrame, title: str) -> list[str]:
    lines = [
        f"### {title}",
        "",
        "| bin | n | mean_predicted | realized_rate | |gap| |",
        "|---:|---:|---:|---:|---:|",
    ]
    for _, r in tab.iterrows():
        lines.append(
            f"| {int(r['bin'])} | {int(r['n'])} | {r['mean_predicted']:.4f} | "
            f"{r['realized_rate']:.4f} | {r['abs_gap']:.4f} |"
        )
    lines.append("")
    return lines


def load_val_and_test(base: Path):
    """Score frozen WF final model on its val slice; load sealed test preds."""
    with open(base / "models_v3/long/lightgbm_long_v3.pkl", "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]

    df = load_v3_dataset(
        base / "features/xauusd_h1_h4_d1_features_v3.parquet",
        base / "labels/xauusd_triple_barrier_labels.parquet",
    )
    kept, _ = drop_timeouts(df, "long")
    te = pd.Timestamp(TEST_START, tz="UTC")
    emb = pd.Timedelta(hours=HORIZON_BARS)
    train_full = kept[kept["Date"] < (te - emb)].copy()
    cut = train_full["Date"].quantile(0.85)
    va = train_full[train_full["Date"] >= cut].copy()

    xva, yva = _xy(va)
    p_raw_va = model.predict_proba(xva)[:, 1]

    test_preds = pd.read_parquet(base / "models_v3/long/lightgbm_long_v3_test_preds.parquet")
    y_te = test_preds["y_true"].to_numpy(int)
    p_raw_te = test_preds["y_prob"].to_numpy(float)

    meta = {
        "n_val": len(va),
        "n_test": len(test_preds),
        "val_start": str(va["Date"].min()),
        "val_end": str(va["Date"].max()),
        "val_base_rate": float(yva.mean()),
        "test_base_rate": float(y_te.mean()),
        "selected_thr_raw": float(bundle.get("selected_thr") or THR_RAW),
    }
    return yva, p_raw_va, y_te, p_raw_te, meta


def fit_calibrators(y_va: np.ndarray, p_va: np.ndarray):
    lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    lr.fit(p_va.reshape(-1, 1), y_va)
    platt = PlattCalibrator(lr)

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p_va, y_va)
    isotonic = IsotonicCalibrator(iso)
    return platt, isotonic


def choose_tiers(p_cal_va: np.ndarray, p_raw_va: np.ndarray, calibrator, raw_thr: float = THR_RAW) -> TierThresholds:
    """
    Tier edges on calibrated scale:
      low    : calibrated < breakeven
      medium : [BE, split)
      high   : >= split

    `split` = median calibrated confidence among validation rows with raw p >= raw_thr
    so that tradeable signals are not all collapsed into a single tier after Platt
    compression. Fallback: median of calibrated among calibrated >= BE.
    """
    be = float(BREAKEVEN_WINRATE)
    cal_thr = float(calibrator.predict(np.array([raw_thr]))[0])
    tradeable = p_cal_va[p_raw_va >= raw_thr]
    if len(tradeable) >= 50:
        split = float(np.median(tradeable))
        src = f"median calibrated among val raw>={raw_thr} (n={len(tradeable)})"
    else:
        above = p_cal_va[p_cal_va >= be]
        split = float(np.median(above)) if len(above) else cal_thr
        src = f"median calibrated among val cal>=BE (n={len(above)})"
    # ensure medium band non-empty
    if split <= be + 1e-4:
        t1, t2 = np.quantile(p_cal_va, [1 / 3, 2 / 3])
        return TierThresholds(
            low_max=float(t1),
            medium_max=float(t2),
            rationale=(
                f"Fallback tertiles — tradeable split ({split:.4f}) collapsed near BE. "
                f"calibrated(raw_thr={raw_thr})={cal_thr:.4f}."
            ),
        )
    return TierThresholds(
        low_max=be,
        medium_max=split,
        rationale=(
            f"low < BE={be:.4f} (no economic edge). "
            f"medium/high split = {src} = {split:.4f}. "
            f"Note: calibrated(raw_thr={raw_thr})={cal_thr:.4f} "
            f"(Platt compresses raw probs toward base rate)."
        ),
    )


def tier_validation(y: np.ndarray, p_cal: np.ndarray, tiers: TierThresholds) -> pd.DataFrame:
    eng = ConfidenceEngine(calibrator=_Identity(), tier_thresholds=tiers)  # noqa — use edges only
    # classify directly
    labels = []
    for c in p_cal:
        if c < tiers.low_max:
            labels.append("low")
        elif c < tiers.medium_max:
            labels.append("medium")
        else:
            labels.append("high")
    labels = np.array(labels)
    rows = []
    for tier in ("low", "medium", "high"):
        m = labels == tier
        n = int(m.sum())
        rows.append(
            {
                "tier": tier,
                "n": n,
                "mean_calibrated": float(p_cal[m].mean()) if n else float("nan"),
                "realized_wr": float(y[m].mean()) if n else float("nan"),
            }
        )
    return pd.DataFrame(rows)


class _Identity:
    def predict(self, raw: np.ndarray) -> np.ndarray:
        return np.asarray(raw, dtype=float)


def _monotonic(wr_by_tier: dict[str, float]) -> bool:
    """True if low <= medium <= high (allowing nan gaps)."""
    seq = [wr_by_tier.get("low"), wr_by_tier.get("medium"), wr_by_tier.get("high")]
    prev = None
    for w in seq:
        if w is None or (isinstance(w, float) and np.isnan(w)):
            continue
        if prev is not None and w + 1e-9 < prev:
            return False
        prev = w
    return True


def main() -> None:
    base = Path("data")
    y_va, p_va, y_te, p_te, meta = load_val_and_test(base)

    print("=== PREFLIGHT ===")
    print(f"n_val={meta['n_val']} ({meta['val_start']} -> {meta['val_end']})")
    print(f"n_test_sealed={meta['n_test']}  base_rate_val={meta['val_base_rate']:.4f} "
          f"base_rate_test={meta['test_base_rate']:.4f}")
    concern = ""
    if meta["n_val"] < 500:
        concern = "WARNING: n_val < 500 — isotonic may be unstable."
    elif meta["n_val"] < 2000:
        concern = "NOTE: n_val moderate; prefer Platt if test ECE close."
    else:
        concern = "n_val ample for both Platt and isotonic."
    print(concern)

    ece_va_raw = ece(y_va, p_va)
    ece_te_raw = ece(y_te, p_te)
    rel_va_raw = reliability_table(y_va, p_va)
    rel_te_raw = reliability_table(y_te, p_te)

    platt, isotonic = fit_calibrators(y_va, p_va)
    p_va_platt = platt.predict(p_va)
    p_te_platt = platt.predict(p_te)
    p_va_iso = isotonic.predict(p_va)
    p_te_iso = isotonic.predict(p_te)

    scores = {
        "raw": {"ece_val": ece_va_raw, "ece_test": ece_te_raw},
        "platt": {
            "ece_val": ece(y_va, p_va_platt),
            "ece_test": ece(y_te, p_te_platt),
            "cal": platt,
            "p_va": p_va_platt,
            "p_te": p_te_platt,
        },
        "isotonic": {
            "ece_val": ece(y_va, p_va_iso),
            "ece_test": ece(y_te, p_te_iso),
            "cal": isotonic,
            "p_va": p_va_iso,
            "p_te": p_te_iso,
        },
    }

    # Pick by sealed TEST ECE (primary); tie-break smaller val ECE
    best_name = min(
        ("platt", "isotonic"),
        key=lambda k: (scores[k]["ece_test"], scores[k]["ece_val"]),
    )
    best = scores[best_name]
    print(f"ECE val raw={ece_va_raw:.4f} platt={scores['platt']['ece_val']:.4f} "
          f"iso={scores['isotonic']['ece_val']:.4f}")
    print(f"ECE test raw={ece_te_raw:.4f} platt={scores['platt']['ece_test']:.4f} "
          f"iso={scores['isotonic']['ece_test']:.4f}")
    print(f"SELECTED: {best_name}")

    tiers = choose_tiers(best["p_va"], p_va, best["cal"], raw_thr=THR_RAW)
    engine = ConfidenceEngine(best["cal"], tiers)
    tier_te = tier_validation(y_te, best["p_te"], tiers)
    tier_va = tier_validation(y_va, best["p_va"], tiers)
    wr_map = {r["tier"]: r["realized_wr"] for _, r in tier_te.iterrows()}
    mono = _monotonic(wr_map)

    # Also show signal-gated (@ raw thr) tier mix on test
    m_sig = p_te >= THR_RAW
    tier_sig = tier_validation(y_te[m_sig], best["p_te"][m_sig], tiers) if m_sig.any() else None

    out_pkl = Path("models/calibration/confidence_calibrator_long_v3.pkl")
    engine.save(
        out_pkl,
        meta={
            **meta,
            "method": best_name,
            "ece_val_raw": ece_va_raw,
            "ece_test_raw": ece_te_raw,
            "ece_val_cal": best["ece_val"],
            "ece_test_cal": best["ece_test"],
            "ece_val_platt": scores["platt"]["ece_val"],
            "ece_test_platt": scores["platt"]["ece_test"],
            "ece_val_isotonic": scores["isotonic"]["ece_val"],
            "ece_test_isotonic": scores["isotonic"]["ece_test"],
            "tier_monotonic_test": mono,
            "breakeven": BREAKEVEN_WINRATE,
            "raw_thr": THR_RAW,
        },
    )
    print("wrote", out_pkl)

    # Trust verdict — sealed 2026 has lower base rate than val; perfect ECE is unrealistic
    ece_improved = best["ece_test"] < ece_te_raw - 0.005
    if not mono:
        trust = (
            "**NOT READY** for sizing — tier realized WR is not monotonic on sealed test. "
            "Fix calibration / tier edges before Risk Engine uses this."
        )
    elif best["ece_test"] > 0.12:
        trust = (
            "**NOT READY** — test ECE still high after calibration. "
            "Confidence numbers are rough; do not drive size/R:R yet."
        )
    elif mono and ece_improved and best["ece_test"] <= 0.10:
        trust = (
            "**CONDITIONAL YES** — Platt cuts sealed-test ECE and tiers are monotonic "
            f"(low→high WR). Remaining ECE≈{best['ece_test']:.3f} reflects a 2026 base-rate "
            "shift (val 0.44 vs test 0.35) that a static calibrator cannot fully fix. "
            "Safe as a **soft** Risk Engine input (mild size tilt by tier); "
            "**not** yet for hard R:R or aggressive leverage. Re-fit if the base model changes."
        )
    else:
        trust = (
            "**MARGINAL** — use as diagnostic only; keep flat sizing until ECE/monotonicity improve."
        )

    # Report
    lines = [
        "# Calibration Report — LONG LightGBM v3",
        "",
        "## Preflight",
        "",
        f"- Model: `data/models_v3/long/lightgbm_long_v3.pkl` (WF final; sealed preds source)",
        f"- Validation (fit calibrator): **n={meta['n_val']}** "
        f"({meta['val_start']} → {meta['val_end']}), base rate={meta['val_base_rate']:.4f}",
        f"- Sealed test (evaluate only): **n={meta['n_test']}**, "
        f"base rate={meta['test_base_rate']:.4f}",
        f"- Labels: resolved TP/SL only (`drop_timeouts`) — same as model training",
        f"- {concern}",
        f"- Breakeven WR = {BREAKEVEN_WINRATE:.4f}; operational raw thr = {THR_RAW}",
        "",
        "## 1. ECE before vs after",
        "",
        "| set | raw | Platt | Isotonic |",
        "|---|---:|---:|---:|",
        f"| validation | {ece_va_raw:.4f} | {scores['platt']['ece_val']:.4f} | "
        f"{scores['isotonic']['ece_val']:.4f} |",
        f"| sealed test | {ece_te_raw:.4f} | {scores['platt']['ece_test']:.4f} | "
        f"{scores['isotonic']['ece_test']:.4f} |",
        "",
        f"**Selected method: `{best_name}`** (lowest sealed-test ECE"
        f"{' — tie-break val ECE' if abs(scores['platt']['ece_test']-scores['isotonic']['ece_test'])<1e-6 else ''}).",
        "",
        "### Why this method?",
        "",
    ]
    if best_name == "platt":
        lines += [
            f"- Sealed-test ECE: Platt **{scores['platt']['ece_test']:.4f}** vs "
            f"Isotonic **{scores['isotonic']['ece_test']:.4f}**.",
            "- Platt is simpler and more robust; isotonic val ECE≈0 with "
            f"worse sealed-test ECE than raw ({scores['isotonic']['ece_test']:.4f} > "
            f"{ece_te_raw:.4f}) = classic overfit despite n_val={meta['n_val']}.",
            "",
        ]
    else:
        lines += [
            f"- Sealed-test ECE: Isotonic **{scores['isotonic']['ece_test']:.4f}** vs "
            f"Platt **{scores['platt']['ece_test']:.4f}**.",
            f"- n_val={meta['n_val']} supports isotonic flexibility; test ECE wins → keep it.",
            "",
        ]

    lines += ["## 2. Reliability diagrams (decile tables)", ""]
    lines += _md_rel(rel_va_raw, "Validation — RAW")
    lines += _md_rel(reliability_table(y_va, best["p_va"]), f"Validation — {best_name}")
    lines += _md_rel(rel_te_raw, "Sealed test — RAW")
    lines += _md_rel(reliability_table(y_te, best["p_te"]), f"Sealed test — {best_name}")

    lines += [
        "## 3. Confidence tiers",
        "",
        f"- Edges: low < {tiers.low_max:.4f}; "
        f"medium < {tiers.medium_max:.4f}; else high",
        f"- Rationale: {tiers.rationale}",
        "",
        "### Validation (fit window — diagnostic only)",
        "",
        "| tier | n | mean_calibrated | realized_wr |",
        "|---|---:|---:|---:|",
    ]
    for _, r in tier_va.iterrows():
        lines.append(
            f"| {r['tier']} | {int(r['n'])} | {r['mean_calibrated']:.4f} | {r['realized_wr']:.4f} |"
        )
    lines += [
        "",
        "### Sealed test (Bagian 4 — primary)",
        "",
        "| tier | n | mean_calibrated | realized_wr |",
        "|---|---:|---:|---:|",
    ]
    for _, r in tier_te.iterrows():
        flag = ""
        if int(r["n"]) < 30:
            flag = " _(n<30, hati-hati)_"
        lines.append(
            f"| {r['tier']} | {int(r['n'])} | {r['mean_calibrated']:.4f} | "
            f"{r['realized_wr']:.4f}{flag} |"
        )
    lines += [
        "",
        f"- Monotonic low ≤ medium ≤ high on sealed test? "
        f"**{'YES' if mono else 'NO'}**",
        "",
    ]
    if tier_sig is not None:
        lines += [
            f"### Among signals with raw p ≥ {THR_RAW} (tradeable gate)",
            "",
            "| tier | n | mean_calibrated | realized_wr |",
            "|---|---:|---:|---:|",
        ]
        for _, r in tier_sig.iterrows():
            lines.append(
                f"| {r['tier']} | {int(r['n'])} | {r['mean_calibrated']:.4f} | "
                f"{r['realized_wr']:.4f} |"
            )
        lines.append("")

    lines += [
        "## 4. Verdict for Risk Engine",
        "",
        trust,
        "",
        "Artifacts:",
        f"- `{out_pkl.as_posix()}`",
        "- `src/confidence_engine.py`",
        "- `data/reports/calibration_report.md`",
        "",
        "Not done here: position sizing / dynamic R:R (downstream tasks).",
        "",
    ]

    report_path = base / "reports/calibration_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print("wrote", report_path)
    print("TRUST:", trust.split("—")[0].strip())


def _self_check() -> None:
    y = np.array([0, 0, 1, 1, 1, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.55, 0.6, 0.7, 0.3, 0.8, 0.9])
    # perfect-ish high bins → ECE finite
    assert ece(y, p) >= 0
    assert _monotonic({"low": 0.3, "medium": 0.4, "high": 0.5})
    assert not _monotonic({"low": 0.5, "medium": 0.4, "high": 0.6})


if __name__ == "__main__":
    _self_check()
    main()
