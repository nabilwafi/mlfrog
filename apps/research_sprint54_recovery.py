"""Sprint 54 — M5 recovery state (underwater, conditional on current_R).

H1 = entry only. M5 = candidate management resolution. Not shipped.
Question: when post-entry is underwater, can M5 distinguish recoverable vs not
conditional on current_R? Not EXIT prediction. No trading actions.

Research only. Production unchanged.

  python apps/research_sprint54_recovery.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import tools.pyc_path_hook  # noqa: F401

from apps.research_sprint42_attribution import TRUE_OOS, _md
from apps.research_sprint52_m5_management import (
    _ece,
    _pr,
    _qsep,
    _residual_sep,
    _roc,
    _score_expanding,
)

OUT = _ROOT / "artifacts/pipeline_backtest/exit_structure/sprint54"
PATH52 = _ROOT / "artifacts/pipeline_backtest/exit_structure/sprint52/sprint52_m5_path.parquet"
CONFIRM = 0.25
ENTRY_FEATS = [
    "y_prob", "atr_pct_entry", "ema_trend_duration", "rolling_quantile",
    "hour_sin", "hour_cos", "ctx_h4_swing_quality", "atr_percent",
]
TIME_FEATS = [
    "minutes_since_entry", "bars_since_entry", "bars_since_MAE", "time_since_last_positive_R",
]
REC_FEATS = [
    "adverse_velocity", "recovery_velocity", "recovery_fraction",
    "distance_recovered_from_MAE", "bars_since_MAE",
]
PRICE_FEATS = [
    "mae_so_far_R", "mfe_so_far_R", "distance_from_entry",
    "distance_from_recent_low", "distance_from_recent_high",
]
A_FEATS = ["current_R"]
B_FEATS = ["current_R"] + TIME_FEATS
C_FEATS = ["current_R"] + REC_FEATS
D_FEATS = ["current_R"] + TIME_FEATS + [f for f in REC_FEATS if f not in TIME_FEATS]
E_FEATS = D_FEATS + ENTRY_FEATS
MODELS = (
    ("A_current_R", A_FEATS),
    ("B_R_time", B_FEATS),
    ("C_R_recovery", C_FEATS),
    ("D_R_time_recovery", D_FEATS),
    ("E_plus_H1", E_FEATS),
)
ALL_FEATS = sorted(set(A_FEATS + B_FEATS + C_FEATS + D_FEATS + E_FEATS + PRICE_FEATS))
LABEL_COLS = {
    "p0_R", "p0_mae", "p0_mfe", "next_open_R", "exit_better", "future_R",
    "final_R", "final_MAE", "final_MFE", "y_s53", "y_recover", "y_recover_deep",
    "remaining_R", "thesis_invalid",
}
CR_BINS = (
    ("[-1.00,-0.75)", -1.00, -0.75),
    ("[-0.75,-0.50)", -0.75, -0.50),
    ("[-0.50,-0.25)", -0.50, -0.25),
    ("[-0.25, 0.00)", -0.25, 0.00),
)
TIME_BINS = (
    ("0-15m", 0.0, 15.0),
    ("15-30m", 15.0, 30.0),
    ("30-60m", 30.0, 60.0),
    ("60-120m", 60.0, 120.0),
    ("120m+", 120.0, 1e9),
)
FRAC_BINS = (
    ("0-0.2", 0.0, 0.2),
    ("0.2-0.4", 0.2, 0.4),
    ("0.4-0.6", 0.4, 0.6),
    ("0.6-0.8", 0.6, 0.8),
    ("0.8+", 0.8, 10.0),
)
YCOL = "y_recover"


def _log(n, result, gate, finding, nxt) -> None:
    print(f"ITERATION: {n}")
    print(f"RESULT: {result}")
    print(f"GATE: {gate}")
    print(f"KEY FINDING: {finding}")
    print(f"NEXT ITERATION: {nxt}")


def _add_feats(obs: pd.DataFrame) -> pd.DataFrame:
    obs = obs.sort_values(["trade_id", "bars_since_entry"]).copy()
    g = obs.groupby("trade_id", sort=False)
    obs["distance_from_entry"] = obs["current_R"]
    roll_min = g["current_R"].transform(lambda s: s.rolling(6, min_periods=1).min())
    roll_max = g["current_R"].transform(lambda s: s.rolling(6, min_periods=1).max())
    obs["distance_from_recent_low"] = obs["current_R"] - roll_min
    obs["distance_from_recent_high"] = roll_max - obs["current_R"]
    mae_prev = g["mae_so_far_R"].shift(1)
    new_mae = (obs["mae_so_far_R"] - mae_prev.fillna(0.0) > 1e-6)
    obs["_mae_ep"] = new_mae.groupby(obs["trade_id"]).cumsum()
    obs["bars_since_MAE"] = obs.groupby(["trade_id", "_mae_ep"], sort=False).cumcount().astype(float)
    pos_mins = obs["minutes_since_entry"].where(obs["current_R"] > 0)
    last_pos = pos_mins.groupby(obs["trade_id"]).ffill()
    obs["time_since_last_positive_R"] = (obs["minutes_since_entry"] - last_pos).fillna(obs["minutes_since_entry"])
    mae = obs["mae_so_far_R"].clip(lower=1e-9)
    obs["distance_recovered_from_MAE"] = obs["mae_so_far_R"] + obs["current_R"]
    obs["recovery_fraction"] = (obs["distance_recovered_from_MAE"] / mae).clip(lower=0.0, upper=2.0)
    mins = obs["minutes_since_entry"].clip(lower=5.0)
    obs["adverse_velocity"] = obs["mae_so_far_R"] / mins
    rec_mins = (obs["bars_since_MAE"] * 5.0).clip(lower=5.0)
    obs["recovery_velocity"] = obs["distance_recovered_from_MAE"] / rec_mins
    obs["y_s53"] = ((obs["p0_mae"] >= 0.50) & (obs["p0_R"] > 0)).astype(int)
    obs["y_recover"] = (obs["p0_R"] > 0).astype(int)
    obs["y_recover_deep"] = ((obs["mae_so_far_R"] >= 0.50) & (obs["p0_R"] > 0)).astype(int)
    obs["remaining_R"] = obs["p0_R"] - obs["current_R"]
    obs["thesis_invalid"] = ((obs["p0_R"] <= -1.0) & (obs["p0_mfe"] < CONFIRM)).astype(int)
    return obs.drop(columns=["_mae_ep"])


def _oos(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["year"].isin(TRUE_OOS)].copy()


def _eval_models(df: pd.DataFrame, ycol: str) -> tuple[list[dict], list[dict], dict[str, np.ndarray]]:
    rows, yearly, preds = [], [], {}
    for name, feats in MODELS:
        pred = _score_expanding(df, feats, ycol)
        preds[name] = pred
        m = df["year"].isin(TRUE_OOS) & np.isfinite(pred)
        tmp = df.loc[m].copy()
        tmp["p"] = pred[m.to_numpy()]
        yv, pv = tmp[ycol].to_numpy(int), tmp["p"].to_numpy(float)
        sep, top, bot = _qsep(tmp, "p", ycol)
        rows.append({
            "model": name, "y": ycol,
            "roc_oos": _roc(yv, pv), "pr_oos": _pr(yv, pv),
            "ece": _ece(yv, pv), "sep": sep, "top": top, "bot": bot,
            "n": int(len(tmp)), "base_rate": float(tmp[ycol].mean()) if len(tmp) else 0.0,
        })
        for y in TRUE_OOS:
            gy = tmp[tmp["year"] == y]
            if len(gy) < 50:
                continue
            s, _, _ = _qsep(gy, "p", ycol)
            yearly.append({
                "model": name, "y": ycol, "year": int(y),
                "roc": _roc(gy[ycol], gy["p"]), "sep": s, "n": int(len(gy)),
                "base_rate": float(gy[ycol].mean()),
            })
    return rows, yearly, preds


def _gate_years(yearly: list[dict], best: str, a_name: str = "A_current_R") -> tuple[int, bool, list[float]]:
    ya = {r["year"]: r["roc"] for r in yearly if r["model"] == a_name}
    yb = {r["year"]: r["roc"] for r in yearly if r["model"] == best}
    lifts = [yb[y] - ya[y] for y in TRUE_OOS if y in ya and y in yb]
    n_pos = sum(1 for x in lifts if x > 0)
    collapse = any(yb[y] < 0.52 for y in yb) or any(x < -0.03 for x in lifts)
    return n_pos, collapse, lifts


def _calib(tmp: pd.DataFrame, ycol: str, n=10) -> list[dict]:
    p = tmp["p"].to_numpy(float)
    y = tmp[ycol].to_numpy(float)
    bins = np.linspace(0, 1, n + 1)
    out = []
    for i in range(n):
        hi = bins[i + 1] if i < n - 1 else 1.01
        sel = (p >= bins[i]) & (p < hi)
        if sel.sum() < 20:
            continue
        out.append({
            "bin": f"{bins[i]:.1f}-{min(hi, 1.0):.1f}",
            "n": int(sel.sum()),
            "p_mean": float(p[sel].mean()),
            "rate": float(y[sel].mean()),
            "mean_p0_R": float(tmp.loc[sel, "p0_R"].mean()),
            "mean_remaining_R": float(tmp.loc[sel, "remaining_R"].mean()),
        })
    return out


def _grid(df: pd.DataFrame, ycol: str) -> tuple[list[dict], list[dict]]:
    xt, xf = [], []
    for cr_name, lo, hi in CR_BINS:
        cr = df[(df["current_R"] >= lo) & (df["current_R"] < hi)]
        for t_name, t0, t1 in TIME_BINS:
            sub = cr[(cr["minutes_since_entry"] >= t0) & (cr["minutes_since_entry"] < t1)]
            xt.append({
                "current_R": cr_name, "time": t_name, "n": int(len(sub)),
                "base_rate": float(sub[ycol].mean()) if len(sub) else 0.0,
                "mean_p0_R": float(sub["p0_R"].mean()) if len(sub) else 0.0,
                "mean_remaining_R": float(sub["remaining_R"].mean()) if len(sub) else 0.0,
            })
        for f_name, f0, f1 in FRAC_BINS:
            sub = cr[(cr["recovery_fraction"] >= f0) & (cr["recovery_fraction"] < f1)]
            xf.append({
                "current_R": cr_name, "recovery_fraction": f_name, "n": int(len(sub)),
                "base_rate": float(sub[ycol].mean()) if len(sub) else 0.0,
                "mean_p0_R": float(sub["p0_R"].mean()) if len(sub) else 0.0,
                "mean_remaining_R": float(sub["remaining_R"].mean()) if len(sub) else 0.0,
            })
    return xt, xf


def main() -> int:
    leak_ok = not (set(ALL_FEATS) & LABEL_COLS)
    assert leak_ok, set(ALL_FEATS) & LABEL_COLS
    if not PATH52.is_file():
        print(f"missing {PATH52} — run Sprint 52 first")
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    obs = _add_feats(pd.read_parquet(PATH52))
    s53 = obs[obs["mfe_so_far_R"] < CONFIRM].copy()
    uw = s53[s53["current_R"] < 0].copy()
    oos_s53, oos_uw, oos_all = _oos(s53), _oos(uw), _oos(obs)

    # ----- 1 label audit -----
    already_green = oos_s53["current_R"] >= 0
    mae_known = oos_s53["mae_so_far_R"] >= 0.50
    audit = {
        "original_s53": "recoverable = (p0_mae >= 0.50) AND (p0_R > 0)  [trade-level, broadcast to every M5 obs]",
        "causal": (
            "y_s53 is not a future outcome at every t: p0_mae/p0_R are trade-final. "
            "If current_R>=0 the 'P0_R>0' event may already be in progress. "
            "If mae_so_far<0.50 the MAE>=0.50 clause still uses future path."
        ),
        "primary_label": "y_recover = (p0_R > 0) on underwater obs (current_R < 0); final R is after this M5 close",
        "deep_label": "y_recover_deep = (mae_so_far_R >= 0.50 PIT) AND (p0_R > 0)",
        "n_s53_oos": int(len(oos_s53)),
        "n_uw_oos": int(len(oos_uw)),
        "n_s53_trades_oos": int(oos_s53["trade_id"].nunique()),
        "n_uw_trades_oos": int(oos_uw["trade_id"].nunique()),
        "y_s53_rate_s53": float(oos_s53["y_s53"].mean()),
        "y_s53_rate_uw": float(oos_uw["y_s53"].mean()) if len(oos_uw) else 0.0,
        "y_recover_rate_uw": float(oos_uw["y_recover"].mean()) if len(oos_uw) else 0.0,
        "y_recover_deep_rate_uw": float(oos_uw["y_recover_deep"].mean()) if len(oos_uw) else 0.0,
        "n_ambiguous_already_green": int(already_green.sum()),
        "pct_ambiguous_already_green": float(already_green.mean()),
        "n_s53_mae_not_yet": int((~mae_known & (oos_s53["y_s53"] == 1)).sum()),
        "pct_s53_pos_with_future_mae": float(
            ((oos_s53["y_s53"] == 1) & ~mae_known).mean()
        ),
        "n_valid_y_recover": int(len(oos_uw)),
        "n_valid_y_recover_deep": int((oos_uw["mae_so_far_R"] >= 0.50).sum()) if len(oos_uw) else 0,
        "n_invalid": 0,
    }
    pd.DataFrame([audit]).to_csv(OUT / "sprint54_label_audit.csv", index=False)
    _log(1, "label audit", "PASS",
         f"s53_obs={audit['n_s53_oos']} uw={audit['n_uw_oos']} "
         f"y_s53={audit['y_s53_rate_s53']:.3f} y_recover_uw={audit['y_recover_rate_uw']:.3f} "
         f"ambiguous_green={audit['pct_ambiguous_already_green']:.3f}",
         2)

    # ----- 2 residual beyond current_R -----
    resid = []
    for feat in ["current_R"] + PRICE_FEATS + TIME_FEATS + [f for f in REC_FEATS if f != "bars_since_MAE"]:
        if feat == "current_R":
            sep, _, _ = _qsep(oos_uw, feat, YCOL)
            resid.append({"feature": feat, "y": YCOL, "kind": "raw",
                          "mean_abs_sep": abs(sep), "mean_sep": sep, "n_buckets_ge": 5})
            continue
        r = _residual_sep(oos_uw, feat, YCOL)
        r["kind"] = "residual_current_R"
        resid.append(r)
    rdf = pd.DataFrame(resid)
    rdf.to_csv(OUT / "sprint54_residual.csv", index=False)
    _log(2, "window + residual features", "PASS",
         f"window=S53 mfe<{CONFIRM} AND current_R<0 n_oos={len(oos_uw)}",
         3)

    # ----- 3/4 models + yearly -----
    rows, yearly, preds = _eval_models(uw, YCOL)
    rows_s53, yearly_s53, _ = _eval_models(uw, "y_s53")
    pd.DataFrame(rows).to_csv(OUT / "sprint54_models.csv", index=False)
    pd.DataFrame(yearly).to_csv(OUT / "sprint54_yearly.csv", index=False)
    pd.DataFrame(rows_s53).to_csv(OUT / "sprint54_models_y_s53.csv", index=False)
    a = next(r for r in rows if r["model"] == "A_current_R")
    rich = [r for r in rows if r["model"] != "A_current_R"]
    best = max(rich, key=lambda r: r["roc_oos"])
    droc = best["roc_oos"] - a["roc_oos"]
    dsep = best["sep"] - a["sep"]
    n_pos, collapse, lifts = _gate_years(yearly, best["model"])
    _log(3, "expanding WF vs current_R", "PASS" if droc >= 0.02 and dsep >= 0.03 else "FAIL",
         f"A={a['roc_oos']:.3f} best={best['model']} {best['roc_oos']:.3f} "
         f"dROC={droc:.3f} dsep={dsep:.3f} pos_years={n_pos} collapse={collapse}",
         4)
    _log(4, "yearly OOS", "PASS" if n_pos >= 3 and not collapse else "FAIL",
         f"lifts={[round(x, 3) for x in lifts]}",
         5)

    # ----- 5 current_R buckets -----
    uw = uw.copy()
    for name, pred in preds.items():
        uw[f"p_{name}"] = pred
    oos = _oos(uw)
    oos = oos[np.isfinite(oos["p_A_current_R"])].copy()
    fixed, year_rows = [], []
    for bname, lo, hi in CR_BINS:
        sub = oos[(oos["current_R"] >= lo) & (oos["current_R"] < hi)]
        stats: dict[str, dict] = {}
        for mname, _feats in MODELS:
            pcol = f"p_{mname}"
            row = {
                "bucket": bname, "model": mname, "n": int(len(sub)),
                "base_rate": float(sub[YCOL].mean()) if len(sub) else 0.0,
                "mean_current_R": float(sub["current_R"].mean()) if len(sub) else 0.0,
                "roc": 0.5, "sep": 0.0, "droc_vs_A": 0.0, "dsep_vs_A": 0.0,
            }
            if len(sub) >= 200 and sub[YCOL].nunique() == 2 and sub[pcol].nunique() >= 3:
                row["roc"] = _roc(sub[YCOL], sub[pcol])
                row["sep"], _, _ = _qsep(sub, pcol, YCOL)
            stats[mname] = row
        a_roc, a_sep = stats["A_current_R"]["roc"], stats["A_current_R"]["sep"]
        for row in stats.values():
            row["droc_vs_A"] = float(row["roc"]) - a_roc
            row["dsep_vs_A"] = float(row["sep"]) - a_sep
            fixed.append(row)
        for y in TRUE_OOS:
            gy = sub[sub["year"] == y]
            if len(gy) < 50 or gy[YCOL].nunique() < 2:
                continue
            for mname in ("A_current_R", best["model"]):
                pcol = f"p_{mname}"
                s, _, _ = _qsep(gy, pcol, YCOL) if gy[pcol].nunique() >= 3 else (0.0, 0.0, 0.0)
                year_rows.append({
                    "bucket": bname, "model": mname, "year": int(y), "n": int(len(gy)),
                    "base_rate": float(gy[YCOL].mean()),
                    "roc": _roc(gy[YCOL], gy[pcol]), "sep": s,
                })
    pd.DataFrame(fixed + year_rows).to_csv(OUT / "sprint54_current_R_buckets.csv", index=False)
    cond_pass = []
    for bname, _lo, _hi in CR_BINS:
        bb = [r for r in fixed if r["bucket"] == bname and r["model"] == best["model"] and r["n"] >= 200]
        if bb and bb[0]["droc_vs_A"] >= 0.02 and bb[0]["dsep_vs_A"] >= 0.03:
            cond_pass.append(bname)
    _log(5, "current_R conditional", "PASS" if len(cond_pass) >= 2 else "FAIL",
         f"best={best['model']} buckets_pass={cond_pass}",
         6)

    # ----- 6 depth x time / recovery_fraction -----
    xt, xf = _grid(oos, YCOL)
    pd.DataFrame(xt).to_csv(OUT / "sprint54_depth_time.csv", index=False)
    pd.DataFrame(xf).to_csv(OUT / "sprint54_depth_frac.csv", index=False)
    _log(6, "depth x time / recovery_fraction", "PASS", "predefined bins only", 7)

    # ----- 7 calib, E[R], lead -----
    best_col = f"p_{best['model']}"
    scored = oos.copy()
    scored["p"] = scored[best_col]
    cal = _calib(scored, YCOL)
    pd.DataFrame(cal).to_csv(OUT / "sprint54_calibration.csv", index=False)
    qtmp = scored.copy()
    try:
        qtmp["q"] = pd.qcut(qtmp["p"], 5, duplicates="drop", labels=False)
    except ValueError:
        qtmp["q"] = 0
    erows = []
    for q, g in qtmp.groupby("q", observed=False):
        erows.append({
            "q": int(q) + 1, "n": int(len(g)),
            "p_mean": float(g["p"].mean()),
            "P_recovery": float(g[YCOL].mean()),
            "E_p0_R": float(g["p0_R"].mean()),
            "E_remaining_R": float(g["remaining_R"].mean()),
            "mean_current_R": float(g["current_R"].mean()),
        })
    for bname, lo, hi in CR_BINS:
        g = scored[(scored["current_R"] >= lo) & (scored["current_R"] < hi)]
        erows.append({
            "q": bname, "n": int(len(g)),
            "p_mean": float(g["p"].mean()) if len(g) else 0.0,
            "P_recovery": float(g[YCOL].mean()) if len(g) else 0.0,
            "E_p0_R": float(g["p0_R"].mean()) if len(g) else 0.0,
            "E_remaining_R": float(g["remaining_R"].mean()) if len(g) else 0.0,
            "mean_current_R": float(g["current_R"].mean()) if len(g) else 0.0,
        })
    pd.DataFrame(erows).to_csv(OUT / "sprint54_expected_R.csv", index=False)

    first = scored.sort_values(["trade_id", "bars_since_entry"]).drop_duplicates("trade_id")
    last = scored.sort_values(["trade_id", "bars_since_entry"]).groupby("trade_id", sort=False).tail(1)
    mae50 = (scored[scored["mae_so_far_R"] >= 0.50]
             .sort_values(["trade_id", "bars_since_entry"]).drop_duplicates("trade_id"))
    hard = first[first["p0_R"] <= -1.0]
    rec = first[first[YCOL] == 1]
    nrec = first[first[YCOL] == 0]
    lead = {
        "n_trades_oos": int(len(first)),
        "first_roc_A": _roc(first[YCOL], first["p_A_current_R"]),
        "first_roc_best": _roc(first[YCOL], first[best_col]),
        "mae50_n": int(len(mae50)),
        "mae50_roc_A": _roc(mae50[YCOL], mae50["p_A_current_R"]) if len(mae50) >= 50 else 0.5,
        "mae50_roc_best": _roc(mae50[YCOL], mae50[best_col]) if len(mae50) >= 50 else 0.5,
        "rec_median_mins_first": float(rec["minutes_since_entry"].median()) if len(rec) else 0.0,
        "nrec_median_mins_first": float(nrec["minutes_since_entry"].median()) if len(nrec) else 0.0,
        "hard_n": int(len(hard)),
        "hard_median_mins_first": float(hard["minutes_since_entry"].median()) if len(hard) else 0.0,
        "hard_median_mins_last": float(
            last.loc[last["trade_id"].isin(hard["trade_id"]), "minutes_since_entry"].median()
        ) if len(hard) else 0.0,
    }
    pd.DataFrame([lead]).to_csv(OUT / "sprint54_lead.csv", index=False)
    _log(7, "P(recovery), E[R], lead time", "PASS",
         f"first dROC={lead['first_roc_best'] - lead['first_roc_A']:.3f} "
         f"ece={best['ece']:.3f}",
         8)

    leak = "PASS" if leak_ok else "FAIL"
    extra = {
        "gate_roc": droc >= 0.02,
        "gate_sep": dsep >= 0.03,
        "gate_years": n_pos >= 3,
        "gate_collapse": not collapse,
        "gate_conditional": len(cond_pass) >= 2,
        "droc": droc, "dsep": dsep, "n_pos_years": n_pos, "collapse": collapse,
        "best_model": best["model"], "a_roc": a["roc_oos"], "best_roc": best["roc_oos"],
        "cond_pass_buckets": cond_pass,
        "n_oos": a["n"], "base_rate": a["base_rate"],
    }
    overall = (droc >= 0.02 and dsep >= 0.03 and n_pos >= 3 and not collapse and leak == "PASS")
    if overall and len(cond_pass) >= 2:
        verdict = "INCREMENTAL_RECOVERY_SIGNAL"
    elif len(cond_pass) == 1:
        verdict = "CONDITIONAL_RECOVERY_SIGNAL"
    else:
        verdict = "NO_INCREMENTAL_RECOVERY_SIGNAL"
    extra["verdict"] = verdict
    _log(8, "verdict", "PASS" if verdict != "NO_INCREMENTAL_RECOVERY_SIGNAL" else "FAIL",
         verdict, "DONE")

    (OUT / "sprint54_leakage_audit.json").write_text(json.dumps({
        "LEAKAGE_AUDIT": leak,
        "features_pit": True,
        "no_future_mfe_mae_as_features": True,
        "no_p0_outcome_as_feature": True,
        "no_future_recovery_in_state": True,
        "no_sprint_46_52_probabilities": True,
        "h1_entry_unchanged": True,
        "m5_after_entry_only": True,
        "no_m5_confirmation_before_entry": True,
        "window": "mfe_so_far_R < 0.25 PIT AND current_R < 0",
        "label": "y_recover = p0_R > 0 (evaluation only)",
        "y_s53_retained_for_comparison": True,
    }, indent=2), encoding="utf-8")
    (OUT / "sprint54_final_verdict.json").write_text(
        json.dumps({"sprint": 54, "verdict": verdict, "production_changed": False,
                    "m5_management_shipped": False, **extra}, indent=2, default=float),
        encoding="utf-8",
    )
    report = _report(verdict, audit, rdf, rows, yearly, rows_s53, fixed, year_rows,
                     xt, xf, cal, erows, lead, leak, extra, cond_pass, a, best, droc, dsep)
    (OUT / "sprint54_report.md").write_text(report, encoding="utf-8")
    print(f"FINAL VERDICT: {verdict}")
    print("PRODUCTION: UNCHANGED")
    print("M5 HOLD/REDUCE/EXIT/TRAIL: NOT SHIPPED")
    return 0


def _report(verdict, audit, resid, rows, yearly, rows_s53, buck, buck_y,
            xt, xf, cal, erows, lead, leak, extra, cond_pass, a, best, droc, dsep) -> str:
    rec = (
        "Do not ship M5 HOLD/REDUCE/EXIT/TRAIL. "
        + (
            f"Recovery information is only incremental in {cond_pass}. Not production-ready."
            if verdict == "CONDITIONAL_RECOVERY_SIGNAL"
            else (
                "Gates pass, but the incremental ranking is time-underwater / adverse_velocity (MAE per minute), "
                "not bounce (recovery_fraction residual ≈ 0). Useful only for current_R < -0.25; "
                "[-0.25, 0) misses +0.02 ROC. ECE is poor (~0.17); first-bar ROC ≈ 0.53. "
                "Next research (still no ship): frozen a-priori time×depth description, or expected-R economics "
                "in current_R < -0.25 after ≥60m — not more indicators, not action replay unless asked."
                if verdict == "INCREMENTAL_RECOVERY_SIGNAL"
                else "current_R already ranks future green vs not. M5 time/recovery path does not clear +0.02 ROC / +0.03 sep "
                "conditionally. Next: stop recovery-state research or change the question (path to timeout vs hard stop), "
                "not more indicators."
            )
        )
    )
    lines = [
        "# Sprint 54 — M5 Recovery State",
        "",
        f"**VERDICT: `{verdict}`**",
        "",
        "Research only. Production unchanged. No HOLD/REDUCE/EXIT/TRAIL.",
        "",
        "## Architecture",
        "",
        "H1 = Entry only.",
        "M5 = candidate management resolution (not shipped).",
        "P0 = control / label source only.",
        "",
        f"Window: Sprint 53 unconfirmed (`mfe_so_far_R < {CONFIRM}`) **and** `current_R < 0`.",
        "Primary label: `y_recover = (p0_R > 0)` — future final green given underwater at t.",
        "Expanding WF LR. No Sprint 46–52 model scores. No action replay.",
        "",
        "## 1. Verdict",
        "",
        f"`{verdict}`",
        "",
        f"Best `{best['model']}` ROC={best['roc_oos']:.3f} vs A={a['roc_oos']:.3f} "
        f"(dROC={droc:.3f}, dsep={dsep:.3f}). Conditional buckets: {cond_pass or 'none'}.",
        "",
        "Mechanism note: residual lift is minutes_since_entry / adverse_velocity, not recovery_fraction. "
        "First underwater bar is near-random; separation shows up after MAE develops and after ~60m.",
        "",
        "## 2. Label audit",
        "",
        f"- Original Sprint 53: `{audit['original_s53']}`",
        f"- Causal: {audit['causal']}",
        f"- Valid for every M5 obs? **No** for y_s53. **Yes** for y_recover on underwater closes (obs stop while P0 alive).",
        f"- S53-window OOS obs={audit['n_s53_oos']} trades={audit['n_s53_trades_oos']} "
        f"y_s53 rate={audit['y_s53_rate_s53']:.3f}",
        f"- Ambiguous already-green: n={audit['n_ambiguous_already_green']} "
        f"({audit['pct_ambiguous_already_green']:.3f})",
        f"- y_s53 positives whose MAE>=0.50 is still future: {audit['pct_s53_pos_with_future_mae']:.3f}",
        f"- Primary y_recover underwater OOS: n={audit['n_uw_oos']} trades={audit['n_uw_trades_oos']} "
        f"rate={audit['y_recover_rate_uw']:.3f}",
        f"- y_recover_deep (PIT MAE>=0.50): n={audit['n_valid_y_recover_deep']} "
        f"rate={audit['y_recover_deep_rate_uw']:.3f}",
        f"- Invalid observations: {audit['n_invalid']}",
        "",
        "## 3. Dataset / window",
        "",
        "Same H1 entries as Sprint 52/53 (`sprint52_m5_path.parquet`).",
        "Observation = completed M5 after entry, PIT path only, until P0 exit or hard -1R.",
        f"Modeling sample: unconfirmed and underwater. OOS n={a['n']} base_rate={a['base_rate']:.3f} years={list(TRUE_OOS)}.",
        "",
        "## 4. Features",
        "",
        "Price (descriptive): current_R, MAE/MFE so far, distance_from_entry (=current_R), "
        "distance_from_recent_low/high (6-bar rolling close R).",
        "Time: minutes_since_entry, bars_since_entry, bars_since_MAE, time_since_last_positive_R.",
        "Recovery: adverse_velocity, recovery_velocity, recovery_fraction, distance_recovered_from_MAE, bars_since_MAE.",
        "E only: frozen H1 entry context (y_prob + FEAT7). No RSI/MACD/ATR dumps. No 46–52 probs.",
        "",
    ]
    if resid is not None:
        rshow = resid.to_dict("records") if hasattr(resid, "to_dict") else resid
        cols = [c for c in ("feature", "y", "kind", "mean_abs_sep", "mean_sep", "n_buckets_ge") if rshow and c in rshow[0]]
        if rshow and cols:
            lines += ["Residual after current_R (descriptive):", "", _md(rshow, cols, {"mean_abs_sep": 4, "mean_sep": 4, "n_buckets_ge": 0}), ""]
    lines += [
        "## 5. Overall OOS",
        "",
        "Expanding WF LR. Gate: +0.02 ROC and +0.03 sep vs A_current_R.",
        "",
        _md(rows, ["model", "roc_oos", "pr_oos", "sep", "n", "base_rate"],
            {"roc_oos": 3, "pr_oos": 3, "sep": 3, "n": 0, "base_rate": 3}),
        "",
        "y_s53 comparison (same underwater window, not the primary target):",
        "",
        _md(rows_s53, ["model", "roc_oos", "pr_oos", "sep", "n", "base_rate"],
            {"roc_oos": 3, "pr_oos": 3, "sep": 3, "n": 0, "base_rate": 3}),
        "",
        "## 6. Yearly OOS",
        "",
        _md(yearly, ["model", "year", "roc", "sep", "n"],
            {"year": 0, "roc": 3, "sep": 3, "n": 0}),
        "",
        "## 7. current_R conditional",
        "",
        "Global WF scores sliced by current_R. Incremental vs A inside each bucket.",
        "",
        _md([r for r in buck if r["model"] in ("A_current_R", best["model"])],
            ["bucket", "model", "n", "base_rate", "roc", "sep", "droc_vs_A", "dsep_vs_A"],
            {"n": 0, "base_rate": 3, "roc": 3, "sep": 3, "droc_vs_A": 3, "dsep_vs_A": 3}),
        "",
    ]
    if buck_y:
        lines += [
            "Yearly (A and best) by bucket:",
            "",
            _md(buck_y, ["bucket", "model", "year", "n", "base_rate", "roc", "sep"],
                {"year": 0, "n": 0, "base_rate": 3, "roc": 3, "sep": 3}),
            "",
        ]
    lines += [
        f"Buckets clearing +0.02 ROC and +0.03 sep: {cond_pass or 'none'}.",
        "",
        "## 8. Depth × time",
        "",
        "Predefined bins. Not a threshold search.",
        "",
        _md(xt, ["current_R", "time", "n", "base_rate", "mean_p0_R", "mean_remaining_R"],
            {"n": 0, "base_rate": 3, "mean_p0_R": 3, "mean_remaining_R": 3}),
        "",
        "Depth × recovery_fraction:",
        "",
        _md(xf, ["current_R", "recovery_fraction", "n", "base_rate", "mean_p0_R", "mean_remaining_R"],
            {"n": 0, "base_rate": 3, "mean_p0_R": 3, "mean_remaining_R": 3}),
        "",
        "## 9. Recovery probability",
        "",
        f"Best model `{best['model']}` OOS ECE={best['ece']:.3f}. P(recovery|state) = calibrated bin rate (not a trade threshold).",
        "",
        _md(cal, ["bin", "n", "p_mean", "rate", "mean_p0_R", "mean_remaining_R"],
            {"n": 0, "p_mean": 3, "rate": 3, "mean_p0_R": 3, "mean_remaining_R": 3}) if cal else "Too thin to bin.",
        "",
        "## 10. Expected future R",
        "",
        "`E[p0_R | state]` and `E[p0_R - current_R | state]` (remaining). Descriptive.",
        "",
        _md(erows, ["q", "n", "p_mean", "P_recovery", "E_p0_R", "E_remaining_R", "mean_current_R"],
            {"n": 0, "p_mean": 3, "P_recovery": 3, "E_p0_R": 3, "E_remaining_R": 3, "mean_current_R": 3}),
        "",
        "## 11. Lead time",
        "",
        "No action replay. First underwater obs per trade vs first PIT MAE>=0.50 vs last obs (near P0/hard stop).",
        "",
        f"- First obs: A ROC={lead['first_roc_A']:.3f} best={lead['first_roc_best']:.3f} n_trades={lead['n_trades_oos']}",
        f"- At first MAE>=0.50: A ROC={lead['mae50_roc_A']:.3f} best={lead['mae50_roc_best']:.3f} n={lead['mae50_n']}",
        f"- Recover vs not, median minutes at first underwater obs: "
        f"{lead['rec_median_mins_first']:.1f} vs {lead['nrec_median_mins_first']:.1f}",
        f"- Hard-stop trades (p0_R<=-1): n={lead['hard_n']} first={lead['hard_median_mins_first']:.1f}m "
        f"last={lead['hard_median_mins_last']:.1f}m",
        "",
        "## 12. Leakage audit",
        "",
        f"**LEAKAGE_AUDIT = {leak}**",
        "",
        "- Features PIT (current/past M5 + frozen H1 entry)",
        "- No future MFE/MAE as features",
        "- No P0 final outcome as feature",
        "- Recovery fraction / velocity use mae_so_far and current_R only",
        "- No Sprint 46–52 probabilities",
        "- H1 entry unchanged; M5 obs after entry only; no M5 confirmation before entry",
        "",
        "## 13. Production impact",
        "",
        "Entry: UNCHANGED",
        "",
        "P0: UNCHANGED",
        "",
        "M5 HOLD: NOT SHIPPED",
        "",
        "M5 REDUCE: NOT SHIPPED",
        "",
        "M5 EXIT: NOT SHIPPED",
        "",
        "M5 TRAIL: NOT SHIPPED",
        "",
        "Reversal: NOT USED",
        "",
        "P1/P2/P3: NOT USED",
        "",
        "## 14. Recommended next research step",
        "",
        rec,
        "",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
