"""Sprint 55 — M5 price-state vs time-state attribution.

H1 = entry only. M5 observes from the first completed bar after entry.
Question: is the Sprint 54 recovery lift PRICE / position state, or TIME / trade age?
Not EXIT prediction. No timer rule. No actions.

Research only. Production unchanged.

  python apps/research_sprint55_price_vs_time.py
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
from apps.research_sprint52_m5_management import _ece, _pr, _qsep, _residual_sep, _roc, _score_expanding
from apps.research_sprint54_recovery import (
    CONFIRM,
    CR_BINS,
    PATH52,
    TIME_BINS,
    YCOL,
    _add_feats,
    _calib,
    _grid,
    _oos,
)

OUT = _ROOT / "artifacts/pipeline_backtest/exit_structure/sprint55"

# B: price/position only. No clock age.
# adverse_velocity is MAE increment / bar (not MAE/minutes — that embeds time).
B_FEATS = [
    "current_R", "mae_so_far_R", "mfe_so_far_R", "distance_from_entry",
    "distance_from_recent_low", "distance_from_recent_high",
    "adverse_velocity", "recovery_velocity", "recovery_fraction",
    "distance_recovered_from_MAE", "bars_since_MAE",
]
C_FEATS = [
    "current_R", "minutes_since_entry", "bars_since_entry",
    "bars_since_MAE", "time_since_last_positive_R",
]
A_FEATS = ["current_R"]
D_FEATS = list(dict.fromkeys(B_FEATS + C_FEATS))
E_FEATS = [
    "adverse_velocity", "recovery_velocity", "recovery_fraction",
    "distance_recovered_from_MAE", "distance_from_recent_low", "distance_from_recent_high",
]
MODELS = (
    ("A_current_R", A_FEATS),
    ("B_price_state", B_FEATS),
    ("C_time_state", C_FEATS),
    ("D_full_state", D_FEATS),
    ("E_price_dynamics", E_FEATS),
)
CLOCK = {"minutes_since_entry", "bars_since_entry", "time_since_last_positive_R"}
LABEL_COLS = {
    "p0_R", "p0_mae", "p0_mfe", "next_open_R", "exit_better", "future_R",
    "final_R", "final_MAE", "final_MFE", "y_s53", "y_recover", "y_recover_deep",
    "remaining_R", "thesis_invalid",
}


def _log(n, result, gate, finding, nxt) -> None:
    print(f"ITERATION: {n}")
    print(f"RESULT: {result}")
    print(f"GATE: {gate}")
    print(f"KEY FINDING: {finding}")
    print(f"NEXT ITERATION: {nxt}")


def _detime_velocities(obs: pd.DataFrame) -> pd.DataFrame:
    """Replace Sprint 54 clock velocities with PIT bar increments (no minutes)."""
    obs = obs.sort_values(["trade_id", "bars_since_entry"]).copy()
    g = obs.groupby("trade_id", sort=False)
    dmae = (obs["mae_so_far_R"] - g["mae_so_far_R"].shift(1).fillna(0.0)).clip(lower=0.0)
    obs["adverse_velocity"] = dmae.groupby(obs["trade_id"]).transform(
        lambda s: s.rolling(6, min_periods=1).mean())
    obs["recovery_velocity"] = obs["distance_recovered_from_MAE"] / obs["bars_since_MAE"].clip(lower=1.0)
    return obs


def _eval(df: pd.DataFrame, models=MODELS) -> tuple[list[dict], list[dict], dict[str, np.ndarray]]:
    rows, yearly, preds = [], [], {}
    for name, feats in models:
        pred = _score_expanding(df, feats, YCOL)
        preds[name] = pred
        m = df["year"].isin(TRUE_OOS) & np.isfinite(pred)
        tmp = df.loc[m].copy()
        tmp["p"] = pred[m.to_numpy()]
        yv, pv = tmp[YCOL].to_numpy(int), tmp["p"].to_numpy(float)
        sep, top, bot = _qsep(tmp, "p", YCOL)
        rows.append({
            "model": name, "roc_oos": _roc(yv, pv), "pr_oos": _pr(yv, pv),
            "ece": _ece(yv, pv), "sep": sep, "top": top, "bot": bot,
            "n": int(len(tmp)), "base_rate": float(tmp[YCOL].mean()) if len(tmp) else 0.0,
        })
        for y in TRUE_OOS:
            gy = tmp[tmp["year"] == y]
            if len(gy) < 50:
                continue
            s, _, _ = _qsep(gy, "p", YCOL)
            yearly.append({
                "model": name, "year": int(y), "roc": _roc(gy[YCOL], gy["p"]),
                "sep": s, "n": int(len(gy)), "base_rate": float(gy[YCOL].mean()),
            })
    return rows, yearly, preds


def _lifts(yearly: list[dict], rich: str, base: str) -> tuple[int, bool, list[float]]:
    ya = {r["year"]: r["roc"] for r in yearly if r["model"] == base}
    yb = {r["year"]: r["roc"] for r in yearly if r["model"] == rich}
    lifts = [yb[y] - ya[y] for y in TRUE_OOS if y in ya and y in yb]
    n_pos = sum(1 for x in lifts if x > 0)
    collapse = any(yb[y] < 0.52 for y in yb) or any(x < -0.03 for x in lifts)
    return n_pos, collapse, lifts


def _inc(rich: dict, base: dict, yearly: list[dict]) -> dict:
    droc = rich["roc_oos"] - base["roc_oos"]
    dsep = rich["sep"] - base["sep"]
    n_pos, collapse, lifts = _lifts(yearly, rich["model"], base["model"])
    ok = droc >= 0.02 and dsep >= 0.03 and n_pos >= 3 and not collapse
    return {
        "pair": f"{base['model']} → {rich['model']}",
        "droc": droc, "dsep": dsep, "n_pos_years": n_pos, "collapse": collapse,
        "lifts": lifts, "gate": "PASS" if ok else "FAIL", "pass": ok,
    }


def _slice_metrics(df: pd.DataFrame, pcol: str) -> dict:
    if len(df) < 50 or df[YCOL].nunique() < 2 or df[pcol].nunique() < 3:
        br = float(df[YCOL].mean()) if len(df) else 0.0
        return {"n": int(len(df)), "base_rate": br, "roc": 0.5, "sep": 0.0}
    sep, _, _ = _qsep(df, pcol, YCOL)
    return {
        "n": int(len(df)), "base_rate": float(df[YCOL].mean()),
        "roc": _roc(df[YCOL], df[pcol]), "sep": sep,
    }


def _classify(inc: dict[str, dict]) -> str:
    ab, ac, bd, cd = inc["A→B"], inc["A→C"], inc["B→D"], inc["C→D"]
    b_ok, c_ok = ab["pass"], ac["pass"]
    if not b_ok and not c_ok:
        return "WEAK_OR_UNSTABLE"
    if b_ok and not c_ok:
        return "BOTH_INCREMENTAL" if bd["pass"] else "PRICE_DOMINANT"
    if c_ok and not b_ok:
        return "BOTH_INCREMENTAL" if cd["pass"] else "TIME_DOMINANT"
    # both B and C beat A
    if bd["pass"] and cd["pass"]:
        return "BOTH_INCREMENTAL"
    if cd["pass"] and not bd["pass"]:
        return "PRICE_DOMINANT"
    if bd["pass"] and not cd["pass"]:
        return "TIME_DOMINANT"
    return "PRICE_DOMINANT" if ab["droc"] >= ac["droc"] else "TIME_DOMINANT"


def main() -> int:
    all_feats = set(A_FEATS + B_FEATS + C_FEATS + D_FEATS + E_FEATS)
    leak_ok = not (all_feats & LABEL_COLS)
    assert leak_ok, all_feats & LABEL_COLS
    assert not (set(B_FEATS) & CLOCK), "B must not contain clock-age features"
    if not PATH52.is_file():
        print(f"missing {PATH52} — run Sprint 52 first")
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    obs = _detime_velocities(_add_feats(pd.read_parquet(PATH52)))
    uw = obs[(obs["mfe_so_far_R"] < CONFIRM) & (obs["current_R"] < 0)].copy()
    oos_all = _oos(uw)

    label = {
        "definition": "y_recover = (p0_R > 0) at observation t with current_R < 0; P0 final R is after this M5 close",
        "window": "Sprint 53 unconfirmed (mfe_so_far_R < 0.25 PIT) AND current_R < 0",
        "exclusions": "already-green bars (current_R >= 0); confirmed thesis (mfe_so_far >= 0.25); hard-stop/P0-dead bars",
        "n_oos": int(len(oos_all)),
        "n_trades_oos": int(oos_all["trade_id"].nunique()),
        "base_rate": float(oos_all[YCOL].mean()) if len(oos_all) else 0.0,
        "m5_starts": "first completed M5 after H1 entry (searchsorted right); no wait for H1 close",
    }
    pd.DataFrame([label]).to_csv(OUT / "sprint55_label.csv", index=False)
    _log(1, "label + window (Sprint 54)", "PASS",
         f"n={label['n_oos']} rate={label['base_rate']:.3f} starts=first_M5", 2)

    rows, yearly, preds = _eval(uw)
    by = {r["model"]: r for r in rows}
    a, b, c, d, e = (by[k] for k in (
        "A_current_R", "B_price_state", "C_time_state", "D_full_state", "E_price_dynamics"))
    for r in rows:
        r["droc_vs_A"] = r["roc_oos"] - a["roc_oos"]
        r["dsep_vs_A"] = r["sep"] - a["sep"]
    pd.DataFrame(rows).to_csv(OUT / "sprint55_models.csv", index=False)
    pd.DataFrame(yearly).to_csv(OUT / "sprint55_yearly.csv", index=False)

    inc = {
        "A→B": _inc(b, a, yearly),
        "A→C": _inc(c, a, yearly),
        "A→D": _inc(d, a, yearly),
        "B→D": _inc(d, b, yearly),
        "C→D": _inc(d, c, yearly),
        "A→E": _inc(e, a, yearly),
    }
    pd.DataFrame([{**v, "lifts": ",".join(f"{x:.3f}" for x in v["lifts"])} for v in inc.values()]).to_csv(
        OUT / "sprint55_incremental.csv", index=False)

    yinc = []
    ya = {r["year"]: r for r in yearly if r["model"] == "A_current_R"}
    for name in ("B_price_state", "C_time_state", "D_full_state"):
        for r in yearly:
            if r["model"] != name or r["year"] not in ya:
                continue
            yinc.append({
                "pair": f"A → {name}", "year": r["year"], "n": r["n"],
                "roc_A": ya[r["year"]]["roc"], "roc": r["roc"],
                "droc": r["roc"] - ya[r["year"]]["roc"],
                "sep_A": ya[r["year"]]["sep"], "sep": r["sep"],
                "dsep": r["sep"] - ya[r["year"]]["sep"],
            })
    yb = {r["year"]: r for r in yearly if r["model"] == "B_price_state"}
    for r in yearly:
        if r["model"] != "D_full_state" or r["year"] not in yb:
            continue
        yinc.append({
            "pair": "B → D", "year": r["year"], "n": r["n"],
            "roc_A": yb[r["year"]]["roc"], "roc": r["roc"],
            "droc": r["roc"] - yb[r["year"]]["roc"],
            "sep_A": yb[r["year"]]["sep"], "sep": r["sep"],
            "dsep": r["sep"] - yb[r["year"]]["sep"],
        })
    pd.DataFrame(yinc).to_csv(OUT / "sprint55_yearly_incremental.csv", index=False)

    verdict = _classify(inc)
    _log(2, "WF attribution A/B/C/D", inc["A→B"]["gate"] + "/" + inc["A→C"]["gate"],
         f"A={a['roc_oos']:.3f} B={b['roc_oos']:.3f} C={c['roc_oos']:.3f} D={d['roc_oos']:.3f} "
         f"class={verdict}", 3)

    uw = uw.copy()
    for name, pred in preds.items():
        uw[f"p_{name}"] = pred
    oos = _oos(uw)
    oos = oos[np.isfinite(oos["p_A_current_R"])].copy()
    names = ("A_current_R", "B_price_state", "C_time_state", "D_full_state")

    buck = []
    for bname, lo, hi in CR_BINS:
        sub = oos[(oos["current_R"] >= lo) & (oos["current_R"] < hi)]
        stats = {m: {**_slice_metrics(sub, f"p_{m}"), "bucket": bname, "model": m} for m in names}
        aroc, asep = stats["A_current_R"]["roc"], stats["A_current_R"]["sep"]
        for m, row in stats.items():
            row["droc_vs_A"] = row["roc"] - aroc
            row["dsep_vs_A"] = row["sep"] - asep
            buck.append(row)
    pd.DataFrame(buck).to_csv(OUT / "sprint55_current_R_buckets.csv", index=False)

    lead_rows = []
    first = oos.sort_values(["trade_id", "bars_since_entry"]).drop_duplicates("trade_id")
    later = oos.merge(first[["trade_id", "bars_since_entry"]].rename(columns={"bars_since_entry": "_f"}),
                      on="trade_id")
    later = later[later["bars_since_entry"] > later["_f"]]
    mae50 = (oos[oos["mae_so_far_R"] >= 0.50]
             .sort_values(["trade_id", "bars_since_entry"]).drop_duplicates("trade_id"))
    for stage, sdf in (("first_underwater", first), ("first_mae_ge_0.50", mae50), ("subsequent", later)):
        for m in names:
            lead_rows.append({"stage": stage, "model": m, **_slice_metrics(sdf, f"p_{m}")})
    pd.DataFrame(lead_rows).to_csv(OUT / "sprint55_lead.csv", index=False)

    xt, _xf = _grid(oos, YCOL)
    pd.DataFrame(xt).to_csv(OUT / "sprint55_depth_time.csv", index=False)

    # robustness: residual after current_R; within time; within depth
    resid = []
    price_dyn = [f for f in B_FEATS if f != "current_R"]
    time_dyn = [f for f in C_FEATS if f != "current_R"]
    for feat in price_dyn + time_dyn:
        r = _residual_sep(oos, feat, YCOL)
        r["kind"] = "residual_current_R"
        r["family"] = "price" if feat in price_dyn else "time"
        resid.append(r)
    pd.DataFrame(resid).to_csv(OUT / "sprint55_residual.csv", index=False)

    tctrl, dctrl = [], []
    for t_name, t0, t1 in TIME_BINS:
        sub = oos[(oos["minutes_since_entry"] >= t0) & (oos["minutes_since_entry"] < t1)]
        for m in names:
            tctrl.append({"control": "time", "bucket": t_name, "model": m, **_slice_metrics(sub, f"p_{m}")})
    for bname, lo, hi in CR_BINS:
        sub = oos[(oos["current_R"] >= lo) & (oos["current_R"] < hi)]
        for m in names:
            dctrl.append({"control": "depth", "bucket": bname, "model": m, **_slice_metrics(sub, f"p_{m}")})
    pd.DataFrame(tctrl + dctrl).to_csv(OUT / "sprint55_robustness.csv", index=False)

    erows, cal_rows = [], []
    for m in names:
        scored = oos.copy()
        scored["p"] = scored[f"p_{m}"]
        try:
            scored["q"] = pd.qcut(scored["p"], 5, duplicates="drop", labels=False)
        except ValueError:
            scored["q"] = 0
        for q, g in scored.groupby("q", observed=False):
            erows.append({
                "model": m, "q": int(q) + 1, "n": int(len(g)),
                "p_mean": float(g["p"].mean()),
                "P_recovery": float(g[YCOL].mean()),
                "E_p0_R": float(g["p0_R"].mean()),
                "E_remaining_R": float(g["remaining_R"].mean()),
            })
        for cal in _calib(scored, YCOL):
            cal_rows.append({"model": m, **cal})
    pd.DataFrame(erows).to_csv(OUT / "sprint55_expected_R.csv", index=False)
    pd.DataFrame(cal_rows).to_csv(OUT / "sprint55_calibration.csv", index=False)

    leak = "PASS" if leak_ok else "FAIL"
    extra = {
        "verdict": verdict,
        "inc": {k: {kk: vv for kk, vv in v.items() if kk != "lifts"} for k, v in inc.items()},
        "a_roc": a["roc_oos"], "b_roc": b["roc_oos"], "c_roc": c["roc_oos"], "d_roc": d["roc_oos"],
        "n_oos": a["n"], "base_rate": a["base_rate"],
        "no_h1_features": True, "no_timer_rule": True,
    }
    _log(3, "conditional / lead / robustness", "PASS", verdict, "DONE")
    (OUT / "sprint55_leakage_audit.json").write_text(json.dumps({
        "LEAKAGE_AUDIT": leak,
        "m5_current_past_only": True,
        "current_R_current_price_only": True,
        "mae_mfe_so_far_pit": True,
        "velocity_pit_no_clock_in_B": True,
        "no_future_recovery_in_features": True,
        "no_p0_outcome_as_feature": True,
        "no_sprint_46_52_probabilities": True,
        "no_future_h1_close": True,
        "no_h1_model_features_in_m5": True,
        "m5_starts_first_completed_bar_after_entry": True,
        "no_h1_confirmation_after_entry": True,
        "no_60m_activation_rule": True,
        "adverse_velocity": "6-bar mean of PIT MAE increment (not MAE/minutes)",
    }, indent=2), encoding="utf-8")
    (OUT / "sprint55_final_verdict.json").write_text(
        json.dumps({"sprint": 55, "verdict": verdict, "production_changed": False,
                    "m5_management_shipped": False, **extra}, indent=2, default=float),
        encoding="utf-8",
    )
    report = _report(verdict, label, rows, inc, yearly, yinc, buck, lead_rows, xt,
                     resid, tctrl, dctrl, erows, cal_rows, leak, a, b, c, d, e)
    (OUT / "sprint55_report.md").write_text(report, encoding="utf-8")
    print(f"FINAL VERDICT: {verdict}")
    print("PRODUCTION: UNCHANGED")
    print("M5 HOLD/REDUCE/EXIT/TRAIL: NOT SHIPPED")
    return 0


def _report(verdict, label, rows, inc, yearly, yinc, buck, lead, xt,
            resid, tctrl, dctrl, erows, cal, leak, a, b, c, d, e) -> str:
    ab, ac, bd, cd = inc["A→B"], inc["A→C"], inc["B→D"], inc["C→D"]
    nxt = {
        "PRICE_DOMINANT": (
            "Price/position state carries the incremental ranking without a clock. "
            "Next (still no ship): expected-R economics of a frozen price-state description. No timer rule."
        ),
        "TIME_DOMINANT": (
            "The Sprint 54 lift is trade age / time-since-green, not M5 bounce. "
            "Next (still no ship): stop treating recovery_fraction as the driver; if continued, "
            "describe depth×age only — not a 60m activation rule, not more indicators."
        ),
        "BOTH_INCREMENTAL": (
            "Price and time both add. Next (still no ship): keep them as separate state channels; "
            "do not collapse into a timer. No HOLD/REDUCE/EXIT."
        ),
        "WEAK_OR_UNSTABLE": (
            "Neither channel clears the incremental gates once clock is removed from velocity. "
            "Next: stop this line or change the question. No ship."
        ),
    }[verdict]
    return "\n".join([
        "# Sprint 55 — M5 Price-State vs Time-State",
        "",
        "## VERDICT",
        "",
        f"**`{verdict}`**",
        "",
        f"A ROC={a['roc_oos']:.3f}. B_price {b['roc_oos']:.3f} (Δ{ab['droc']:+.3f}, {ab['gate']}). "
        f"C_time {c['roc_oos']:.3f} (Δ{ac['droc']:+.3f}, {ac['gate']}). "
        f"D_full {d['roc_oos']:.3f}. Time after price {bd['gate']}; price after time {cd['gate']}.",
        "",
        "Not a management-works conclusion. No timer rule. Production unchanged.",
        "",
        "## 1. Research Question",
        "",
        "Does M5 provide incremental recovery information from PRICE / POSITION STATE, "
        "or is the Sprint 54 signal primarily TIME / TRADE AGE?",
        "",
        "## 2. Architecture",
        "",
        "H1 opens the position. After entry, M5 reads position state from the first completed M5 bar. "
        "No wait for next H1 close, no H1 confirmation, no 60m activation. "
        "H1 model probability/features are not in A–D (E is price dynamics only). "
        "60m+ in Sprint 54 was an empirical pattern, not an execution rule.",
        "",
        "## 3. Dataset / Window",
        "",
        "Same H1 entries as Sprint 52/53/54 (`sprint52_m5_path.parquet`).",
        f"Window: {label['window']}.",
        f"M5 starts: {label['m5_starts']}.",
        f"OOS n={label['n_oos']} trades={label['n_trades_oos']} years={list(TRUE_OOS)}.",
        "",
        "## 4. Label",
        "",
        f"{label['definition']}",
        f"Base rate={label['base_rate']:.3f}. Exclusions: {label['exclusions']}.",
        "Sprint 54 `y_s53` is not reused as the target.",
        "",
        "## 5. Models",
        "",
        "Expanding WF LR (same protocol as Sprint 53/54). Raw scores, not calibrated probabilities.",
        "",
        "- A_current_R: current_R",
        "- B_price_state: current_R + MAE/MFE/distances + bar-PIT velocities + recovery_fraction + bars_since_MAE. "
        "**No minutes / bars_since_entry / time_since_last_positive_R.** "
        "adverse_velocity = 6-bar mean MAE increment (Sprint 54 used MAE/minutes; that embeds clock and is not in B).",
        "- C_time_state: current_R + minutes_since_entry + bars_since_entry + bars_since_MAE + time_since_last_positive_R. No velocities.",
        "- D_full_state: B ∪ C",
        "- E_price_dynamics (optional): velocities + distances, no current_R",
        "",
        "## 6. Overall OOS Results",
        "",
        _md(rows, ["model", "roc_oos", "pr_oos", "sep", "n", "base_rate", "droc_vs_A", "dsep_vs_A", "ece"],
            {"roc_oos": 3, "pr_oos": 3, "sep": 3, "n": 0, "base_rate": 3,
             "droc_vs_A": 3, "dsep_vs_A": 3, "ece": 3}),
        "",
        "ECE is a calibration diagnostic on raw LR scores (class_weight=balanced). Not a trading probability.",
        "",
        "## 7. Incremental Attribution",
        "",
        "Gates vs baseline: ΔROC ≥ +0.020, Δsep ≥ +0.030, ≥3 OOS years with positive lift, no yearly collapse.",
        "",
        f"- Price incremental (A → B): ΔROC={ab['droc']:+.3f} Δsep={ab['dsep']:+.3f} "
        f"years={ab['n_pos_years']} collapse={ab['collapse']} **{ab['gate']}**",
        f"- Time incremental (A → C): ΔROC={ac['droc']:+.3f} Δsep={ac['dsep']:+.3f} "
        f"years={ac['n_pos_years']} collapse={ac['collapse']} **{ac['gate']}**",
        f"- Time after price (B → D): ΔROC={bd['droc']:+.3f} Δsep={bd['dsep']:+.3f} **{bd['gate']}**",
        f"- Price after time (C → D): ΔROC={cd['droc']:+.3f} Δsep={cd['dsep']:+.3f} **{cd['gate']}**",
        "",
        _md(list(inc.values()), ["pair", "droc", "dsep", "n_pos_years", "collapse", "gate"],
            {"droc": 3, "dsep": 3, "n_pos_years": 0}),
        "",
        "## 8. Yearly Stability",
        "",
        _md(yearly, ["model", "year", "roc", "sep", "n"],
            {"year": 0, "roc": 3, "sep": 3, "n": 0}),
        "",
        "Yearly incremental:",
        "",
        _md(yinc, ["pair", "year", "n", "roc_A", "roc", "droc", "dsep"],
            {"year": 0, "n": 0, "roc_A": 3, "roc": 3, "droc": 3, "dsep": 3}),
        "",
        "## 9. Current_R Conditional Results",
        "",
        "Global WF scores sliced by current_R. Not a new threshold search.",
        "",
        _md(buck, ["bucket", "model", "n", "base_rate", "roc", "sep", "droc_vs_A", "dsep_vs_A"],
            {"n": 0, "base_rate": 3, "roc": 3, "sep": 3, "droc_vs_A": 3, "dsep_vs_A": 3}),
        "",
        "## 10. Lead-Time Results",
        "",
        "M5 starts at first completed bar. Stages are descriptive, not a timer.",
        "",
        _md(lead, ["stage", "model", "n", "base_rate", "roc", "sep"],
            {"n": 0, "base_rate": 3, "roc": 3, "sep": 3}),
        "",
        "## 11. Depth × Time",
        "",
        "Predefined Sprint 54 bins. Not optimized.",
        "",
        _md(xt, ["current_R", "time", "n", "base_rate", "mean_p0_R", "mean_remaining_R"],
            {"n": 0, "base_rate": 3, "mean_p0_R": 3, "mean_remaining_R": 3}),
        "",
        "## 12. Price-State Robustness",
        "",
        "Residual after current_R (price family) and scores inside elapsed-time buckets.",
        "",
        _md([r for r in resid if r.get("family") == "price"],
            ["feature", "mean_abs_sep", "mean_sep", "n_buckets_ge"],
            {"mean_abs_sep": 4, "mean_sep": 4, "n_buckets_ge": 0}),
        "",
        "Within time buckets (elapsed-time controlled):",
        "",
        _md([r for r in tctrl if r["model"] in ("A_current_R", "B_price_state")],
            ["bucket", "model", "n", "base_rate", "roc", "sep"],
            {"n": 0, "base_rate": 3, "roc": 3, "sep": 3}),
        "",
        "## 13. Time-State Robustness",
        "",
        "Residual after current_R (time family) and scores inside depth buckets.",
        "",
        _md([r for r in resid if r.get("family") == "time"],
            ["feature", "mean_abs_sep", "mean_sep", "n_buckets_ge"],
            {"mean_abs_sep": 4, "mean_sep": 4, "n_buckets_ge": 0}),
        "",
        "Within depth buckets (price-depth controlled):",
        "",
        _md([r for r in dctrl if r["model"] in ("A_current_R", "C_time_state")],
            ["bucket", "model", "n", "base_rate", "roc", "sep"],
            {"n": 0, "base_rate": 3, "roc": 3, "sep": 3}),
        "",
        "## 14. Expected Future R",
        "",
        "Raw-score quintiles. Descriptive. Not an action threshold.",
        "",
        _md(erows, ["model", "q", "n", "p_mean", "P_recovery", "E_p0_R", "E_remaining_R"],
            {"q": 0, "n": 0, "p_mean": 3, "P_recovery": 3, "E_p0_R": 3, "E_remaining_R": 3}),
        "",
        "## 15. Calibration",
        "",
        "Scores are **raw** expanding-WF LR probabilities (balanced). Not calibrated. "
        f"OOS ECE A={a['ece']:.3f} B={b['ece']:.3f} C={c['ece']:.3f} D={d['ece']:.3f} E={e['ece']:.3f}. "
        "Do not use as P for sizing or execution.",
        "",
        _md(cal, ["model", "bin", "n", "p_mean", "rate", "mean_p0_R", "mean_remaining_R"],
            {"n": 0, "p_mean": 3, "rate": 3, "mean_p0_R": 3, "mean_remaining_R": 3}) if cal else "Too thin.",
        "",
        "## 16. Leakage Audit",
        "",
        f"**LEAKAGE_AUDIT = {leak}**",
        "",
        "- M5 features current/past only; MAE/MFE so_far PIT",
        "- B velocities are bar increments / bars_since_MAE — no minutes_since_entry",
        "- No P0 outcome or future recovery as features",
        "- No Sprint 46–52 probabilities; no H1 model features in A–D",
        "- M5 starts at first completed bar after entry; no H1 confirmation; no 60m gate",
        "",
        "## 17. Production Impact",
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
        "## 18. Final Interpretation",
        "",
        f"1. Price-driven? **{'yes' if ab['pass'] else 'no'}** (A→B {ab['gate']}, ΔROC={ab['droc']:+.3f}).",
        f"2. Time-driven? **{'yes' if ac['pass'] else 'no'}** (A→C {ac['gate']}, ΔROC={ac['droc']:+.3f}).",
        f"3. Price after time? **{'yes' if cd['pass'] else 'no'}** (C→D {cd['gate']}, ΔROC={cd['droc']:+.3f}).",
        f"4. Time after price? **{'yes' if bd['pass'] else 'no'}** (B→D {bd['gate']}, ΔROC={bd['droc']:+.3f}).",
        "5. Yearly 2022–2026: see §8.",
        "6. current_R conditioning: see §9 (Sprint 54 lift was in current_R < -0.25).",
        "7. Lead-time: see §10. First bar vs later vs first MAE≥0.50 — not a 60m switch.",
        "8. Independent of H1 model: A–D use position/M5 state only (no y_prob / FEAT7).",
        "",
        f"Classification: **`{verdict}`**. This is attribution, not 'M5 management works'.",
        "",
        "## 19. Recommended Next Research Step",
        "",
        "Do not ship M5 HOLD/REDUCE/EXIT/TRAIL. " + nxt,
        "",
    ]) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
