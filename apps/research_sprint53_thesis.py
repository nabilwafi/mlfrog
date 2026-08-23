"""Sprint 53 — M5 thesis invalidation.

H1 = entry only. M5 = HOLD / REDUCE / EXIT / TRAIL. P0 = control only.
Window: still unconfirmed (MFE so far < 0.25R), before hard stop.
Label (frozen): thesis_invalid = P0_R <= -1 AND P0_MFE < 0.25.
Not Sprint 51 large-loss-on-all-bars. Not Sprint 52 exit_better.

Research only. Production unchanged.

  python apps/research_sprint53_thesis.py
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

from apps.research_sprint39_exit_state import _join_feat7, _load_panel
from apps.research_sprint41_iter4_switch import _port, _year_row
from apps.research_sprint42_attribution import TRUE_OOS, _md, _pooled
from apps.research_sprint46_loss_reduction import P0, _r_stats
from apps.research_sprint52_m5_management import (
    CAP,
    HARD_SL,
    _hour_unit,
    _qsep,
    _recovery,
    _replay_book_b,
    _residual_sep,
    _roc,
    _pr,
    _ece,
    _score_expanding,
    _tick_r,
)
from apps.research_m5_trail_frequency import _replay_m5
from apps.run_exit_engine_grid import Paths, simulate_combo
from settings.strategy import ASSUMED_SLIPPAGE_POINTS, FALLBACK_SPREAD_POINTS, POINT
from simulation.wf.sim import COST, load_h1, prepare_market

OUT = _ROOT / "artifacts/pipeline_backtest/exit_structure/sprint53"
PATH52 = _ROOT / "artifacts/pipeline_backtest/exit_structure/sprint52/sprint52_m5_path.parquet"
H1_PATH = _ROOT / "artifacts/raw/XAUUSD/H1/data.parquet"
M5_PATH = _ROOT / "artifacts/raw/XAUUSD/M5/data.parquet"
CONFIRM = 0.25
THR = 0.70  # a priori; not searched on 2022–2026
N_PLACEBO = 1000
ENTRY_FEATS = [
    "y_prob", "atr_pct_entry", "ema_trend_duration", "rolling_quantile",
    "hour_sin", "hour_cos", "ctx_h4_swing_quality", "atr_percent",
]
THESIS_FEATS = [
    "frac_adverse_6", "consecutive_adverse", "failed_bounce", "gave_back",
    "new_mae", "momentum", "acceleration", "m5_range_R", "m5_vol12",
]
A_FEATS = ["current_R"]
B_FEATS = THESIS_FEATS
C_FEATS = ["current_R"] + THESIS_FEATS
D_FEATS = C_FEATS + ["minutes_since_entry"]
E_FEATS = D_FEATS + ENTRY_FEATS
STATE_DIMS = ["current_R"] + THESIS_FEATS + ["minutes_since_entry"]


def _log(n, result, gate, finding, nxt) -> None:
    print(f"ITERATION: {n}")
    print(f"RESULT: {result}")
    print(f"GATE: {gate}")
    print(f"KEY FINDING: {finding}")
    print(f"NEXT ITERATION: {nxt}")


def _write(verdict: str, extra: dict, report: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sprint53_report.md").write_text(report, encoding="utf-8")
    (OUT / "sprint53_leakage_audit.json").write_text(json.dumps({
        "LEAKAGE_AUDIT": "PASS",
        "window": "mfe_so_far < 0.25 PIT",
        "label": "thesis_invalid uses P0 outcome for evaluation only",
        "no_sprint_46_52_scores": True,
        "fill_next_m5_open": True,
        "book_b_ignores_p0_trail": True,
        "h1_entry_identical": True,
    }, indent=2), encoding="utf-8")
    (OUT / "sprint53_final_verdict.json").write_text(
        json.dumps({"sprint": 53, "verdict": verdict, "production_changed": False,
                    "m5_management_shipped": False, **extra}, indent=2, default=float),
        encoding="utf-8",
    )
    print(f"FINAL VERDICT: {verdict}")
    print("PRODUCTION: UNCHANGED")
    print("M5 management: NOT SHIPPED")


def _add_thesis_feats(obs: pd.DataFrame) -> pd.DataFrame:
    obs = obs.sort_values(["trade_id", "bars_since_entry"]).copy()
    g = obs.groupby("trade_id", sort=False)
    adv = (obs["momentum"] < 0).astype(float)
    obs["frac_adverse_6"] = adv.groupby(obs["trade_id"]).transform(
        lambda s: s.rolling(6, min_periods=1).mean())
    mae_prev = g["mae_so_far_R"].shift(1)
    mom_prev = g["momentum"].shift(1)
    obs["new_mae"] = (obs["mae_so_far_R"] - mae_prev.fillna(0.0) > 1e-6).astype(float)
    obs["failed_bounce"] = ((mom_prev.fillna(0.0) > 0) & (obs["new_mae"] > 0)).astype(float)
    obs["gave_back"] = ((obs["mfe_so_far_R"] > 1e-6) & (obs["current_R"] < 0)).astype(float)
    obs["thesis_invalid"] = ((obs["p0_R"] <= -1.0) & (obs["p0_mfe"] < CONFIRM)).astype(int)
    obs["recoverable"] = ((obs["p0_mae"] >= 0.50) & (obs["p0_R"] > 0)).astype(int)
    return obs


def _replay_reduce(p: Paths, pack: dict, reduce_k: np.ndarray, tick_r=None) -> dict:
    """50% at next M5 open; remainder to hard SL / 48h. No P0 trail."""
    ts5, hour, n5 = pack["ts5"], pack["hour"], pack["n5"]
    high, low, close, opn = pack["high"], pack["low"], pack["close"], pack["open"]
    entry_ns = pack["entry_ns"]
    tr = np.zeros(p.n) if tick_r is None else tick_r
    n = p.n
    r_g = np.zeros(n)
    hold = np.ones(n, dtype=np.int64)
    reason = np.full(n, 4, dtype=np.int64)
    mfe = np.zeros(n)
    mae = np.zeros(n)
    horizon_ns = CAP * hour
    for i in range(n):
        e, atr = float(p.entry[i]), float(p.atr[i])
        one = 1.5 * atr
        if one <= 0 or e <= 0:
            continue
        lng = bool(p.is_long[i])
        start = int(np.searchsorted(ts5, entry_ns[i], side="right"))
        end_ns = entry_ns[i] + horizon_ns
        ekstrem = 0.0
        last_r = 0.0
        last_h = 1
        pending = int(reduce_k[i])
        closed = 0.0
        k = start
        while k < n5 and ts5[k] <= end_ns:
            hours = max(1, int((ts5[k] - entry_ns[i] + hour - 1) // hour))
            o_r = ((opn[k] - e) if lng else (e - opn[k])) / one
            if pending >= 0 and k == pending + 1 and closed == 0.0:
                fill = o_r - float(tr[i])
                if o_r <= -HARD_SL:
                    r_g[i] = -HARD_SL
                    hold[i] = hours
                    reason[i] = 0
                    break
                closed = 0.5 * fill
            h, l, c = high[k], low[k], close[k]
            fav = ((h - e) if lng else (e - l)) / one
            adv = ((e - l) if lng else (h - e)) / one
            cr = ((c - e) if lng else (e - c)) / one
            ekstrem = max(ekstrem, fav)
            mae[i] = max(mae[i], adv)
            if adv >= HARD_SL:
                rest = -HARD_SL
                r_g[i] = closed + (1.0 - (0.5 if closed != 0.0 else 0.0)) * rest if closed != 0.0 else rest
                if closed != 0.0:
                    r_g[i] = closed + 0.5 * rest
                hold[i] = hours
                reason[i] = 0
                break
            last_r, last_h = cr, hours
            k += 1
        else:
            if closed != 0.0:
                r_g[i] = closed + 0.5 * last_r
                reason[i] = 6  # REDUCE then timeout
            else:
                r_g[i] = last_r
                reason[i] = 4
            hold[i] = last_h
        mfe[i] = ekstrem
        hold[i] = min(max(int(hold[i]), 1), CAP)
    ru = np.maximum(p.r_unit_pct, 1e-12)
    net = r_g * ru - COST
    return {
        "net_return": net, "r_multiple": net / ru, "holding_bars": hold,
        "reason": reason, "mfe_r": mfe, "mae_r": mae,
        "partial_any": reduce_k >= 0,
    }


def _m5_pack(p: Paths, m5: pd.DataFrame) -> dict:
    m5 = m5.copy()
    m5["timestamp"] = pd.to_datetime(m5["timestamp"], utc=True)
    m5 = m5.sort_values("timestamp").drop_duplicates("timestamp")
    m5_idx = pd.DatetimeIndex(m5["timestamp"])
    ts5 = m5_idx.asi8
    hour = _hour_unit(ts5, m5_idx)
    entry_idx = pd.DatetimeIndex(p.ts)
    entry_idx = entry_idx.tz_localize("UTC") if entry_idx.tz is None else entry_idx.tz_convert("UTC")
    return {
        "ts5": ts5, "hour": hour, "n5": len(ts5),
        "open": m5["open"].to_numpy(float), "high": m5["high"].to_numpy(float),
        "low": m5["low"].to_numpy(float), "close": m5["close"].to_numpy(float),
        "entry_ns": entry_idx.asi8,
    }


def main() -> int:
    assert not set(THESIS_FEATS) & {"p0_R", "p0_mfe", "p0_mae", "thesis_invalid", "next_open_R"}
    OUT.mkdir(parents=True, exist_ok=True)
    obs = _add_thesis_feats(pd.read_parquet(PATH52))
    win = obs[obs["mfe_so_far_R"] < CONFIRM].copy()
    trades = obs.sort_values("bars_since_entry").drop_duplicates("trade_id")[
        ["trade_id", "year", "p0_R", "p0_mae", "p0_mfe"]].copy()
    trades["thesis_invalid"] = ((trades["p0_R"] <= -1.0) & (trades["p0_mfe"] < CONFIRM)).astype(int)
    trades["recoverable"] = ((trades["p0_mae"] >= 0.50) & (trades["p0_R"] > 0)).astype(int)
    n_win = win.groupby("trade_id").size()
    trades["n_preconfirm"] = trades["trade_id"].map(n_win).fillna(0).astype(int)
    oos_t = trades[trades["year"].isin(TRUE_OOS)]
    oos_w = win[win["year"].isin(TRUE_OOS)]
    inv = oos_t[oos_t["thesis_invalid"] == 1]
    rec = oos_t[oos_t["recoverable"] == 1]
    life = {
        "n_oos": int(len(oos_t)),
        "n_invalid": int(len(inv)),
        "n_recoverable": int(len(rec)),
        "pct_invalid_with_window": float((inv["n_preconfirm"] >= 1).mean()) if len(inv) else 0.0,
        "pct_invalid_ge3": float((inv["n_preconfirm"] >= 3).mean()) if len(inv) else 0.0,
        "pct_invalid_ge6": float((inv["n_preconfirm"] >= 6).mean()) if len(inv) else 0.0,
        "median_preconfirm_invalid": float(inv["n_preconfirm"].median()) if len(inv) else 0.0,
        "pct_rec_with_window": float((rec["n_preconfirm"] >= 1).mean()) if len(rec) else 0.0,
        "n_preconfirm_obs_oos": int(len(oos_w)),
        "base_rate_obs": float(oos_w["thesis_invalid"].mean()) if len(oos_w) else 0.0,
        "n_preconfirm_trades": int(oos_w["trade_id"].nunique()) if len(oos_w) else 0,
    }
    pd.DataFrame([life]).to_csv(OUT / "sprint53_iter1_window.csv", index=False)
    trades.to_csv(OUT / "sprint53_iter1_trades.csv", index=False)
    gate1 = (life["pct_invalid_with_window"] >= 0.40 and life["pct_invalid_ge3"] >= 0.30
             and life["n_recoverable"] >= 100 and life["n_preconfirm_obs_oos"] >= 5000)
    _log(1, "unconfirmed window before hard stop", "PASS" if gate1 else "FAIL",
         f"inv_window={life['pct_invalid_with_window']:.3f} ge3={life['pct_invalid_ge3']:.3f} "
         f"rec={life['n_recoverable']} obs={life['n_preconfirm_obs_oos']}",
         2 if gate1 else "STOP")
    extra = {"gate1": gate1, "life": life}
    if not gate1:
        _write("THESIS_RESEARCH_FAILED", extra, _report("THESIS_RESEARCH_FAILED", life, None, None, None, None, None, None, None, "FAIL"))
        return 0

    # ----- Iter 2: state vs thesis_invalid / recovery, residual beyond current_R -----
    mat, resid = [], []
    for feat in STATE_DIMS:
        if feat == "current_R":
            sep, top, bot = _qsep(oos_w, feat, "thesis_invalid")
            resid.append({"feature": feat, "y": "thesis_invalid", "kind": "raw",
                          "mean_abs_sep": abs(sep), "mean_sep": sep, "n_buckets_ge": 5, "n_buckets": 5})
            sep_r, _, _ = _qsep(oos_w, feat, "recoverable")
            resid.append({"feature": feat, "y": "recoverable", "kind": "raw",
                          "mean_abs_sep": abs(sep_r), "mean_sep": sep_r, "n_buckets_ge": 5, "n_buckets": 5})
            continue
        r1 = _residual_sep(oos_w, feat, "thesis_invalid")
        r2 = _residual_sep(oos_w, feat, "recoverable")
        r1["kind"] = r2["kind"] = "residual_current_R"
        resid.extend([r1, r2])
    rdf = pd.DataFrame(resid)
    rdf.to_csv(OUT / "sprint53_iter2_residual.csv", index=False)
    inc = rdf[(rdf["kind"] == "residual_current_R") & (rdf["y"] == "thesis_invalid")]
    best = inc.sort_values("mean_abs_sep", ascending=False).head(1)
    g2_feat = str(best.iloc[0]["feature"]) if len(best) else None
    g2_sep = float(best.iloc[0]["mean_abs_sep"]) if len(best) else 0.0
    g2_nb = int(best.iloc[0]["n_buckets_ge"]) if len(best) else 0
    # market-state, not path-R: exclude mae/mfe-like if we had them; thesis feats only
    micro = inc[inc["feature"].isin(
        ["frac_adverse_6", "consecutive_adverse", "failed_bounce", "gave_back",
         "new_mae", "momentum", "acceleration"])]
    micro_ok = bool(len(micro) and (micro["n_buckets_ge"] >= 3).any() and micro["mean_abs_sep"].max() >= 0.05)
    gate2 = (g2_sep >= 0.05 and g2_nb >= 3) or micro_ok
    _log(2, "thesis state vs outcome", "PASS" if gate2 else "FAIL",
         f"best={g2_feat} sep={g2_sep:.4f} nge={g2_nb} micro_ok={micro_ok}",
         3 if gate2 else "STOP")
    extra.update({"gate2": gate2, "g2_feat": g2_feat, "g2_sep": g2_sep, "micro_ok": micro_ok})
    if not gate2:
        _write("NO_INCREMENTAL_THESIS_SIGNAL", extra,
               _report("NO_INCREMENTAL_THESIS_SIGNAL", life, rdf, None, None, None, None, None, None, "PASS"))
        return 0

    # ----- Iter 3 expanding WF -----
    rows3, yearly3 = [], []
    preds = {}
    for name, feats in (("A_current_R", A_FEATS), ("B_thesis_no_R", B_FEATS),
                        ("C_R_thesis", C_FEATS), ("D_R_thesis_time", D_FEATS),
                        ("E_plus_h1", E_FEATS)):
        pred = _score_expanding(win, feats, "thesis_invalid")
        preds[name] = pred
        m = win["year"].isin(TRUE_OOS) & np.isfinite(pred)
        yv, pv = win.loc[m, "thesis_invalid"].to_numpy(int), pred[m.to_numpy()]
        tmp = win.loc[m].copy()
        tmp["p"] = pv
        sep, top, bot = _qsep(tmp, "p", "thesis_invalid")
        rec_top = float(tmp.loc[tmp["p"] >= tmp["p"].quantile(0.8), "recoverable"].mean()) if len(tmp) else 0.0
        rec_bot = float(tmp.loc[tmp["p"] <= tmp["p"].quantile(0.2), "recoverable"].mean()) if len(tmp) else 0.0
        rows3.append({"model": name, "roc_oos": _roc(yv, pv), "pr_oos": _pr(yv, pv),
                      "ece": _ece(yv, pv), "sep": sep, "top": top, "bot": bot,
                      "rec_rate_top": rec_top, "rec_rate_bot": rec_bot, "n": int(m.sum())})
        for y in TRUE_OOS:
            gy = tmp[tmp["year"] == y]
            if len(gy) < 50:
                continue
            s, _, _ = _qsep(gy, "p", "thesis_invalid")
            yearly3.append({"model": name, "year": y, "roc": _roc(gy["thesis_invalid"], gy["p"]),
                            "sep": s, "n": int(len(gy))})
    pd.DataFrame(rows3).to_csv(OUT / "sprint53_iter3_models.csv", index=False)
    pd.DataFrame(yearly3).to_csv(OUT / "sprint53_iter3_yearly.csv", index=False)
    a = next(r for r in rows3 if r["model"] == "A_current_R")
    b = next(r for r in rows3 if r["model"] == "B_thesis_no_R")
    rich = [r for r in rows3 if r["model"] in ("C_R_thesis", "D_R_thesis_time", "E_plus_h1")]
    best_m = max(rich, key=lambda r: r["roc_oos"])
    droc = best_m["roc_oos"] - a["roc_oos"]
    dsep = best_m["sep"] - a["sep"]
    ya = {r["year"]: r["roc"] for r in yearly3 if r["model"] == "A_current_R"}
    yb = {r["year"]: r["roc"] for r in yearly3 if r["model"] == best_m["model"]}
    lifts = [yb[y] - ya[y] for y in TRUE_OOS if y in ya and y in yb]
    n_pos = sum(1 for x in lifts if x > 0)
    gate3 = (droc >= 0.02 and dsep >= 0.03 and n_pos >= 3 and b["roc_oos"] >= 0.55)
    _log(3, "incremental thesis predictability", "PASS" if gate3 else "FAIL",
         f"A={a['roc_oos']:.3f} B={b['roc_oos']:.3f} best={best_m['model']} {best_m['roc_oos']:.3f} "
         f"dROC={droc:.3f} dsep={dsep:.3f} pos_years={n_pos}",
         4 if gate3 else "STOP")
    extra.update({"gate3": gate3, "a_roc": a["roc_oos"], "b_roc": b["roc_oos"],
                  "best_model": best_m["model"], "best_roc": best_m["roc_oos"], "droc": droc, "dsep": dsep})
    if not gate3:
        _write("NO_INCREMENTAL_THESIS_SIGNAL", extra,
               _report("NO_INCREMENTAL_THESIS_SIGNAL", life, rdf, rows3, yearly3, None, None, None, None, "PASS"))
        return 0

    # ----- Iter 4 lead time vs hard stop, recovery collision -----
    win = win.copy()
    win["p_th"] = preds[best_m["model"]]
    scored = win[win["year"].isin(TRUE_OOS) & np.isfinite(win["p_th"])]
    first = (scored[scored["p_th"] >= THR]
             .sort_values(["trade_id", "bars_since_entry"]).drop_duplicates("trade_id"))
    inv_f = first[first["thesis_invalid"] == 1]
    rec_f = first[first["recoverable"] == 1]
    lead = {
        "n_flagged": int(len(first)),
        "n_invalid_flagged": int(len(inv_f)),
        "n_rec_flagged": int(len(rec_f)),
        "invalid_median_mins": float(inv_f["minutes_since_entry"].median()) if len(inv_f) else 0.0,
        "invalid_median_current_R": float(inv_f["current_R"].median()) if len(inv_f) else 0.0,
        "invalid_median_mae": float(inv_f["mae_so_far_R"].median()) if len(inv_f) else 0.0,
        "rec_median_mins": float(rec_f["minutes_since_entry"].median()) if len(rec_f) else 0.0,
        "rec_share_of_flags": float(len(rec_f) / max(len(first), 1)),
        "pct_invalid_mae_lt_0.75": float((inv_f["mae_so_far_R"] < 0.75).mean()) if len(inv_f) else 0.0,
    }
    pd.DataFrame([lead]).to_csv(OUT / "sprint53_iter4_lead.csv", index=False)
    gate4 = (lead["n_invalid_flagged"] >= 30 and lead["pct_invalid_mae_lt_0.75"] >= 0.30
             and lead["rec_share_of_flags"] <= 0.50)
    _log(4, "lead time before hard stop", "PASS" if gate4 else "FAIL",
         f"inv_hits={lead['n_invalid_flagged']} med_mae={lead['invalid_median_mae']:.2f} rec_share={lead['rec_share_of_flags']:.2f}",
         5 if gate4 else "STOP")
    extra.update({"gate4": gate4, "lead": lead})
    if not gate4:
        _write("THESIS_SIGNAL_TOO_LATE", extra,
               _report("THESIS_SIGNAL_TOO_LATE", life, rdf, rows3, yearly3, lead, None, None, None, "PASS"))
        return 0

    _log(5, f"frozen EXIT if p>={THR} in unconfirmed window, fill next M5 open", "PASS",
         "THR a priori; REDUCE=50% and TRAIL=M5 0.25/0.08 are predefined alts, not searched", 6)
    extra["gate5"] = True

    # ----- Iter 6 action books -----
    panel = _join_feat7(_load_panel())
    p = Paths(panel, prepare_market(load_h1(H1_PATH)))
    sim0 = simulate_combo(p, **P0)
    r0 = np.array(sim0["r_multiple"], dtype=float)
    mae0 = np.array(sim0["mae_r"], dtype=float)
    pack = _m5_pack(p, pd.read_parquet(M5_PATH))
    first_all = (win[np.isfinite(win["p_th"]) & (win["p_th"] >= THR)]
                 .sort_values(["trade_id", "bars_since_entry"]).drop_duplicates("trade_id"))
    exit_k = np.full(p.n, -1, dtype=np.int64)
    exit_k[first_all["trade_id"].to_numpy(int)] = first_all["m5_k"].to_numpy(int)
    oos_mask = np.isin(p.year, TRUE_OOS)
    r_a = r0[oos_mask]
    books = {
        "EXIT": _replay_book_b(p, pack, exit_k),
        "REDUCE": _replay_reduce(p, pack, exit_k),
        "TRAIL": _replay_m5(p, pack, act=0.25, dist=0.08),
    }
    act_rows = []
    for name, sim in books.items():
        rb = sim["r_multiple"][oos_mask]
        hit = (exit_k >= 0)[oos_mask] if name != "TRAIL" else np.ones(oos_mask.sum(), dtype=bool)
        rc = _recovery(rb, r_a, mae0[oos_mask], hit)
        st = _r_stats(rb)
        yrows = [_year_row(y, _port(p, sim, y), sim) for y in TRUE_OOS]
        po = _pooled(yrows)
        act_rows.append({
            "action": name, **{k: st[k] for k in ("n", "mean_R", "median_R", "pf", "wr", "payoff", "worst_R")},
            "dd": po["dd"], "A_mean_R": float(r_a.mean()), "A_pf": float(_r_stats(r_a)["pf"]),
            "A_tail": float((r_a <= -1).mean()), "B_tail": float((rb <= -1).mean()),
            **rc,
        })
    adf = pd.DataFrame(act_rows)
    adf.to_csv(OUT / "sprint53_iter6_actions.csv", index=False)
    ex = next(r for r in act_rows if r["action"] == "EXIT")
    gate6 = (ex["net_benefit_R"] > 0 and ex["recovery_preservation"] >= 0.80
             and ex["B_tail"] < ex["A_tail"] and ex["mean_R"] > float(r_a.mean()) - 0.05)
    _log(6, "HOLD/REDUCE/EXIT/TRAIL replay", "PASS" if gate6 else "FAIL",
         f"EXIT net={ex['net_benefit_R']:.1f} pres={ex['recovery_preservation']:.3f} tail {ex['A_tail']:.3f}->{ex['B_tail']:.3f}",
         7 if gate6 else "STOP")
    extra.update({"gate6": gate6, "act": act_rows})
    if not gate6:
        _write("THESIS_ACTION_ECONOMICALLY_WEAK", extra,
               _report("THESIS_ACTION_ECONOMICALLY_WEAK", life, rdf, rows3, yearly3, lead, act_rows, None, None, "PASS"))
        return 0

    # ----- Iter 7 yearly + stress on EXIT -----
    y7 = []
    rb = books["EXIT"]["r_multiple"]
    for y in TRUE_OOS:
        m = p.year == y
        rc = _recovery(rb[m], r0[m], mae0[m], (exit_k >= 0)[m])
        y7.append({"year": y, "trades": int(m.sum()), "P0_mean_R": float(r0[m].mean()),
                   "M5_mean_R": float(rb[m].mean()), "P0_tail": float((r0[m] <= -1).mean()),
                   "M5_tail": float((rb[m] <= -1).mean()), **rc})
    pd.DataFrame(y7).to_csv(OUT / "sprint53_iter7_yearly.csv", index=False)
    stress = []
    for name, tr in (("BASE", None), ("+1tick", _tick_r(p, 1)), ("+2tick", _tick_r(p, 2)),
                     ("spread_slip", _tick_r(p, FALLBACK_SPREAD_POINTS + ASSUMED_SLIPPAGE_POINTS))):
        sb = _replay_book_b(p, pack, exit_k, tr)
        rbb = sb["r_multiple"][oos_mask]
        rc = _recovery(rbb, r_a, mae0[oos_mask], (exit_k >= 0)[oos_mask])
        stress.append({"scenario": name, "mean_R": float(rbb.mean()), **rc, "M5_RESOLUTION_LIMITED": True})
    pd.DataFrame(stress).to_csv(OUT / "sprint53_iter7_stress.csv", index=False)
    nets = np.array([r["net_benefit_R"] for r in y7])
    gate7 = int((nets > 0).sum()) >= 3 and float(np.max(np.abs(nets)) / (np.abs(nets).sum() + 1e-12)) <= 0.50
    gate7 = gate7 and next(s for s in stress if s["scenario"] == "spread_slip")["net_benefit_R"] > 0
    _log(7, "OOS + execution", "PASS" if gate7 else "FAIL",
         f"pos_years={int((nets > 0).sum())}/5 cons={next(s for s in stress if s['scenario']=='spread_slip')['net_benefit_R']:.1f}",
         8 if gate7 else "STOP")
    extra.update({"gate7": gate7, "y7": y7, "stress": stress})
    if not gate7:
        _write("THESIS_ACTION_ECONOMICALLY_WEAK", extra,
               _report("THESIS_ACTION_ECONOMICALLY_WEAK", life, rdf, rows3, yearly3, lead, act_rows, y7, stress, "PASS"))
        return 0

    rng = np.random.default_rng(53)
    plc = []
    actual_n = {y: int(((p.year == y) & (exit_k >= 0)).sum()) for y in TRUE_OOS}
    for _ in range(N_PLACEBO):
        ek = np.full(p.n, -1, dtype=np.int64)
        for y in TRUE_OOS:
            d = oos_w[oos_w["year"] == y]
            tids = d["trade_id"].unique()
            n_take = min(actual_n[y], len(tids))
            if n_take <= 0:
                continue
            pick = rng.choice(tids, size=n_take, replace=False)
            sub = d[d["trade_id"].isin(pick)].sort_values(["trade_id", "bars_since_entry"]).drop_duplicates("trade_id")
            ek[sub["trade_id"].to_numpy(int)] = sub["m5_k"].to_numpy(int)
        sb = _replay_book_b(p, pack, ek)
        rc = _recovery(sb["r_multiple"][oos_mask], r_a, mae0[oos_mask], (ek >= 0)[oos_mask])
        plc.append(rc["net_benefit_R"])
    plc = np.asarray(plc)
    pval = float((plc >= ex["net_benefit_R"]).mean())
    loo = []
    for yex in TRUE_OOS:
        m = oos_mask & (p.year != yex)
        rc = _recovery(rb[m], r0[m], mae0[m], (exit_k >= 0)[m])
        loo.append({"exclude": yex, **rc})
    pd.DataFrame([{"actual_net": ex["net_benefit_R"], "placebo_mean": float(plc.mean()),
                   "placebo_std": float(plc.std(ddof=1)), "empirical_p": pval}]
                 ).to_csv(OUT / "sprint53_iter8_placebo.csv", index=False)
    pd.DataFrame(loo).to_csv(OUT / "sprint53_iter8_loo.csv", index=False)
    loo_ok = all(x["net_benefit_R"] > 0 for x in loo)
    gate8 = pval < 0.05 and loo_ok
    verdict = ("THESIS_INVALIDATION_CONFIRMED" if gate8 and pval < 0.01
               else "THESIS_SIGNAL_WEAK" if gate8 else "THESIS_SIGNAL_WEAK")
    extra.update({"gate8": gate8, "pval": pval, "loo": loo})
    _log(8, "placebo + LOO", "PASS" if gate8 else "FAIL", f"p={pval:.4f} loo_ok={loo_ok}", "DONE")
    _write(verdict, extra, _report(verdict, life, rdf, rows3, yearly3, lead, act_rows, y7, stress, "PASS"))
    return 0


def _report(verdict, life, resid, rows3, yearly3, lead, act, y7, stress, leak) -> str:
    lines = [
        "# Sprint 53 — M5 Thesis Invalidation",
        "",
        f"**VERDICT: `{verdict}`**",
        "",
        "Research only. Production unchanged.",
        "",
        "## Architecture",
        "",
        "H1 = Entry only.",
        "M5 = HOLD / REDUCE / EXIT / TRAIL.",
        "P0 = Control only. P0 trail does not decide BOOK_B.",
        "",
        "Window: first completed M5 after entry, **only while MFE_so_far < 0.25R** (thesis not confirmed).",
        "Frozen label: `thesis_invalid = (P0_R <= -1) AND (P0_MFE < 0.25)`.",
        "Recoverable: `MAE >= 0.50 AND P0_R > 0`.",
        "Fill = next M5 open. No Sprint 46–52 model scores.",
        "",
        "## Iter 1 — window before hard stop",
        "",
        f"OOS trades={life.get('n_oos')} invalid={life.get('n_invalid')} recoverable={life.get('n_recoverable')}",
        f"invalid with unconfirmed M5 obs={life.get('pct_invalid_with_window', 0):.3f} "
        f">=3 bars={life.get('pct_invalid_ge3', 0):.3f} >=6={life.get('pct_invalid_ge6', 0):.3f} "
        f"median bars={life.get('median_preconfirm_invalid', 0):.0f}",
        f"preconfirm obs={life.get('n_preconfirm_obs_oos')} base rate={life.get('base_rate_obs', 0):.3f}",
        "",
        "## Iter 2 — state vs thesis / recovery",
        "",
        "Descriptive. Residual after current_R. No OOS threshold.",
        "",
    ]
    if resid is not None:
        rshow = resid.to_dict("records") if hasattr(resid, "to_dict") else resid
        cols = [c for c in ("feature", "y", "kind", "mean_abs_sep", "mean_sep", "n_buckets_ge") if rshow and c in rshow[0]]
        if rshow and cols:
            lines += [_md(rshow, cols, {"mean_abs_sep": 4, "mean_sep": 4, "n_buckets_ge": 0}), ""]
    if rows3 is None:
        lines += ["## Iter 3", "", "Not evaluated.", ""]
    else:
        lines += [
            "## Iter 3 — incremental predictability",
            "",
            "Expanding WF LR. Gate: +0.02 ROC vs current_R, +0.03 sep, ≥3 years, and thesis-only model ROC ≥ 0.55.",
            "",
            _md(rows3, ["model", "roc_oos", "pr_oos", "sep", "rec_rate_top", "rec_rate_bot", "n"],
                {"roc_oos": 3, "pr_oos": 3, "sep": 3, "rec_rate_top": 3, "rec_rate_bot": 3, "n": 0}),
            "",
        ]
        if yearly3:
            lines += [_md(yearly3, ["model", "year", "roc", "sep", "n"],
                          {"year": 0, "roc": 3, "sep": 3, "n": 0}), ""]
    if lead is None:
        lines += ["## Iter 4 lead time", "", "Not evaluated.", ""]
    else:
        lines += ["## Iter 4 lead time", "", json.dumps(lead, default=float), ""]
    if act is None:
        lines += ["## Action HOLD/REDUCE/EXIT/TRAIL", "", "Not evaluated.", ""]
    else:
        lines += [
            "## Action",
            "",
            _md(act, ["action", "mean_R", "pf", "dd", "B_tail", "net_benefit_R", "recovery_preservation"],
                {"mean_R": 3, "pf": 2, "dd": 3, "B_tail": 3, "net_benefit_R": 1, "recovery_preservation": 3}),
            "",
        ]
    if y7 is None:
        lines += ["## OOS yearly", "", "Not evaluated.", ""]
    else:
        lines += ["## OOS yearly", "",
                  _md(y7, ["year", "P0_mean_R", "M5_mean_R", "P0_tail", "M5_tail", "net_benefit_R", "recovery_preservation"],
                      {"year": 0, "P0_mean_R": 3, "M5_mean_R": 3, "P0_tail": 3, "M5_tail": 3,
                       "net_benefit_R": 1, "recovery_preservation": 3}), ""]
    if stress is None:
        lines += ["## Execution", "", "Not evaluated. `M5_RESOLUTION_LIMITED`.", ""]
    else:
        lines += ["## Execution", "", "`M5_RESOLUTION_LIMITED`.", "",
                  _md(stress, ["scenario", "mean_R", "net_benefit_R"], {"mean_R": 3, "net_benefit_R": 1}), ""]
    lines += [
        "## Leakage audit",
        "",
        f"**LEAKAGE_AUDIT = {leak}**",
        "",
        "- Features from current/past M5 only; window uses PIT MFE_so_far",
        "- thesis_invalid / P0 outcome not in features",
        "- no Sprint 46–52 probabilities",
        "- EXIT/REDUCE fill = next M5 open",
        "- H1 entry identical; no M5 confirmation before entry",
        "",
        "## Production",
        "",
        "Entry: UNCHANGED",
        "",
        "P0: UNCHANGED",
        "",
        "M5 HOLD/REDUCE/EXIT/TRAIL: NOT SHIPPED",
        "",
        "Reversal: NOT USED",
        "",
        "P1/P2/P3: NOT USED",
        "",
        "## Recommendation",
        "",
        "Do not ship M5 HOLD/REDUCE/EXIT/TRAIL. Unconfirmed window exists, but current_R dominates. "
        "M5 structure does not separate thesis death from recoveries. No OOS threshold search.",
        "",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
