"""Sprint 49 — Executable early loss suppression (research only).

No retrospective -0.50 fill. No P0/entry change. No P1/P2/P3/reversal.

  python apps/research_sprint49_early_suppression.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import tools.pyc_path_hook  # noqa: F401

from apps.research_sprint39_exit_state import _atr_pctile, _join_feat7, _load_panel, session_of_hour
from apps.research_sprint40_iter1_time_aware_exit import STARTING
from apps.research_sprint41_iter1_diff import STATE_FEATS
from apps.research_sprint41_iter4_switch import _port, _year_row
from apps.research_sprint42_attribution import TRUE_OOS, _md, _pooled
from apps.research_sprint46_loss_reduction import FOLDS, P0, _fit, _mc, _r_stats
from apps.research_sprint47_loss_action import P0_ACT, P0_DIST, _net, _sim_from, _tick_r
from apps.run_exit_engine_grid import FEAT7, Paths, simulate_combo
from settings.strategy import ASSUMED_SLIPPAGE_POINTS, FALLBACK_SPREAD_POINTS, POINT
from simulation.wf.sim import COST, entry_indices, load_h1, prepare_market

OUT = _ROOT / "artifacts/pipeline_backtest/loss_reduction/sprint49"
H1_PATH = _ROOT / "artifacts/raw/XAUUSD/H1/data.parquet"
REF_N_OOS = 5614
CHECKS = (0.25, 0.35, 0.40, 0.45, 0.50)  # frozen audit list
STOP_REASONS = {0, 1, 2}
THR = 0.70  # a priori for NEW model; not searched on OOS
A4_SL = -0.75  # a priori
CONS_PTS = float(FALLBACK_SPREAD_POINTS + ASSUMED_SLIPPAGE_POINTS)
N_PLACEBO = 1000
MAE_BINS = (
    ("0.50_0.60R", 0.50, 0.60),
    ("0.60_0.75R", 0.60, 0.75),
    ("0.75_1.00R", 0.75, 1.00),
    ("1.00_1.25R", 1.00, 1.25),
    (">=1.25R", 1.25, 10.0),
)
MODEL_FEATS = STATE_FEATS + ["bars_in_trade", "consec_adverse", "range_R", "cr_accel", "ctx_h4_swing_quality"]
REASON = {0: "SL", 1: "TRAIL", 2: "BE", 3: "TP", 4: "TIMEOUT"}


def _log(n, result, gate, finding, nxt) -> None:
    print(f"ITERATION: {n}")
    print(f"RESULT: {result}")
    print(f"GATE: {gate}")
    print(f"KEY FINDING: {finding}")
    print(f"NEXT ITERATION: {nxt}")


def _roc(y, p) -> float:
    y = np.asarray(y, dtype=int)
    if len(y) < 2 or y.min() == y.max():
        return 0.5
    return float(roc_auc_score(y, p))


def _pr(y, p) -> float:
    y = np.asarray(y, dtype=int)
    if len(y) < 2 or y.min() == y.max():
        return float(y.mean())
    return float(average_precision_score(y, p))


def _px_to_r(px, entry, one_r, is_long) -> float:
    return float((px - entry) / one_r if is_long else (entry - px) / one_r)


def _a4_next(p: Paths, i: int, j: int, last: int, sl0: float) -> tuple[float, int]:
    """Tighter stop from bar j+1 only. Never fills on the signal bar."""
    sl = float(sl0)
    extreme = 0.0
    for t in range(1, j + 1):
        f = p.fav[i, t]
        if np.isfinite(f):
            extreme = max(extreme, float(f))
    for k in range(j + 1, last + 1):
        fav, adv = p.fav[i, k], p.adv[i, k]
        if np.isfinite(fav):
            extreme = max(extreme, float(fav))
        if extreme >= P0_ACT and np.isfinite(p.atr_over_r[i, k]):
            sl = max(sl, extreme - P0_DIST * float(p.atr_over_r[i, k]))
        if np.isfinite(adv) and float(adv) >= -sl:
            return sl, k
    cr = p.close_r[i, last]
    return (float(cr) if np.isfinite(cr) else sl), last


def _econ(r, r0, trades, trig) -> dict:
    ll = trades["large_loss_1"].to_numpy(dtype=bool)
    rec = trades["p0_R"].to_numpy() > 0
    tid = trades["trade_id"].to_numpy(dtype=int)
    hit = np.array([int(t) in trig for t in tid])
    g075 = (trades["mae_R"].to_numpy() >= 0.75) & rec
    g1 = (trades["mae_R"].to_numpy() >= 1.0) & rec
    destroyed = g075 & (r <= 0)
    ts = float(((r - r0)[ll & hit]).sum()) if (ll & hit).any() else 0.0
    rl = float(((r0 - r)[rec & hit]).sum()) if (rec & hit).any() else 0.0
    st = _r_stats(r)
    return {
        **st,
        "p_le_m1_05": float((r <= -1.05).mean()),
        "tail_saved_R": ts,
        "recovery_lost_R": rl,
        "execution_cost_R": 0.0,
        "net_benefit_R": ts - rl,
        "n_triggered": int(hit.sum()),
        "recovery_preservation": float(1.0 - destroyed.sum() / g075.sum()) if g075.sum() else 1.0,
        "n_rec_mae075": int(g075.sum()),
        "n_rec_destroyed": int(destroyed.sum()),
        "recovery_preservation_mae1": float(1.0 - (g1 & (r <= 0)).sum() / g1.sum()) if g1.sum() else 1.0,
        "n_rec_mae1": int(g1.sum()),
        "n_rec_destroyed_mae1": int((g1 & (r <= 0)).sum()),
        "tail_loss_rate": float((r <= -1.00).mean()),
    }


def _write(verdict: str, extra: dict, report: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sprint49_report.md").write_text(report, encoding="utf-8")
    (OUT / "sprint49_final_verdict.json").write_text(
        json.dumps({"sprint": 49, "verdict": verdict, "production_changed": False,
                    "a1_shipped": False, **extra}, indent=2, default=float),
        encoding="utf-8",
    )
    print(f"FINAL VERDICT: {verdict}")
    print("PRODUCTION: UNCHANGED")
    print("Loss action: NOT SHIPPED")


def _score_expanding(obs: pd.DataFrame, model: str) -> np.ndarray:
    X = obs[MODEL_FEATS].astype(float).fillna(0.0).to_numpy()
    y = obs["large_loss_1"].astype(int).to_numpy()
    years = obs["year"].to_numpy()
    pred = np.full(len(obs), np.nan)
    for te, tr_years, _va in FOLDS:
        tr = np.isin(years, tr_years)
        te_m = years == te
        if tr.sum() < 80 or te_m.sum() < 20 or y[tr].min() == y[tr].max():
            continue
        pred[te_m] = _fit(model, X[tr], y[tr], X[te_m])
    m22, tr21 = years == 2022, years == 2021
    if tr21.sum() >= 80 and m22.sum() >= 20 and y[tr21].min() != y[tr21].max():
        pred[m22] = _fit(model, X[tr21], y[tr21], X[m22])
    # 2021 in-sample diagnostic only (not OOS)
    if tr21.sum() >= 80 and y[tr21].min() != y[tr21].max():
        pred[tr21] = _fit(model, X[tr21], y[tr21], X[tr21])
    return pred


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    h1 = load_h1(H1_PATH)
    mkt = prepare_market(h1)
    open_px = h1["open"].to_numpy(dtype=float)
    close_s, atr_s = mkt["close"], mkt["atr"]
    ema = pd.Series(close_s).ewm(span=20, adjust=False).mean().to_numpy()
    atr_pct_bar = _atr_pctile(atr_s)
    n_bars = len(open_px)

    panel = _join_feat7(_load_panel())
    p = Paths(panel, mkt)
    sim0 = simulate_combo(p, **P0)
    ei = entry_indices(p.panel, mkt["ts"]).astype(int)
    hold = sim0["holding_bars"].astype(int)
    reason = sim0["reason"].astype(int)
    r0_all = sim0["r_multiple"]
    mae_all = sim0["mae_r"]
    one_r = np.maximum(1.5 * p.atr, 1e-12)
    y_prob = p.panel["y_prob"].astype(float).to_numpy()
    f7 = p.panel[list(FEAT7)].astype(float).to_numpy()
    f7i = {name: k for k, name in enumerate(FEAT7)}
    hour0 = pd.DatetimeIndex(p.ts).hour.to_numpy()
    h4 = p.panel["ctx_h4_swing_quality"].astype(float).to_numpy() if "ctx_h4_swing_quality" in p.panel.columns else np.zeros(p.n)

    trades = pd.DataFrame({
        "trade_id": np.arange(p.n),
        "year": p.year,
        "p0_R": r0_all,
        "mae_R": mae_all,
        "holding_bars": hold,
        "exit_reason": [REASON.get(int(x), "TIMEOUT") for x in reason],
        "large_loss_1": (r0_all <= -1.00).astype(int),
    })
    oos = trades[trades["year"].isin(TRUE_OOS)].copy()
    print(f"oos trades={len(oos)} ref={REF_N_OOS}")
    gate1 = len(oos) == REF_N_OOS
    if not gate1:
        _write("RESEARCH_STOP", {"gate1": False, "n_oos": int(len(oos))},
               f"# Sprint 49\n\n**RESEARCH_STOP** universe {len(oos)} != {REF_N_OOS}\n")
        return 1

    # ----- Iter 1: timing audit -----
    hits = {c: [] for c in CHECKS}  # list of dicts
    ll_survived_bar1 = 0
    n_ll = 0
    for i in range(p.n):
        last = int(hold[i])
        ll = bool(r0_all[i] <= -1.00)
        if ll:
            n_ll += 1
            if last > 1:
                ll_survived_bar1 += 1
        seen = {c: False for c in CHECKS}
        prev_cr = 0.0
        prev_mom = 0.0
        consec = 0
        for j in range(1, last + 1):
            cr = p.close_r[i, j]
            if not np.isfinite(cr):
                continue
            mom = float(cr - prev_cr)
            if cr < prev_cr:
                consec += 1
            else:
                consec = 0
            accel = mom - prev_mom
            adv = float(p.adv[i, j]) if np.isfinite(p.adv[i, j]) else 0.0
            mfe = float(np.nanmax(p.fav[i, 1:j + 1]))
            mae = float(np.nanmax(p.adv[i, 1:j + 1]))
            executable = last > j  # survived this bar → next-open exists
            same_sl = (last == j) and (int(reason[i]) in STOP_REASONS)
            idx = int(ei[i] + j)
            nxt = idx + 1
            open_n = np.nan
            if 0 <= nxt < n_bars:
                open_n = _px_to_r(open_px[nxt], p.entry[i], one_r[i], bool(p.is_long[i]))
            hr = int((hour0[i] + j) % 24)
            sign = 1.0 if p.is_long[i] else -1.0
            ema_dist = sign * (close_s[idx] - ema[idx]) / max(float(p.atr[i]), 1e-12) if 0 <= idx < n_bars else 0.0
            feat = {
                "trade_id": i, "year": int(p.year[i]), "j": j, "current_R": float(cr),
                "executable": bool(executable), "same_bar_sl": bool(same_sl),
                "open_next_R": open_n, "bars_to_exit": int(last - j),
                "large_loss_1": int(ll), "p0_R": float(r0_all[i]), "mae_R": float(mae_all[i]),
                "y_prob": float(y_prob[i]),
                "atr_pct_entry": float(f7[i, f7i["atr_percentile_252"]]),
                "atr_pct_now": float(atr_pct_bar[idx]) if 0 <= idx < n_bars and np.isfinite(atr_pct_bar[idx]) else 0.5,
                "ema_dist_atr": float(ema_dist),
                "ema_trend_duration": float(f7[i, f7i["ema_trend_duration"]]),
                "rolling_quantile": float(f7[i, f7i["rolling_quantile"]]),
                "hour_sin": float(f7[i, f7i["hour_sin"]]),
                "hour_cos": float(f7[i, f7i["hour_cos"]]),
                "hour_sin_now": math.sin(2 * math.pi * hr / 24),
                "hour_cos_now": math.cos(2 * math.pi * hr / 24),
                "mfe_so_far_R": mfe, "mae_so_far_R": mae,
                "drawdown_from_MFE_R": mfe - float(cr),
                "mom_1R": mom, "bars_in_trade": j,
                "consec_adverse": consec, "range_R": adv, "cr_accel": accel,
                "ctx_h4_swing_quality": float(h4[i]),
            }
            for c in CHECKS:
                if (not seen[c]) and cr <= -c:
                    seen[c] = True
                    hits[c].append(feat)
            prev_cr, prev_mom = float(cr), mom

    rows1 = []
    for c in CHECKS:
        df = pd.DataFrame(hits[c])
        for scope, sub in (("all", df), ("oos", df[df["year"].isin(TRUE_OOS)] if len(df) else df),
                           ("oos_ll", df[(df["year"].isin(TRUE_OOS)) & (df["large_loss_1"] == 1)] if len(df) else df),
                           ("y2021_ll", df[(df["year"] == 2021) & (df["large_loss_1"] == 1)] if len(df) else df)):
            if sub is None or not len(sub):
                rows1.append({"checkpoint": c, "scope": scope, "n": 0, "executable_pct": 0.0,
                              "same_bar_sl_pct": 1.0, "median_bars_to_sl": 0.0, "median_current_R": 0.0,
                              "median_action_R": 0.0, "median_R_to_sl": 0.0})
                continue
            exe = sub["executable"].to_numpy(dtype=bool)
            bte = sub.loc[exe, "bars_to_exit"].to_numpy() if exe.any() else np.array([0.0])
            nxt = sub.loc[exe, "open_next_R"].to_numpy(dtype=float) if exe.any() else np.array([np.nan])
            rows1.append({
                "checkpoint": c, "scope": scope, "n": int(len(sub)),
                "executable_pct": float(exe.mean()),
                "same_bar_sl_pct": float(sub["same_bar_sl"].mean()),
                "median_bars_to_sl": float(np.median(bte)) if len(bte) else 0.0,
                "median_current_R": float(sub["current_R"].median()),
                "median_action_R": float(np.nanmedian(nxt)) if np.isfinite(nxt).any() else 0.0,
                "median_R_to_sl": float(np.median(-1.0 - sub.loc[exe, "current_R"])) if exe.any() else 0.0,
            })
    pd.DataFrame(rows1).to_csv(OUT / "sprint49_iter1_timing_audit.csv", index=False)

    oos_ll_n = int(((oos["large_loss_1"] == 1)).sum())
    oos_ll_surv = float(ll_survived_bar1 / n_ll) if n_ll else 0.0
    # gate on OOS large-loss at each ck
    oos_ll_rows = [r for r in rows1 if r["scope"] == "oos_ll"]
    best_exec = max((r["executable_pct"] for r in oos_ll_rows), default=0.0)
    y21_ll = [r for r in rows1 if r["scope"] == "y2021_ll"]
    ck_star = None
    for r in sorted(y21_ll, key=lambda x: x["checkpoint"]):
        if r["n"] >= 20 and r["executable_pct"] >= 0.30 and r["same_bar_sl_pct"] <= 0.60:
            ck_star = r["checkpoint"]
            break
    if ck_star is None:
        cand = [r for r in y21_ll if r["n"] >= 20 and r["executable_pct"] >= 0.20]
        if cand:
            ck_star = max(cand, key=lambda x: x["executable_pct"])["checkpoint"]

    window = best_exec >= 0.25 and any(r["executable_pct"] >= 0.25 and r["same_bar_sl_pct"] <= 0.75 for r in oos_ll_rows)
    _log(1, "execution-timing audit", "PASS" if window else "FAIL",
         f"oos_ll_survived_bar1={oos_ll_surv:.2f} best_exec={best_exec:.2f} ck_star={ck_star}",
         2 if window else "STOP")
    if not window or ck_star is None:
        _write("NO_ACTIONABLE_WINDOW", {
            "gate1": True, "gate2": False, "best_executable_pct": best_exec, "ck_star": ck_star,
            "oos_ll_n": oos_ll_n,
        }, "\n".join([
            "# Sprint 49 — Executable Early Loss Suppression",
            "", "## 1. Executive Verdict",
            "", "Question: Can we reduce P0 tail losses using an executable PIT deterioration signal?",
            "", "**Verdict: `NO_ACTIONABLE_WINDOW`**",
            "", f"OOS large-loss n={oos_ll_n}. Best checkpoint executable_pct={best_exec:.2f}. ck_star={ck_star}.",
            "Useful deterioration is too close to P0 SL for a realistic next-open action.",
            "", "## 12. Production Decision",
            "", "Entry: UNCHANGED", "", "P0: UNCHANGED", "", "Loss action: NOT SHIPPED",
            "", "Reversal: NOT USED", "", "P1/P2/P3: NOT USED",
            "", "## 13. Next Step", "", "KEEP_P0_AND_STOP_OR_REFRAME_LOSS_RESEARCH",
        ]))
        return 0

    # ----- Iter 2/3: obs at first executable ck_star -----
    raw = pd.DataFrame(hits[ck_star])
    obs = raw[raw["executable"]].copy().drop_duplicates("trade_id")
    obs["p_loss"] = np.nan
    pred_lr = _score_expanding(obs, "lr")
    pred_lgb = _score_expanding(obs, "lgbm")
    m21 = obs["year"].to_numpy() == 2021
    auc21_lr, auc21_lgb = _roc(obs.loc[m21, "large_loss_1"], pred_lr[m21]), _roc(obs.loc[m21, "large_loss_1"], pred_lgb[m21])
    model_name = "lgbm" if auc21_lgb >= auc21_lr else "lr"
    obs["p_loss"] = pred_lgb if model_name == "lgbm" else pred_lr
    oos_obs = obs[obs["year"].isin(TRUE_OOS) & np.isfinite(obs["p_loss"])].copy()
    y_o, p_o = oos_obs["large_loss_1"].to_numpy(dtype=int), oos_obs["p_loss"].to_numpy()
    roc, pr = _roc(y_o, p_o), _pr(y_o, p_o)
    q = pd.qcut(oos_obs["p_loss"], 5, duplicates="drop")
    bucket = oos_obs.groupby(q, observed=False)["large_loss_1"].agg(["mean", "size"]).reset_index()
    top_ll = float(bucket["mean"].iloc[-1]) if len(bucket) else 0.0
    bot_ll = float(bucket["mean"].iloc[0]) if len(bucket) else 0.0
    sep = top_ll - bot_ll
    pd.DataFrame([{"model": model_name, "auc_2021_lr": auc21_lr, "auc_2021_lgb": auc21_lgb,
                   "roc_oos": roc, "pr_oos": pr, "top_ll": top_ll, "bottom_ll": bot_ll, "sep": sep,
                   "n_obs_oos": int(len(oos_obs)), "base_ll": float(y_o.mean()) if len(y_o) else 0.0,
                   "ck_star": ck_star, "thr": THR}]).to_csv(OUT / "sprint49_iter3_model.csv", index=False)
    gate3 = roc >= 0.60 and sep >= 0.10
    _log(2, "deterioration state", "PASS", f"frozen label LARGE_LOSS=P0_R<=-1; ck={ck_star}; n_exec={len(obs)}", 3)
    _log(3, "model/separation", "PASS" if gate3 else "FAIL",
         f"model={model_name} roc={roc:.3f} pr={pr:.3f} top={top_ll:.3f} bot={bot_ll:.3f}",
         4 if gate3 else "STOP")
    if not gate3:
        _write("NO_PREDICTIVE_LOSS_SIGNAL", {
            "gate1": True, "gate2": True, "gate3": False, "roc": roc, "sep": sep, "ck_star": ck_star,
        }, "\n".join([
            "# Sprint 49 — Executable Early Loss Suppression",
            "", "## 1. Executive Verdict",
            "", "**Verdict: `NO_PREDICTIVE_LOSS_SIGNAL`**",
            "", f"Actionable window at ck=-{ck_star:.2f}R exists, but PIT separation is weak (ROC={roc:.3f}, top-bot={sep:.3f}).",
            "", "## 12. Production Decision",
            "", "Entry: UNCHANGED", "", "P0: UNCHANGED", "", "Loss action: NOT SHIPPED",
            "", "## 13. Next Step", "", "KEEP_P0_AND_STOP_OR_REFRAME_LOSS_RESEARCH",
        ]))
        return 0

    # ----- Iter 4: actionable filter at frozen thr -----
    flagged = oos_obs[oos_obs["p_loss"] >= THR].copy()
    act_pct = float(flagged["executable"].mean()) if len(flagged) else 0.0
    rows4 = []
    if len(oos_obs):
        oos_obs = oos_obs.copy()
        oos_obs["q"] = pd.qcut(oos_obs["p_loss"], 5, duplicates="drop", labels=False)
        for qn, g in oos_obs.groupby("q"):
            rows4.append({
                "risk_bucket": int(qn) + 1, "n": int(len(g)),
                "large_loss_rate": float(g["large_loss_1"].mean()),
                "actionable_pct": float(g["executable"].mean()),
                "median_action_R": float(np.nanmedian(g["open_next_R"])),
                "median_R_to_SL": float(np.median(-1.0 - g["current_R"])),
                "median_current_R": float(g["current_R"].median()),
            })
    pd.DataFrame(rows4).to_csv(OUT / "sprint49_iter4_actionable.csv", index=False)
    gate4_timing = len(flagged) >= 30 and act_pct >= 0.80
    _log(4, "actionable window filter", "PASS" if gate4_timing else "FAIL",
         f"flagged={len(flagged)} actionable_pct={act_pct:.2f} (obs already executable-only)",
         5 if gate4_timing else "STOP")
    if not gate4_timing:
        _write("NO_ACTIONABLE_WINDOW", {
            "gate1": True, "gate2": True, "gate3": True, "gate4": False,
            "n_flagged": int(len(flagged)), "ck_star": ck_star,
        }, "\n".join([
            "# Sprint 49", "", "**Verdict: `NO_ACTIONABLE_WINDOW`**",
            f"Flagged n={len(flagged)} actionable_pct={act_pct:.2f} at p>={THR}.",
            "", "Entry: UNCHANGED", "", "P0: UNCHANGED", "", "Loss action: NOT SHIPPED",
        ]))
        return 0

    # ----- Iter 5–6: actions with next-open fill -----
    trig = {}
    for row in flagged.itertuples(index=False):
        tid = int(row.trade_id)
        if tid in trig:
            continue
        fill = float(row.open_next_R) if np.isfinite(row.open_next_R) else None
        if fill is None:
            continue
        trig[tid] = {"j": int(row.j), "fill": fill, "year": int(row.year), "current_R": float(row.current_R)}
    tick1, tick2, tickc = _tick_r(p, 1.0), _tick_r(p, 2.0), _tick_r(p, CONS_PTS)

    def _packs(tr_map, extra_tick):
        r = np.array(sim0["r_multiple"], dtype=float)
        h = np.array(sim0["holding_bars"], dtype=np.int64)
        ru = np.maximum(p.r_unit_pct, 1e-12)
        out = {a: {"r": r.copy(), "hold": h.copy()} for a in ("A0", "A1", "A2", "A3", "A4")}
        for i, inf in tr_map.items():
            j, fill = int(inf["j"]), float(inf["fill"])
            last = int(h[i])
            g = fill - float(extra_tick[i])
            g0 = r[i] + COST / ru[i]
            out["A1"]["r"][i] = _net(g, ru[i]); out["A1"]["hold"][i] = max(j + 1, 1)
            out["A2"]["r"][i] = _net(0.50 * g + 0.50 * g0, ru[i])
            out["A3"]["r"][i] = _net(0.75 * g + 0.25 * g0, ru[i])
            g4, h4 = _a4_next(p, i, j, last, A4_SL - float(extra_tick[i]))
            out["A4"]["r"][i] = _net(g4, ru[i]); out["A4"]["hold"][i] = max(h4, 1)
        return out

    packs = _packs(trig, np.zeros(p.n))
    r0 = oos["p0_R"].to_numpy()
    rows6 = []
    for name in ("A0", "A1", "A2", "A3", "A4"):
        rr = np.array([packs[name]["r"][int(t)] for t in oos["trade_id"]])
        oos[f"{name}_R"] = rr
        e = _econ(rr, r0, oos, trig if name != "A0" else {})
        if name == "A0":
            e["net_benefit_R"] = 0.0
            e["tail_saved_R"] = 0.0
            e["recovery_lost_R"] = 0.0
        rows6.append({"action": name, **e})
    pd.DataFrame(rows6).to_csv(OUT / "sprint49_iter6_actions.csv", index=False)
    surv = [r["action"] for r in rows6 if r["action"] != "A0" and r["net_benefit_R"] > 0]
    best = max((r for r in rows6 if r["action"] != "A0"), key=lambda x: x["net_benefit_R"])["action"]
    gate4 = rows6[[i for i, r in enumerate(rows6) if r["action"] == best][0]]["net_benefit_R"] > 0
    _log(5, "action design", "done", f"family A0–A4; best={best} surv={surv}", 6)
    _log(6, "realistic economics", "PASS" if gate4 else "FAIL",
         f"best={best} net={next(r['net_benefit_R'] for r in rows6 if r['action']==best):.2f}",
         7 if gate4 else "STOP")
    if not gate4:
        _write("ACTIONABLE_SIGNAL_BUT_ECONOMICALLY_WEAK", {
            "gate1": True, "gate2": True, "gate3": True, "gate4": False,
            "best": best, "net": next(r["net_benefit_R"] for r in rows6 if r["action"] == best),
            "ck_star": ck_star, "thr": THR, "model": model_name, "roc": roc,
        }, _report_weak(rows6, roc, pr, top_ll, bot_ll, ck_star, model_name, flagged, oos_ll_rows))
        return 0

    # ----- Iter 7 recovery -----
    best_row = next(r for r in rows6 if r["action"] == best)
    gate5 = best_row["recovery_preservation"] >= 0.80
    rec_rows = [{
        "action": r["action"], "recoverable_trades": r["n_rec_mae075"],
        "incorrectly_cut": r["n_rec_destroyed"], "recovery_lost_R": r["recovery_lost_R"],
        "recovery_preservation": r["recovery_preservation"],
        "recovery_preservation_mae1": r["recovery_preservation_mae1"],
    } for r in rows6]
    pd.DataFrame(rec_rows).to_csv(OUT / "sprint49_iter7_recovery.csv", index=False)
    _log(7, "recovery protection", "PASS" if gate5 else "FAIL",
         f"pres={best_row['recovery_preservation']:.3f} mae1={best_row['recovery_preservation_mae1']:.3f}",
         8 if gate5 else "STOP")
    if not gate5:
        _write("ACTIONABLE_SIGNAL_BUT_ECONOMICALLY_WEAK", {
            "gate5": False, "best": best, "preservation": best_row["recovery_preservation"],
        }, "# Sprint 49\n\n**Recovery preservation < 80%. FAIL.**\n\nEntry: UNCHANGED\nP0: UNCHANGED\n")
        return 0

    # ----- Iter 8 yearly + stress + placebo + loo + attr -----
    year_rows = []
    port_map = {}
    for name in ("A0", best):
        sim = _sim_from(sim0, packs[name], p)
        yrows = []
        for y in TRUE_OOS:
            port = _port(p, sim, y)
            yr = _year_row(y, port, sim)
            yrows.append(yr)
        po = _pooled(yrows)
        port_map[name] = {"po": po, "yrows": yrows, "mc": _mc(po["pnl"], name)}
        rmap = dict(zip(oos["trade_id"].astype(int), oos[f"{name}_R"]))
        for y in TRUE_OOS:
            ty = oos[oos["year"] == y]
            r = np.array([rmap[int(t)] for t in ty["trade_id"]])
            b = ty["p0_R"].to_numpy()
            hit = np.array([int(t) in trig for t in ty["trade_id"]])
            ll = ty["large_loss_1"].to_numpy(dtype=bool)
            rec = b > 0
            ts = float(((r - b)[ll & hit]).sum()) if (ll & hit).any() else 0.0
            rl = float(((b - r)[rec & hit]).sum()) if (rec & hit).any() else 0.0
            g075 = (ty["mae_R"].to_numpy() >= 0.75) & rec
            pres = float(1.0 - ((g075 & (r <= 0)).sum() / g075.sum())) if g075.sum() else 1.0
            yw = next(x for x in yrows if x["year"] == y)
            year_rows.append({
                "year": y, "action": name, "trades": int(len(ty)), "trigger_count": int(hit.sum()),
                "large_loss_rate": float((r <= -1).mean()), "P0_tail_rate": float(ll.mean()),
                "tail_saved_R": ts, "recovery_lost_R": rl, "execution_cost_R": 0.0,
                "net_benefit_R": 0.0 if name == "A0" else ts - rl,
                "PF": yw["pf"], "DD": yw["dd"], "recovery_preservation": pres,
            })
    ydf = pd.DataFrame(year_rows)
    ydf.to_csv(OUT / "sprint49_iter8_yearly.csv", index=False)
    yb = ydf[ydf["action"] == best]
    pos_years = int((yb["net_benefit_R"] > 0).sum())
    tot = float(yb["net_benefit_R"].sum())
    max_share = float((yb["net_benefit_R"] / tot).abs().max()) if abs(tot) > 1e-9 else 1.0
    gate6 = pos_years >= 4 and max_share <= 0.50

    stress = [("BASE", np.zeros(p.n)), ("+1tick", tick1), ("+2tick", tick2),
              ("spread_slip", tickc), ("next_open_cons", tickc)]
    rows_st = []
    for lab, tk in stress:
        pk = _packs(trig, tk)
        rr = np.array([pk[best]["r"][int(t)] for t in oos["trade_id"]])
        e = _econ(rr, r0, oos, trig)
        sim = _sim_from(sim0, pk[best], p)
        yrows = [_year_row(y, _port(p, sim, y), sim) for y in TRUE_OOS]
        po = _pooled(yrows)
        rows_st.append({"stress": lab, "action": best, "pf": po["pf"], "dd": po["dd"],
                        "net_benefit_R": e["net_benefit_R"], "tail_saved_R": e["tail_saved_R"],
                        "recovery_lost_R": e["recovery_lost_R"], "tail_rate": e["tail_loss_rate"]})
    pd.DataFrame(rows_st).to_csv(OUT / "sprint49_iter8_stress.csv", index=False)
    gate7 = all(r["net_benefit_R"] > 0 for r in rows_st)

    # placebo among executable ck_star OOS trades
    rng = np.random.default_rng(42)
    elig = obs[(obs["year"].isin(TRUE_OOS))].drop_duplicates("trade_id")
    elig_by_y = elig.groupby("year")["trade_id"].apply(lambda s: [int(x) for x in s]).to_dict()
    n_by_y = {y: sum(1 for t, inf in trig.items() if inf["year"] == y) for y in TRUE_OOS}
    fill_ck = {int(t): float(r) for t, r in zip(elig["trade_id"], elig["open_next_R"])}
    j_ck = {int(t): int(j) for t, j in zip(elig["trade_id"], elig["j"])}
    actual_net = best_row["net_benefit_R"]
    actual_ts = best_row["tail_saved_R"]
    plc_net, plc_ts = [], []
    for _ in range(N_PLACEBO):
        fake = {}
        for y, n in n_by_y.items():
            pool = elig_by_y.get(y, [])
            if n <= 0 or not pool:
                continue
            pick = rng.choice(pool, size=min(n, len(pool)), replace=False)
            for tid in pick:
                if not np.isfinite(fill_ck.get(int(tid), np.nan)):
                    continue
                fake[int(tid)] = {"j" : j_ck.get(int(tid), 1), "fill": fill_ck[int(tid)], "year": y, "current_R": 0.0}
        pk = _packs(fake, np.zeros(p.n))
        rr = np.array([pk[best]["r"][int(t)] for t in oos["trade_id"]])
        st = _econ(rr, r0, oos, fake)
        plc_net.append(st["net_benefit_R"]); plc_ts.append(st["tail_saved_R"])
    plc_net, plc_ts = np.asarray(plc_net), np.asarray(plc_ts)
    p_net = float((plc_net >= actual_net).mean())
    p_ts = float((plc_ts >= actual_ts).mean())
    pd.DataFrame([{"actual_net": actual_net, "placebo_mean": float(plc_net.mean()),
                   "placebo_sd": float(plc_net.std()), "p_net": p_net,
                   "actual_ts": actual_ts, "placebo_ts_mean": float(plc_ts.mean()), "p_ts": p_ts,
                   "n_rep": N_PLACEBO}]).to_csv(OUT / "sprint49_placebo.csv", index=False)

    loo = []
    rb = oos[f"{best}_R"].to_numpy()
    years_a = oos["year"].to_numpy()
    hit_all = np.array([int(t) in trig for t in oos["trade_id"]])
    ll_all = oos["large_loss_1"].to_numpy(dtype=bool)
    rec_all = r0 > 0
    for drop in TRUE_OOS:
        m = years_a != drop
        ts = float(((rb - r0)[m & ll_all & hit_all]).sum())
        rl = float(((r0 - rb)[m & rec_all & hit_all]).sum())
        loo.append({"exclude_year": drop, "net_benefit_R": ts - rl, "tail_saved_R": ts, "recovery_lost_R": rl})
    pd.DataFrame(loo).to_csv(OUT / "sprint49_leave_one_year_out.csv", index=False)
    gate8 = p_net < 0.01 and all(x["net_benefit_R"] >= 0 for x in loo)

    attr = []
    tot_net = actual_net if abs(actual_net) > 1e-9 else 1.0
    for lab, lo, hi in MAE_BINS:
        sub = oos[(oos["mae_R"] >= lo) & (oos["mae_R"] < hi)]
        if not len(sub):
            continue
        r = sub[f"{best}_R"].to_numpy(); b = sub["p0_R"].to_numpy()
        hit = np.array([int(t) in trig for t in sub["trade_id"]])
        ll = sub["large_loss_1"].to_numpy(dtype=bool); recb = b > 0
        ts = float(((r - b)[ll & hit]).sum()) if (ll & hit).any() else 0.0
        rl = float(((b - r)[recb & hit]).sum()) if (recb & hit).any() else 0.0
        attr.append({"bucket": lab, "n": int(len(sub)), "tail_saved_R": ts, "recovery_lost_R": rl,
                     "net_R": ts - rl, "share_of_total_net": (ts - rl) / tot_net})
    pd.DataFrame(attr).to_csv(OUT / "sprint49_attribution.csv", index=False)

    _log(8, "oos+stress+placebo", "PASS" if (gate6 and gate7 and gate8) else "FAIL",
         f"pos_years={pos_years}/5 p_net={p_net:.4f} loo_ok={all(x['net_benefit_R']>=0 for x in loo)}", "FINAL")

    if gate6 and gate7 and gate8:
        verdict = "ACTIONABLE_LOSS_SUPPRESSION_CONFIRMED"
        nxt = "CANDIDATE_FOR_PAPER_EXECUTION_VALIDATION"
    elif (not gate6) or (not gate8):
        verdict = "ACTIONABLE_SIGNAL_BUT_ECONOMICALLY_WEAK"
        nxt = "KEEP_P0_AND_STOP_OR_REFRAME_LOSS_RESEARCH"
    elif not gate7:
        verdict = "EXECUTION_INVALIDATED"
        nxt = "KEEP_P0_AND_STOP_OR_REFRAME_LOSS_RESEARCH"
    else:
        verdict = "ACTIONABLE_SIGNAL_BUT_ECONOMICALLY_WEAK"
        nxt = "KEEP_P0_AND_STOP_OR_REFRAME_LOSS_RESEARCH"

    a0p, bp = port_map["A0"]["po"], port_map[best]["po"]
    report = _final_report(
        verdict, nxt, ck_star, model_name, roc, pr, top_ll, bot_ll, sep,
        rows1, rows4, rows6, yb, rows_st, rec_rows, attr, loo,
        actual_net, float(plc_net.mean()), float(plc_net.std()), p_net,
        best, a0p, bp, port_map[best]["mc"], port_map["A0"]["mc"],
        gate1, window, gate3, gate4, gate5, gate6, gate7, gate8,
        len(flagged), act_pct, pos_years, max_share, THR,
    )
    _write(verdict, {
        "gate1": True, "gate2": bool(window), "gate3": bool(gate3), "gate4": bool(gate4),
        "gate5": bool(gate5), "gate6": bool(gate6), "gate7": bool(gate7), "gate8": bool(gate8),
        "ck_star": ck_star, "thr": THR, "model": model_name, "best_action": best,
        "net_benefit_R": actual_net, "roc": roc, "placebo_p": p_net,
        "data_availability": "EXECUTION_DATA_LIMITED",
        "recommendation": nxt,
    }, report)
    return 0


def _report_weak(rows6, roc, pr, top_ll, bot_ll, ck_star, model_name, flagged, oos_ll_rows) -> str:
    return "\n".join([
        "# Sprint 49 — Executable Early Loss Suppression",
        "", "## 1. Executive Verdict",
        "", "Question: Can we reduce P0 tail losses using an executable PIT deterioration signal?",
        "", "**Verdict: `ACTIONABLE_SIGNAL_BUT_ECONOMICALLY_WEAK`**",
        "", f"ck=-{ck_star:.2f} model={model_name} ROC={roc:.3f} PR={pr:.3f} top/bot LL={top_ll:.3f}/{bot_ll:.3f}",
        f"flagged={len(flagged)}",
        "", _md(rows6, ["action", "mean_R", "pf", "tail_loss_rate", "tail_saved_R", "recovery_lost_R",
                        "net_benefit_R", "recovery_preservation"],
                {"mean_R": 3, "pf": 2, "tail_loss_rate": 3, "tail_saved_R": 2, "recovery_lost_R": 2,
                 "net_benefit_R": 2, "recovery_preservation": 3}),
        "", "## 12. Production Decision",
        "", "Entry: UNCHANGED", "", "P0: UNCHANGED", "", "Loss action: NOT SHIPPED",
        "", "Reversal: NOT USED", "", "P1/P2/P3: NOT USED",
        "", "## 13. Next Step", "", "KEEP_P0_AND_STOP_OR_REFRAME_LOSS_RESEARCH",
    ])


def _final_report(verdict, nxt, ck, model, roc, pr, top, bot, sep, rows1, rows4, rows6, yb,
                  rows_st, rec_rows, attr, loo, actual_net, plc_mean, plc_sd, p_net,
                  best, a0p, bp, mc_b, mc_0, g1, g2, g3, g4, g5, g6, g7, g8,
                  n_flag, act_pct, pos_years, max_share, thr) -> str:
    oos_ll = [r for r in rows1 if r["scope"] == "oos_ll"]
    return "\n".join([
        "# Sprint 49 — Executable Early Loss Suppression",
        "", "## 1. Executive Verdict",
        "", "Question: Can we reduce P0 tail losses using an executable PIT deterioration signal?",
        "", f"**Verdict: `{verdict}`**",
        "", "No retrospective −0.50 fill. Next-open only if the trade survived the signal bar.",
        "", "## 2. Actionable Window",
        "", f"Frozen observation (selected on 2021 LL only): first current_R ≤ **−{ck:.2f}R**, survived bar, next-open fill.",
        f"Flagged p≥{thr:.2f}: n={n_flag}, actionable_pct={act_pct:.2f} (universe is executable-only).",
        "", _md(oos_ll, ["checkpoint", "n", "executable_pct", "same_bar_sl_pct", "median_bars_to_sl",
                         "median_current_R", "median_action_R"],
                {"checkpoint": 2, "n": 0, "executable_pct": 3, "same_bar_sl_pct": 3,
                 "median_bars_to_sl": 2, "median_current_R": 3, "median_action_R": 3}),
        "", "## 3. Predictive Signal",
        "", f"NEW model (not Sprint 46 retune): **{model}** expanding WF. ROC={roc:.3f} PR={pr:.3f} top LL={top:.3f} bottom LL={bot:.3f} sep={sep:.3f}.",
        "", "## 4. Action Comparison",
        "", _md(rows6, ["action", "mean_R", "pf", "tail_loss_rate", "tail_saved_R", "recovery_lost_R",
                        "execution_cost_R", "net_benefit_R", "recovery_preservation"],
                {"mean_R": 3, "pf": 2, "tail_loss_rate": 3, "tail_saved_R": 2, "recovery_lost_R": 2,
                 "execution_cost_R": 2, "net_benefit_R": 2, "recovery_preservation": 3}),
        f"", f"Best by net: **{best}**. Portfolio P0 PF={a0p['pf']:.2f} DD={a0p['dd']:.3f} → {best} PF={bp['pf']:.2f} DD={bp['dd']:.3f}.",
        "", "## 5. Yearly OOS",
        "", _md(yb.to_dict("records"), ["year", "trades", "trigger_count", "large_loss_rate", "net_benefit_R",
                                        "PF", "DD", "recovery_preservation"],
                {"year": 0, "trades": 0, "trigger_count": 0, "large_loss_rate": 3, "net_benefit_R": 2,
                 "PF": 2, "DD": 3, "recovery_preservation": 3}),
        f"", f"positive_net_years={pos_years}/5 max_year_share={max_share:.2f}",
        "", "## 6. Execution Stress",
        "", "Tick/bid-ask path absent (`EXECUTION_DATA_LIMITED`). Conservative points = 19+5.",
        "", _md(rows_st, ["stress", "pf", "dd", "net_benefit_R", "tail_rate"],
                {"pf": 2, "dd": 3, "net_benefit_R": 2, "tail_rate": 3}),
        "", "## 7. Recovery Protection",
        "", _md(rec_rows, ["action", "recoverable_trades", "incorrectly_cut", "recovery_lost_R",
                           "recovery_preservation", "recovery_preservation_mae1"],
                {"recoverable_trades": 0, "incorrectly_cut": 0, "recovery_lost_R": 2,
                 "recovery_preservation": 3, "recovery_preservation_mae1": 3}),
        "", "## 8. Attribution",
        "", _md(attr, ["bucket", "n", "tail_saved_R", "recovery_lost_R", "net_R", "share_of_total_net"],
                {"n": 0, "tail_saved_R": 2, "recovery_lost_R": 2, "net_R": 2, "share_of_total_net": 3}),
        "", "## 9. Placebo",
        "", f"actual net={actual_net:.2f} placebo mean={plc_mean:.2f} sd={plc_sd:.2f} empirical p={p_net:.4f} (n=1000)",
        "", "## 10. Leave-One-Year-Out",
        "", _md(loo, ["exclude_year", "net_benefit_R", "tail_saved_R", "recovery_lost_R"],
                {"exclude_year": 0, "net_benefit_R": 2, "tail_saved_R": 2, "recovery_lost_R": 2}),
        "", "## 11. Leakage Audit",
        "", "- no future labels in features",
        "- no future MAE/MFE",
        "- no final outcome in trigger (p from expanding WF, frozen thr)",
        "- no retrospective fill (next-open only; never max(current_R, −ck))",
        "- no same-bar SL rewrite (require last > j)",
        "- Sprint 46/47 model not retrained; not used as the action trigger",
        "- 2021 used only to freeze checkpoint + pick LR vs LGBM; not as OOS claim",
        "", "## 12. Production Decision",
        "", "Entry: UNCHANGED", "", "P0: UNCHANGED", "", "Loss action: NOT SHIPPED unless ALL gates pass",
        "", "Reversal: NOT USED", "", "P1/P2/P3: NOT USED",
        "", "## Gate Results",
        "", f"- G1 data: {'PASS' if g1 else 'FAIL'}",
        f"- G2 timing: {'PASS' if g2 else 'FAIL'}",
        f"- G3 separation: {'PASS' if g3 else 'FAIL'}",
        f"- G4 economics: {'PASS' if g4 else 'FAIL'}",
        f"- G5 recovery: {'PASS' if g5 else 'FAIL'}",
        f"- G6 yearly: {'PASS' if g6 else 'FAIL'}",
        f"- G7 stress: {'PASS' if g7 else 'FAIL'}",
        f"- G8 placebo/LOO: {'PASS' if g8 else 'FAIL'}",
        "", "## 13. Next Step",
        "", nxt,
    ])


if __name__ == "__main__":
    raise SystemExit(main())
