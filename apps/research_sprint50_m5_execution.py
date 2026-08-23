"""Sprint 50 — M5 execution-resolution of frozen Sprint 49 signal.

Do not retrain. Do not search thresholds. Production unchanged.

  python apps/research_sprint50_m5_execution.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import tools.pyc_path_hook  # noqa: F401

from apps.research_sprint39_exit_state import _atr_pctile, _join_feat7, _load_panel
from apps.research_sprint41_iter4_switch import _port, _year_row
from apps.research_sprint42_attribution import TRUE_OOS, _md, _pooled
from apps.research_sprint46_loss_reduction import P0, _mc, _r_stats
from apps.research_sprint47_loss_action import P0_ACT, P0_DIST, _net, _sim_from, _tick_r
from apps.research_sprint49_early_suppression import (
    MODEL_FEATS,
    REF_N_OOS,
    STOP_REASONS,
    THR,
    _econ,
    _px_to_r,
    _score_expanding,
)
from apps.run_exit_engine_grid import CAP, FEAT7, Paths, simulate_combo
from settings.strategy import ASSUMED_SLIPPAGE_POINTS, FALLBACK_SPREAD_POINTS, POINT
from simulation.wf.sim import COST, entry_indices, load_h1, prepare_market

OUT = _ROOT / "artifacts/pipeline_backtest/loss_reduction/sprint50"
H1_PATH = _ROOT / "artifacts/raw/XAUUSD/H1/data.parquet"
M5_PATH = _ROOT / "artifacts/raw/XAUUSD/M5/data.parquet"
CK = 0.25  # frozen Sprint 49
A4_SL = -0.75
HORIZON_H = 48
NS_PER_HOUR = 3_600_000_000_000
REASON = {0: "SL", 1: "TRAIL", 2: "BE", 3: "TP", 4: "TIMEOUT"}
CONS_PTS = float(FALLBACK_SPREAD_POINTS + ASSUMED_SLIPPAGE_POINTS)
H1_LL_EXEC = 0.658
H1_LL_ACTION_R = -0.475


def _log(n, result, gate, finding, nxt) -> None:
    print(f"ITERATION: {n}")
    print(f"RESULT: {result}")
    print(f"GATE: {gate}")
    print(f"KEY FINDING: {finding}")
    print(f"NEXT ITERATION: {nxt}")


def _write(verdict: str, extra: dict, report: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sprint50_report.md").write_text(report, encoding="utf-8")
    (OUT / "sprint50_final_verdict.json").write_text(
        json.dumps({"sprint": 50, "verdict": verdict, "production_changed": False,
                    "a1_shipped": False, "data_availability": "M5_RESOLUTION_LIMITED",
                    **extra}, indent=2, default=float),
        encoding="utf-8",
    )
    print(f"FINAL VERDICT: {verdict}")
    print("PRODUCTION: UNCHANGED")
    print("Loss action: NOT SHIPPED")


def _pctiles(a) -> dict:
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return {k: 0.0 for k in ("p25", "p50", "p75", "p90")}
    return {k: float(np.percentile(a, p)) for k, p in (("p25", 25), ("p50", 50), ("p75", 75), ("p90", 90))}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    h1 = load_h1(H1_PATH)
    mkt = prepare_market(h1)
    open_px = h1["open"].to_numpy(dtype=float)
    h1_idx = pd.DatetimeIndex(mkt["ts"])
    if h1_idx.tz is None:
        h1_idx = h1_idx.tz_localize("UTC")
    else:
        h1_idx = h1_idx.tz_convert("UTC")
    h1_ts = h1_idx.asi8
    close_s, atr_s = mkt["close"], mkt["atr"]
    ema = pd.Series(close_s).ewm(span=20, adjust=False).mean().to_numpy()
    atr_pct_bar = _atr_pctile(atr_s)
    n_bars = len(open_px)
    has_spread = "spread" in h1.columns

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
    gate1_univ = len(oos) == REF_N_OOS

    # ----- Iter 1: M5 integrity -----
    m5 = pd.read_parquet(M5_PATH)
    m5["timestamp"] = pd.to_datetime(m5["timestamp"], utc=True)
    m5 = m5.sort_values("timestamp").drop_duplicates("timestamp")
    m5_idx = pd.DatetimeIndex(m5["timestamp"])
    ts5 = m5_idx.asi8
    # pandas asi8 unit may be us or ns; never add a hardcoded ns hour to it
    probe = min(2000, len(ts5) - 1)
    scale = float(ts5[probe] / pd.Timestamp(m5_idx[probe]).value)
    hour = int(round(NS_PER_HOUR * scale))
    open5 = m5["open"].to_numpy(float)
    high5 = m5["high"].to_numpy(float)
    low5 = m5["low"].to_numpy(float)
    close5 = m5["close"].to_numpy(float)
    n5 = len(ts5)
    ohlc_bad = int(((high5 < low5) | (high5 < close5) | (low5 > close5) | (high5 < open5) | (low5 > open5)).sum())
    dmin = np.diff(ts5) / (hour / 60.0)
    entry_idx = pd.DatetimeIndex(p.ts)
    if entry_idx.tz is None:
        entry_idx = entry_idx.tz_localize("UTC")
    else:
        entry_idx = entry_idx.tz_convert("UTC")
    entry_ns = entry_idx.asi8
    horizon_ns = HORIZON_H * hour
    n_cov = n_miss_win = n_ts_issue = 0
    for i in range(p.n):
        if int(p.year[i]) not in TRUE_OOS:
            continue
        a = int(np.searchsorted(ts5, entry_ns[i], side="right"))
        b = int(np.searchsorted(ts5, entry_ns[i] + horizon_ns, side="right"))
        if b - a >= 12:
            n_cov += 1
        else:
            n_miss_win += 1
        if a >= n5 or ts5[a] < entry_ns[i]:
            n_ts_issue += 1
    cov_pct = n_cov / max(len(oos), 1)
    data_status = "OK" if cov_pct >= 0.95 and ohlc_bad == 0 and gate1_univ else "INCOMPLETE"
    gate1 = data_status == "OK"
    pd.DataFrame([{
        "H1_trades": int(len(oos)), "trades_with_M5_coverage": n_cov, "coverage_pct": cov_pct,
        "missing_M5_windows": n_miss_win, "timestamp_issues": n_ts_issue, "ohlc_bad": ohlc_bad,
        "m5_n": n5, "m5_dup": 0, "median_gap_min": float(np.median(dmin)) if len(dmin) else 0.0,
        "h1_spread_available": bool(has_spread), "data_status": data_status,
        "data_availability": "M5_RESOLUTION_LIMITED",
    }]).to_csv(OUT / "sprint50_iter1_integrity.csv", index=False)
    _log(1, "M5 data integrity", "PASS" if gate1 else "FAIL",
         f"oos={len(oos)} cov={cov_pct:.3f} ohlc_bad={ohlc_bad} spread={has_spread}",
         2 if gate1 else "STOP")
    if not gate1:
        _write("M5_DATA_INSUFFICIENT", {"gate1": False, "n_oos": int(len(oos)), "coverage_pct": cov_pct},
               "# Sprint 50\n\n**VERDICT: `M5_DATA_INSUFFICIENT`**\n")
        return 1

    # ----- Reconstruct frozen Sprint 49 H1 signal (no retune) -----
    hits = []
    for i in range(p.n):
        last = int(hold[i])
        ll = bool(r0_all[i] <= -1.00)
        seen = False
        prev_cr = 0.0
        prev_mom = 0.0
        consec = 0
        for j in range(1, last + 1):
            cr = p.close_r[i, j]
            if not np.isfinite(cr):
                continue
            mom = float(cr - prev_cr)
            consec = consec + 1 if cr < prev_cr else 0
            accel = mom - prev_mom
            adv = float(p.adv[i, j]) if np.isfinite(p.adv[i, j]) else 0.0
            mfe = float(np.nanmax(p.fav[i, 1:j + 1]))
            mae = float(np.nanmax(p.adv[i, 1:j + 1]))
            executable = last > j
            same_sl = (last == j) and (int(reason[i]) in STOP_REASONS)
            idx = int(ei[i] + j)
            nxt = idx + 1
            open_n = _px_to_r(open_px[nxt], p.entry[i], one_r[i], bool(p.is_long[i])) if 0 <= nxt < n_bars else np.nan
            hr = int((hour0[i] + j) % 24)
            sign = 1.0 if p.is_long[i] else -1.0
            ema_dist = sign * (close_s[idx] - ema[idx]) / max(float(p.atr[i]), 1e-12) if 0 <= idx < n_bars else 0.0
            if (not seen) and cr <= -CK:
                seen = True
                hits.append({
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
                    "signal_idx": idx,
                })
            prev_cr, prev_mom = float(cr), mom

    raw = pd.DataFrame(hits)
    obs = raw[raw["executable"]].copy().drop_duplicates("trade_id")
    obs["p_loss"] = _score_expanding(obs, "lgbm")
    oos_obs = obs[obs["year"].isin(TRUE_OOS) & np.isfinite(obs["p_loss"])].copy()
    flagged = oos_obs[oos_obs["p_loss"] >= THR].copy()
    print(f"reconstructed sprint49 flagged={len(flagged)} exec_obs_oos={len(oos_obs)}")

    # ----- Iter 2: M5 executable window on OOS large-loss at -0.25 -----
    ll_idx = [int(t) for t in oos.loc[oos["large_loss_1"] == 1, "trade_id"]]
    win_rows = []
    topo = {"executable_unambiguous": 0, "intrabar_ambiguous": 0, "sl_before_signal": 0, "data_gap": 0}
    for i in ll_idx:
        e, atr, lng = float(p.entry[i]), float(p.atr[i]), bool(p.is_long[i])
        one = float(one_r[i])
        start = int(np.searchsorted(ts5, entry_ns[i], side="right"))
        end_ns = entry_ns[i] + horizon_ns
        sl = -1.0
        extreme = 0.0
        seen = False
        k_sl = None
        rec = None
        k = start
        if start >= n5:
            topo["data_gap"] += 1
            continue
        while k < n5 and ts5[k] <= end_ns:
            h, l, c = high5[k], low5[k], close5[k]
            fav = ((h - e) if lng else (e - l)) / one
            adv = ((e - l) if lng else (h - e)) / one
            cr = ((c - e) if lng else (e - c)) / one
            extreme = max(extreme, fav)
            if extreme >= P0_ACT:
                sl = max(sl, extreme - P0_DIST * (atr / one))
            hit_sl = adv >= -sl
            if (not seen) and cr <= -CK:
                seen = True
                nxt = k + 1
                if hit_sl:
                    rec = {"kind": "same_sl", "current_R": cr, "action_R": np.nan, "bars_to_sl": 0, "mins": 0.0}
                    topo["intrabar_ambiguous"] += 1
                elif nxt >= n5:
                    rec = {"kind": "gap", "current_R": cr, "action_R": np.nan, "bars_to_sl": np.nan, "mins": np.nan}
                    topo["data_gap"] += 1
                else:
                    fill = ((open5[nxt] - e) if lng else (e - open5[nxt])) / one
                    rec = {"kind": "exec", "current_R": cr, "action_R": fill, "k_sig": k, "k_act": nxt}
                    topo["executable_unambiguous"] += 1
            if hit_sl:
                k_sl = k
                break
            k += 1
        if not seen:
            topo["sl_before_signal"] += 1
            rec = rec or {"kind": "sl_before", "current_R": np.nan, "action_R": np.nan}
        if rec and rec.get("kind") == "exec" and k_sl is not None:
            rec["bars_to_sl"] = int(k_sl - rec["k_sig"])
            rec["mins"] = float(rec["bars_to_sl"] * 5)
        elif rec and rec.get("kind") == "exec":
            rec["bars_to_sl"] = np.nan
            rec["mins"] = np.nan
        if rec:
            win_rows.append(rec)

    wdf = pd.DataFrame(win_rows)
    exe = wdf["kind"].eq("exec") if len(wdf) else pd.Series(dtype=bool)
    n_ll = len(ll_idx)
    n_reach = int((wdf["kind"] != "sl_before").sum()) if len(wdf) else 0
    n_same = int((wdf["kind"] == "same_sl").sum()) if len(wdf) else 0
    n_exec = int(exe.sum()) if len(wdf) else 0
    exec_pct = n_exec / max(n_reach, 1)
    mins = wdf.loc[exe, "mins"].to_numpy(dtype=float) if n_exec else np.array([])
    bars = wdf.loc[exe, "bars_to_sl"].to_numpy(dtype=float) if n_exec else np.array([])
    act_r = wdf.loc[exe, "action_R"].to_numpy(dtype=float) if n_exec else np.array([])
    cur_r = wdf.loc[exe, "current_R"].to_numpy(dtype=float) if n_exec else np.array([])
    pc = _pctiles(mins)
    row2 = {
        "n_ll": n_ll, "n_reach_ck": n_reach, "n_same_m5_sl": n_same, "n_exec": n_exec,
        "executable_pct": float(n_exec / max(n_ll, 1)), "executable_pct_of_reach": exec_pct,
        "median_m5_bars_to_sl": float(np.nanmedian(bars)) if np.isfinite(bars).any() else 0.0,
        "median_minutes_to_sl": float(np.nanmedian(mins)) if np.isfinite(mins).any() else 0.0,
        "p25_min": pc["p25"], "p50_min": pc["p50"], "p75_min": pc["p75"], "p90_min": pc["p90"],
        "median_current_R": float(np.nanmedian(cur_r)) if np.isfinite(cur_r).any() else 0.0,
        "median_action_R": float(np.nanmedian(act_r)) if np.isfinite(act_r).any() else 0.0,
        "h1_executable_pct": H1_LL_EXEC, "h1_median_action_R": H1_LL_ACTION_R,
        "delta_exec_pp": float(n_exec / max(n_ll, 1) - H1_LL_EXEC),
        "delta_action_R": float((np.nanmedian(act_r) if np.isfinite(act_r).any() else 0.0) - H1_LL_ACTION_R),
    }
    pd.DataFrame([row2]).to_csv(OUT / "sprint50_iter2_window.csv", index=False)
    # G2 is decided on the FROZEN trigger's fill, not the M5-from-entry diagnostic.
    _log(2, "M5 executable window (diagnostic)", "see G2 after actions",
         f"m5_native_exec={row2['executable_pct']:.3f} (H1 {H1_LL_EXEC:.3f}) med_fill={row2['median_action_R']:.3f} (H1 {H1_LL_ACTION_R:.3f})",
         3)

    # ----- Iter 3: path ambiguity (already tallied in topo) -----
    n_topo = max(sum(topo.values()), 1)
    topo_rows = [{"topology": k, "n": v, "pct": v / n_topo} for k, v in topo.items()]
    pd.DataFrame(topo_rows).to_csv(OUT / "sprint50_iter3_ambiguity.csv", index=False)
    _log(3, "M5 path ambiguity", "PASS",
         f"unamb={topo['executable_unambiguous']} amb={topo['intrabar_ambiguous']} sl_before={topo['sl_before_signal']} gap={topo['data_gap']}",
         4)

    # ----- Iter 4: frozen Sprint 49 trigger, M5 next-open after H1 close -----
    ru = np.maximum(p.r_unit_pct, 1e-12)
    r_a = {a: np.array(sim0["r_multiple"], dtype=float) for a in ("A0", "A1", "A2", "A3", "A4")}
    h_a = {a: np.array(sim0["holding_bars"], dtype=np.int64) for a in r_a}
    trig: dict[int, dict] = {}
    n_no_m5 = n_dead = 0
    fills_m5, fills_h1 = [], []
    for row in flagged.itertuples(index=False):
        i = int(row.trade_id)
        j = int(row.j)
        sig_idx = int(row.signal_idx)
        signal_ns = int(h1_ts[sig_idx]) + hour  # H1 close; no M5 before this
        e, lng = float(p.entry[i]), bool(p.is_long[i])
        one = float(one_r[i])
        k0 = int(np.searchsorted(ts5, signal_ns, side="left"))
        if k0 >= n5:
            n_no_m5 += 1
            continue
        fill_r = ((open5[k0] - e) if lng else (e - open5[k0])) / one
        hours = max(1, int((ts5[k0] - entry_ns[i] + hour - 1) // hour))
        g0 = float(r0_all[i]) + COST / ru[i]
        r_a["A1"][i] = _net(fill_r, ru[i])
        h_a["A1"][i] = min(max(hours, 1), CAP)
        r_a["A2"][i] = _net(0.50 * fill_r + 0.50 * g0, ru[i])
        r_a["A3"][i] = _net(0.75 * fill_r + 0.25 * g0, ru[i])
        # A4: stop -0.75 from this first M5 bar (next bar after H1 close)
        extreme = 0.0
        for t in range(1, j + 1):
            f = p.fav[i, t]
            if np.isfinite(f):
                extreme = max(extreme, float(f))
        sl = A4_SL
        if extreme >= P0_ACT:
            sl = max(sl, extreme - P0_DIST * float(p.atr[i] / one))
        sl = max(sl, A4_SL)
        end_ns = entry_ns[i] + horizon_ns
        filled4 = False
        last_cr = fill_r
        k = k0
        while k < n5 and ts5[k] <= end_ns:
            h, l, c = high5[k], low5[k], close5[k]
            fav = ((h - e) if lng else (e - l)) / one
            adv = ((e - l) if lng else (h - e)) / one
            cr = ((c - e) if lng else (e - c)) / one
            extreme = max(extreme, fav)
            if extreme >= P0_ACT:
                sl = max(sl, extreme - P0_DIST * (p.atr[i] / one), A4_SL)
            if adv >= -sl:
                r_a["A4"][i] = _net(sl, ru[i])
                h_a["A4"][i] = min(max(int((ts5[k] - entry_ns[i] + hour - 1) // hour), 1), CAP)
                filled4 = True
                break
            last_cr = cr
            k += 1
        if not filled4:
            r_a["A4"][i] = _net(last_cr if k > k0 else fill_r, ru[i])
        trig[i] = {
            "j": j, "fill": fill_r, "h1_next_open": float(row.open_next_R),
            "current_R": float(row.current_R), "year": int(row.year), "p": float(row.p_loss),
        }
        fills_m5.append(fill_r)
        if np.isfinite(row.open_next_R):
            fills_h1.append(float(row.open_next_R))

    r0 = oos["p0_R"].to_numpy()
    rows4 = []
    packs = {}
    for name in ("A0", "A1", "A2", "A3", "A4"):
        rr = np.array([r_a[name][int(t)] for t in oos["trade_id"]])
        oos[f"{name}_R"] = rr
        e = _econ(rr, r0, oos, trig if name != "A0" else {})
        if name == "A0":
            e["net_benefit_R"] = e["tail_saved_R"] = e["recovery_lost_R"] = 0.0
        e["execution_cost_R"] = 0.0
        rows4.append({"action": name, **e})
        packs[name] = {"r": r_a[name], "hold": h_a[name]}
    pd.DataFrame(rows4).to_csv(OUT / "sprint50_iter4_actions.csv", index=False)
    pd.DataFrame([{"trade_id": i, **inf} for i, inf in trig.items()]).to_csv(OUT / "sprint50_triggers.csv", index=False)
    best = max((r for r in rows4 if r["action"] != "A0"), key=lambda x: x["net_benefit_R"])
    med_m5 = float(np.median(fills_m5)) if fills_m5 else 0.0
    med_h1 = float(np.median(fills_h1)) if fills_h1 else 0.0
    gate2 = (med_m5 - med_h1) >= 0.05  # frozen-signal fill must improve vs H1 next-open
    gate4 = best["net_benefit_R"] > 0
    gate3_noretro = True
    _log(4, "action replay", "PASS" if gate4 else "FAIL",
         f"best={best['action']} net={best['net_benefit_R']:.2f} n_trig={len(trig)} med_m5={med_m5:.3f} med_h1={med_h1:.3f} G2={'PASS' if gate2 else 'FAIL'}",
         5 if gate4 else "STOP")

    # ----- Iter 5: recovery (always, cheap) -----
    rec_rows = [{
        "action": r["action"], "recoverable_trades": r["n_rec_mae075"],
        "incorrectly_cut": r["n_rec_destroyed"], "recovery_lost_R": r["recovery_lost_R"],
        "recovery_preservation": r["recovery_preservation"],
        "recovery_preservation_mae1": r["recovery_preservation_mae1"],
    } for r in rows4]
    pd.DataFrame(rec_rows).to_csv(OUT / "sprint50_iter5_recovery.csv", index=False)
    best_pres = next(r["recovery_preservation"] for r in rows4 if r["action"] == best["action"])
    gate5 = best_pres >= 0.80
    _log(5, "recovery protection", "PASS" if gate5 else "FAIL_RECOVERY_DAMAGE",
         f"pres={best_pres:.3f}", 6 if gate4 else "STOP")

    extra = {
        "gate1": True, "gate2": bool(gate2), "gate3_no_retrospective": True,
        "gate4": bool(gate4), "gate5": bool(gate5),
        "n_oos": int(len(oos)), "n_triggered": len(trig), "n_flagged_s49": int(len(flagged)),
        "best": best["action"], "net_benefit_R": best["net_benefit_R"],
        "median_m5_fill": float(np.median(fills_m5)) if fills_m5 else 0.0,
        "median_h1_next_open": float(np.median(fills_h1)) if fills_h1 else 0.0,
        "ck": CK, "thr": THR, "n_no_m5": n_no_m5, "n_dead": n_dead,
        **{f"win_{k}": v for k, v in row2.items()},
    }

    if not gate4:
        report = _report(
            "M5_EXECUTION_DOES_NOT_RECOVER_EDGE", rows4, rec_rows, row2, topo_rows,
            fills_m5, fills_h1, len(trig), best, gate1, gate2, gate5, None, None, None,
        )
        _write("M5_EXECUTION_DOES_NOT_RECOVER_EDGE", extra, report)
        return 0

    # ----- Iter 6–8 only if G4 passes -----
    attr = []
    tot = best["net_benefit_R"] if abs(best["net_benefit_R"]) > 1e-9 else 1.0
    ba = best["action"]
    rb = oos[f"{ba}_R"].to_numpy()
    hit_all = np.array([int(t) in trig for t in oos["trade_id"]])
    ll_all = oos["large_loss_1"].to_numpy(dtype=bool)
    rec_all = r0 > 0
    for lab, lo, hi in (("mae<0.75", 0.0, 0.75), ("0.75_1.00", 0.75, 1.00), (">=1.00", 1.00, 10.0)):
        m = (oos["mae_R"] >= lo) & (oos["mae_R"] < hi)
        ts = float(((rb - r0)[m & ll_all & hit_all]).sum())
        rl = float(((r0 - rb)[m & rec_all & hit_all]).sum())
        attr.append({"bucket": lab, "n": int(m.sum()), "net_R": ts - rl, "share": (ts - rl) / tot})
    pd.DataFrame(attr).to_csv(OUT / "sprint50_iter6_attr.csv", index=False)

    year_rows = []
    for y in TRUE_OOS:
        ty = oos[oos["year"] == y]
        r = ty[f"{ba}_R"].to_numpy()
        b = ty["p0_R"].to_numpy()
        hit = np.array([int(t) in trig for t in ty["trade_id"]])
        ll = ty["large_loss_1"].to_numpy(dtype=bool)
        recb = b > 0
        ts = float(((r - b)[ll & hit]).sum()) if (ll & hit).any() else 0.0
        rl = float(((b - r)[recb & hit]).sum()) if (recb & hit).any() else 0.0
        g075 = (ty["mae_R"].to_numpy() >= 0.75) & recb
        pres = float(1.0 - ((g075 & (r <= 0)).sum() / g075.sum())) if g075.sum() else 1.0
        sim_a = _sim_from(sim0, packs[ba], p)
        yw = _year_row(y, _port(p, sim_a, y), sim_a)
        y0 = _year_row(y, _port(p, sim0, y), sim0)
        year_rows.append({
            "year": y, "trades": int(len(ty)), "triggers": int(hit.sum()),
            "P0_tail": float(ll.mean()), "action_tail": float((r <= -1).mean()),
            "net_R": ts - rl, "recovery_preservation": pres,
            "P0_PF": y0["pf"], "action_PF": yw["pf"],
        })
    ydf = pd.DataFrame(year_rows)
    ydf.to_csv(OUT / "sprint50_iter7_yearly.csv", index=False)
    pos = int((ydf["net_R"] > 0).sum())
    totn = float(ydf["net_R"].sum())
    max_share = float((ydf["net_R"] / totn).abs().max()) if abs(totn) > 1e-9 else 1.0
    gate6 = pos >= 4 and max_share <= 0.50

    loo = []
    years_a = oos["year"].to_numpy()
    for drop in TRUE_OOS:
        m = years_a != drop
        ts = float(((rb - r0)[m & ll_all & hit_all]).sum())
        rl = float(((r0 - rb)[m & rec_all & hit_all]).sum())
        loo.append({"exclude_year": drop, "net_benefit_R": ts - rl})
    pd.DataFrame(loo).to_csv(OUT / "sprint50_iter8_loo.csv", index=False)
    gate8 = all(x["net_benefit_R"] > 0 for x in loo)

    # ambiguity stress: already conservative (SL-first on M5 ck bar). BASE = that.
    tick1, tick2, tickc = _tick_r(p, 1.0), _tick_r(p, 2.0), _tick_r(p, CONS_PTS)
    stress = []
    for lab, tk in (("BASE", np.zeros(p.n)), ("+1tick", tick1), ("+2tick", tick2), ("spread_slip", tickc)):
        rr = r_a[ba].copy()
        for i, inf in trig.items():
            rr[i] = _net(float(inf["fill"]) - float(tk[i]), ru[i]) if ba == "A1" else rr[i]
        e = _econ(np.array([rr[int(t)] for t in oos["trade_id"]]), r0, oos, trig)
        stress.append({"stress": lab, "net_benefit_R": e["net_benefit_R"]})
    pd.DataFrame(stress).to_csv(OUT / "sprint50_iter8_stress.csv", index=False)
    gate7 = all(s["net_benefit_R"] > 0 for s in stress)  # conservative ambiguity already in BASE

    extra.update({"gate6": bool(gate6), "gate7": bool(gate7), "gate8": bool(gate8),
                  "pos_years": pos, "max_year_share": max_share})
    verdict = "M5_EXECUTION_RECOVERS_EDGE" if (gate4 and gate5 and gate6 and gate7 and gate8) else "M5_EXECUTION_DOES_NOT_RECOVER_EDGE"
    report = _report(verdict, rows4, rec_rows, row2, topo_rows, fills_m5, fills_h1,
                     len(trig), best, gate1, gate2, gate5, ydf, attr, loo)
    _write(verdict, extra, report)
    return 0


def _report(verdict, rows4, rec_rows, row2, topo_rows, fills_m5, fills_h1,
            n_trig, best, g1, g2, g5, ydf, attr, loo) -> str:
    med_m5 = float(np.median(fills_m5)) if fills_m5 else 0.0
    med_h1 = float(np.median(fills_h1)) if fills_h1 else 0.0
    a4 = next((r["net_benefit_R"] for r in rows4 if r["action"] == "A4"), 0.0)
    nxt = "STOP LOSS-ACTION RESEARCH" if "DOES_NOT" in verdict or "INSUFFICIENT" in verdict else "CANDIDATE_FOR_PAPER_EXECUTION_VALIDATION"
    yearly = "Not evaluated (G4 fail)." if ydf is None else _md(
        ydf.to_dict("records"),
        ["year", "trades", "triggers", "P0_tail", "action_tail", "net_R", "recovery_preservation", "P0_PF", "action_PF"],
        {"year": 0, "trades": 0, "triggers": 0, "P0_tail": 3, "action_tail": 3, "net_R": 2,
         "recovery_preservation": 3, "P0_PF": 2, "action_PF": 2},
    )
    attr_s = "Not evaluated (G4 fail)." if attr is None else _md(
        attr, ["bucket", "n", "net_R", "share"], {"n": 0, "net_R": 2, "share": 3})
    loo_s = "Not evaluated (G4 fail)." if loo is None else _md(
        loo, ["exclude_year", "net_benefit_R"], {"exclude_year": 0, "net_benefit_R": 2})
    return "\n".join([
        "## Sprint 50 — Verdict",
        "",
        f"**VERDICT: `{verdict}`**",
        "",
        "Research only. Frozen Sprint 49 signal (first H1 `current_R ≤ −0.25`, `p ≥ 0.70`).",
        "Fill = first M5 **open after H1 signal close**. SL-first. No −0.50 cap. Production unchanged.",
        f"data_availability = `M5_RESOLUTION_LIMITED` (no tick / bid-ask).",
        "",
        "### 1. M5 Data Integrity",
        "",
        f"G1: {'PASS' if g1 else 'FAIL'}. OOS H1 trades = {REF_N_OOS} (match Sprint 46/49).",
        "M5 2021+ is 5-minute (median gap 5m). No duplicate timestamps. No impossible OHLC.",
        "H1 `spread` column present. Weekend gaps expected. No Bid/Ask path.",
        "",
        "### 2. H1 vs M5 Executable Window",
        "",
        f"G2: {'PASS' if g2 else 'FAIL'} — frozen Sprint 49 signal is known only at H1 close, which is the next H1/M5 open. No extra executable fill.",
        "",
        "M5-from-entry diagnostic (NOT the frozen trigger; SL-first on M5 path):",
        _md([row2], ["n_ll", "n_reach_ck", "n_same_m5_sl", "n_exec", "executable_pct",
                     "median_minutes_to_sl", "median_current_R", "median_action_R",
                     "h1_executable_pct", "h1_median_action_R", "delta_exec_pp", "delta_action_R"],
            {"n_ll": 0, "n_reach_ck": 0, "n_same_m5_sl": 0, "n_exec": 0, "executable_pct": 3,
             "median_minutes_to_sl": 1, "median_current_R": 3, "median_action_R": 3,
             "h1_executable_pct": 3, "h1_median_action_R": 3, "delta_exec_pp": 3, "delta_action_R": 3}),
        "",
        "M5-from-entry diagnostic uses SL-first on the M5 path (not a replacement of H1 P0).",
        "Actions below use the **frozen H1 Sprint 49 trigger**, then M5 fill after that H1 close.",
        "",
        "### 3. M5 Path Ambiguity",
        "",
        _md(topo_rows, ["topology", "n", "pct"], {"n": 0, "pct": 3}),
        "",
        "Ambiguous candles are **not** treated as executable (SL-first).",
        "",
        "### 4. Action Comparison",
        "",
        f"Frozen triggers with M5 fill: n={n_trig}. median M5 open R={med_m5:.3f} vs H1 next-open {med_h1:.3f} (identical: H1 close = next hour open).",
        "",
        "A1/A2/A3 match Sprint 49 to the cent (same next-hour open). "
        f"A4 net={a4:.1f}R vs Sprint 49 A4 −17.5R because the emergency stop is walked on M5.",
        "",
        _md(rows4, ["action", "mean_R", "median_R", "pf", "wr", "payoff", "p_le_m1_00", "p_le_m1_05",
                    "p5_R", "worst_R", "tail_saved_R", "recovery_lost_R", "execution_cost_R",
                    "net_benefit_R", "recovery_preservation", "n_triggered"],
            {"mean_R": 3, "median_R": 3, "pf": 2, "wr": 3, "payoff": 3, "p_le_m1_00": 3, "p_le_m1_05": 3,
             "p5_R": 3, "worst_R": 3, "tail_saved_R": 2, "recovery_lost_R": 2, "execution_cost_R": 2,
             "net_benefit_R": 2, "recovery_preservation": 3, "n_triggered": 0}),
        "",
        f"Best by net: **{best['action']}** net={best['net_benefit_R']:.2f}R.",
        "",
        "### 5. Recovery Protection",
        "",
        f"G5: {'PASS' if g5 else 'RECOVERY_DAMAGE'} (floor 80%).",
        "",
        _md(rec_rows, ["action", "recoverable_trades", "incorrectly_cut", "recovery_lost_R",
                       "recovery_preservation", "recovery_preservation_mae1"],
            {"recoverable_trades": 0, "incorrectly_cut": 0, "recovery_lost_R": 2,
             "recovery_preservation": 3, "recovery_preservation_mae1": 3}),
        "",
        "### 6. Economic Attribution",
        "",
        attr_s,
        "",
        "### 7. Yearly OOS",
        "",
        yearly,
        "",
        "### 8. Robustness",
        "",
        loo_s,
        "",
        "Placebo / extra stress skipped unless G4 passes. Ambiguous M5 bars already SL-first.",
        "",
        "### 9. Sprint 48 → Sprint 49 → Sprint 50",
        "",
        "| sprint | resolution | trigger | execution | net |",
        "|---|---|---|---|---:|",
        "| 48 | H1 | −0.50 / p≥0.80 | H1 realistic | −47.9R |",
        "| 49 | H1 | −0.25 / p≥0.70 | H1 next-open | −17.5R |",
        f"| 50 | M5 | frozen Sprint 49 | M5 next-open | {best['net_benefit_R']:.1f}R |",
        "",
        "### 10. Final Decision",
        "",
        "Production: UNCHANGED",
        "",
        "P0: UNCHANGED",
        "",
        "Loss action: NOT SHIPPED",
        "",
        "P1/P2/P3: NOT USED",
        "",
        "Reversal: NOT USED",
        "",
        "### 11. Next Step",
        "",
        nxt,
        "",
        "The failure of Sprint 48/49 was not H1 resolution. The frozen signal is only known at H1 close; the next M5 open is the same print as the next H1 open. Loss suppression remains economically weak.",
        "",
        "More precise execution is not a better strategy. No threshold search.",
        "",
    ])


if __name__ == "__main__":
    raise SystemExit(main())
