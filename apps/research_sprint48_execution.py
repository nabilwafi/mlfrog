"""Sprint 48 — A1 execution validation (research only).

Frozen Sprint 47 A1. Do not retune trigger/model/action. Do not change P0/entry.

  python apps/research_sprint48_execution.py
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

from apps.research_sprint40_iter1_time_aware_exit import STARTING
from apps.research_sprint41_iter4_switch import _port, _year_row
from apps.research_sprint42_attribution import TRUE_OOS, _md, _pooled
from apps.research_sprint46_loss_reduction import P0, _build_obs, _mc, _r_stats
from apps.research_sprint47_loss_action import (
    CK,
    THR,
    _frozen_p_loss,
    _net,
    _sim_from,
    _tick_r,
    _triggers,
)
from apps.run_exit_engine_grid import Paths, simulate_combo
from apps.research_sprint39_exit_state import _join_feat7, _load_panel
from settings.strategy import ASSUMED_SLIPPAGE_POINTS, FALLBACK_SPREAD_POINTS, POINT
from simulation.wf.sim import COST, entry_indices, load_h1, prepare_market

OUT = _ROOT / "artifacts/pipeline_backtest/loss_reduction/sprint48"
H1_PATH = _ROOT / "artifacts/raw/XAUUSD/H1/data.parquet"
REF_NET = 253.41
REF_N_OOS, REF_N_TRIG = 5614, 715
OVER_BINS = (
    ("<=0.02R", 0.0, 0.02),
    ("0.02_0.05R", 0.02, 0.05),
    ("0.05_0.10R", 0.05, 0.10),
    ("0.10_0.20R", 0.10, 0.20),
    (">0.20R", 0.20, 99.0),
)
STOP_REASONS = {0, 1, 2}  # SL / TRAIL / BE — filled at stop before close
CONS_PTS = float(FALLBACK_SPREAD_POINTS + ASSUMED_SLIPPAGE_POINTS)


def _log(gate, result, finding) -> None:
    print(f"GATE: {gate}")
    print(f"RESULT: {result}")
    print(f"KEY FINDING: {finding}")


def _px_to_r(px: np.ndarray, entry: np.ndarray, one_r: np.ndarray, is_long: np.ndarray) -> np.ndarray:
    return np.where(is_long, (px - entry) / one_r, (entry - px) / one_r)


def _apply_fills(p: Paths, sim0: dict, trig: dict, fill_g: dict[int, float | None]) -> dict:
    r = np.array(sim0["r_multiple"], dtype=float)
    h = np.array(sim0["holding_bars"], dtype=np.int64)
    ru = np.maximum(p.r_unit_pct, 1e-12)
    for i, g in fill_g.items():
        if g is None:
            continue
        r[i] = _net(float(g), ru[i])
        h[i] = max(int(trig[i]["j"]), 1)
    return {"r": r, "hold": h}


def _tail_extra(r: np.ndarray) -> dict:
    return {
        "p_le_m0_75": float((r <= -0.75).mean()),
        "p_le_m1_00": float((r <= -1.00).mean()),
        "p_le_m1_05": float((r <= -1.05).mean()),
        "p_le_m1_10": float((r <= -1.10).mean()),
        "p_le_m1_25": float((r <= -1.25).mean()),
    }


def _econ(r: np.ndarray, r0: np.ndarray, trades: pd.DataFrame, trig: dict, r_ref: np.ndarray | None) -> dict:
    ll = trades["large_loss_1"].to_numpy(dtype=bool)
    rec = trades["p0_R"].to_numpy() > 0
    tid = trades["trade_id"].to_numpy(dtype=int)
    hit = np.array([int(t) in trig for t in tid])
    g075 = (trades["mae_R"].to_numpy() >= 0.75) & rec
    g1 = (trades["mae_R"].to_numpy() >= 1.0) & rec
    destroyed = g075 & (r <= 0)
    destroyed1 = g1 & (r <= 0)
    ts = float(((r - r0)[ll & hit]).sum()) if (ll & hit).any() else 0.0
    rl = float(((r0 - r)[rec & hit]).sum()) if (rec & hit).any() else 0.0
    # execution_cost = extra adverse vs analytical A1; attribution only (already in r)
    exec_c = float((r_ref - r)[hit].sum()) if r_ref is not None and hit.any() else 0.0
    st = _r_stats(r)
    return {
        **st,
        **_tail_extra(r),
        "tail_saved_R": ts,
        "recovery_lost_R": rl,
        "execution_cost_R": exec_c,
        "net_benefit_R": ts - rl,  # cost already inside action_R
        "n_triggered": int(hit.sum()),
        "recoverable_trades": int(g075.sum()),
        "recovery_destroyed": int(destroyed.sum()),
        "recovery_preserved_R": float(r[g075 & (r > 0)].sum()) if (g075 & (r > 0)).any() else 0.0,
        "recovery_preservation_pct": float(1.0 - destroyed.sum() / g075.sum()) if g075.sum() else 1.0,
        "recoverable_mae1": int(g1.sum()),
        "recovery_destroyed_mae1": int(destroyed1.sum()),
        "recovery_preservation_mae1": float(1.0 - destroyed1.sum() / g1.sum()) if g1.sum() else 1.0,
        "tail_loss_rate": float((r <= -1.00).mean()),
        "p0_tail_loss_rate": float(ll.mean()),
    }


def _over_stats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {k: 0.0 for k in ("n", "median", "p75", "p90", "p95", "p99", "max")}
    return {
        "n": int(len(x)),
        "median": float(np.median(x)),
        "p75": float(np.percentile(x, 75)),
        "p90": float(np.percentile(x, 90)),
        "p95": float(np.percentile(x, 95)),
        "p99": float(np.percentile(x, 99)),
        "max": float(x.max()),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    h1 = load_h1(H1_PATH)
    mkt = prepare_market(h1)
    open_px = h1["open"].to_numpy(dtype=float)
    spread_pts = h1["spread"].to_numpy(dtype=float)
    ts_mkt = pd.DatetimeIndex(mkt["ts"])
    n_bars = len(open_px)

    panel = _join_feat7(_load_panel())
    p = Paths(panel, mkt)
    sim0 = simulate_combo(p, **P0)
    trades, obs = _build_obs(p, mkt, sim0)
    obs = obs.copy()
    obs["p_loss"] = _frozen_p_loss(obs)
    trig_all = _triggers(obs)
    oos = trades[trades["year"].isin(TRUE_OOS)].copy()
    trig = {k: v for k, v in trig_all.items() if v["year"] in TRUE_OOS}
    r0 = oos["p0_R"].to_numpy()
    print(f"oos trades={len(oos)} triggered={len(trig)}")

    tick_data = False  # no bid/ask/tick path in repo; H1 OHLC + bar spread only
    ei = entry_indices(p.panel, mkt["ts"]).astype(int)
    hold = sim0["holding_bars"].astype(int)
    reason = sim0["reason"].astype(int)
    one_r = np.maximum(1.5 * p.atr, 1e-12)
    tick1 = _tick_r(p, 1.0)
    tick2 = _tick_r(p, 2.0)
    tick_cons = _tick_r(p, CONS_PTS)

    # per-trigger execution anatomy
    rows_te = []
    fill_a: dict[int, float | None] = {}
    fill_b: dict[int, float | None] = {}
    fill_c: dict[int, float | None] = {}
    fill_d: dict[int, float | None] = {}
    fill_e: dict[int, float | None] = {}
    fill_worst: dict[int, float | None] = {}
    n_ambiguous = 0
    n_same_bar_stop = 0
    n_gap = 0
    n_delay_worse = 0
    gap_r_sum = 0.0
    delay_r_sum = 0.0
    spread_r_sum = 0.0
    residual_sl = 0
    residual_sl_r = 0.0

    for tid, inf in trig.items():
        i, j = int(tid), int(inf["j"])
        cr = float(inf["current_R"])
        idx = int(ei[i] + j)
        nxt = idx + 1
        bar_ts = ts_mkt[idx] if 0 <= idx < n_bars else p.ts.iloc[i]
        sig_ts = bar_ts + pd.Timedelta(hours=1)
        same_stop = int(hold[i]) == j and int(reason[i]) in STOP_REASONS
        adv_j = float(p.adv[i, j]) if np.isfinite(p.adv[i, j]) else np.nan
        open_next_r = np.nan
        if 0 <= nxt < n_bars:
            open_next_r = float(_px_to_r(np.array([open_px[nxt]]), p.entry[i:i+1], one_r[i:i+1], p.is_long[i:i+1])[0])
        sp = float(spread_pts[idx]) if 0 <= idx < n_bars else 0.0
        sp_r = (max(sp, 0.0) * POINT) / one_r[i]
        anal = max(cr, -CK)
        close_g = cr
        # BASE: live/paper on_bar is SL-first; A1 at close only if stop not already tagged
        can_close = not same_stop
        b_g = close_g if can_close else None
        c_g = (close_g - tick1[i]) if can_close else None
        d_g = (close_g - tick2[i]) if can_close else None
        e_g = (close_g - tick_cons[i]) if can_close else None
        worst_g = None
        if can_close:
            base_w = open_next_r if np.isfinite(open_next_r) else close_g
            worst_g = float(base_w) - tick_cons[i]
        fill_a[i] = anal
        fill_b[i] = b_g
        fill_c[i] = c_g
        fill_d[i] = d_g
        fill_e[i] = e_g
        fill_worst[i] = worst_g

        n_ambiguous += 1  # H1 OHLC cannot reconstruct intrabar sequence
        if same_stop:
            n_same_bar_stop += 1
        if np.isfinite(adv_j) and adv_j > -cr + 1e-12:
            n_gap += 1
            gap_r_sum += float(adv_j + cr)  # extra adverse vs close
        if np.isfinite(open_next_r) and open_next_r < cr:
            n_delay_worse += 1
            delay_r_sum += float(cr - open_next_r)
        spread_r_sum += sp_r

        exec_b = close_g if can_close else float(sim0["r_multiple"][i]) + COST / max(p.r_unit_pct[i], 1e-12)
        slip_r = 0.0 if not can_close else 0.0
        over = float(cr - exec_b)  # + = worse than trigger close
        if can_close:
            over = 0.0  # BASE fills at trigger close
            over_vs_anal = float(anal - close_g)  # + = close worse than -0.50 cap
        else:
            over_vs_anal = float(anal - (float(sim0["r_multiple"][i]) + COST / max(p.r_unit_pct[i], 1e-12)))

        rows_te.append({
            "trade_id": i,
            "year": int(inf["year"]),
            "side": "long" if p.is_long[i] else "short",
            "trigger_bar": j,
            "trigger_bar_ts": str(bar_ts),
            "signal_available_timestamp": str(sig_ts),
            "actual_execution_timestamp": str(sig_ts if can_close else bar_ts),
            "execution_ambiguous": True,
            "same_bar_stop": bool(same_stop),
            "p0_reason": int(reason[i]),
            "p0_hold": int(hold[i]),
            "trigger_current_R": cr,
            "trigger_probability": float(inf["p"]),
            "trigger_R": cr,
            "analytical_fill_R": anal,
            "close_R": close_g,
            "next_open_R": float(open_next_r) if np.isfinite(open_next_r) else np.nan,
            "bar_adv_R": adv_j,
            "spread_points": sp,
            "spread_R": float(sp_r),
            "execution_R_base": float(exec_b) if can_close else np.nan,
            "execution_slippage_R": slip_r,
            "overshoot_vs_trigger_R": over,
            "overshoot_vs_analytical_R": over_vs_anal,
            "a1_executable_base": bool(can_close),
        })

    te = pd.DataFrame(rows_te)
    te.to_csv(OUT / "sprint48_trigger_execution.csv", index=False)

    # residual P0 SL never triggered
    oos_sl = oos[(oos["exit_reason"] == "SL") & ~oos["trade_id"].isin(trig.keys())]
    residual_sl = int(len(oos_sl))
    residual_sl_r = float(oos_sl["p0_R"].sum()) if len(oos_sl) else 0.0

    packs = {
        "A": _apply_fills(p, sim0, trig, fill_a),
        "B": _apply_fills(p, sim0, trig, fill_b),
        "C": _apply_fills(p, sim0, trig, fill_c),
        "D": _apply_fills(p, sim0, trig, fill_d),
        "E": _apply_fills(p, sim0, trig, fill_e),
        "WORST": _apply_fills(p, sim0, trig, fill_worst),
    }
    rA = np.array([packs["A"]["r"][int(t)] for t in oos["trade_id"]])
    scen_r = {"P0": r0, "A": rA}
    for name in ("B", "C", "D", "E", "WORST"):
        scen_r[name] = np.array([packs[name]["r"][int(t)] for t in oos["trade_id"]])

    labels = {
        "P0": "P0 baseline",
        "A": "analytical max(current_R,-0.50)",
        "B": "BASE close fill, SL-first (project on_bar)",
        "C": "BASE +1 tick",
        "D": "BASE +2 ticks",
        "E": "conservative spread+slip 19+5 pts",
        "F": "historical bid/ask (SKIPPED — no tick data)",
        "WORST": "next-open + conservative pts, SL-first",
    }

    # GATE 1
    gate1 = len(oos) == REF_N_OOS and len(trig) == REF_N_TRIG and (not tick_data)
    _log("1 DATA", "PASS" if len(oos) == REF_N_OOS and len(trig) == REF_N_TRIG else "EXECUTION_DATA_FAIL",
         f"oos={len(oos)} trig={len(trig)} tick_data={tick_data} ambiguous={n_ambiguous}")
    if len(oos) != REF_N_OOS or len(trig) != REF_N_TRIG:
        (OUT / "sprint48_final_verdict.json").write_text(json.dumps({
            "sprint": 48, "verdict": "EXECUTION_DATA_FAIL", "production_changed": False,
            "n_oos": int(len(oos)), "n_trig": int(len(trig)),
        }, indent=2), encoding="utf-8")
        print("FINAL VERDICT: EXECUTION_DATA_FAIL")
        return 1

    econ = {}
    for name, r in scen_r.items():
        econ[name] = _econ(r, r0, oos, trig if name != "P0" else {}, rA if name not in ("P0", "A") else None)
        if name == "P0":
            econ[name]["execution_cost_R"] = 0.0
            econ[name]["net_benefit_R"] = 0.0
            econ[name]["tail_saved_R"] = 0.0
            econ[name]["recovery_lost_R"] = 0.0

    # GATE 2 reproduction
    net_a = econ["A"]["net_benefit_R"]
    gate2 = abs(net_a - REF_NET) < 2.0
    _log("2 REPRO", "PASS" if gate2 else "REPRODUCTION_FAIL", f"net_A={net_a:.4f} ref={REF_NET}")
    if not gate2:
        pd.DataFrame([{"scenario": k, **v} for k, v in econ.items()]).to_csv(OUT / "sprint48_execution_scenarios.csv", index=False)
        (OUT / "sprint48_final_verdict.json").write_text(json.dumps({
            "sprint": 48, "verdict": "REPRODUCTION_FAIL", "production_changed": False,
            "net_A": net_a, "ref": REF_NET,
        }, indent=2), encoding="utf-8")
        print("FINAL VERDICT: REPRODUCTION_FAIL")
        return 1

    rows_scen = []
    for name in ("P0", "A", "B", "C", "D", "E", "F", "WORST"):
        if name == "F":
            rows_scen.append({
                "scenario": "F", "label": labels["F"], "status": "SKIPPED",
                "net_benefit_R": np.nan, "tail_saved_R": np.nan, "n": len(oos),
            })
            continue
        e = econ[name]
        rows_scen.append({"scenario": name, "label": labels[name], "status": "OK", **e})
    pd.DataFrame(rows_scen).to_csv(OUT / "sprint48_execution_scenarios.csv", index=False)

    # overshoot vs analytical cap (how much worse close is than -0.50 fill)
    over_anal = te["overshoot_vs_analytical_R"].to_numpy(dtype=float)
    over_rows = [{"scope": "vs_analytical_cap", **_over_stats(over_anal)}]
    for lab, lo, hi in OVER_BINS:
        m = (over_anal >= lo) & (over_anal < hi) if hi < 90 else (over_anal >= lo)
        sub = over_anal[m]
        over_rows.append({"scope": "vs_analytical_cap", "bucket": lab, "n": int(m.sum()),
                          "share": float(m.mean()), "sum_R": float(sub.sum()) if sub.size else 0.0})
    # next-open overshoot vs close (delayed detection)
    delay = (te["close_R"] - te["next_open_R"]).to_numpy(dtype=float)
    delay = delay[np.isfinite(delay)]
    over_rows.append({"scope": "next_open_vs_close", **_over_stats(delay)})
    pd.DataFrame(over_rows).to_csv(OUT / "sprint48_overshoot_analysis.csv", index=False)

    rec_rows = []
    for name in ("P0", "A", "B", "C", "D", "E", "WORST"):
        e = econ[name]
        rec_rows.append({
            "scenario": name,
            "recoverable_trades": e["recoverable_trades"],
            "recovery_destroyed": e["recovery_destroyed"],
            "recovery_lost_R": e["recovery_lost_R"],
            "recovery_preserved_R": e["recovery_preserved_R"],
            "recovery_preservation_pct": e["recovery_preservation_pct"],
            "recoverable_mae1": e["recoverable_mae1"],
            "recovery_destroyed_mae1": e["recovery_destroyed_mae1"],
            "recovery_preservation_mae1": e["recovery_preservation_mae1"],
        })
    pd.DataFrame(rec_rows).to_csv(OUT / "sprint48_recovery_preservation.csv", index=False)

    # portfolio + yearly
    port_rows, year_rows, yports = [], [], {}
    for name in ("P0", "A", "B", "C", "D", "E", "WORST"):
        sim = _sim_from(sim0, packs[name] if name != "P0" else {"r": sim0["r_multiple"], "hold": sim0["holding_bars"]}, p)
        yrows = []
        for y in TRUE_OOS:
            port = _port(p, sim, y)
            yr = _year_row(y, port, sim)
            yrows.append(yr)
            if name in ("P0", "B"):
                yports[(name, y)] = port
        po = _pooled(yrows)
        mc = _mc(po["pnl"], name)
        port_rows.append({
            "book": "BOOK_A" if name == "P0" else "BOOK_B",
            "scenario": name,
            "pf": po["pf"], "ret": po["ret"], "dd": po["dd"], "wr": po["wr"],
            "payoff": po["payoff"], "avg_r": po["avg_r"], "max_loss": po["max_loss_streak"],
            **{k: mc[k] for k in ("prob_ruin", "median_dd", "p95_dd", "p99_dd", "worst_dd")},
        })
        rmap = dict(zip(oos["trade_id"].astype(int), scen_r[name]))
        for y in TRUE_OOS:
            ty = oos[oos["year"] == y]
            r = np.array([rmap[int(t)] for t in ty["trade_id"]])
            b = ty["p0_R"].to_numpy()
            hit = np.array([int(t) in trig for t in ty["trade_id"]])
            ll = ty["large_loss_1"].to_numpy(dtype=bool)
            rec = b > 0
            ts = float(((r - b)[ll & hit]).sum()) if (ll & hit).any() else 0.0
            rl = float(((b - r)[rec & hit]).sum()) if (rec & hit).any() else 0.0
            exec_c = float((rA[oos["year"].to_numpy() == y][hit] - r[hit]).sum()) if name not in ("P0", "A") and hit.any() else 0.0
            g075 = (ty["mae_R"].to_numpy() >= 0.75) & rec
            pres = float(1.0 - ((g075 & (r <= 0)).sum() / g075.sum())) if g075.sum() else 1.0
            yrow = next(x for x in yrows if x["year"] == y)
            year_rows.append({
                "year": y, "scenario": name, "trades": int(len(ty)),
                "trigger_count": int(hit.sum()),
                "P0_tail_rate": float(ll.mean()),
                "A1_tail_rate": float((r <= -1).mean()),
                "tail_saved_R": ts, "recovery_lost_R": rl, "execution_cost_R": exec_c,
                "net_benefit_R": ts - rl,
                "PF": yrow["pf"], "DD": yrow["dd"], "Return": yrow["ret"],
                "WR": yrow["wr"], "AvgR": yrow["avg_r"],
                "recovery_preservation": pres,
            })
    # attach P0 PF/DD onto A1 yearly rows
    ydf = pd.DataFrame(year_rows)
    p0y = ydf[ydf["scenario"] == "P0"].set_index("year")
    ydf["P0_PF"] = ydf["year"].map(p0y["PF"])
    ydf["P0_DD"] = ydf["year"].map(p0y["DD"])
    ydf["A1_PF"] = ydf["PF"]
    ydf["A1_DD"] = ydf["DD"]
    ydf.to_csv(OUT / "sprint48_yearly_oos.csv", index=False)
    pd.DataFrame(port_rows).to_csv(OUT / "sprint48_portfolio_comparison.csv", index=False)

    mc_keep = [r for r in port_rows if r["scenario"] in ("P0", "B", "E", "WORST")]
    pd.DataFrame(mc_keep).to_csv(OUT / "sprint48_mc_risk.csv", index=False)

    # leave-one-year-out on BASE (B)
    loo = []
    rb = scen_r["B"]
    years_arr = oos["year"].to_numpy()
    hit_all = np.array([int(t) in trig for t in oos["trade_id"]])
    ll_all = oos["large_loss_1"].to_numpy(dtype=bool)
    rec_all = r0 > 0
    for drop in TRUE_OOS:
        m = years_arr != drop
        ts = float(((rb - r0)[m & ll_all & hit_all]).sum())
        rl = float(((r0 - rb)[m & rec_all & hit_all]).sum())
        exec_c = float((rA - rb)[m & hit_all].sum())
        loo.append({"exclude_year": drop, "net_benefit_R": ts - rl, "tail_saved_R": ts,
                    "recovery_lost_R": rl, "execution_cost_R": exec_c})
    pd.DataFrame(loo).to_csv(OUT / "sprint48_leave_one_year_out.csv", index=False)

    # failure modes (counts on triggered set / OOS)
    rec_cut = int(((r0 > 0) & (scen_r["B"] <= 0) & hit_all).sum())
    rec_cut_r = float(((r0 - scen_r["B"])[(r0 > 0) & hit_all]).sum())
    n_trig = len(trig)
    same_ids = set(te.loc[te["same_bar_stop"], "trade_id"].astype(int))
    same_mask = np.array([int(t) in same_ids for t in oos["trade_id"]])
    fail = [
        {"mode": "A_GAP_THROUGH_TRIGGER", "count": n_gap, "pct": n_gap / n_trig,
         "R_impact": gap_r_sum, "note": "H1 bar adv worse than close; sequence unknown"},
        {"mode": "B_SPREAD_EXPANSION", "count": int((te["spread_points"] > FALLBACK_SPREAD_POINTS).sum()),
         "pct": float((te["spread_points"] > FALLBACK_SPREAD_POINTS).mean()),
         "R_impact": float(spread_r_sum), "note": "H1 bar spread points; not bid/ask path"},
        {"mode": "C_DELAYED_DETECTION", "count": n_delay_worse, "pct": n_delay_worse / n_trig,
         "R_impact": delay_r_sum, "note": "next H1 open worse than trigger close"},
        {"mode": "D_OHLC_AMBIGUITY", "count": n_ambiguous, "pct": 1.0,
         "R_impact": 0.0, "note": "all triggers; no tick path"},
        {"mode": "E_RESIDUAL_P0_SL", "count": residual_sl, "pct": residual_sl / len(oos),
         "R_impact": residual_sl_r, "note": "never triggered A1, still P0 SL"},
        {"mode": "F_RECOVERY_DESTRUCTION", "count": rec_cut, "pct": rec_cut / n_trig,
         "R_impact": rec_cut_r, "note": "P0_R>0 triggered trades with BASE R<=0"},
        {"mode": "SAME_BAR_STOP_BLOCKS_A1", "count": n_same_bar_stop, "pct": n_same_bar_stop / n_trig,
         "R_impact": float((rA - r0)[same_mask].sum()) if n_same_bar_stop else 0.0,
         "note": "P0 already SL/TRAIL on trigger bar; A1 at close is too late"},
    ]
    pd.DataFrame(fail).to_csv(OUT / "sprint48_execution_failure_modes.csv", index=False)

    # gates 3-8 on BASE / conservative
    p0_ll = econ["P0"]["p_le_m1_00"]
    gate3 = econ["B"]["p_le_m1_00"] < p0_ll
    gate4 = econ["B"]["net_benefit_R"] > 0
    gate5 = econ["E"]["net_benefit_R"] > 0
    gate6 = econ["B"]["recovery_preservation_pct"] >= 0.80 and econ["E"]["recovery_preservation_pct"] >= 0.80
    yb = ydf[ydf["scenario"] == "B"]
    tail_ok_years = int((yb["A1_tail_rate"] <= yb["P0_tail_rate"] + 1e-12).sum())
    net_pos_years = int((yb["net_benefit_R"] > 0).sum())
    total_net_b = float(yb["net_benefit_R"].sum())
    max_share = float((yb["net_benefit_R"] / total_net_b).abs().max()) if abs(total_net_b) > 1e-9 else 1.0
    gate7 = tail_ok_years == 5
    gate8 = all(x["net_benefit_R"] > 0 for x in loo)

    deg = 1.0 - (econ["B"]["net_benefit_R"] / econ["A"]["net_benefit_R"]) if econ["A"]["net_benefit_R"] else 1.0
    same_bar_share = n_same_bar_stop / n_trig
    dominant_fail = same_bar_share >= 0.50 or (over_anal[over_anal >= 0.20].sum() > 0.50 * max(over_anal.sum(), 1e-9))

    _log("3 TAIL", "PASS" if gate3 else "FAIL", f"P0={p0_ll:.3f} B={econ['B']['p_le_m1_00']:.3f} E={econ['E']['p_le_m1_00']:.3f}")
    _log("4 ECON", "PASS" if gate4 else "FAIL", f"net_B={econ['B']['net_benefit_R']:.2f}")
    _log("5 CONS", "PASS" if gate5 else "FAIL", f"net_E={econ['E']['net_benefit_R']:.2f}")
    _log("6 REC", "PASS" if gate6 else "FAIL", f"pres_B={econ['B']['recovery_preservation_pct']:.3f}")
    _log("7 YEAR", "PASS" if gate7 else "FAIL", f"tail_ok={tail_ok_years}/5 net_pos={net_pos_years}/5 max_share={max_share:.2f}")
    _log("8 LOO", "PASS" if gate8 else "FAIL", f"loo={[round(x['net_benefit_R'], 2) for x in loo]}")

    if not (gate3 and gate4 and gate6):
        verdict = "EXECUTION_INVALIDATED"
    elif not (gate5 and gate7 and gate8) or deg > 0.50 or dominant_fail:
        verdict = "EXECUTION_FRAGILE"
    else:
        verdict = "EXECUTION_VALIDATED"

    rec_map = {
        "EXECUTION_VALIDATED": "OHLC-conservative A1 still positive — still not production; next is paper/live fill log, not a new action search",
        "EXECUTION_FRAGILE": "analytical edge survives; realistic fill materially cuts it — still not production",
        "EXECUTION_INVALIDATED": "realistic execution removes the A1 edge — keep P0",
        "EXECUTION_DATA_LIMITED": "no tick/bid-ask path; cannot honestly validate executable fills",
    }

    p0p = next(r for r in port_rows if r["scenario"] == "P0")
    bp = next(r for r in port_rows if r["scenario"] == "B")
    ep = next(r for r in port_rows if r["scenario"] == "E")
    wp = next(r for r in port_rows if r["scenario"] == "WORST")

    lines = [
        "# Sprint 48 — Conditional Loss Action Execution Validation",
        "",
        "## Verdict",
        "",
        f"**`{verdict}`**",
        "",
        "Production Entry = UNCHANGED. Production P0 = UNCHANGED. A1 = NOT SHIPPED.",
        "",
        "## Frozen Configuration",
        "",
        "- Entry: FEAT7 top 21% LGBM WF, unchanged",
        "- P0: activation 0.25R, distance 0.08 ATR, no TP/BEP, horizon 48",
        "- A1 only: first current_R ≤ −0.50R AND p ≥ 0.80 → full close",
        "- Model / checkpoint / threshold: Sprint 46/47 frozen (not retrained)",
        "- Lot 0.01, max_open 5, heat 3R, capital $280/year",
        "",
        "## OOS Scope",
        "",
        f"2022–2026 only. trades={len(oos)} (ref {REF_N_OOS}). triggered={len(trig)} (ref {REF_N_TRIG}). 2021 unused.",
        "",
        "## Execution Data Availability",
        "",
        "- Tick / bid-ask path: **absent**. Scenario F skipped.",
        "- Available: H1 OHLC + MT5 bar `spread` (points). M5 OHLC exists but is not used to invent a new fill engine.",
        "- Project fill convention: `COST=1.5e-4` already in P0/A1 net; paper close applies **no extra close slippage**; live/paper `on_bar` is **SL-first**, TIMEOUT/close at bar close.",
        "- Conservative points: `FALLBACK_SPREAD_POINTS=19` + `ASSUMED_SLIPPAGE_POINTS=5` (settings/strategy.py).",
        "- POINT=0.01. H1 timestamp = bar open; signal available at bar close (+1h).",
        "",
        f"**data_availability = EXECUTION_DATA_LIMITED** (no executable bid/ask). Conservative OHLC scenarios still run.",
        "",
        "## Analytical Reproduction",
        "",
        f"Scenario A net_benefit_R = **{net_a:.2f}** (ref {REF_NET}). Gate 2: **{'PASS' if gate2 else 'FAIL'}**.",
        f"A tail_saved={econ['A']['tail_saved_R']:.2f} recovery_lost={econ['A']['recovery_lost_R']:.2f} P(R≤−1)={econ['A']['p_le_m1_00']:.3f} preservation={econ['A']['recovery_preservation_pct']:.3f}.",
        "",
        "## Realistic Execution",
        "",
        "BASE does **not** use `max(current_R, −0.50R)`. Fill = trigger-bar **close_R**, and only if P0 has not already stopped out on that bar (SL-first).",
        "",
        _md(rows_scen, ["scenario", "status", "mean_R", "p_le_m1_00", "tail_saved_R", "recovery_lost_R",
                        "execution_cost_R", "net_benefit_R", "recovery_preservation_pct"],
            {"mean_R": 3, "p_le_m1_00": 3, "tail_saved_R": 2, "recovery_lost_R": 2,
             "execution_cost_R": 2, "net_benefit_R": 2, "recovery_preservation_pct": 3}),
        "",
        f"Degradation vs analytical net: BASE {deg:.1%}. same-bar stop blocks {n_same_bar_stop}/{n_trig} triggers.",
        "",
        "## Overshoot / Gap Analysis",
        "",
        "Positive overshoot_vs_analytical = close worse than the −0.50 cap (analytical optimism).",
        "",
        _md([r for r in over_rows if r.get("scope") == "vs_analytical_cap" and "bucket" not in r],
            ["n", "median", "p75", "p90", "p95", "p99", "max"],
            {"n": 0, "median": 3, "p75": 3, "p90": 3, "p95": 3, "p99": 3, "max": 3}),
        "",
        _md([r for r in over_rows if "bucket" in r],
            ["bucket", "n", "share", "sum_R"], {"n": 0, "share": 3, "sum_R": 2}),
        "",
        "## Tail Reduction",
        "",
        f"P0 P(R≤−1)={p0_ll:.3f}. BASE={econ['B']['p_le_m1_00']:.3f}. conservative E={econ['E']['p_le_m1_00']:.3f}. WORST={econ['WORST']['p_le_m1_00']:.3f}.",
        f"P0 P(R≤−1.05)={econ['P0']['p_le_m1_05']:.3f} BASE={econ['B']['p_le_m1_05']:.3f}. worst_R P0={econ['P0']['worst_R']:.3f} BASE={econ['B']['worst_R']:.3f}.",
        "",
        "## Recovery Preservation",
        "",
        _md(rec_rows, ["scenario", "recoverable_trades", "recovery_destroyed", "recovery_lost_R",
                       "recovery_preservation_pct", "recovery_preservation_mae1"],
            {"recoverable_trades": 0, "recovery_destroyed": 0, "recovery_lost_R": 2,
             "recovery_preservation_pct": 3, "recovery_preservation_mae1": 3}),
        "",
        "## Economic Attribution",
        "",
        "net_benefit_R = tail_saved_R − recovery_lost_R. execution_cost_R is extra adverse vs Scenario A, **already inside** action_R (not subtracted twice).",
        "",
        f"- BASE net **{econ['B']['net_benefit_R']:+.2f}R** (tail {econ['B']['tail_saved_R']:+.2f}, rec lost {econ['B']['recovery_lost_R']:.2f}, vs-A cost {econ['B']['execution_cost_R']:.2f})",
        f"- E net **{econ['E']['net_benefit_R']:+.2f}R**",
        f"- WORST (next-open+24pts) net **{econ['WORST']['net_benefit_R']:+.2f}R**",
        "",
        "## Yearly OOS",
        "",
        _md(ydf[ydf["scenario"] == "B"].to_dict("records"),
            ["year", "trades", "trigger_count", "P0_tail_rate", "A1_tail_rate", "net_benefit_R",
             "recovery_preservation", "P0_PF", "A1_PF", "P0_DD", "A1_DD"],
            {"year": 0, "trades": 0, "trigger_count": 0, "P0_tail_rate": 3, "A1_tail_rate": 3,
             "net_benefit_R": 2, "recovery_preservation": 3, "P0_PF": 2, "A1_PF": 2, "P0_DD": 3, "A1_DD": 3}),
        "",
        f"tail_improved_years={tail_ok_years}/5 positive_net_years={net_pos_years}/5 max_year_share_of_net={max_share:.2f}",
        "",
        "## Portfolio Impact",
        "",
        f"P0 PF={p0p['pf']:.2f} DD={p0p['dd']:.3f} AvgR={p0p['avg_r']:.3f} WR={p0p['wr']:.3f}",
        f"BASE PF={bp['pf']:.2f} DD={bp['dd']:.3f} AvgR={bp['avg_r']:.3f} WR={bp['wr']:.3f}",
        f"E PF={ep['pf']:.2f} DD={ep['dd']:.3f}",
        "",
        "## Monte Carlo",
        "",
        _md(mc_keep, ["scenario", "prob_ruin", "median_dd", "p95_dd", "p99_dd", "worst_dd"],
            {"prob_ruin": 3, "median_dd": 3, "p95_dd": 3, "p99_dd": 3, "worst_dd": 3}),
        "",
        "## Leave-One-Year-Out",
        "",
        _md(loo, ["exclude_year", "net_benefit_R", "tail_saved_R", "recovery_lost_R", "execution_cost_R"],
            {"exclude_year": 0, "net_benefit_R": 2, "tail_saved_R": 2, "recovery_lost_R": 2, "execution_cost_R": 2}),
        "",
        "## Execution Failure Modes",
        "",
        _md(fail, ["mode", "count", "pct", "R_impact"], {"count": 0, "pct": 3, "R_impact": 2}),
        "",
        "## Leakage Audit",
        "",
        "- Trigger uses PIT close_R and frozen expanding LGBM scores only.",
        "- No future MAE/MFE/labels in the trigger.",
        "- Realistic fill uses the trigger bar close or next open; not the −0.50 cap.",
        "- Same-bar P0 stop is not rewritten into a −0.50 fill.",
        "- 2021 not used as OOS evidence.",
        "",
        "## Gate Results",
        "",
        f"- GATE 1 data integrity: PASS (universe matched; tick data limited)",
        f"- GATE 2 analytical reproduction: {'PASS' if gate2 else 'FAIL'} (net {net_a:.2f})",
        f"- GATE 3 tail reduction (BASE): {'PASS' if gate3 else 'FAIL'}",
        f"- GATE 4 economic value (BASE): {'PASS' if gate4 else 'FAIL'}",
        f"- GATE 5 conservative (E): {'PASS' if gate5 else 'FAIL'}",
        f"- GATE 6 recovery ≥80%: {'PASS' if gate6 else 'FAIL'}",
        f"- GATE 7 yearly tail: {'PASS' if gate7 else 'FAIL'} ({tail_ok_years}/5)",
        f"- GATE 8 leave-one-year-out: {'PASS' if gate8 else 'FAIL'}",
        "",
        "## Production Decision",
        "",
        "Entry: UNCHANGED",
        "",
        "P0: UNCHANGED",
        "",
        "A1: NOT SHIPPED",
        "",
        "## Recommendation",
        "",
        rec_map[verdict] + ".",
        "",
    ]
    (OUT / "sprint48_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "sprint48_final_verdict.json").write_text(json.dumps({
        "sprint": 48,
        "verdict": verdict,
        "production_changed": False,
        "a1_shipped": False,
        "data_availability": "EXECUTION_DATA_LIMITED",
        "n_oos": int(len(oos)),
        "n_triggered": int(len(trig)),
        "n_ambiguous": n_ambiguous,
        "n_same_bar_stop": n_same_bar_stop,
        "net_analytical": econ["A"]["net_benefit_R"],
        "net_base": econ["B"]["net_benefit_R"],
        "net_conservative": econ["E"]["net_benefit_R"],
        "net_worst": econ["WORST"]["net_benefit_R"],
        "p_le_m1_p0": p0_ll,
        "p_le_m1_base": econ["B"]["p_le_m1_00"],
        "recovery_preservation_base": econ["B"]["recovery_preservation_pct"],
        "degradation_vs_analytical": deg,
        "gate1": True,
        "gate2": bool(gate2),
        "gate3": bool(gate3),
        "gate4": bool(gate4),
        "gate5": bool(gate5),
        "gate6": bool(gate6),
        "gate7": bool(gate7),
        "gate8": bool(gate8),
        "tail_improved_years": tail_ok_years,
        "positive_net_years": net_pos_years,
        "max_year_share_of_net": max_share,
        "recommendation": rec_map[verdict],
    }, indent=2, default=float), encoding="utf-8")
    print(f"FINAL VERDICT: {verdict}")
    print("PRODUCTION: UNCHANGED")
    print("A1: NOT SHIPPED")
    print(f"NEXT: {rec_map[verdict]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
