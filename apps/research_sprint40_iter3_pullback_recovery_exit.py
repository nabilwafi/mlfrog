"""Sprint 40 — Pullback / Recovery Exit (ITER 3).

Frozen production entry/risk/multi-trade (same as Iter 1/2).
Exit overlay only, no ML.

Goal:
- shallow pullback -> hold
- deep pullback -> exit sooner
- recovery after pullback -> hold

Implementation (no ML, quantile TRAIN ONLY):
1) Estimate quantile thresholds from Sprint 39 `exit_state_dataset.parquet`
   using only TRAIN years (2021-2023) and only `kind == "bar"` observations.
2) During exit replay, when trail updates (extreme >= 0.25R), choose
   trail distance based on:
   - drawdown_from_MFE_R = extreme - current_R at bar j
   - current_R at bar j
   - bars_in_trade (j) for late pullback variant

Candidates:
  A) deep pullback tighten dist=0.06; recovery hold with dist=0.12
  B) same as A but deep pullback applies only when bars_in_trade is late
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import tools.pyc_path_hook  # noqa: F401

from apps.report_adaptive_vol_risk import iter6_montecarlo
from apps.report_multi_trade_engine import run_multi
from apps.run_exit_engine_grid import FEAT7, Paths, build_entry_panel
from apps.research_sprint40_iter1_time_aware_exit import (
    CFG_PROD,
    EXIT_KW_BASE,
    STARTING,
    TimePolicy,
    _year_metrics,
    _top_bot_mean,
    simulate_timeaware_combo,
)
from simulation.wf.sim import COST, load_h1, prepare_market


OUT = _ROOT / "artifacts" / "pipeline_backtest" / "exit_structure" / "sprint40" / "iter3_pullback_recovery_exit"


@dataclass(frozen=True)
class PullbackThr:
    dd_deep: float
    cur_low: float
    cur_rec: float
    j_late: float


def _orders_for_year(p: Paths, year: int) -> np.ndarray:
    idx = np.where(p.year == year)[0]
    return idx[np.argsort(pd.DatetimeIndex(p.ts.iloc[idx]).as_unit("ns").asi8, kind="stable")]


def _load_thresholds() -> PullbackThr:
    ds = _ROOT / "artifacts" / "pipeline_backtest" / "rolling_wf" / "sprint39_exit_state" / "exit_state_dataset.parquet"
    df = pd.read_parquet(ds)
    bar = df[(df["kind"] == "bar") & (df["year"] >= 2021) & (df["year"] <= 2023)].copy()
    # Train-only quantiles
    dd = bar["drawdown_from_MFE_R"].astype(float).to_numpy()
    cr = bar["current_R"].astype(float).to_numpy()
    j = bar["bars_in_trade"].astype(float).to_numpy()
    dd_deep = float(np.nanquantile(dd, 0.70))
    cur_low = float(np.nanquantile(cr, 0.30))
    cur_rec = float(np.nanquantile(cr, 0.70))
    j_late = float(np.nanquantile(j, 0.70))
    print(f"Thresholds (train 2021-2023): dd_deep={dd_deep:.3f} cur_low={cur_low:.3f} cur_rec={cur_rec:.3f} j_late={j_late:.1f}")
    return PullbackThr(dd_deep=dd_deep, cur_low=cur_low, cur_rec=cur_rec, j_late=j_late)


def simulate_pullback_combo(
    p: Paths,
    *,
    act: float,
    dist_normal: float,
    dist_tight: float,
    dist_wide: float,
    thr: PullbackThr,
    late_only: bool,
    recovery_wide: bool,
) -> dict[str, np.ndarray]:
    """Replay trail like simulate_timeaware_combo but with dist chosen by pullback state."""

    n = p.n
    T = 48

    sl = np.full(n, -1.0)
    extreme = np.zeros(n)
    mae = np.zeros(n)
    realized = np.zeros(n)
    frac = np.ones(n)
    active = np.ones(n, dtype=bool)

    exit_r = np.zeros(n)
    exit_off = np.zeros(n, dtype=np.int64)
    reason = np.full(n, 4, dtype=np.int64)
    trail_moved = np.zeros(n, dtype=bool)
    be_moved = np.zeros(n, dtype=bool)
    partial_any = np.zeros(n, dtype=bool)

    for j in range(1, T + 1):
        ended = active & (p.last_off < j)
        if ended.any():
            lo = p.last_off[ended]
            exit_r[ended] = p.close_r[ended, lo]
            exit_off[ended] = lo
            reason[ended] = 4
            active[ended] = False

        m = active
        if not m.any():
            break

        fav_j = p.fav[m, j]
        adv_j = p.adv[m, j]
        extreme[m] = np.maximum(extreme[m], fav_j)
        mae[m] = np.maximum(mae[m], adv_j)

        act_m = m.copy()
        act_m[m] = extreme[m] >= act

        if act_m.any():
            idx = np.where(act_m)[0]
            extreme_sel = extreme[act_m]
            cr_sel = p.close_r[act_m, j]
            drawdown_sel = extreme_sel - cr_sel

            dist_idx = np.full(len(idx), dist_normal, dtype=float)

            deep = (drawdown_sel >= thr.dd_deep) & (cr_sel <= thr.cur_low)
            if late_only:
                deep = deep & (j >= int(np.floor(thr.j_late)))

            rec = (drawdown_sel >= thr.dd_deep) & (cr_sel >= thr.cur_rec)

            if deep.any():
                dist_idx[deep] = dist_tight
            if rec.any():
                dist_idx[rec] = dist_wide if recovery_wide else dist_normal

            new_sl = extreme_sel - dist_idx * p.atr_over_r[act_m, j]
            moved = new_sl > sl[act_m]
            idx2 = np.where(act_m)[0][moved]
            sl[idx2] = new_sl[moved]
            trail_moved[idx2] = True

        hit_sl = m.copy()
        hit_sl[m] = adv_j >= -sl[m]
        if hit_sl.any():
            exit_r[hit_sl] = sl[hit_sl]
            exit_off[hit_sl] = j
            r_sl = np.where(sl[hit_sl] > 1e-9, 1, np.where(trail_moved[hit_sl], 1, 0))
            reason[hit_sl] = r_sl
            active[hit_sl] = False

    if active.any():
        lo = p.last_off[active]
        exit_r[active] = p.close_r[active, lo]
        exit_off[active] = lo
        active[:] = False

    net_return = exit_r * p.r_unit_pct - COST
    r_multiple = net_return / p.r_unit_pct

    return {
        "net_return": net_return,
        "r_multiple": r_multiple,
        "holding_bars": np.maximum(exit_off, 1),
        "reason": reason,
        "mfe_r": extreme,
        "mae_r": mae,
        "partial_any": partial_any,
    }


def _pnl_stream_from_sim(p: Paths, sim: dict[str, np.ndarray], years: list[int]) -> tuple[np.ndarray, bool]:
    """Concatenate r_multiple for taken trades across years (isolated $280 per year)."""
    stream = []
    any_blown = False
    for y in years:
        o = _orders_for_year(p, y)
        port = run_multi(
            order=o,
            p=p,
            sim=sim,
            lot_mult=np.ones(p.n, dtype=float),
            atr_pct=np.ones(p.n, dtype=float),
            probs=np.full(p.n, 0.5, dtype=float),
            trend_ok=np.ones(p.n, dtype=bool),
            cfg=CFG_PROD,
            starting=STARTING,
            fixed_lot_01=True,
            daily_on_below=80.0,
        )
        taken = np.asarray(port["taken"], dtype=int)
        if taken.size:
            stream.append(sim["r_multiple"][taken])
        any_blown = any_blown or bool(port["blown"])
    stream = np.concatenate(stream) if stream else np.array([], dtype=float)
    return stream, any_blown


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    args = ap.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)

    p = build_entry_panel(top_pct=0.21)
    h1 = load_h1(_ROOT / "artifacts" / "raw" / "XAUUSD" / "H1" / "data.parquet")
    mkt = prepare_market(h1)
    pp = Paths(p, mkt)
    years = sorted(set(int(y) for y in pp.year))

    thr = _load_thresholds()

    # baseline replay
    sim_base = simulate_timeaware_combo(
        pp,
        act=EXIT_KW_BASE["act"],
        dist_normal=EXIT_KW_BASE["dist"],
        dist_wide=EXIT_KW_BASE["dist"],
        dist_tight=EXIT_KW_BASE["dist"],
        policy=TimePolicy(
            name="baseline",
            n=None,
            kind="baseline",
            dist_normal=EXIT_KW_BASE["dist"],
            dist_wide=EXIT_KW_BASE["dist"],
            dist_tight=EXIT_KW_BASE["dist"],
            delayed=False,
            hybrid_mfe_thr=None,
        ),
    )

    base_year_rows = []
    for y in years:
        base_year_rows.append(_year_metrics(year=y, order=_orders_for_year(pp, y), p=pp, sim=sim_base))

    base_pnl_stream, base_blown = _pnl_stream_from_sim(pp, sim_base, years)

    candidates = [
        ("pullback_exit_A", dict(late_only=False, recovery_wide=True)),
        ("pullback_exit_B", dict(late_only=True, recovery_wide=True)),
    ]

    rows = []
    for name, kw in candidates:
        print(f"Candidate {name}...")
        sim = simulate_pullback_combo(
            pp,
            act=EXIT_KW_BASE["act"],
            dist_normal=0.08,
            dist_tight=0.06,
            dist_wide=0.12,
            thr=thr,
            late_only=bool(kw["late_only"]),
            recovery_wide=bool(kw["recovery_wide"]),
        )

        year_rows = []
        for y in years:
            year_rows.append(_year_metrics(year=y, order=_orders_for_year(pp, y), p=pp, sim=sim))

        pnl_stream, blown = _pnl_stream_from_sim(pp, sim, years)
        mc = iter6_montecarlo(base_pnl_stream, pnl_stream, base_blown, starting=STARTING)
        mc_dyn = mc[mc["policy"] == "dynamic"].iloc[0] if not mc.empty else None
        mc_ruin = float(mc_dyn["prob_ruin"]) if mc_dyn is not None else 1.0
        mc_worst_dd = float(mc_dyn["worst_dd"]) if mc_dyn is not None else 1.0

        rows.append(
            {
                "candidate": name,
                "PF_2024": float(next(r for r in year_rows if r["year"] == 2024)["pf"]),
                "DD_2026": float(next(r for r in year_rows if r["year"] == 2026)["dd"]),
                "Return_2026": float(next(r for r in year_rows if r["year"] == 2026)["ret"]),
                "MC_ruin": mc_ruin,
                "MC_worst_dd": mc_worst_dd,
                "yearly": year_rows,
                "thr": {
                    "dd_deep": thr.dd_deep,
                    "cur_low": thr.cur_low,
                    "cur_rec": thr.cur_rec,
                    "j_late": thr.j_late,
                },
            }
        )

    base_pf_2024 = float(next(r for r in base_year_rows if r["year"] == 2024)["pf"])
    base_dd_2026 = float(next(r for r in base_year_rows if r["year"] == 2026)["dd"])

    verdict = "STOP"
    best = None
    for r in rows:
        pf_ok = r["PF_2024"] >= 2.0
        dd_ok = r["DD_2026"] <= base_dd_2026 - 0.01
        pos_ok = r["Return_2026"] > 0
        if pf_ok and dd_ok and pos_ok:
            verdict = "PASS"
            best = r
            break

    if verdict == "STOP":
        any_pf_gain = any(r["PF_2024"] > base_pf_2024 for r in rows)
        any_dd_good = any(r["DD_2026"] < base_dd_2026 for r in rows)
        any_pf_only = any(r["PF_2024"] > base_pf_2024 and r["DD_2026"] > base_dd_2026 for r in rows)
        if any_pf_gain and any_dd_good:
            verdict = "WEAK"
        elif any_pf_only:
            verdict = "FAIL"
        else:
            verdict = "FAIL"

    best_name = best["candidate"] if best else None

    lines = [
        "# Sprint 40 — Iter 3 Pullback / Recovery Exit",
        "",
        "Quantile thresholds (TRAIN years 2021-2023, kind=bar):",
        f"- dd_deep={thr.dd_deep:.3f}  cur_low={thr.cur_low:.3f}  cur_rec={thr.cur_rec:.3f}  j_late={thr.j_late:.1f}",
        "",
        "Baseline:",
    ]
    for r in base_year_rows:
        lines.append(
            f"- {r['year']}: PF {r['pf']:.2f}  DD {r['dd']*100:.1f}%  Return {r['ret']*100:.0f}%  Trades {r['trades']}  Trail% {r['trail_pct']:.1%}"
        )

    lines += ["", "Candidates tested:"]
    for r in rows:
        lines.append(
            f"- {r['candidate']}: PF_2024 {r['PF_2024']:.2f}  DD_2026 {r['DD_2026']*100:.1f}%  Return_2026 {r['Return_2026']*100:.0f}%  MC_ruin {r['MC_ruin']:.3f}"
        )

    lines += [
        "",
        f"Iter 3 verdict: {verdict}" + (f" (best {best_name})" if best_name else ""),
        "",
        "Next: Iter 4 (volatility-aware exit) if verdict != STOP.",
    ]

    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "decision.json").write_text(
        json.dumps(
            {
                "iter": 3,
                "next_iter": 4 if verdict != "STOP" else None,
                "verdict": verdict,
                "base_pf_2024": base_pf_2024,
                "base_dd_2026": base_dd_2026,
                "best": best_name,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    (OUT / "candidates.json").write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

