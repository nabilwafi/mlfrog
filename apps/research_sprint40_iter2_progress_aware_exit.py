"""Sprint 40 — Progress-Aware Exit (ITER 2).

Frozen: same production entry/risk/multi-trade as Iter 1.
Exit overlay only: dist changes based on in-trade progress (MFE so far in R).

Candidates:
1) baseline
2) wider-after-MFE>=0.5R
3) wider-after-MFE>=1.0R
4) tighter-after-MFE>=1.0R
5) time+MFE hybrid (uses iter1 best time N=2, and MFE>=0.5R)

No ML; fixed/simple thresholds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import tools.pyc_path_hook  # noqa: F401

from apps.report_adaptive_vol_risk import iter6_montecarlo
from apps.run_exit_engine_grid import FEAT7, Paths, build_entry_panel
from apps.research_sprint40_iter1_time_aware_exit import (
    CFG_PROD,
    EXIT_KW_BASE,
    STARTING,
    TimePolicy,
    _top_bot_mean,
    _year_metrics,
    simulate_timeaware_combo,
)
from simulation.wf.sim import load_h1, prepare_market


OUT = _ROOT / "artifacts" / "pipeline_backtest" / "exit_structure" / "sprint40" / "iter2_progress_aware_exit"


def _build_entry_and_paths():
    print("Build production entry panel (FEAT7 top 21%)...")
    panel = build_entry_panel(top_pct=0.21)
    h1 = load_h1(_ROOT / "artifacts" / "raw" / "XAUUSD" / "H1" / "data.parquet")
    mkt = prepare_market(h1)
    p = Paths(panel, mkt)
    return p


def _orders_for_year(p: Paths, year: int) -> np.ndarray:
    idx = np.where(p.year == year)[0]
    return idx[np.argsort(pd.DatetimeIndex(p.ts.iloc[idx]).as_unit("ns").asi8, kind="stable")]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-n", type=int, default=2, help="time component for candidate 5")
    args = ap.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)

    p = _build_entry_and_paths()
    years = sorted(set(int(y) for y in p.year))

    print("Baseline exit replay...")
    sim_base = simulate_timeaware_combo(
        p,
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
        o = _orders_for_year(p, y)
        base_year_rows.append(_year_metrics(year=y, order=o, p=p, sim=sim_base))

    base_pf_2024 = float(next(r for r in base_year_rows if r["year"] == 2024)["pf"])
    base_dd_2026 = float(next(r for r in base_year_rows if r["year"] == 2026)["dd"])
    base_ret_2026 = float(next(r for r in base_year_rows if r["year"] == 2026)["ret"])

    # MC stream: concatenate r_multiple for taken trades across years (isolated per year)
    base_pnl_stream = []
    for y in years:
        o = _orders_for_year(p, y)
        port = _year_metrics(year=y, order=o, p=p, sim=sim_base)
        # _year_metrics already ran run_multi; we need taken stream again for MC.
        # To keep code minimal, recompute quickly here by rerunning run_multi with same args:
        from apps.report_multi_trade_engine import run_multi

        sim = sim_base
        # order o indexes are in p-array space
        taken_port = run_multi(
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
        taken = np.asarray(taken_port["taken"], dtype=int)
        base_pnl_stream.append(sim["r_multiple"][taken])
    base_pnl_stream = np.concatenate(base_pnl_stream) if base_pnl_stream else np.array([], dtype=float)
    base_blown = bool(any(r["blown"] for r in base_year_rows))

    candidates: list[TimePolicy] = [
        TimePolicy(
            name="baseline",
            n=None,
            kind="baseline",
            dist_normal=EXIT_KW_BASE["dist"],
            dist_wide=EXIT_KW_BASE["dist"],
            dist_tight=EXIT_KW_BASE["dist"],
            delayed=False,
            hybrid_mfe_thr=None,
        ),
        TimePolicy(
            name="progress_wide_after_mfe_0.5",
            n=None,
            kind="progress_mfe_wide",
            dist_normal=0.08,
            dist_wide=0.12,
            dist_tight=0.06,
            delayed=False,
            hybrid_mfe_thr=0.50,
        ),
        TimePolicy(
            name="progress_wide_after_mfe_1.0",
            n=None,
            kind="progress_mfe_wide",
            dist_normal=0.08,
            dist_wide=0.12,
            dist_tight=0.06,
            delayed=False,
            hybrid_mfe_thr=1.00,
        ),
        TimePolicy(
            name="progress_tight_after_mfe_1.0",
            n=None,
            kind="progress_mfe_tight",
            dist_normal=0.08,
            dist_wide=0.12,
            dist_tight=0.06,
            delayed=False,
            hybrid_mfe_thr=1.00,
        ),
        TimePolicy(
            name="time+MFE_hybrid_N2_mfe_0.5",
            n=int(args.time_n),
            kind="hybrid_time_mfe_after_n",
            dist_normal=0.08,
            dist_wide=0.12,
            dist_tight=0.06,
            delayed=False,
            hybrid_mfe_thr=0.50,
        ),
    ]

    rows = []
    for cand in candidates:
        print(f"Candidate {cand.name}...")
        sim = simulate_timeaware_combo(
            p,
            act=EXIT_KW_BASE["act"],
            dist_normal=cand.dist_normal,
            dist_wide=cand.dist_wide,
            dist_tight=cand.dist_tight,
            policy=cand,
        )

        year_rows = []
        for y in years:
            o = _orders_for_year(p, y)
            year_rows.append(_year_metrics(year=y, order=o, p=p, sim=sim))

        pf_2024 = float(next(r for r in year_rows if r["year"] == 2024)["pf"])
        dd_2026 = float(next(r for r in year_rows if r["year"] == 2026)["dd"])
        ret_2026 = float(next(r for r in year_rows if r["year"] == 2026)["ret"])
        trades_2024 = int(next(r for r in year_rows if r["year"] == 2024)["trades"])

        # MC
        pnl_stream = []
        for y in years:
            o = _orders_for_year(p, y)
            from apps.report_multi_trade_engine import run_multi

            taken_port = run_multi(
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
            taken = np.asarray(taken_port["taken"], dtype=int)
            pnl_stream.append(sim["r_multiple"][taken])
        pnl_stream = np.concatenate(pnl_stream) if pnl_stream else np.array([], dtype=float)

        mc = iter6_montecarlo(base_pnl_stream, pnl_stream, base_blown, starting=STARTING)
        mc_dyn = mc[mc["policy"] == "dynamic"].iloc[0] if not mc.empty else None
        mc_ruin = float(mc_dyn["prob_ruin"]) if mc_dyn is not None else 1.0
        mc_worst_dd = float(mc_dyn["worst_dd"]) if mc_dyn is not None else 1.0

        rows.append(
            {
                "candidate": cand.name,
                "PF_2024": pf_2024,
                "DD_2026": dd_2026,
                "Return_2026": ret_2026,
                "Trades_2024": trades_2024,
                "MC_ruin": mc_ruin,
                "MC_worst_dd": mc_worst_dd,
                "yearly": year_rows,
            }
        )

    # Decision
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
        # any candidate improves 2024 PF while not worsening DD2026 drastically
        any_pf_gain = any(r["PF_2024"] > base_pf_2024 for r in rows if r["candidate"] != "baseline")
        any_dd_good = any(r["DD_2026"] < base_dd_2026 for r in rows if r["candidate"] != "baseline")
        if any_pf_gain and any_dd_good:
            verdict = "WEAK"
        elif any(r["DD_2026"] > base_dd_2026 and r["PF_2024"] > base_pf_2024 for r in rows if r["candidate"] != "baseline"):
            verdict = "FAIL"
        else:
            verdict = "FAIL"

    best_name = best["candidate"] if best else None

    lines = [
        "# Sprint 40 — Iter 2 Progress-Aware Exit",
        "",
        "Frozen production stack; exit overlay only.",
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
        f"Iter 2 verdict: {verdict}" + (f" (best {best_name})" if best_name else ""),
        "",
        "Next: Iter 3 (pullback / recovery exit) if verdict != STOP.",
    ]

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "decision.json").write_text(
        json.dumps(
            {
                "iter": 2,
                "next_iter": 3 if verdict != "STOP" else None,
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

