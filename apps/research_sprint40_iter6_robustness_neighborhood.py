"""Sprint 40 — Iter 6 Robustness Neighborhood (exit structure only).

Neighborhood around best policy family from Iter 5:
- delayed activation after N bars (delayed trail move)

Grid:
- activation: 0.20 / 0.25 / 0.30
- distance: 0.06 / 0.08 / 0.10 / 0.12
- time threshold: 4 / 6 / 8 / 10

We look for a stable REGION (multiple parameter combos passing),
and do NOT rely on one best single point.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import tools.pyc_path_hook  # noqa: F401

from apps.report_adaptive_vol_risk import iter6_montecarlo
from apps.report_multi_trade_engine import run_multi
from apps.research_sprint40_iter1_time_aware_exit import (
    CFG_PROD,
    EXIT_KW_BASE,
    STARTING,
    TimePolicy,
    _year_metrics,
    simulate_timeaware_combo,
)
from apps.run_exit_engine_grid import Paths, build_entry_panel
from simulation.wf.sim import COST, entry_indices, load_h1, prepare_market


OUT = _ROOT / "artifacts" / "pipeline_backtest" / "exit_structure" / "sprint40" / "iter6_robustness_neighborhood"


def _orders_for_year(p: Paths, year: int) -> np.ndarray:
    idx = np.where(p.year == year)[0]
    return idx[np.argsort(pd.DatetimeIndex(p.ts.iloc[idx]).as_unit("ns").asi8, kind="stable")]


def _pnl_stream_from_sim(p: Paths, sim: dict[str, np.ndarray], years: list[int]) -> tuple[np.ndarray, bool]:
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
    ap.add_argument("--limit", type=int, default=0, help="debug: limit number of configs")
    args = ap.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)

    years_focus = [2023, 2024, 2025, 2026]

    activation_grid = [0.20, 0.25, 0.30]
    distance_grid = [0.06, 0.08, 0.10, 0.12]
    time_grid = [4, 6, 8, 10]

    configs = list(product(activation_grid, distance_grid, time_grid))
    if args.limit and args.limit > 0:
        configs = configs[: args.limit]

    print(f"Configs: {len(configs)}")

    # Build production frozen entry panel and market once
    print("Build production entry panel (FEAT7 top 21%)...")
    panel = build_entry_panel(top_pct=0.21)
    h1 = load_h1(_ROOT / "artifacts" / "raw" / "XAUUSD" / "H1" / "data.parquet")
    mkt = prepare_market(h1)
    p = Paths(panel, mkt)

    # Baseline
    print("Baseline replay...")
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
    for y in years_focus:
        base_year_rows.append(_year_metrics(year=y, order=_orders_for_year(p, y), p=p, sim=sim_base))
    base_pf_2024 = float(next(r for r in base_year_rows if r["year"] == 2024)["pf"])
    base_dd_2026 = float(next(r for r in base_year_rows if r["year"] == 2026)["dd"])

    base_pnl_stream, base_blown = _pnl_stream_from_sim(p, sim_base, years_focus)

    passing = []
    all_rows = []

    for k, (act, dist, n_time) in enumerate(configs, start=1):
        if k % 8 == 0:
            print(f"progress {k}/{len(configs)}")
        policy = TimePolicy(
            name=f"delayed_activate_N{n_time}",
            n=int(n_time),
            kind="baseline",
            dist_normal=float(dist),
            dist_wide=float(dist),
            dist_tight=float(dist),  # only one dist for this family
            delayed=True,
            hybrid_mfe_thr=None,
        )
        sim = simulate_timeaware_combo(
            p,
            act=float(act),
            dist_normal=float(dist),
            dist_wide=float(dist),
            dist_tight=float(dist),
            policy=policy,
        )

        year_rows = []
        for y in years_focus:
            year_rows.append(_year_metrics(year=y, order=_orders_for_year(p, y), p=p, sim=sim))

        pf_2024 = float(next(r for r in year_rows if r["year"] == 2024)["pf"])
        dd_2026 = float(next(r for r in year_rows if r["year"] == 2026)["dd"])
        ret_2026 = float(next(r for r in year_rows if r["year"] == 2026)["ret"])

        row = {
            "activation": float(act),
            "distance": float(dist),
            "time_threshold": int(n_time),
            "PF_2024": pf_2024,
            "DD_2026": dd_2026,
            "Return_2026": ret_2026,
            "yearly": year_rows,
        }
        all_rows.append(row)

        pf_ok = pf_2024 >= 2.0
        dd_ok = dd_2026 <= base_dd_2026 - 0.01
        pos_ok = ret_2026 > 0
        if pf_ok and dd_ok and pos_ok:
            passing.append(row)

    # MC only for passing region (should be few)
    mc_rows = []
    for r in passing:
        act = r["activation"]
        dist = r["distance"]
        n_time = r["time_threshold"]
        policy = TimePolicy(
            name=f"delayed_activate_N{n_time}",
            n=int(n_time),
            kind="baseline",
            dist_normal=float(dist),
            dist_wide=float(dist),
            dist_tight=float(dist),
            delayed=True,
            hybrid_mfe_thr=None,
        )
        sim = simulate_timeaware_combo(
            p,
            act=float(act),
            dist_normal=float(dist),
            dist_wide=float(dist),
            dist_tight=float(dist),
            policy=policy,
        )
        pnl_stream, blown = _pnl_stream_from_sim(p, sim, years_focus)
        mc = iter6_montecarlo(base_pnl_stream, pnl_stream, base_blown, starting=STARTING)
        mc_dyn = mc[mc["policy"] == "dynamic"].iloc[0] if not mc.empty else None
        mc_rows.append(
            {
                "activation": float(act),
                "distance": float(dist),
                "time_threshold": int(n_time),
                "MC_ruin": float(mc_dyn["prob_ruin"]) if mc_dyn is not None else 1.0,
                "MC_worst_dd": float(mc_dyn["worst_dd"]) if mc_dyn is not None else 1.0,
            }
        )

    mc_df = pd.DataFrame(mc_rows)
    passing_df = pd.DataFrame(passing)
    if not mc_df.empty and not passing_df.empty:
        passing_df = passing_df.merge(mc_df, on=["activation", "distance", "time_threshold"], how="left")

    # Verdict: PASS if multiple configs pass (stable region)
    verdict = "FAIL"
    best = None
    if len(passing_df) >= 3:
        verdict = "PASS"
        best = passing_df.sort_values(["PF_2024", "DD_2026"], ascending=[False, True]).iloc[0].to_dict()
    elif len(passing_df) >= 1:
        verdict = "WEAK"
        best = passing_df.sort_values(["PF_2024", "DD_2026"], ascending=[False, True]).iloc[0].to_dict()

    summary_lines = [
        "# Sprint 40 — Iter 6 Robustness Neighborhood",
        "",
        f"Years focus: {years_focus}",
        "",
        "Baseline:",
    ]
    for r in base_year_rows:
        summary_lines.append(
            f"- {r['year']}: PF {r['pf']:.2f}  DD {r['dd']*100:.1f}%  Return {r['ret']*100:.0f}%"
        )

    summary_lines += [
        "",
        f"Passing configs: {len(passing_df)} (PF_2024 >=2.0, DD_2026 <= base-1pp, Return_2026 > 0)",
        "",
        "Passing (with MC if computed):",
    ]
    if len(passing_df):
        summary_lines.append(passing_df.to_string(index=False))
    else:
        summary_lines.append("(none)")

    summary_lines += [
        "",
        f"Iter 6 verdict: {verdict}" + (f" best={best.get('activation')},{best.get('distance')},N={best.get('time_threshold')}" if best else ""),
        "",
        "Next: Iter 7 (STOP) or further analysis per user decision.",
    ]

    (OUT / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    (OUT / "decision.json").write_text(
        json.dumps(
            {
                "iter": 6,
                "verdict": verdict,
                "n_passing": int(len(passing_df)),
                "best": best,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (OUT / "all_configs.json").write_text(json.dumps(all_rows, indent=2, default=str), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

