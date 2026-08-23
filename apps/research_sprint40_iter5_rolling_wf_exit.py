"""Sprint 40 — Iter 5 Rolling WF Confirmation.

Minimum requirement:
- Evaluate candidates that look promising on years 2023-2026.
- No picking based on pooled performance.

Implementation note:
Entry panels are already built in a rolling-WF manner via `build_entry_panel()`.
Here we re-measure exit overlay performance by isolating calendar years
and requiring stability across 2023-2026.
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

from apps.research_sprint40_iter1_time_aware_exit import (
    EXIT_KW_BASE,
    TimePolicy,
    _year_metrics,
    simulate_timeaware_combo,
)
from apps.run_exit_engine_grid import Paths, build_entry_panel
from simulation.wf.sim import load_h1, prepare_market


OUT = _ROOT / "artifacts" / "pipeline_backtest" / "exit_structure" / "sprint40" / "iter5_rolling_wf_exit"


def _orders_for_year(p: Paths, year: int) -> np.ndarray:
    idx = np.where(p.year == year)[0]
    return idx[np.argsort(pd.DatetimeIndex(p.ts.iloc[idx]).as_unit("ns").asi8, kind="stable")]


def main(argv: list[str] | None = None) -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    years_focus = [2023, 2024, 2025, 2026]

    # Rebuild identical production frozen entry panel and market
    print("Build production entry panel (FEAT7 top 21%)...")
    panel = build_entry_panel(top_pct=0.21)
    h1 = load_h1(_ROOT / "artifacts" / "raw" / "XAUUSD" / "H1" / "data.parquet")
    mkt = prepare_market(h1)
    p = Paths(panel, mkt)

    # Production baseline
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

    base_rows = []
    for y in years_focus:
        base_rows.append(_year_metrics(year=y, order=_orders_for_year(p, y), p=p, sim=sim_base))

    base_pf_2024 = float(next(r for r in base_rows if r["year"] == 2024)["pf"])
    base_dd_2026 = float(next(r for r in base_rows if r["year"] == 2026)["dd"])

    # Candidate(s) from prior iterations
    candidates = [
        TimePolicy(
            name="delayed_activate_2",
            n=2,
            kind="baseline",  # dist stays normal; activation delayed
            dist_normal=0.08,
            dist_wide=0.12,
            dist_tight=0.06,
            delayed=True,
            hybrid_mfe_thr=None,
        )
    ]

    rows = []
    for cand in candidates:
        print(f"Candidate {cand.name} WF confirmation...")
        sim = simulate_timeaware_combo(
            p,
            act=EXIT_KW_BASE["act"],
            dist_normal=cand.dist_normal,
            dist_wide=cand.dist_wide,
            dist_tight=cand.dist_tight,
            policy=cand,
        )
        year_rows = []
        for y in years_focus:
            year_rows.append(_year_metrics(year=y, order=_orders_for_year(p, y), p=p, sim=sim))
        pf_2024 = float(next(r for r in year_rows if r["year"] == 2024)["pf"])
        dd_2026 = float(next(r for r in year_rows if r["year"] == 2026)["dd"])
        ret_2026 = float(next(r for r in year_rows if r["year"] == 2026)["ret"])
        min_trades = int(min(r["trades"] for r in year_rows))
        min_pf = float(min(r["pf"] for r in year_rows))
        rows.append(
            {
                "candidate": cand.name,
                "yearly": year_rows,
                "PF_2024": pf_2024,
                "DD_2026": dd_2026,
                "Return_2026": ret_2026,
                "min_trades": min_trades,
                "min_pf": min_pf,
            }
        )

    # Verdict gate for next iter 6 robustness
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
        verdict = "WEAK"

    summary_lines = [
        "# Sprint 40 — Iter 5 Rolling WF Confirmation",
        "",
        f"Focus years: {years_focus}",
        "",
        "Baseline (prod a0.25/d0.08):",
    ]
    for r in base_rows:
        summary_lines.append(
            f"- {r['year']}: PF {r['pf']:.2f}  DD {r['dd']*100:.1f}%  Return {r['ret']*100:.0f}%  Trades {r['trades']}"
        )

    summary_lines += ["", "Candidates:"]
    for r in rows:
        summary_lines.append(
            f"- {r['candidate']}: PF_2024 {r['PF_2024']:.2f}  DD_2026 {r['DD_2026']*100:.1f}%  Return_2026 {r['Return_2026']*100:.0f}%  minPF {r['min_pf']:.2f}"
        )

    summary_lines += [
        "",
        f"Iter 5 verdict: {verdict}" + (f" (best {best['candidate']})" if best else ""),
        "",
        "Next: Iter 6 robustness if verdict != STOP.",
    ]

    (OUT / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    (OUT / "decision.json").write_text(
        json.dumps(
            {
                "iter": 5,
                "next_iter": 6 if verdict != "STOP" else None,
                "verdict": verdict,
                "base_pf_2024": base_pf_2024,
                "base_dd_2026": base_dd_2026,
                "best": best["candidate"] if best else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (OUT / "candidates.json").write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

