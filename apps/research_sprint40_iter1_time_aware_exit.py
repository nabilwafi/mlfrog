"""Sprint 40 — Exit Structure / Time-Aware Exit (ITER 1).

Frozen:
- entry/risk/multi-trade frozen (production config)
- exit overlay only (no ML, no future_* features)

Candidates (time-aware trail semantics):
- baseline: trail activate 0.25R, dist=0.08 ATR
- wider-after-N: dist=0.08 before N bars, dist=0.12 at j>=N
- tighter-after-N: dist=0.08 before N bars, dist=0.06 at j>=N
- delayed-activate-N: trail updates only when j>=N
- hybrid-time+MFE-after-N: at j>=N and extreme>=0.50R, use dist=0.12 else dist=0.08

Metrics (isolated $280 per calendar year):
- PF, DD, Return, WR, Payoff, AvgR, MedianR, Trades, Trail%
Also computes MC ruin bootstrap probability/worst dd.

Example:
  python apps/research_sprint40_iter1_time_aware_exit.py --Ns 2,4,6,8,12
  python apps/research_sprint40_iter1_time_aware_exit.py --Ns 2
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import tools.pyc_path_hook  # noqa: F401

from apps.report_adaptive_vol_risk import iter6_montecarlo
from apps.report_multi_trade_engine import EngineCfg, run_multi
from apps.run_exit_engine_grid import (
    CAP,
    FEAT7,
    Paths,
    build_entry_panel,
)
from simulation.wf.sim import COST, load_h1, prepare_market


OUT = _ROOT / "artifacts" / "pipeline_backtest" / "exit_structure" / "sprint40"
STARTING = 280.0

# production exit fixed horizon
EXIT_KW_BASE = dict(act=0.25, dist=0.08, tp=None, partials=(), tmax=None, be=None)

# production multi-trade config
CFG_PROD = EngineCfg(
    name="parallel_mo5_d0.0_cd0_h3.0",
    family="parallel",
    max_positions=5,
    min_distance_atr=0.0,
    cooldown_bars=0,
    heat_budget_r=3.0,
)


@dataclass(frozen=True)
class TimePolicy:
    name: str
    n: int | None
    kind: str
    dist_normal: float
    dist_wide: float
    dist_tight: float
    delayed: bool
    hybrid_mfe_thr: float | None


def _parse_int_list(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def _top_bot_mean(x: np.ndarray, q: float = 0.10) -> tuple[float, float]:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (0.0, 0.0)
    k = max(1, int(round(q * len(x))))
    order = np.argsort(x)
    bot = x[order[:k]]
    top = x[order[-k:]]
    return (float(np.mean(top)), float(np.mean(bot)))


def simulate_timeaware_combo(
    p: Paths,
    *,
    act: float,
    dist_normal: float,
    dist_wide: float,
    dist_tight: float,
    policy: TimePolicy,
) -> dict[str, np.ndarray]:
    """Vectorized replay like `simulate_combo`, but dist/activation depend on bar age.

    ponytail: stop levels never loosen (monotonic tightening), matching replay_trail.
    """

    n = p.n
    T = CAP

    sl = np.full(n, -1.0)  # trail stop line in R-units (bar ATR semantics)
    extreme = np.zeros(n)  # best favorable excursion so far in R-units
    mae = np.zeros(n)
    frac = np.ones(n)
    realized = np.zeros(n)
    active = np.ones(n, dtype=bool)

    exit_r = np.zeros(n)
    exit_off = np.zeros(n, dtype=np.int64)
    reason = np.full(n, 4, dtype=np.int64)  # default TIMEOUT

    trail_moved = np.zeros(n, dtype=bool)
    be_moved = np.zeros(n, dtype=bool)  # unused (no BE in this sprint)
    partial_any = np.zeros(n, dtype=bool)  # unused (no TP/partials)

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

        # trail updates: activation by extreme>=act, plus optional delayed activation
        act_m = m.copy()
        act_m[m] = extreme[m] >= act
        if policy.delayed:
            act_m[m] = act_m[m] & (j >= int(policy.n or 0))

        if act_m.any():
            dist_idx: np.ndarray
            if policy.kind == "baseline":
                dist_idx = np.full(int(act_m.sum()), dist_normal, dtype=float)
            else:
                idx = np.where(act_m)[0]
                extreme_sel = extreme[act_m]
                if policy.kind == "wider_after_n":
                    assert policy.n is not None
                    dist_idx = np.full(len(idx), dist_normal, dtype=float)
                    if j >= policy.n:
                        dist_idx[:] = dist_wide
                elif policy.kind == "tighter_after_n":
                    assert policy.n is not None
                    dist_idx = np.full(len(idx), dist_normal, dtype=float)
                    if j >= policy.n:
                        dist_idx[:] = dist_tight
                elif policy.kind == "hybrid_time_mfe_after_n":
                    assert policy.n is not None
                    dist_idx = np.full(len(idx), dist_normal, dtype=float)
                    cond = (j >= policy.n) & (extreme_sel >= float(policy.hybrid_mfe_thr or 0.5))
                    if cond.any():
                        dist_idx[cond] = dist_wide
                elif policy.kind == "progress_mfe_wide":
                    dist_idx = np.full(len(idx), dist_normal, dtype=float)
                    cond = extreme_sel >= float(policy.hybrid_mfe_thr or 0.5)
                    if cond.any():
                        dist_idx[cond] = dist_wide
                elif policy.kind == "progress_mfe_tight":
                    dist_idx = np.full(len(idx), dist_normal, dtype=float)
                    cond = extreme_sel >= float(policy.hybrid_mfe_thr or 1.0)
                    if cond.any():
                        dist_idx[cond] = dist_tight
                else:
                    raise ValueError(f"unknown policy kind {policy.kind}")

            new_sl = extreme[act_m] - dist_idx * p.atr_over_r[act_m, j]
            moved = new_sl > sl[act_m]
            idx2 = np.where(act_m)[0][moved]
            sl[idx2] = new_sl[moved]
            trail_moved[idx2] = True

        # SL hit (pessimistic: before BE/TP; both are off)
        hit_sl = m.copy()
        hit_sl[m] = adv_j >= -sl[m]
        if hit_sl.any():
            exit_r[hit_sl] = sl[hit_sl]
            exit_off[hit_sl] = j
            # mirror run_exit_engine_grid.simulate_combo semantics (no BE/TP/partials)
            r_sl = np.where(
                sl[hit_sl] > 1e-9,
                1,
                np.where(trail_moved[hit_sl], 1, 0),
            )
            reason[hit_sl] = r_sl
            active[hit_sl] = False

    if active.any():  # safety: exit at last valid close
        lo = p.last_off[active]
        exit_r[active] = p.close_r[active, lo]
        exit_off[active] = lo
        active[:] = False

    # no partials, no TP, no BE in this sprint
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


def _year_metrics(
    *,
    year: int,
    order: np.ndarray,
    p: Paths,
    sim: dict[str, np.ndarray],
) -> dict[str, Any]:
    taken_port = run_multi(
        order=order,
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
    r = sim["r_multiple"][taken] if taken.size else np.array([], dtype=float)
    reasons = sim["reason"][taken] if taken.size else np.array([], dtype=int)

    trail_pct = float(np.mean(reasons == 1)) if reasons.size else 0.0
    wr = float(np.mean(r > 0)) if r.size else 0.0

    wins = r[r > 0]
    losses = r[r < 0]
    payoff = float(wins.mean() / abs(losses.mean())) if losses.size else (float("inf") if wins.size else 0.0)

    top_mean, bot_mean = _top_bot_mean(r, q=0.10) if r.size else (0.0, 0.0)

    return {
        "year": int(year),
        "trades": int(taken_port["n_trades"]),
        "pf": float(taken_port["profit_factor"]) if np.isfinite(taken_port["profit_factor"]) else 0.0,
        "dd": float(taken_port["max_drawdown"]),
        "ret": float(taken_port["total_return"]),
        "wr": wr,
        "payoff": payoff,
        "avg_r": float(np.mean(r)) if r.size else 0.0,
        "median_r": float(np.median(r)) if r.size else 0.0,
        "top10_mean_r": top_mean,
        "bot10_mean_r": bot_mean,
        "trail_pct": trail_pct,
        "blown": bool(taken_port["blown"]),
        "final_equity": float(taken_port["final_equity"]),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ns", type=str, default="2,4,6,8,12")
    ap.add_argument("--Ns-limit", type=int, default=0, help="debug: keep only first K N values")
    args = ap.parse_args(argv)

    Ns = _parse_int_list(args.Ns)
    Ns = Ns[: args.Ns_limit] if args.Ns_limit and len(Ns) else Ns

    OUT.mkdir(parents=True, exist_ok=True)
    run_dir = OUT / "iter1_time_aware_exit"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("Build production entry panel (FEAT7 top 21%)...")
    panel = build_entry_panel(top_pct=0.21)
    h1 = load_h1(_ROOT / "artifacts" / "raw" / "XAUUSD" / "H1" / "data.parquet")
    mkt = prepare_market(h1)
    p = Paths(panel, mkt)

    order_all = np.argsort(pd.DatetimeIndex(p.ts).as_unit("ns").asi8, kind="stable")
    years = sorted(set(int(y) for y in p.year))

    # baseline simulation (fixed dist 0.08)
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
            dist_normal=0.08,
            dist_wide=0.08,
            dist_tight=0.08,
            delayed=False,
            hybrid_mfe_thr=None,
        ),
    )

    base_year_rows = []
    for y in years:
        idx = np.where(p.year == y)[0]
        o = idx[np.argsort(pd.DatetimeIndex(p.ts.iloc[idx]).as_unit("ns").asi8, kind="stable")]
        # o is indices into p arrays; run_multi expects order over p-index-space
        base_year_rows.append(_year_metrics(year=y, order=o, p=p, sim=sim_base))

    baseline = {
        "policy": "baseline_a0.25_d0.08",
        "features": list(FEAT7),
        "years": years,
        "baseline_yearly": base_year_rows,
    }

    # Validate against known 2024/2026 (production reference)
    ref_2024_pf = 1.96
    ref_2026_dd = 0.328
    pf_2024 = float(next(r for r in base_year_rows if r["year"] == 2024)["pf"])
    dd_2026 = float(next(r for r in base_year_rows if r["year"] == 2026)["dd"])
    print(f"Baseline check: pf_2024={pf_2024:.2f} (ref {ref_2024_pf:.2f}), dd_2026={dd_2026:.3f} (ref {ref_2026_dd:.3f})")

    # Candidate generation
    candidates: list[TimePolicy] = []
    for N in Ns:
        candidates.append(
            TimePolicy(
                name=f"wider_after_{N}",
                n=N,
                kind="wider_after_n",
                dist_normal=0.08,
                dist_wide=0.12,
                dist_tight=0.06,
                delayed=False,
                hybrid_mfe_thr=None,
            )
        )
        candidates.append(
            TimePolicy(
                name=f"tighter_after_{N}",
                n=N,
                kind="tighter_after_n",
                dist_normal=0.08,
                dist_wide=0.12,
                dist_tight=0.06,
                delayed=False,
                hybrid_mfe_thr=None,
            )
        )
        candidates.append(
            TimePolicy(
                name=f"delayed_activate_{N}",
                n=N,
                kind="baseline",  # dist stays normal; activation is delayed
                dist_normal=0.08,
                dist_wide=0.12,
                dist_tight=0.06,
                delayed=True,
                hybrid_mfe_thr=None,
            )
        )
        candidates.append(
            TimePolicy(
                name=f"hybrid_time_mfe_after_{N}",
                n=N,
                kind="hybrid_time_mfe_after_n",
                dist_normal=0.08,
                dist_wide=0.12,
                dist_tight=0.06,
                delayed=False,
                hybrid_mfe_thr=0.50,
            )
        )

    # Evaluate all candidates
    rows: list[dict[str, Any]] = []
    mc_rows: list[dict[str, Any]] = []

    # MC needs pooled pnl stream (we'll use concatenated year-wise pnl, trade-level)
    base_pnl_stream = []
    for y in years:
        idx = np.where(p.year == y)[0]
        o = idx[np.argsort(pd.DatetimeIndex(p.ts.iloc[idx]).as_unit("ns").asi8, kind="stable")]
        port = run_multi(
            order=o,
            p=p,
            sim=sim_base,
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
        base_pnl_stream.append(sim_base["r_multiple"][taken])
    base_pnl_stream = np.concatenate(base_pnl_stream) if base_pnl_stream else np.array([], dtype=float)

    base_blown = bool(any(r["blown"] for r in base_year_rows))

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
            idx = np.where(p.year == y)[0]
            o = idx[np.argsort(pd.DatetimeIndex(p.ts.iloc[idx]).as_unit("ns").asi8, kind="stable")]
            year_rows.append(_year_metrics(year=y, order=o, p=p, sim=sim))

        # aggregate pass/fail signals for 2024 and 2026
        pf_2024 = float(next(r for r in year_rows if r["year"] == 2024)["pf"])
        dd_2026 = float(next(r for r in year_rows if r["year"] == 2026)["dd"])
        ret_2026 = float(next(r for r in year_rows if r["year"] == 2026)["ret"])
        trades_2024 = int(next(r for r in year_rows if r["year"] == 2024)["trades"])
        trades_2026 = int(next(r for r in year_rows if r["year"] == 2026)["trades"])

        # trade-level pnl stream for MC
        pnl_stream = []
        for y in years:
            idx = np.where(p.year == y)[0]
            o = idx[np.argsort(pd.DatetimeIndex(p.ts.iloc[idx]).as_unit("ns").asi8, kind="stable")]
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
            pnl_stream.append(sim["r_multiple"][taken])
        pnl_stream = np.concatenate(pnl_stream) if pnl_stream else np.array([], dtype=float)

        mc = iter6_montecarlo(base_pnl_stream, pnl_stream, base_blown, starting=STARTING)
        mc_dyn = mc[mc["policy"] == "dynamic"].iloc[0] if not mc.empty else None
        mc_ruin = float(mc_dyn["prob_ruin"]) if mc_dyn is not None else 1.0
        mc_worst_dd = float(mc_dyn["worst_dd"]) if mc_dyn is not None else 1.0

        row = {
            "candidate": cand.name,
            "n": cand.n,
            "type": cand.kind,
            "PF_2024": pf_2024,
            "DD_2026": dd_2026,
            "Return_2026": ret_2026,
            "Trades_2024": trades_2024,
            "Trades_2026": trades_2026,
            "MC_ruin": mc_ruin,
            "MC_worst_dd": mc_worst_dd,
            "yearly": year_rows,
        }

        rows.append(row)

    # Decision heuristics for Iter 1:
    # PASS if PF2024 >= 2.0, DD2026 <= base DD2026 - 0.01, Return2026 > 0, and PF overall not worse.
    base_pf_2024 = float(next(r for r in base_year_rows if r["year"] == 2024)["pf"])
    base_dd_2026 = float(next(r for r in base_year_rows if r["year"] == 2026)["dd"])

    verdict = "STOP"
    best = None
    for r in sorted(rows, key=lambda x: (x["PF_2024"], -x["DD_2026"], x["Return_2026"]), reverse=False):
        pf_ok = r["PF_2024"] >= 2.0
        dd_ok = r["DD_2026"] <= base_dd_2026 - 0.01
        pos_ok = r["Return_2026"] > 0
        if pf_ok and dd_ok and pos_ok:
            verdict = "PASS"
            best = r
            break

    if verdict == "STOP":
        # If any candidate beats baseline PF on 2024 without DD blow-up, mark WEAK.
        any_pf = any(r["PF_2024"] > base_pf_2024 for r in rows)
        any_dd_bad = any(r["DD_2026"] > base_dd_2026 for r in rows)
        if any_pf and not all((r["DD_2026"] >= base_dd_2026) for r in rows):
            verdict = "WEAK"
        else:
            verdict = "FAIL"

    best_name = best["candidate"] if best else None

    summary_lines = [
        "# Sprint 40 — Iter 1 Time-Aware Exit",
        "",
        "Frozen production stack (entry/risk/multi-trade). Exit overlay only.",
        "",
        f"Baseline (trail a0.25/d0.08, fixed lot 0.01, parallel mo5, heat 3R, daily_on_below 80):",
        "",
    ]
    for r in base_year_rows:
        summary_lines.append(
            f"- {r['year']}: PF {r['pf']:.2f}  DD {r['dd']*100:.1f}%  Return {r['ret']*100:.0f}%  Trades {r['trades']}  Trail% {r['trail_pct']:.1%}"
        )

    summary_lines += [
        "",
        "Candidates tested:",
    ]
    for r in rows:
        summary_lines.append(
            f"- {r['candidate']}: PF_2024 {r['PF_2024']:.2f}  DD_2026 {r['DD_2026']*100:.1f}%  Return_2026 {r['Return_2026']*100:.0f}%  MC_ruin {r['MC_ruin']:.3f}"
        )

    summary_lines += [
        "",
        f"Iter 1 verdict: {verdict}" + (f" (best {best_name})" if best_name else ""),
        "",
        "Next: Iter 2 (progress-aware exit) only if not STOP.",
    ]

    (run_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    (run_dir / "baseline.json").write_text(json.dumps(baseline, indent=2, default=str), encoding="utf-8")
    (run_dir / "candidates.json").write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")

    decision = {
        "iter": 1,
        "next_iter": 2 if verdict != "STOP" else None,
        "verdict": verdict,
        "base_pf_2024": base_pf_2024,
        "base_dd_2026": base_dd_2026,
        "best": best_name,
    }
    (run_dir / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")

    return 0 if verdict != "STOP" else 0


if __name__ == "__main__":
    raise SystemExit(main())

