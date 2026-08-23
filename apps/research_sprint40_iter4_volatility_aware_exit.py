"""Sprint 40 — Volatility-Aware Exit (ITER 4).

Frozen: production entry/risk/multi-trade.
Exit overlay only.

Use only PIT volatility features:
- atr_pct_entry
- atr_pct_now
- rolling_quantile

Candidates (fixed/simple):
- baseline
- wider trail when HIGH VOL (atr_pct_now >= q70)
- tighter trail when HIGH VOL (atr_pct_now >= q70)
- wider trail when LOW VOL (atr_pct_now <= q30)
- time + volatility: wider dist when j >= N (fixed N=2) AND HIGH VOL

No ML, no future_*.
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
from apps.run_exit_engine_grid import CAP, Paths, build_entry_panel
from apps.research_sprint40_iter1_time_aware_exit import (
    CFG_PROD,
    EXIT_KW_BASE,
    STARTING,
    TimePolicy,
    _year_metrics,
    simulate_timeaware_combo,
)
from simulation.wf.sim import COST, entry_indices, load_h1, prepare_market


OUT = _ROOT / "artifacts" / "pipeline_backtest" / "exit_structure" / "sprint40" / "iter4_volatility_aware_exit"


@dataclass(frozen=True)
class VolThr:
    atr_pct_now_hi: float
    atr_pct_now_lo: float
    n_time: int


def _orders_for_year(p: Paths, year: int) -> np.ndarray:
    idx = np.where(p.year == year)[0]
    return idx[np.argsort(pd.DatetimeIndex(p.ts.iloc[idx]).as_unit("ns").asi8, kind="stable")]


def _atr_pctile(atr: np.ndarray, win: int = 252) -> np.ndarray:
    # Same construction as Sprint 39 (point-in-time atr percentile)
    s = pd.Series(atr)
    return s.rolling(win, min_periods=50).apply(lambda x: float(np.mean(x <= x[-1])), raw=True).to_numpy()


def _load_thresholds(n_time: int) -> VolThr:
    ds = _ROOT / "artifacts" / "pipeline_backtest" / "rolling_wf" / "sprint39_exit_state" / "exit_state_dataset.parquet"
    df = pd.read_parquet(ds)
    bar = df[(df["kind"] == "bar") & (df["year"] >= 2021) & (df["year"] <= 2023)].copy()
    atr_now = bar["atr_pct_now"].astype(float).to_numpy()
    atr_now_hi = float(np.nanquantile(atr_now, 0.70))
    atr_now_lo = float(np.nanquantile(atr_now, 0.30))
    print(f"Vol thresholds (train 2021-2023): atr_pct_now_hi={atr_now_hi:.3f} atr_pct_now_lo={atr_now_lo:.3f} n_time={n_time}")
    return VolThr(atr_pct_now_hi=atr_now_hi, atr_pct_now_lo=atr_now_lo, n_time=n_time)


def simulate_volaware_combo(
    p: Paths,
    *,
    act: float,
    dist_normal: float,
    dist_wide: float,
    dist_tight: float,
    atr_pct_bar: np.ndarray,
    ei: np.ndarray,
    thr: VolThr,
    mode: str,
) -> dict[str, np.ndarray]:
    """Dist chosen at trail update time based on atr_pct_now from market bars."""

    n = p.n
    T = CAP

    sl = np.full(n, -1.0)
    extreme = np.zeros(n)
    mae = np.zeros(n)
    active = np.ones(n, dtype=bool)

    exit_r = np.zeros(n)
    exit_off = np.zeros(n, dtype=np.int64)
    reason = np.full(n, 4, dtype=np.int64)

    trail_moved = np.zeros(n, dtype=bool)
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
            dist_idx = np.full(len(idx), dist_normal, dtype=float)

            # atr_pct_now at bar j for these entries
            bar_idx = ei[act_m] + j
            atr_now_sel = atr_pct_bar[bar_idx]

            if mode == "baseline":
                pass
            elif mode == "wider_high":
                mask = atr_now_sel >= thr.atr_pct_now_hi
                if mask.any():
                    dist_idx[mask] = dist_wide
            elif mode == "tighter_high":
                mask = atr_now_sel >= thr.atr_pct_now_hi
                if mask.any():
                    dist_idx[mask] = dist_tight
            elif mode == "wider_low":
                mask = atr_now_sel <= thr.atr_pct_now_lo
                if mask.any():
                    dist_idx[mask] = dist_wide
            elif mode == "time_and_vol_wide":
                mask = (j >= thr.n_time) & (atr_now_sel >= thr.atr_pct_now_hi)
                if np.any(mask):
                    dist_idx[mask] = dist_wide
            else:
                raise ValueError(f"unknown mode {mode}")

            new_sl = extreme_sel - dist_idx * p.atr_over_r[act_m, j]
            moved = new_sl > sl[act_m]
            idx2 = np.where(act_m)[0][moved]
            sl[idx2] = new_sl[moved]
            trail_moved[idx2] = True

        hit_sl = m.copy()
        hit_sl[m] = p.adv[m, j] >= -sl[m]
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


def _pnl_stream(p: Paths, sim: dict[str, np.ndarray], years: list[int]) -> tuple[np.ndarray, bool]:
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
    ap.add_argument("--time-n", type=int, default=2)
    args = ap.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)

    p = build_entry_panel(top_pct=0.21)
    h1 = load_h1(_ROOT / "artifacts" / "raw" / "XAUUSD" / "H1" / "data.parquet")
    mkt = prepare_market(h1)
    pp = Paths(p, mkt)
    years = sorted(set(int(y) for y in pp.year))

    thr = _load_thresholds(n_time=int(args.time_n))

    # Build atr_pct_now lookup over market bars
    atr_pct_bar = _atr_pctile(mkt["atr"])
    ei = entry_indices(pp.panel, mkt["ts"])

    # Baseline for reference
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

    base_pnl_stream, base_blown = _pnl_stream(pp, sim_base, years)

    modes = [
        ("baseline", "baseline"),
        ("wider_high_vol", "wider_high"),
        ("tighter_high_vol", "tighter_high"),
        ("wider_low_vol", "wider_low"),
        ("time_plus_vol_wide", "time_and_vol_wide"),
    ]

    rows = []
    for cand_name, mode in modes[1:]:
        print(f"Candidate {cand_name}...")
        sim = simulate_volaware_combo(
            pp,
            act=EXIT_KW_BASE["act"],
            dist_normal=0.08,
            dist_wide=0.12,
            dist_tight=0.06,
            atr_pct_bar=atr_pct_bar,
            ei=ei,
            thr=thr,
            mode=mode,
        )

        year_rows = []
        for y in years:
            year_rows.append(_year_metrics(year=y, order=_orders_for_year(pp, y), p=pp, sim=sim))

        pnl_stream, blown = _pnl_stream(pp, sim, years)
        mc = iter6_montecarlo(base_pnl_stream, pnl_stream, base_blown, starting=STARTING)
        mc_dyn = mc[mc["policy"] == "dynamic"].iloc[0] if not mc.empty else None
        mc_ruin = float(mc_dyn["prob_ruin"]) if mc_dyn is not None else 1.0

        rows.append(
            {
                "candidate": cand_name,
                "PF_2024": float(next(r for r in year_rows if r["year"] == 2024)["pf"]),
                "DD_2026": float(next(r for r in year_rows if r["year"] == 2026)["dd"]),
                "Return_2026": float(next(r for r in year_rows if r["year"] == 2026)["ret"]),
                "MC_ruin": mc_ruin,
                "yearly": year_rows,
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
        "# Sprint 40 — Iter 4 Volatility-Aware Exit",
        "",
        f"Vol thresholds (TRAIN 2021-2023): hi {thr.atr_pct_now_hi:.3f}  lo {thr.atr_pct_now_lo:.3f}  n_time {thr.n_time}",
        "",
        "Baseline:",
    ]
    for r in base_year_rows:
        lines.append(
            f"- {r['year']}: PF {r['pf']:.2f}  DD {r['dd']*100:.1f}%  Return {r['ret']*100:.0f}%  Trades {r['trades']}"
        )

    lines += ["", "Candidates tested:"]
    for r in rows:
        lines.append(
            f"- {r['candidate']}: PF_2024 {r['PF_2024']:.2f}  DD_2026 {r['DD_2026']*100:.1f}%  Return_2026 {r['Return_2026']*100:.0f}%  MC_ruin {r['MC_ruin']:.3f}"
        )

    lines += [
        "",
        f"Iter 4 verdict: {verdict}" + (f" (best {best_name})" if best_name else ""),
        "",
        "Next: Iter 5 rolling WF if verdict != STOP.",
    ]

    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "decision.json").write_text(
        json.dumps(
            {
                "iter": 4,
                "next_iter": 5 if verdict != "STOP" else None,
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

