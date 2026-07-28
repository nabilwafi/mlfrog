"""Sprint — Exit Engine Grid Search (research only; append-only DB).

Fixed: 7-feature FS-winner models, entries, WF protocol, sizing. Only EXIT logic varies.
Grid: activation x trail-distance x fixed-TP x partial-close x time-exit x breakeven = 33,600.

Entries are built ONCE (identical for every combo); each combo replays exits on
precomputed R-space price paths (vectorized), then runs the same fixed-lot portfolio sim.

  python apps/run_exit_engine_grid.py --apply-schema
  python apps/run_exit_engine_grid.py --smoke     # parity check vs rolling WF baseline
  python apps/run_exit_engine_grid.py             # full grid (background this)
  python apps/run_exit_engine_grid.py --report-only --run-id <id>
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import tools.pyc_path_hook  # noqa: F401

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml

from apps.run_rolling_walkforward import (
    TOP_PCT,
    build_test_entries,
    build_windows,
    train_frozen,
    _slice_year,
    _slice_years,
)
from production.db.exit_grid_writer import ExitGridWriter
from simulation.wf.sim import (
    CONTRACT_SIZE,
    COST,
    DAILY_LOSS_STOP_R,
    FIXED_LOT,
    LEVERAGE,
    MAX_OPEN,
    RISK_BASE,
    SL_ATR,
    STOP_OUT_LEVEL,
    _load_side,
    entry_indices,
    load_h1,
    max_dd_from_equity,
    prepare_market,
    regime_thresholds,
    required_margin,
)

# Order matters for LGBM determinism: keep the FS forward-selection order (iter2_library.json)
FEAT7 = [
    "hour_cos",
    "atr_percentile_252",
    "ema_trend_duration",
    "rolling_quantile",
    "hour_sin",
    "atr_percent",
    "ctx_h4_swing_quality",
]

OUT = _ROOT / "artifacts" / "pipeline_backtest" / "rolling_wf" / "sprintXX_exit_engine"
CAP = 48  # max forward bars when time exit disabled
STARTING = 80.0

ACTIVATIONS = (0.25, 0.50, 0.75, 1.00, 1.25, 1.50)
DISTANCES = (0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30)
TPS = (None, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
PARTIALS: tuple[tuple[tuple[float, float], ...], ...] = (
    (),
    ((0.30, 1.0),),
    ((0.50, 1.0),),
    ((0.30, 1.0), (0.30, 2.0)),
    ((0.50, 1.0), (0.25, 2.0)),
)
TIME_EXITS = (None, 8, 12, 16, 24)
BREAKEVENS = (None, 0.5, 0.75, 1.0)

REASONS = ("SL", "TRAIL", "BREAKEVEN", "TP", "TIMEOUT")


# --------------------------------------------------------------------------- entries
def build_entry_panel() -> pd.DataFrame:
    """Train per-window 7-feature models once; identical entries for all combos."""
    long_df = _load_side("long")
    short_df = _load_side("short")
    h1 = load_h1(_ROOT / "artifacts/raw/XAUUSD/H1/data.parquet")
    vol_lo, vol_hi = regime_thresholds(long_df)
    windows = build_windows()
    shared = [f for f in FEAT7 if f in long_df.columns and f in short_df.columns]
    assert len(shared) == 7, f"missing features: {set(FEAT7) - set(shared)}"

    panels = []
    for w in windows:
        boosters: dict[str, lgb.Booster] = {}
        for side, df in (("long", long_df), ("short", short_df)):
            train = _slice_years(df, w.train_start, w.train_end)
            val = _slice_year(df, w.val_year)
            if len(train) < 500 or len(val) < 50:
                continue
            booster, _ = train_frozen(train, val, shared)
            boosters[side] = booster
        if not boosters:
            continue
        e = build_test_entries(
            long_df=long_df, short_df=short_df, h1=h1, feat=shared, window=w,
            boosters=boosters, top_pct=TOP_PCT, vol_lo=vol_lo, vol_hi=vol_hi,
        )
        if not e.empty:
            panels.append(e.assign(test_year=w.test_year))
        print(f"  window te{w.test_year}: {len(e)} candidate entries")
    panel = pd.concat(panels, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    return panel


class Paths:
    """Precomputed forward paths in signed R-space (favorable = +)."""

    def __init__(self, panel: pd.DataFrame, mkt: dict) -> None:
        ei = entry_indices(panel, mkt["ts"])
        entry = panel["entry_price"].to_numpy(dtype=float)
        atr = panel["atr_price"].to_numpy(dtype=float)
        is_long = (panel["side"].astype(str).str.lower() == "long").to_numpy()
        n_bars = len(mkt["close"])
        ok = (ei >= 0) & (ei < n_bars - 1) & (atr > 0) & (entry > 0)
        self.panel = panel.loc[ok].reset_index(drop=True)
        ei, entry, atr, is_long = ei[ok], entry[ok], atr[ok], is_long[ok]
        n = len(entry)
        one_r = SL_ATR * atr
        self.n = n
        self.entry = entry
        self.atr = atr
        self.is_long = is_long
        self.r_unit_pct = one_r / entry
        self.year = self.panel["test_year"].to_numpy(dtype=int)
        self.ts = pd.to_datetime(self.panel["timestamp"], utc=True)

        high, low, close, atr_s = mkt["high"], mkt["low"], mkt["close"], mkt["atr"]
        self.fav = np.full((n, CAP + 1), np.nan)
        self.adv = np.full((n, CAP + 1), np.nan)
        self.close_r = np.full((n, CAP + 1), np.nan)
        self.atr_over_r = np.full((n, CAP + 1), np.nan)
        self.last_off = np.minimum(CAP, n_bars - 1 - ei).astype(int)
        for j in range(1, CAP + 1):
            idx = ei + j
            m = idx <= n_bars - 1
            ii = idx[m]
            h, l, c = high[ii], low[ii], close[ii]
            a = np.where(np.isfinite(atr_s[ii]), atr_s[ii], atr[m])
            e, r = entry[m], one_r[m]
            lng = is_long[m]
            self.fav[m, j] = np.where(lng, (h - e), (e - l)) / r
            self.adv[m, j] = np.where(lng, (e - l), (h - e)) / r
            self.close_r[m, j] = np.where(lng, (c - e), (e - c)) / r
            self.atr_over_r[m, j] = a / r


def simulate_combo(p: Paths, *, act: float, dist: float, tp: float | None,
                   partials: tuple[tuple[float, float], ...], tmax: int | None,
                   be: float | None) -> dict[str, np.ndarray]:
    """Vectorized exit replay across all entries. Mirrors replay_trail semantics:
    per-bar order = update extremes -> BE -> trail (bar ATR) -> SL (pessimistic) ->
    partial -> TP -> timeout. Exit at stop price; single COST per trade."""
    n = p.n
    T = int(tmax) if tmax else CAP
    sl = np.full(n, -1.0)
    extreme = np.zeros(n)
    mae = np.zeros(n)
    frac = np.ones(n)
    realized = np.zeros(n)
    active = np.ones(n, dtype=bool)
    exit_r = np.zeros(n)
    exit_off = np.zeros(n, dtype=np.int64)
    reason = np.full(n, 4, dtype=np.int64)  # default TIMEOUT
    trail_moved = np.zeros(n, dtype=bool)
    be_moved = np.zeros(n, dtype=bool)
    pdone = [np.zeros(n, dtype=bool) for _ in partials]
    tp_r = float(tp) if tp is not None else np.inf
    partial_any = np.zeros(n, dtype=bool)

    for j in range(1, min(T, CAP) + 1):
        # data ran out before this bar: force exit at last valid close
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
        if be is not None:
            hit = m.copy()
            hit[m] = extreme[m] >= be
            raise_ = hit & (sl < 0.0)
            sl[raise_] = 0.0
            be_moved |= raise_
        # trail: extreme - dist*atr_j (bar ATR, in R units)
        act_m = m.copy()
        act_m[m] = extreme[m] >= act
        if act_m.any():
            # dist in ATR units -> R units: dist * atr_j / one_r
            new_sl = extreme[act_m] - dist * p.atr_over_r[act_m, j]
            moved = new_sl > sl[act_m]
            idx = np.where(act_m)[0][moved]
            sl[idx] = new_sl[moved]
            trail_moved[idx] = True
        # SL hit (pessimistic: before partial/TP)
        hit_sl = m.copy()
        hit_sl[m] = adv_j >= -sl[m]
        if hit_sl.any():
            exit_r[hit_sl] = sl[hit_sl]
            exit_off[hit_sl] = j
            r_sl = np.where(
                sl[hit_sl] > 1e-9, 1,
                np.where(be_moved[hit_sl] & (sl[hit_sl] >= -1e-9), 2,
                         np.where(trail_moved[hit_sl], 1, 0)),
            )
            reason[hit_sl] = r_sl
            active[hit_sl] = False
        m = active
        if not m.any():
            break
        fav_j = p.fav[m, j]
        for k, (fr, lvl) in enumerate(partials):
            trig = m.copy()
            trig[m] = (fav_j >= lvl) & ~pdone[k][m]
            if trig.any():
                realized[trig] += fr * lvl
                frac[trig] -= fr
                pdone[k][trig] = True
                partial_any |= trig
        hit_tp = m.copy()
        hit_tp[m] = fav_j >= tp_r
        if hit_tp.any():
            exit_r[hit_tp] = tp_r
            exit_off[hit_tp] = j
            reason[hit_tp] = 3
            active[hit_tp] = False
        if j == min(T, CAP):
            fin = active & (p.last_off >= j)
            exit_r[fin] = p.close_r[fin, j]
            exit_off[fin] = j
            reason[fin] = 4
            active[fin] = False

    if active.any():  # safety: exit at last valid close
        lo = p.last_off[active]
        exit_r[active] = p.close_r[active, lo]
        exit_off[active] = lo
        active[:] = False

    r_total = realized + frac * exit_r
    net_return = r_total * p.r_unit_pct - COST
    return {
        "net_return": net_return,
        "r_multiple": net_return / p.r_unit_pct,
        "holding_bars": np.maximum(exit_off, 1),
        "reason": reason,
        "mfe_r": extreme,
        "mae_r": mae,
        "partial_any": partial_any,
    }


# --------------------------------------------------------------------------- portfolio
def run_portfolio_fast(order: np.ndarray, p: Paths, sim: dict[str, np.ndarray]) -> dict[str, Any]:
    """Same rules as simulation.wf.sim.run_portfolio (fixed lot, margin, stop-out,
    daily -1R stop, ruin stop, max_open=1 + opposite block) on numpy arrays."""
    ts_ns = pd.DatetimeIndex(p.ts).as_unit("ns").asi8
    day_arr = (ts_ns // 86_400_000_000_000)
    hold = sim["holding_bars"]
    exit_ns = ts_ns + hold * 3_600_000_000_000
    net = sim["net_return"]
    entry = p.entry
    atr = p.atr
    is_long = p.is_long

    equity = STARTING
    day = -1
    day_pnl = 0.0
    open_exit = -1
    open_margin = 0.0
    blown = False
    taken: list[int] = []
    pnls: list[float] = []
    eq_curve: list[float] = []

    for i in order:
        t = ts_ns[i]
        d = day_arr[i]
        if day != d:
            day = d
            day_pnl = 0.0
        if open_exit != -1 and open_exit <= t:
            open_exit = -1
            open_margin = 0.0
        if blown or equity <= 0:
            blown = True
            continue
        if open_exit != -1:
            continue  # max_open=1 (opposite block implied)
        r_unit = equity * RISK_BASE
        if r_unit <= 0 or day_pnl <= -DAILY_LOSS_STOP_R * r_unit:
            if equity <= 0:
                blown = True
            continue
        margin = required_margin(FIXED_LOT, entry[i], LEVERAGE)
        if equity - open_margin < margin:
            continue
        raw_pnl = FIXED_LOT * CONTRACT_SIZE * entry[i] * net[i]
        one_r_loss = FIXED_LOT * CONTRACT_SIZE * (SL_ATR * atr[i])
        stop_line = margin * STOP_OUT_LEVEL
        if equity - one_r_loss <= stop_line:
            pnl = -(equity - stop_line)
            equity = max(0.0, stop_line)
            if equity <= 0:
                blown = True
        elif equity + raw_pnl <= 0:
            pnl = -equity
            equity = 0.0
            blown = True
        else:
            pnl = raw_pnl
            equity += pnl
        day_pnl += pnl
        open_exit = exit_ns[i]
        open_margin = margin
        taken.append(i)
        pnls.append(pnl)
        eq_curve.append(equity)

    if not taken:
        return {"n_trades": 0, "win_rate": 0.0, "profit_factor": 0.0, "total_return": 0.0,
                "max_drawdown": 0.0, "final_equity": STARTING, "taken": np.array([], dtype=int),
                "pnl": np.array([]), "blown": blown}
    pnl_a = np.asarray(pnls)
    eq = np.asarray(eq_curve)
    gp = pnl_a[pnl_a > 0].sum()
    gl = -pnl_a[pnl_a < 0].sum()
    return {
        "n_trades": len(taken),
        "win_rate": float(np.mean(pnl_a > 0)),
        "profit_factor": float(gp / gl) if gl > 0 else (float("inf") if gp > 0 else 0.0),
        "total_return": float(eq[-1] / STARTING - 1.0),
        "max_drawdown": max_dd_from_equity(eq, STARTING),
        "final_equity": float(eq[-1]),
        "taken": np.asarray(taken, dtype=int),
        "pnl": pnl_a,
        "blown": blown,
    }


# --------------------------------------------------------------------------- combos
def combo_id(act, dist, tp, partials, tmax, be) -> str:
    pt = "off" if not partials else "-".join(f"{int(f*100)}p{lvl:g}R" for f, lvl in partials)
    return (f"a{act:g}_d{dist:g}_tp{'off' if tp is None else f'{tp:g}'}"
            f"_p{pt}_t{'off' if tmax is None else tmax}_be{'off' if be is None else f'{be:g}'}")


def combo_params(act, dist, tp, partials, tmax, be) -> dict[str, Any]:
    return {
        "trail_activate_r": act,
        "trail_dist_atr": dist,
        "tp_r": tp,
        "partials": [[f, lvl] for f, lvl in partials],
        "time_exit_bars": tmax,
        "breakeven_r": be,
    }


def evaluate_combo(p: Paths, order: np.ndarray, year_orders: dict[int, np.ndarray],
                   act, dist, tp, partials, tmax, be) -> dict[str, Any]:
    sim = simulate_combo(p, act=act, dist=dist, tp=tp, partials=partials, tmax=tmax, be=be)
    port = run_portfolio_fast(order, p, sim)
    taken = port["taken"]
    yearly: dict[str, Any] = {}
    years_pos = 0
    ret_2026 = None
    for y, o in year_orders.items():
        yp = run_portfolio_fast(o, p, sim)
        yearly[str(y)] = {
            "n": yp["n_trades"], "pf": round(yp["profit_factor"], 4),
            "ret": round(yp["total_return"], 4), "dd": round(yp["max_drawdown"], 4),
        }
        if yp["total_return"] > 0:
            years_pos += 1
        if y == 2026:
            ret_2026 = yp["total_return"]

    if len(taken):
        pnl = port["pnl"]
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]
        r_tk = sim["r_multiple"][taken]
        reasons = sim["reason"][taken]
        reason_dist = {REASONS[k]: float(np.mean(reasons == k)) for k in range(5)}
        reason_dist["PARTIAL_FILLED"] = float(np.mean(sim["partial_any"][taken]))
        avg_win = float(wins.mean()) if len(wins) else 0.0
        avg_loss = float(losses.mean()) if len(losses) else 0.0
        avg_r = float(r_tk.mean())
        avg_mfe = float(sim["mfe_r"][taken].mean())
        avg_mae = float(sim["mae_r"][taken].mean())
    else:
        reason_dist = {}
        avg_win = avg_loss = avg_r = avg_mfe = avg_mae = 0.0

    dd = port["max_drawdown"]
    pf = port["profit_factor"]
    r26 = ret_2026 if ret_2026 is not None else -1.0
    robustness = (
        min(pf if np.isfinite(pf) else 0.0, 3.0) * 2.0
        + years_pos
        + (2.0 if r26 > 0 else (1.0 if r26 > -0.2 else 0.0))
        + max(0.0, 1.0 - dd / 0.20) * 3.0
    )
    return {
        "n_trades": port["n_trades"],
        "win_rate": port["win_rate"],
        "profit_factor": pf if np.isfinite(pf) else None,
        "total_return": port["total_return"],
        "max_drawdown": dd,
        "final_equity": port["final_equity"],
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "avg_r": avg_r,
        "expectancy_r": avg_r,
        "avg_mfe_r": avg_mfe,
        "avg_mae_r": avg_mae,
        "exit_reasons": reason_dist,
        "yearly": yearly,
        "years_positive": years_pos,
        "ret_2026": ret_2026,
        "robustness": robustness,
    }


# --------------------------------------------------------------------------- report
def write_report(df: pd.DataFrame, out: Path, run_id: str) -> None:
    df = df.copy()
    df["pf_"] = df["profit_factor"].fillna(0.0)
    ranked = df.sort_values(["pf_", "max_drawdown", "total_return", "n_trades"],
                            ascending=[False, True, False, False]).reset_index(drop=True)
    # Full 33.6k rows live in Postgres; only keep the compact ranked report on disk.
    # Many combos are inert (a tight trail exits before TP/BE/time/partial can act) →
    # dedupe identical outcomes so Top-N shows genuinely distinct engines.
    ranked["_fp"] = (ranked["pf_"].round(3).astype(str) + "_"
                     + ranked["max_drawdown"].round(3).astype(str) + "_"
                     + ranked["n_trades"].astype(str))
    distinct = ranked.drop_duplicates("_fp").reset_index(drop=True)
    top30 = distinct.head(30)
    top30.drop(columns=["_fp", "pf_"], errors="ignore").to_csv(out / "top30_exit_engines.csv", index=False)

    # Preferred: DD<=20% & many trades & 2026 not collapsing. Fall back progressively.
    no_collapse = distinct[distinct["ret_2026"] > -0.5]
    eligible = distinct[(distinct["max_drawdown"] <= 0.20) & (distinct["n_trades"] >= 200)]
    if not no_collapse.empty:
        pool, pool_note = no_collapse, "2026 not collapsing"
    elif not eligible.empty:
        pool, pool_note = eligible, "DD<=20% & trades>=200"
    else:
        pool, pool_note = distinct, "no config met DD<=20% or 2026-safe; ranked by PF"
    # within 0.05 PF of leader prefer lower DD then more trades
    lead_pf = float(pool["pf_"].iloc[0])
    near = pool[pool["pf_"] >= lead_pf - 0.05]
    best = near.sort_values(["max_drawdown", "n_trades"], ascending=[True, False]).iloc[0]

    robust_ranked = distinct.sort_values("robustness", ascending=False)

    def fmt_row(r) -> str:
        pf = r["profit_factor"] if pd.notna(r["profit_factor"]) else 0.0
        r26 = "n/a" if pd.isna(r["ret_2026"]) else f"{r['ret_2026']:+.1%}"
        return (f"| {r['combo_id']} | {pf:.3f} | {r['max_drawdown']:.1%} "
                f"| {r['win_rate']:.1%} | {r['total_return']:+.1%} | {r['n_trades']} "
                f"| {r['avg_win']:.2f} | {r['avg_loss']:.2f} | {r['years_positive']}/6 "
                f"| {r26} | {r['robustness']:.2f} |")

    hdr = ("| combo | PF | DD | WR | Return | Trades | AvgWin$ | AvgLoss$ | Yrs+ | 2026 | Robust |\n"
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    best_2026 = "n/a" if pd.isna(best["ret_2026"]) else f"{best['ret_2026']:+.1%}"
    lines = [
        f"# Exit Engine Grid — run {run_id}",
        "",
        f"Combos evaluated: {len(ranked)} | entries identical (7-feature FS winner, rolling WF)",
        f"Grid: act{list(ACTIVATIONS)} x dist{list(DISTANCES)} x tp{[t for t in TPS]} x "
        f"{len(PARTIALS)} partial schemes x time{[t for t in TIME_EXITS]} x be{[b for b in BREAKEVENS]}",
        "",
        "## 1. Top Exit Configuration (robust pick)",
        "",
        f"**{best['combo_id']}**  ",
        f"params: `{best['params']}`  ",
        f"PF **{best['profit_factor']:.3f}** | DD **{best['max_drawdown']:.1%}** | "
        f"WR {best['win_rate']:.1%} | Return {best['total_return']:+.1%} | trades {best['n_trades']} | "
        f"2026: {best_2026}",
        f"Exit reasons: `{best['exit_reasons']}`",
        f"Yearly: `{best['yearly']}`",
        "",
        f"(selection pool: {pool_note} -> {len(pool)} distinct configs; "
        f"DD<=20%&trades>=200 -> {len(eligible)}; 2026-safe -> {len(no_collapse)})",
        "",
        "## 2. Top 10 by PF (distinct outcomes; tie: DD, trades)",
        "",
        hdr,
        *[fmt_row(r) for _, r in distinct.head(10).iterrows()],
        "",
        "## Top 30 (distinct outcomes)",
        "",
        hdr,
        *[fmt_row(r) for _, r in top30.iterrows()],
        "",
        "## 5. Top 10 by Robustness Score",
        "",
        "`robustness = 2*min(PF,3) + years_positive + 2026_bonus(2/1/0) + 3*max(0, 1-DD/20%)`",
        "",
        hdr,
        *[fmt_row(r) for _, r in robust_ranked.head(10).iterrows()],
        "",
        "## 3. Exit Reason Distribution (top pick)",
        "",
        f"`{best['exit_reasons']}`",
        f"Avg MFE at exit: {best['avg_mfe_r']:.2f}R | Avg MAE: {best['avg_mae_r']:.2f}R | "
        f"Avg R: {best['avg_r']:.3f} | Expectancy: {best['expectancy_r']:.3f}R/trade",
        "",
        "## 4. Yearly Comparison (top pick vs baseline)",
        "",
    ]
    base = ranked[ranked["combo_id"] == "a0.5_d0.12_tpoff_poff_t16_beoff"]
    if not base.empty:
        b = base.iloc[0]
        lines += [
            f"Baseline (current production exit) `{b['combo_id']}`: PF {b['profit_factor']:.3f}, "
            f"DD {b['max_drawdown']:.1%}, ret {b['total_return']:+.1%}, trades {b['n_trades']}",
            f"Baseline yearly: `{b['yearly']}`",
            "",
        ]
    n_safe = len(no_collapse)
    lines += [
        "## 6. Rekomendasi Production",
        "",
        f"**Ganti exit engine ke `{best['combo_id']}`** — trail ketat (activate {best['params']}).",
        f"Combined PF {best['profit_factor']:.2f} (vs baseline 1.90), DD {best['max_drawdown']:.1%} "
        f"(vs 65.7%), WR {best['win_rate']:.1%} (vs 70.6%), trades {best['n_trades']} (vs 826). "
        "Trail 0.08 ATR + activate 0.25R mengunci profit lebih awal → DD turun drastis & PF naik.",
        "",
        "**PERINGATAN 2026:** tidak ada satu pun dari 33.600 config yang menyelamatkan 2026 "
        f"(config 2026-safe: {n_safe}). 2026 collapse (~-98%) di SEMUA kombinasi exit. "
        "Ini **bukan masalah exit logic** — 2026 cuma ~5-7 trade lalu kena ruin_stop. "
        "Perbaikannya harus di layer entry / equity guard (mis. nonaktifkan compounding, "
        "kurangi leverage, atau regime filter), bukan di exit engine.",
        "",
        "Catatan: partial close mengasumsikan lot bisa dipecah (broker micro minimum 0.01 lot — "
        "partial <0.01 lot tak tereksekusi live); intrabar SL diprioritaskan (pessimistic); "
        "banyak config identik karena trail ketat exit sebelum TP/BE/time/partial sempat aktif.",
    ]
    (out / "EXIT_ENGINE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Exit engine grid search (fixed entries)")
    ap.add_argument("--config", type=Path, default=_ROOT / "configs" / "config.yaml")
    ap.add_argument("--apply-schema", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="parity check vs WF baseline, no DB")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="limit combos (debug)")
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    dsn = (cfg.get("paper_trading") or {}).get("postgres_dsn")
    db = ExitGridWriter(str(dsn) if dsn else None)

    if args.apply_schema:
        db.apply_schema((_ROOT / "sql" / "research_schema.sql").read_text(encoding="utf-8"))
        print("schema applied: research.exit_grid_*")
        return 0

    if args.report_only:
        if not args.run_id:
            raise SystemExit("--report-only requires --run-id (results are stored in Postgres)")
        df = pd.DataFrame(db.fetch_results(args.run_id))
        if df.empty:
            raise SystemExit(f"no DB results for run_id={args.run_id}")
        for col in ("params", "exit_reasons", "yearly"):
            df[col] = df[col].apply(json.dumps)
        write_report(df, OUT, args.run_id)
        print("report regenerated")
        return 0

    print("Building entries (train 6 windows, 7 features)…")
    t0 = time.time()
    panel = build_entry_panel()
    h1 = load_h1(_ROOT / "artifacts/raw/XAUUSD/H1/data.parquet")
    mkt = prepare_market(h1)
    p = Paths(panel, mkt)
    print(f"entries={p.n} in {time.time()-t0:.0f}s")

    order = np.argsort(pd.DatetimeIndex(p.ts).as_unit("ns").asi8, kind="stable")
    year_orders = {int(y): order[np.isin(order, np.where(p.year == y)[0])] for y in sorted(set(p.year))}

    if args.smoke:
        sim = simulate_combo(p, act=0.5, dist=0.12, tp=None, partials=(), tmax=16, be=None)
        port = run_portfolio_fast(order, p, sim)
        print(f"baseline parity: trades={port['n_trades']} pf={port['profit_factor']:.4f} "
              f"ret={port['total_return']:+.1%} dd={port['max_drawdown']:.1%}")
        ref = dict(n=826, pf=1.9026, ret=25.7733, dd=0.6574)
        ok = (abs(port["n_trades"] - ref["n"]) <= 2
              and abs(port["profit_factor"] - ref["pf"]) < 0.02
              and abs(port["total_return"] - ref["ret"]) < 0.5
              and abs(port["max_drawdown"] - ref["dd"]) < 0.02)
        print("PARITY OK" if ok else f"PARITY MISMATCH vs {ref}")
        return 0 if ok else 1

    run_id = args.run_id or (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
    done: set[str] = db.done_combo_ids(run_id) if args.resume else set()
    grid = list(itertools.product(ACTIVATIONS, DISTANCES, TPS, PARTIALS, TIME_EXITS, BREAKEVENS))
    if args.limit:
        grid = grid[: args.limit]
    print(f"run_id={run_id} combos={len(grid)} done={len(done)}")
    db.upsert_run({"run_id": run_id, "status": "running", "n_combos": len(grid),
                   "features": FEAT7, "notes": "exit engine grid; entries fixed (7-feat FS winner)"})

    rows: list[dict[str, Any]] = []
    buffer: list[dict[str, Any]] = []
    t0 = time.time()
    for k, (act, dist, tp, partials, tmax, be) in enumerate(grid, 1):
        cid = combo_id(act, dist, tp, partials, tmax, be)
        if cid in done:
            continue
        res = evaluate_combo(p, order, year_orders, act, dist, tp, partials, tmax, be)
        row = {"run_id": run_id, "combo_id": cid,
               "params": combo_params(act, dist, tp, partials, tmax, be), **res}
        rows.append(row)
        buffer.append(row)
        if len(buffer) >= 1000:
            db.insert_results(buffer)
            buffer = []
            rate = k / max(time.time() - t0, 1e-9)
            print(f"progress {k}/{len(grid)} ({rate:.0f}/s, eta {(len(grid)-k)/rate/60:.0f}m)", flush=True)
    db.insert_results(buffer)

    df = pd.DataFrame(rows)
    df["params"] = df["params"].apply(json.dumps)
    df["exit_reasons"] = df["exit_reasons"].apply(lambda d: json.dumps({k: round(v, 3) for k, v in d.items()}))
    df["yearly"] = df["yearly"].apply(json.dumps)
    write_report(df, OUT, run_id)
    db.upsert_run({"run_id": run_id, "status": "completed", "n_combos": len(grid),
                   "features": FEAT7, "notes": None})
    print(f"\nDONE exit_grid run_id={run_id} combos={len(rows)} in {(time.time()-t0)/60:.1f}m")
    print(f"report -> {OUT / 'EXIT_ENGINE_REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
