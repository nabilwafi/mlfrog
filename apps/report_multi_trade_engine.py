"""Sprint 37 — Multi Trade Benchmark Engine (measurement only).

Frozen: FEAT7 entries, exit a0.25_d0.08, Adaptive ATR map_conservative, WF.
Only trade-management rules change (max positions / add logic / heat budget).

  python apps/report_multi_trade_engine.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import tools.pyc_path_hook  # noqa: F401

import numpy as np
import pandas as pd

from apps.report_adaptive_vol_risk import (
    _mar,
    _pf,
    _sharpe,
    _ulcer,
    enrich_vol,
    iter6_montecarlo,
    mapping_from_edges,
)
from apps.run_exit_engine_grid import Paths, STARTING, build_entry_panel, simulate_combo
from simulation.wf.sim import (
    CONTRACT_SIZE,
    DAILY_LOSS_STOP_R,
    FIXED_LOT,
    LEVERAGE,
    RISK_BASE,
    SL_ATR,
    STOP_OUT_LEVEL,
    _load_side,
    load_h1,
    max_dd_from_equity,
    prepare_market,
    required_margin,
)

OUT = _ROOT / "artifacts" / "pipeline_backtest" / "rolling_wf" / "sprint37_multi_trade"
EXIT_KW = dict(act=0.25, dist=0.08, tp=None, partials=(), tmax=None, be=None)
ATR_MAP = {"edges": [0.30, 0.60, 0.80, 0.90], "risks": [1.0, 0.70, 0.40, 0.25, 0.10]}
CAP = 48
HOUR_NS = 3_600_000_000_000
BASELINE_PF = 2.35
BASELINE_DD = 0.158


@dataclass(frozen=True)
class EngineCfg:
    name: str
    family: str  # baseline | parallel | restack | scale_in | pyramid | time_add | atr_ctrl | prob_ctrl | hybrid
    max_positions: int = 1
    min_distance_atr: float = 0.0
    cooldown_bars: int = 0
    min_prob_delta: float = 0.0
    scale_fracs: tuple[float, ...] = ()
    pyramid_add_r: float = 0.0
    time_every_bars: int = 0
    atr_pct_max: float = 1.0
    require_prob_gt_first: bool = False
    require_same_trend: bool = False
    heat_budget_r: float = 1.0


def _atr_mult(atr_pct: np.ndarray) -> np.ndarray:
    return mapping_from_edges(ATR_MAP["edges"], ATR_MAP["risks"])(atr_pct)


def _cache_panel() -> pd.DataFrame:
    path = OUT / "entry_panel.parquet"
    if path.is_file():
        print(f"  load cached entries {path}")
        return pd.read_parquet(path)
    print("  building FEAT7 WF entries (frozen)…")
    panel = build_entry_panel()
    OUT.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(path, index=False)
    return panel


def candidate_engines() -> list[EngineCfg]:
    out: list[EngineCfg] = [
        EngineCfg(name="baseline_single_atr", family="baseline", max_positions=1, heat_budget_r=1.0),
    ]
    heats = (1.0, 1.5, 2.0, 3.0)

    for mo, dist, cd, heat in itertools.product((2, 3, 5), (0.0, 0.5, 1.0), (0, 1, 2), heats):
        out.append(
            EngineCfg(
                name=f"parallel_mo{mo}_d{dist}_cd{cd}_h{heat}",
                family="parallel",
                max_positions=mo,
                min_distance_atr=dist,
                cooldown_bars=cd,
                heat_budget_r=heat,
            )
        )

    for mo, dlt, heat in itertools.product((2, 3, 5), (0.0, 0.02, 0.05), heats):
        out.append(
            EngineCfg(
                name=f"restack_mo{mo}_dp{dlt}_h{heat}",
                family="restack",
                max_positions=mo,
                min_prob_delta=dlt,
                heat_budget_r=heat,
            )
        )

    for fracs, heat in itertools.product(((0.5, 0.3, 0.2), (0.25, 0.25, 0.25, 0.25)), heats):
        tag = "532" if fracs[0] == 0.5 else "2222"
        out.append(
            EngineCfg(
                name=f"scale_in_{tag}_h{heat}",
                family="scale_in",
                max_positions=len(fracs),
                scale_fracs=fracs,
                heat_budget_r=heat,
            )
        )

    for add_r, mo, heat in itertools.product((0.25, 0.5, 1.0), (2, 3, 4), heats):
        out.append(
            EngineCfg(
                name=f"pyramid_a{add_r}_mo{mo}_h{heat}",
                family="pyramid",
                max_positions=mo,
                pyramid_add_r=add_r,
                require_same_trend=True,
                heat_budget_r=heat,
            )
        )

    for every, mo, heat in itertools.product((2, 4, 6, 8), (2, 3), heats):
        out.append(
            EngineCfg(
                name=f"time_add_e{every}_mo{mo}_h{heat}",
                family="time_add",
                max_positions=mo,
                time_every_bars=every,
                require_same_trend=True,
                heat_budget_r=heat,
            )
        )

    for mo, heat in itertools.product((2, 3, 5), heats):
        out.append(
            EngineCfg(
                name=f"atr_ctrl_mo{mo}_h{heat}",
                family="atr_ctrl",
                max_positions=mo,
                atr_pct_max=0.60,
                heat_budget_r=heat,
            )
        )

    for mo, heat in itertools.product((2, 3, 5), heats):
        out.append(
            EngineCfg(
                name=f"prob_ctrl_mo{mo}_h{heat}",
                family="prob_ctrl",
                max_positions=mo,
                require_prob_gt_first=True,
                heat_budget_r=heat,
            )
        )

    for mo, heat in itertools.product((2, 3, 5), heats):
        out.append(
            EngineCfg(
                name=f"hybrid_mo{mo}_h{heat}",
                family="hybrid",
                max_positions=mo,
                min_distance_atr=0.5,
                cooldown_bars=1,
                atr_pct_max=0.60,
                require_prob_gt_first=True,
                require_same_trend=True,
                heat_budget_r=heat,
            )
        )
    return out


def _fav_at(p: Paths, idx: int, now_ns: int, ts_ns: np.ndarray) -> float:
    bars = int(max(1, min(CAP, (now_ns - int(ts_ns[idx])) // HOUR_NS)))
    v = p.fav[idx, bars]
    return float(v) if np.isfinite(v) else 0.0


def run_multi(
    order: np.ndarray,
    p: Paths,
    sim: dict[str, np.ndarray],
    lot_mult: np.ndarray,
    atr_pct: np.ndarray,
    probs: np.ndarray,
    trend_ok: np.ndarray,
    cfg: EngineCfg,
    *,
    starting: float = STARTING,
) -> dict[str, Any]:
    """Multi-pos portfolio; PnL booked at open (parity with run_portfolio_scaled)."""
    ts_ns = pd.DatetimeIndex(p.ts).as_unit("ns").asi8
    day_arr = ts_ns // 86_400_000_000_000
    hold = sim["holding_bars"]
    exit_ns_arr = ts_ns + hold * HOUR_NS
    net = sim["net_return"]
    entry = p.entry
    atr = p.atr
    is_long = p.is_long

    equity = float(starting)
    day = -1
    day_pnl = 0.0
    blown = False
    opens: list[dict[str, Any]] = []
    taken: list[int] = []
    pnls: list[float] = []
    eq_curve: list[float] = []
    add_types: list[str] = []
    heat_obs: list[float] = []
    risk_obs: list[float] = []
    n_open_obs: list[int] = []
    last_entry_ns = -1
    scale_step = 0

    for i in order:
        t = int(ts_ns[i])
        d = int(day_arr[i])
        if day != d:
            day = d
            day_pnl = 0.0
        opens = [op for op in opens if int(op["exit_ns"]) > t]
        if blown or equity <= 0:
            blown = True
            continue

        r_unit = equity * RISK_BASE
        if r_unit <= 0 or day_pnl <= -DAILY_LOSS_STOP_R * r_unit:
            if equity <= 0:
                blown = True
            continue

        n_open = len(opens)
        n_open_obs.append(n_open)
        open_risk = float(sum(op["risk"] for op in opens))
        heat_obs.append(open_risk)
        risk_obs.append(open_risk)

        same_side = [op for op in opens if bool(op["long"]) == bool(is_long[i])]
        if any(bool(op["long"]) != bool(is_long[i]) for op in opens):
            continue
        if n_open >= int(cfg.max_positions):
            continue

        is_add = n_open > 0
        kind = "entry"
        if is_add:
            if cfg.family == "baseline":
                continue
            if cfg.cooldown_bars > 0 and last_entry_ns >= 0:
                if (t - last_entry_ns) // HOUR_NS < cfg.cooldown_bars:
                    continue
            if cfg.min_distance_atr > 0 and same_side:
                if not all(
                    abs(entry[i] - entry[int(op["idx"])]) >= cfg.min_distance_atr * atr[i]
                    for op in same_side
                ):
                    continue
            if cfg.atr_pct_max < 1.0 and float(atr_pct[i]) >= cfg.atr_pct_max:
                continue
            if cfg.require_same_trend and not bool(trend_ok[i]):
                continue
            if cfg.require_prob_gt_first and same_side:
                if float(probs[i]) <= float(same_side[0]["prob"]):
                    continue
            if cfg.min_prob_delta > 0 and same_side:
                if float(probs[i]) < float(same_side[-1]["prob"]) + cfg.min_prob_delta:
                    continue
            if cfg.family == "pyramid":
                parent = same_side[0] if same_side else None
                if parent is None or _fav_at(p, int(parent["idx"]), t, ts_ns) < float(cfg.pyramid_add_r):
                    continue
                kind = "pyramid"
            elif cfg.family == "time_add":
                if last_entry_ns < 0 or (t - last_entry_ns) // HOUR_NS < int(cfg.time_every_bars):
                    continue
                kind = "time_add"
            elif cfg.family == "restack":
                kind = "restack"
            elif cfg.family == "scale_in":
                kind = "scale_in"
            elif cfg.family in {"atr_ctrl", "prob_ctrl"}:
                kind = "parallel"
            elif cfg.family == "hybrid":
                kind = "hybrid"
            else:
                kind = "parallel"
        else:
            scale_step = 0

        m = float(max(lot_mult[i], 0.0))
        frac = 1.0
        if cfg.scale_fracs:
            if scale_step >= len(cfg.scale_fracs):
                continue
            frac = float(cfg.scale_fracs[scale_step])
        lots = FIXED_LOT * m * frac
        if lots <= 0:
            continue

        # heat = sum of ATR-scaled lot slots (lots / FIXED_LOT)
        risk = lots / max(FIXED_LOT, 1e-12)
        if open_risk + risk > float(cfg.heat_budget_r) + 1e-12:
            continue

        margin = required_margin(lots, entry[i], LEVERAGE)
        open_margin = float(sum(op["margin"] for op in opens))
        if equity - open_margin < margin:
            continue

        raw_pnl = lots * CONTRACT_SIZE * entry[i] * net[i]
        one_r_loss = lots * CONTRACT_SIZE * (SL_ATR * atr[i])
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

        opens.append(
            {
                "idx": int(i),
                "exit_ns": int(exit_ns_arr[i]),
                "lots": lots,
                "margin": margin,
                "risk": risk,
                "long": bool(is_long[i]),
                "prob": float(probs[i]),
                "kind": kind if is_add else "entry",
            }
        )
        last_entry_ns = t
        if cfg.scale_fracs:
            scale_step += 1
        taken.append(int(i))
        pnls.append(float(pnl))
        eq_curve.append(float(equity))
        add_types.append(str(opens[-1]["kind"]))

    empty = {
        "n_trades": 0,
        "win_rate": 0.0,
        "profit_factor": 0.0,
        "total_return": 0.0,
        "max_drawdown": 0.0,
        "final_equity": starting,
        "taken": np.array([], dtype=int),
        "pnl": np.array([]),
        "equity": np.array([]),
        "blown": blown,
        "sharpe": 0.0,
        "mar": 0.0,
        "ulcer": 0.0,
        "avg_positions": 0.0,
        "max_positions": 0,
        "avg_risk": 0.0,
        "peak_risk": 0.0,
        "avg_heat": 0.0,
        "avg_adds": 0.0,
        "avg_pyramid": 0.0,
        "avg_scale_in": 0.0,
        "avg_parallel": 0.0,
        "longest_losing_streak": 0,
    }
    if not taken:
        return empty

    pnl_a = np.asarray(pnls, dtype=float)
    eq = np.asarray(eq_curve, dtype=float)
    dd = max_dd_from_equity(eq, starting)
    tr = float(eq[-1] / starting - 1.0)
    kinds = np.asarray(add_types)
    streak = best = 0
    for x in pnl_a:
        if x < 0:
            streak += 1
            best = max(best, streak)
        else:
            streak = 0

    return {
        "n_trades": len(taken),
        "win_rate": float(np.mean(pnl_a > 0)),
        "profit_factor": _pf(pnl_a),
        "total_return": tr,
        "max_drawdown": dd,
        "final_equity": float(eq[-1]),
        "taken": np.asarray(taken, dtype=int),
        "pnl": pnl_a,
        "equity": eq,
        "blown": blown,
        "sharpe": _sharpe(eq, starting),
        "mar": _mar(tr, dd),
        "ulcer": _ulcer(eq, starting),
        "avg_positions": float(np.mean(n_open_obs)) if n_open_obs else 0.0,
        "max_positions": int(max(n_open_obs) if n_open_obs else 0),
        "avg_risk": float(np.mean(risk_obs)) if risk_obs else 0.0,
        "peak_risk": float(np.max(risk_obs)) if risk_obs else 0.0,
        "avg_heat": float(np.mean(heat_obs)) if heat_obs else 0.0,
        "avg_adds": float(np.mean(kinds != "entry")),
        "avg_pyramid": float(np.mean(kinds == "pyramid")),
        "avg_scale_in": float(np.mean(kinds == "scale_in")),
        "avg_parallel": float(np.mean(np.isin(kinds, ["parallel", "restack", "hybrid", "time_add"]))),
        "longest_losing_streak": int(best),
    }


def _metrics_row(cfg: EngineCfg, port: dict[str, Any], *, y2026: dict[str, Any] | None = None) -> dict[str, Any]:
    row = {
        "engine": cfg.name,
        "family": cfg.family,
        "max_positions": cfg.max_positions,
        "min_distance_atr": cfg.min_distance_atr,
        "cooldown_bars": cfg.cooldown_bars,
        "min_prob_delta": cfg.min_prob_delta,
        "pyramid_add_r": cfg.pyramid_add_r,
        "time_every_bars": cfg.time_every_bars,
        "atr_pct_max": cfg.atr_pct_max,
        "heat_budget_r": cfg.heat_budget_r,
        "pf": port["profit_factor"],
        "dd": port["max_drawdown"],
        "ret": port["total_return"],
        "trades": port["n_trades"],
        "wr": port["win_rate"],
        "sharpe": port["sharpe"],
        "mar": port["mar"],
        "ulcer": port["ulcer"],
        "blown": port["blown"],
        "avg_pos": port["avg_positions"],
        "max_pos": port["max_positions"],
        "avg_risk": port["avg_risk"],
        "peak_risk": port["peak_risk"],
        "avg_heat": port["avg_heat"],
        "avg_adds": port["avg_adds"],
        "avg_pyramid": port["avg_pyramid"],
        "avg_scale_in": port["avg_scale_in"],
        "avg_parallel": port["avg_parallel"],
        "longest_losing_streak": port["longest_losing_streak"],
    }
    if y2026 is not None:
        row["pf_2026"] = y2026["profit_factor"]
        row["dd_2026"] = y2026["max_drawdown"]
        row["ret_2026"] = y2026["total_return"]
        row["trades_2026"] = y2026["n_trades"]
        row["pos_2026"] = float(y2026["total_return"] > 0)
    return row


def _mc_ruin(base_pnl: np.ndarray, port_pnl: np.ndarray, blown: bool) -> tuple[float, float]:
    mc = iter6_montecarlo(base_pnl, port_pnl, blown)
    if mc.empty or "policy" not in mc.columns:
        return (1.0 if len(port_pnl) < 5 else 0.0), 1.0
    dyn = mc[mc["policy"] == "dynamic"]
    if dyn.empty:
        return 1.0, 1.0
    return float(dyn.iloc[0]["prob_ruin"]), float(dyn.iloc[0]["worst_dd"])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="optional cap on engines for smoke")
    ap.add_argument(
        "--full",
        action="store_true",
        help="evaluate entire grid (no early-stop on first success-criteria winner)",
    )
    args = ap.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    print("=== Sprint 37 Multi Trade Benchmark ===")
    panel = _cache_panel()
    h1 = load_h1(_ROOT / "artifacts/raw/XAUUSD/H1/data.parquet")
    mkt = prepare_market(h1)

    long_df = _load_side("long")
    short_df = _load_side("short")
    feat_parts = []
    for side, df in (("long", long_df), ("short", short_df)):
        cols = ["timestamp"] + [c for c in ("atr_percentile_252", "atr_percent") if c in df.columns]
        g = df[cols].copy()
        g["side"] = side
        feat_parts.append(g)
    feats = pd.concat(feat_parts, ignore_index=True)
    feats["timestamp"] = pd.to_datetime(feats["timestamp"], utc=True)
    panel = panel.copy()
    panel["timestamp"] = pd.to_datetime(panel["timestamp"], utc=True)
    panel = panel.merge(feats, on=["timestamp", "side"], how="left")

    p = Paths(panel, mkt)
    d = enrich_vol(p.panel, mkt)
    assert len(d) == p.n
    sim = simulate_combo(p, **EXIT_KW)
    order = np.argsort(pd.DatetimeIndex(p.ts).as_unit("ns").asi8, kind="stable")
    atr_pct = d["atr_pct"].astype(float).fillna(0.5).to_numpy()
    lot_mult = _atr_mult(atr_pct)
    probs = p.panel["y_prob"].astype(float).to_numpy() if "y_prob" in p.panel.columns else np.full(p.n, 0.5)
    if "trend_state" in p.panel.columns:
        ts_ = p.panel["trend_state"].astype(str).to_numpy()
        trend_ok = np.where(
            p.is_long,
            np.isin(ts_, ["UPTREND", "UNKNOWN"]),
            np.isin(ts_, ["DOWNTREND", "UNKNOWN"]),
        )
    else:
        trend_ok = np.ones(p.n, dtype=bool)
    years = sorted(int(y) for y in d["year"].unique())
    print(f"entries={p.n} years={years}")

    engines = candidate_engines()
    if args.limit > 0:
        engines = engines[: args.limit]
    print(f"engines={len(engines)}")

    base_cfg = engines[0]
    base = run_multi(order, p, sim, lot_mult, atr_pct, probs, trend_ok, base_cfg)
    base_y = {}
    for y in years:
        idx = np.where(d["year"].to_numpy() == y)[0]
        o = idx[np.argsort(pd.DatetimeIndex(p.ts.iloc[idx]).as_unit("ns").asi8, kind="stable")]
        base_y[y] = run_multi(o, p, sim, lot_mult, atr_pct, probs, trend_ok, base_cfg)
    print(
        f"BASELINE pf={base['profit_factor']:.3f} dd={base['max_drawdown']:.3f} "
        f"ret={base['total_return']:.3f} n={base['n_trades']} "
        f"2026_ret={base_y.get(2026, {}).get('total_return', float('nan')):.3f}"
    )

    base_ruin, _ = _mc_ruin(base["pnl"], base["pnl"], base["blown"])

    rows = []
    yearly_rows = []
    trade_stat_rows = []
    robust_rows = []
    equity_rows = []
    winner = None

    for k, cfg in enumerate(engines):
        port = run_multi(order, p, sim, lot_mult, atr_pct, probs, trend_ok, cfg)
        yports = {}
        for y in years:
            idx = np.where(d["year"].to_numpy() == y)[0]
            o = idx[np.argsort(pd.DatetimeIndex(p.ts.iloc[idx]).as_unit("ns").asi8, kind="stable")]
            yports[y] = run_multi(o, p, sim, lot_mult, atr_pct, probs, trend_ok, cfg)
            yearly_rows.append(
                {
                    "engine": cfg.name,
                    "family": cfg.family,
                    "year": y,
                    "pf": yports[y]["profit_factor"],
                    "dd": yports[y]["max_drawdown"],
                    "ret": yports[y]["total_return"],
                    "trades": yports[y]["n_trades"],
                    "wr": yports[y]["win_rate"],
                    "blown": yports[y]["blown"],
                }
            )

        y2026 = yports.get(2026, {"profit_factor": 0, "max_drawdown": 1, "total_return": -1, "n_trades": 0})
        row = _metrics_row(cfg, port, y2026=y2026)
        row["pf_gain"] = row["pf"] - base["profit_factor"]
        row["dd_gain"] = base["max_drawdown"] - row["dd"]
        row["trade_gain"] = row["trades"] - base["n_trades"]
        row["ret_gain"] = row["ret"] - base["total_return"]

        # MC only when a candidate could still clear the PF/DD bar (saves ~hours on full grid)
        if row["pf"] > BASELINE_PF and row["dd"] <= BASELINE_DD + 1e-9 and port["n_trades"] >= base["n_trades"]:
            ruin, worst_dd_mc = _mc_ruin(base["pnl"], port["pnl"], port["blown"])
        else:
            ruin, worst_dd_mc = 1.0, 1.0
        row["mc_ruin"] = ruin
        row["mc_worst_dd"] = worst_dd_mc
        row["mc_ruin_vs_base"] = ruin - base_ruin

        pos_years = sum(1 for y in years if yports[y]["total_return"] > 0)
        pf_var = float(np.var([yports[y]["profit_factor"] for y in years]))
        dd_var = float(np.var([yports[y]["max_drawdown"] for y in years]))
        worst_year = min(years, key=lambda y: yports[y]["total_return"])
        row["positive_years"] = pos_years
        row["worst_year"] = worst_year
        row["worst_year_ret"] = yports[worst_year]["total_return"]
        row["var_pf"] = pf_var
        row["var_dd"] = dd_var

        beats = (
            row["pf"] > BASELINE_PF
            and row["dd"] <= BASELINE_DD + 1e-9
            and row["ret"] >= base["total_return"] - 1e-12
            and row["trades"] >= base["n_trades"]
            and row["ret_2026"] > 0
            and not row["blown"]
            and ruin <= base_ruin + 1e-12
        )
        row["beats_baseline"] = beats
        rows.append(row)

        trade_stat_rows.append(
            {
                "engine": cfg.name,
                "family": cfg.family,
                "avg_holding": float(np.mean(sim["holding_bars"][port["taken"]])) if port["n_trades"] else 0.0,
                "avg_mfe": float(np.mean(sim["mfe_r"][port["taken"]])) if port["n_trades"] else 0.0,
                "avg_mae": float(np.mean(sim["mae_r"][port["taken"]])) if port["n_trades"] else 0.0,
                "avg_r": float(np.mean(sim["r_multiple"][port["taken"]])) if port["n_trades"] else 0.0,
            }
        )
        robust_rows.append(
            {
                "engine": cfg.name,
                "family": cfg.family,
                "positive_years": pos_years,
                "worst_year": worst_year,
                "var_pf": pf_var,
                "var_dd": dd_var,
                "mc_ruin": ruin,
                "mc_worst_dd": worst_dd_mc,
                "longest_losing_streak": port["longest_losing_streak"],
            }
        )

        if cfg.family in {"baseline", "parallel", "pyramid", "hybrid", "atr_ctrl"} and (
            cfg.name == "baseline_single_atr" or beats or k % 40 == 0
        ):
            for j, e in enumerate(port["equity"]):
                equity_rows.append({"engine": cfg.name, "i": j, "equity": float(e)})

        if (k + 1) % 25 == 0 or k == 0 or beats:
            print(
                f"  [{k+1}/{len(engines)}] {cfg.name} "
                f"trades={row['trades']} PF={row['pf']:.3f} DD={row['dd']*100:.1f}% "
                f"ret={row['ret']*100:.0f}% beats={beats}"
            )

        if beats and winner is None and cfg.family != "baseline":
            winner = row
            print(
                f"WINNER (success criteria): {cfg.name} "
                f"trades={row['trades']} PF={row['pf']:.3f} DD={row['dd']*100:.1f}% ret={row['ret']*100:.0f}%"
            )
            if not args.full:
                print("early-stop (pass --full to score entire grid)")
                break

    bench = pd.DataFrame(rows)
    yearly = pd.DataFrame(yearly_rows)
    tstats = pd.DataFrame(trade_stat_rows)
    robust = pd.DataFrame(robust_rows)
    equity = pd.DataFrame(equity_rows)

    lead = bench.sort_values(
        by=["beats_baseline", "pf", "dd", "ret", "trades"],
        ascending=[False, False, True, False, False],
    ).reset_index(drop=True)
    lead["winner"] = False
    beaters = lead[(lead["family"] != "baseline") & (lead["beats_baseline"] == True)]
    if not beaters.empty:
        # among success-criteria passers: best by pf → dd → ret → trades
        pick = beaters.iloc[0]["engine"]
        winner = beaters.iloc[0].to_dict()
        lead.loc[lead["engine"] == pick, "winner"] = True
    elif not lead.empty:
        ok = lead[(lead["family"] != "baseline") & (lead["dd"] <= BASELINE_DD)]
        pick = ok.iloc[0]["engine"] if not ok.empty else lead.iloc[0]["engine"]
        lead.loc[lead["engine"] == pick, "winner"] = True
        winner = None

    contribution = lead[
        [
            "engine",
            "family",
            "pf",
            "dd",
            "ret",
            "trades",
            "pf_gain",
            "dd_gain",
            "trade_gain",
            "ret_gain",
            "ret_2026",
            "beats_baseline",
        ]
    ].copy()

    bench.to_csv(OUT / "decision_engine_benchmark.csv", index=False)
    lead.to_csv(OUT / "policy_leaderboard.csv", index=False)
    yearly.to_csv(OUT / "policy_yearly.csv", index=False)
    tstats.to_csv(OUT / "policy_trade_stats.csv", index=False)
    robust.to_csv(OUT / "robustness.csv", index=False)
    contribution.to_csv(OUT / "component_contribution.csv", index=False)
    if not equity.empty:
        equity.to_csv(OUT / "equity_comparison.csv", index=False)
    pd.DataFrame(columns=["ablation", "removed", "pf", "dd", "ret"]).to_csv(OUT / "ablation.csv", index=False)

    top = lead.head(15)
    best = lead.iloc[0].to_dict() if not lead.empty else {}

    if winner is not None:
        stop_reason = (
            f"{'FULL GRID' if args.full else 'SUCCESS'}: {winner['engine']} "
            f"(max_pos={int(winner.get('max_positions', 0))}) "
            f"trades={int(winner['trades'])} PF={float(winner['pf']):.3f} "
            f"DD={float(winner['dd'])*100:.1f}% ret={float(winner['ret'])*100:.0f}% "
            f"— beats Adaptive ATR Risk"
            + ("." if args.full else " — STOP (next sprint: Winner + Edge Quality).")
        )
    else:
        multi = lead[lead["family"] != "baseline"]
        if multi.empty or (
            (multi["pf"] <= BASELINE_PF).all()
            or (multi["dd"] > BASELINE_DD).all()
            or (multi["ret_2026"] <= 0).all()
        ):
            stop_reason = (
                "STOP: Multi Trade tidak memberi nilai tambah vs Adaptive ATR Risk "
                "(no engine meets success criteria)."
            )
        else:
            stop_reason = (
                "NO SUCCESS CRITERIA WINNER — some multi engines improve subsets of metrics "
                "but none clear full bar. STOP per sprint scope."
            )

    lines = [
        "# Sprint 37 — Multi Trade Benchmark",
        "",
        "Frozen: FEAT7 + LGBM + exit `a0.25_d0.08` + Adaptive ATR `map_conservative`.",
        "Only trade-management rules vary.",
        "",
        f"Engines evaluated (this run): **{len(bench)}** / planned grid"
        + (" (`--full`)" if args.full else "")
        + ".",
        "",
        "## Baseline (Adaptive ATR, max_pos=1)",
        "",
        "| Engine | Trades | PF | DD | Return | 2026 ret | MC ruin |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| baseline_single_atr | {base['n_trades']} | {base['profit_factor']:.3f} | "
        f"{base['max_drawdown']*100:.1f}% | {base['total_return']*100:.0f}% | "
        f"{base_y.get(2026, {}).get('total_return', float('nan'))*100:.0f}% | {base_ruin:.3f} |",
        "",
        "## Success bar",
        "",
        f"- PF > {BASELINE_PF}",
        f"- DD ≤ {BASELINE_DD*100:.1f}%",
        "- Return ≥ baseline, Trades ≥ baseline, Positive 2026, MC ruin ≤ baseline, no blow-up",
        "",
        f"**Any full winner?** {'YES — ' + str(winner.get('engine')) if winner else 'NO'}",
        "",
        "## Leaderboard (top 15) — trades / PF / DD / ret",
        "",
        top[
            [
                "engine",
                "family",
                "max_positions",
                "heat_budget_r",
                "trades",
                "pf",
                "dd",
                "ret",
                "ret_2026",
                "beats_baseline",
                "winner",
            ]
        ].to_string(index=False),
        "",
        "## Stop decision",
        "",
        stop_reason,
        "",
        "## Best engine snapshot",
        "",
        "```json",
        json.dumps(
            {
                k: (winner or best).get(k)
                for k in (
                    "engine",
                    "family",
                    "max_positions",
                    "heat_budget_r",
                    "trades",
                    "pf",
                    "dd",
                    "ret",
                    "ret_2026",
                    "mc_ruin",
                    "beats_baseline",
                )
            },
            indent=2,
            default=str,
        ),
        "```",
        "",
        "## Artifacts",
        "",
        "- `decision_engine_benchmark.csv`",
        "- `policy_leaderboard.csv`",
        "- `policy_yearly.csv`",
        "- `policy_trade_stats.csv`",
        "- `robustness.csv`",
        "- `component_contribution.csv`",
        "- `equity_comparison.csv`",
        "- `ablation.csv` (empty — not ALL-stack sprint)",
        "",
    ]
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(stop_reason)
    print(f"wrote {OUT}")
    print("SPRINT37_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
