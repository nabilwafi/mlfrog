"""Walk-forward panels, backtest, Monte Carlo for H4→H1→M5 research."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import lightgbm as lgb
import numpy as np
import pandas as pd

from apps.research_m5_full_execution import _hour_unit, _replay_m5_from_signal, HORIZON_H
from apps.research_sprint40_iter1_time_aware_exit import CFG_PROD, STARTING
from apps.research_sprint41_iter4_switch import _port, _year_row
from apps.research_sprint42_attribution import TRUE_OOS, _pooled
from apps.research_sprint46_loss_reduction import P0
from apps.run_exit_engine_grid import CAP, Paths, simulate_combo
from apps.run_rolling_walkforward import (
    WFWindow,
    build_test_entries,
    build_windows,
    train_frozen,
    _slice_year,
    _slice_years,
)
from production import PRIMARY_TOP_PCT
from research.mtf_h4_h1_m5.h1_features import H1_NATIVE, H1_WITH_D1_SIGNAL, H1_WITH_H4_SIGNAL, PRODUCTION_FEAT7
from research.mtf_h4_h1_m5.h4_engine import attach_h4_signals
from research.mtf_h4_h1_m5.d1_engine import attach_d1_signals
from research.mtf_h4_h1_m5.m5_engine import (
    M5ExecResult,
    build_m15_bars,
    build_m5_features,
    decide_m15_execution,
    decide_m5_execution,
)
from simulation.wf.sim import COST, _load_side, load_h1, prepare_market, regime_thresholds, wilder_atr

Experiment = Literal["h1_baseline", "h4_h1", "d1_h1", "production_feat7"]
M5Strategy = Literal[
    "immediate", "delay_1", "delay_2", "delay_3", "delay_4",
    "momentum", "pullback_recovery", "structure",
]

YEARS = (2021, 2022, 2023, 2024, 2025, 2026)
TOP_PCT = float(PRIMARY_TOP_PCT)
H1_PATH = Path("artifacts/raw/XAUUSD/H1/data.parquet")
H4_PATH = Path("artifacts/raw/XAUUSD/H4/data.parquet")
M15_PATH = Path("artifacts/raw/XAUUSD/M15/data.parquet")
M5_PATH = Path("artifacts/raw/XAUUSD/M5/data.parquet")


def feat_for_experiment(exp: Experiment) -> list[str]:
    if exp == "h1_baseline":
        return list(H1_NATIVE)
    if exp == "h4_h1":
        return list(H1_WITH_H4_SIGNAL)
    if exp == "d1_h1":
        return list(H1_WITH_D1_SIGNAL)
    return list(PRODUCTION_FEAT7)


def build_wf_entry_panel(
    exp: Experiment,
    *,
    h4: pd.DataFrame,
    d1: pd.DataFrame | None = None,
    force: bool = False,
    cache_dir: Path,
) -> pd.DataFrame:
    cache = cache_dir / f"entry_panel_{exp}_top21.parquet"
    if not force and cache.is_file():
        return pd.read_parquet(cache)
    cache_dir.mkdir(parents=True, exist_ok=True)
    long_df = _load_side("long")
    short_df = _load_side("short")
    h1 = load_h1(H1_PATH)
    vol_lo, vol_hi = regime_thresholds(long_df)
    windows = build_windows()
    panels = []
    for w in windows:
        if exp == "h4_h1":
            ldf = attach_h4_signals(long_df, h4, train_start=w.train_start, train_end=w.train_end)
            sdf = attach_h4_signals(short_df, h4, train_start=w.train_start, train_end=w.train_end)
        elif exp == "d1_h1":
            if d1 is None:
                raise ValueError("d1_h1 requires d1 bars")
            ldf = attach_d1_signals(long_df, d1, train_start=w.train_start, train_end=w.train_end)
            sdf = attach_d1_signals(short_df, d1, train_start=w.train_start, train_end=w.train_end)
        else:
            ldf, sdf = long_df, short_df
        feat = feat_for_experiment(exp)
        shared = [f for f in feat if f in ldf.columns and f in sdf.columns]
        missing = set(feat) - set(shared)
        if missing:
            raise ValueError(f"{exp} missing features: {missing}")
        boosters: dict[str, lgb.Booster] = {}
        for side, df in (("long", ldf), ("short", sdf)):
            train = _slice_years(df, w.train_start, w.train_end)
            val = _slice_year(df, w.val_year)
            if len(train) < 500 or len(val) < 50:
                continue
            booster, _ = train_frozen(train, val, shared)
            boosters[side] = booster
        if not boosters:
            continue
        e = build_test_entries(
            long_df=ldf,
            short_df=sdf,
            h1=h1,
            feat=shared,
            window=w,
            boosters=boosters,
            top_pct=TOP_PCT,
            vol_lo=vol_lo,
            vol_hi=vol_hi,
        )
        if not e.empty:
            panels.append(e.assign(test_year=w.test_year, experiment=exp))
    panel = pd.concat(panels, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    panel.to_parquet(cache, index=False)
    return panel


def build_wf_entry_panel_feats(
    feat: list[str],
    *,
    cache_dir: Path,
    tag: str,
    enrich=None,
    force: bool = False,
) -> pd.DataFrame:
    """WF panel for arbitrary feature list; optional enrich(long_df, short_df) hook."""
    cache = cache_dir / f"entry_panel_{tag}_top21.parquet"
    if not force and cache.is_file():
        return pd.read_parquet(cache)
    cache_dir.mkdir(parents=True, exist_ok=True)
    long_df = _load_side("long")
    short_df = _load_side("short")
    if enrich is not None:
        long_df, short_df = enrich(long_df, short_df)
    h1 = load_h1(H1_PATH)
    vol_lo, vol_hi = regime_thresholds(long_df)
    windows = build_windows()
    panels = []
    for w in windows:
        ldf, sdf = long_df, short_df
        shared = [f for f in feat if f in ldf.columns and f in sdf.columns]
        missing = set(feat) - set(shared)
        if missing:
            raise ValueError(f"{tag} missing features: {missing}")
        boosters: dict[str, lgb.Booster] = {}
        for side, df in (("long", ldf), ("short", sdf)):
            train = _slice_years(df, w.train_start, w.train_end)
            val = _slice_year(df, w.val_year)
            if len(train) < 500 or len(val) < 50:
                continue
            booster, _ = train_frozen(train, val, shared)
            boosters[side] = booster
        if not boosters:
            continue
        e = build_test_entries(
            long_df=ldf, short_df=sdf, h1=h1, feat=shared, window=w,
            boosters=boosters, top_pct=TOP_PCT, vol_lo=vol_lo, vol_hi=vol_hi,
        )
        if not e.empty:
            panels.append(e.assign(test_year=w.test_year, experiment=tag))
    panel = pd.concat(panels, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    panel.to_parquet(cache, index=False)
    return panel


def _pack_bars(p: Paths, bars: pd.DataFrame) -> dict:
    bars = bars.copy()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
    bars = bars.sort_values("timestamp").drop_duplicates("timestamp")
    idx = pd.DatetimeIndex(bars["timestamp"])
    ts = idx.asi8
    hour = _hour_unit(ts, idx)
    entry_idx = pd.DatetimeIndex(p.ts)
    entry_idx = entry_idx.tz_localize("UTC") if entry_idx.tz is None else entry_idx.tz_convert("UTC")
    return {
        "ts5": ts,
        "hour": hour,
        "n5": len(ts),
        "open": bars["open"].to_numpy(float),
        "high": bars["high"].to_numpy(float),
        "low": bars["low"].to_numpy(float),
        "close": bars["close"].to_numpy(float),
        "entry_ns": entry_idx.asi8,
    }


def _pack_m15(p: Paths, m15: pd.DataFrame) -> dict:
    return _pack_bars(p, m15)


def _pack_m5(p: Paths, m5: pd.DataFrame) -> dict:
    return _pack_bars(p, m5)


def _replay_pullback_trail(
    p: Paths,
    panel: pd.DataFrame,
    bars: pd.DataFrame,
    *,
    act: float,
    dist: float,
) -> dict:
    """P0 trail on bar closes (M15 or M5): M5 pullback entry, trail from fill bar after H1 close."""
    bars = bars.copy()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
    bars = bars.sort_values("timestamp").drop_duplicates("timestamp")
    ts_b = pd.DatetimeIndex(bars["timestamp"]).asi8
    hour = _hour_unit(ts_b, pd.DatetimeIndex(bars["timestamp"]))
    high = bars["high"].to_numpy(float)
    low = bars["low"].to_numpy(float)
    close = bars["close"].to_numpy(float)
    nb = len(ts_b)
    h1_ns = pd.DatetimeIndex(pd.to_datetime(panel["timestamp"], utc=True)).asi8
    fill_col = "exec_fill_timestamp" if "exec_fill_timestamp" in panel.columns else "m5_fill_timestamp"
    fill_ns = pd.DatetimeIndex(pd.to_datetime(panel[fill_col], utc=True)).asi8
    horizon_ns = HORIZON_H * hour
    n = p.n
    r_g = np.zeros(n)
    hold = np.ones(n, dtype=np.int64)
    reason = np.full(n, 4, dtype=np.int64)
    mfe = np.zeros(n)
    mae = np.zeros(n)
    fill_lag = np.full(n, np.nan)
    for i in range(n):
        atr = float(p.atr[i])
        one = 1.5 * atr
        e = float(p.entry[i])
        if one <= 0 or e <= 0:
            continue
        lng = bool(p.is_long[i])
        signal_ns = int(h1_ns[i] + hour)
        end_ns = signal_ns + horizon_ns
        k_fill = int(np.searchsorted(ts_b, fill_ns[i], side="left"))
        if k_fill >= nb:
            continue
        if int(ts_b[k_fill]) < signal_ns:
            k_fill = int(np.searchsorted(ts_b, signal_ns, side="left"))
        if k_fill >= nb:
            continue
        fill_lag[i] = float(ts_b[k_fill] - signal_ns) / (hour / 60.0)
        sl = -1.0
        extreme = 0.0
        last_r = 0.0
        last_j = 1
        k = k_fill
        while k < nb and ts_b[k] <= end_ns:
            h, l, c = high[k], low[k], close[k]
            fav = ((h - e) if lng else (e - l)) / one
            adv = ((e - l) if lng else (h - e)) / one
            cr = ((c - e) if lng else (e - c)) / one
            extreme = max(extreme, fav)
            mae[i] = max(mae[i], adv)
            if extreme >= act:
                sl = max(sl, extreme - dist * (atr / one))
            hours = max(1, int((ts_b[k] - h1_ns[i] + hour - 1) // hour))
            if adv >= -sl:
                r_g[i] = sl
                hold[i] = hours
                reason[i] = 1 if extreme >= act else 0
                last_r = sl
                break
            last_r, last_j = cr, hours
            k += 1
        else:
            r_g[i] = last_r
            hold[i] = last_j
            reason[i] = 4
        mfe[i] = extreme
        hold[i] = min(max(int(hold[i]), 1), CAP)
    ru = np.maximum(p.r_unit_pct, 1e-12)
    net = r_g * ru - COST
    return {
        "net_return": net, "r_multiple": net / ru, "holding_bars": hold,
        "reason": reason, "mfe_r": mfe, "mae_r": mae,
        "partial_any": np.zeros(n, dtype=bool),
        "fill_lag_mins": fill_lag, "fill_slip_R": np.full(n, np.nan),
    }


def _replay_m5_pullback_trail(
    p: Paths,
    panel: pd.DataFrame,
    m5: pd.DataFrame,
    *,
    act: float,
    dist: float,
) -> dict:
    return _replay_pullback_trail(p, panel, m5, act=act, dist=dist)


def apply_m5_execution(
    panel: pd.DataFrame,
    m5_feat: pd.DataFrame,
    *,
    strategy: M5Strategy,
    m5_exec_kw: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter/adjust entries by M5 rule. Returns (executed_panel, diagnostics)."""
    m5_exec_kw = m5_exec_kw or {}
    if strategy == "immediate":
        out = panel.copy()
        out["m5_delay"] = 0
        out["m5_action"] = "EXECUTE"
        out["m5_reason"] = "immediate"
        return out, pd.DataFrame()

    rows, diag = [], []
    for i, row in panel.iterrows():
        side = str(row["side"])
        atr = float(row["atr_price"])
        one_r = 1.5 * atr
        res: M5ExecResult = decide_m5_execution(
            side=side,
            h1_signal_ts=row["timestamp"],
            h1_ref_price=float(row["entry_price"]),
            one_r=one_r,
            m5=m5_feat,
            strategy=strategy,
            pullback_atr=float(m5_exec_kw.get("pullback_atr", 0.15)),
            recovery_frac=float(m5_exec_kw.get("recovery_frac", 0.5)),
        )
        diag.append({"timestamp": row["timestamp"], "side": side, **res.__dict__})
        if res.action != "EXECUTE" or res.fill_price is None:
            continue
        nr = row.copy()
        nr["entry_price"] = res.fill_price
        nr["m5_delay"] = res.delay_m5_bars
        nr["m5_action"] = res.action
        nr["m5_reason"] = res.reason
        if res.available_timestamp is not None:
            nr["m5_fill_timestamp"] = res.available_timestamp
            nr["exec_fill_timestamp"] = res.available_timestamp
        rows.append(nr)
    executed = pd.DataFrame(rows).reset_index(drop=True) if rows else panel.iloc[:0].copy()
    return executed, pd.DataFrame(diag)


def apply_m15_execution(
    panel: pd.DataFrame,
    m15_bars: pd.DataFrame,
    *,
    strategy: M5Strategy = "pullback_recovery",
    m15_exec_kw: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter/adjust entries by M15 pullback rule. Returns (executed_panel, diagnostics)."""
    m15_exec_kw = m15_exec_kw or {}
    rows, diag = [], []
    for _, row in panel.iterrows():
        side = str(row["side"])
        atr = float(row["atr_price"])
        one_r = 1.5 * atr
        res = decide_m15_execution(
            side=side,
            h1_signal_ts=row["timestamp"],
            h1_ref_price=float(row["entry_price"]),
            one_r=one_r,
            m15=m15_bars,
            strategy=strategy,
            pullback_atr=float(m15_exec_kw.get("pullback_atr", 0.15)),
            recovery_frac=float(m15_exec_kw.get("recovery_frac", 0.5)),
        )
        diag.append({"timestamp": row["timestamp"], "side": side, **res.__dict__})
        if res.action != "EXECUTE" or res.fill_price is None:
            continue
        nr = row.copy()
        nr["entry_price"] = res.fill_price
        nr["m15_delay"] = res.delay_m5_bars
        nr["m15_action"] = res.action
        nr["m15_reason"] = res.reason
        if res.available_timestamp is not None:
            nr["m15_fill_timestamp"] = res.available_timestamp
            nr["exec_fill_timestamp"] = res.available_timestamp
        rows.append(nr)
    executed = pd.DataFrame(rows).reset_index(drop=True) if rows else panel.iloc[:0].copy()
    return executed, pd.DataFrame(diag)


def score_panel(
    panel: pd.DataFrame,
    *,
    m15: pd.DataFrame,
    m5: pd.DataFrame | None = None,
    exit_tf: Literal["M15", "M5"] = "M15",
    entry_mode: Literal["h1_close", "m5_pullback", "m15_pullback"] = "h1_close",
    cost_mult: float = 1.0,
) -> dict[str, Any]:
    if panel.empty:
        empty = {"yrows": [], "po": _empty_pooled(), "po_oos": _empty_pooled(), "sim": {}}
        return empty
    h1 = load_h1(H1_PATH)
    mkt = prepare_market(h1)
    p = Paths(panel, mkt)
    if exit_tf == "M5":
        if m5 is None:
            raise ValueError("m5 bars required for M5 exit")
        if entry_mode == "m5_pullback":
            if m5 is None:
                raise ValueError("m5 bars required for M5 exit")
            if "m5_fill_timestamp" not in panel.columns and "exec_fill_timestamp" not in panel.columns:
                raise ValueError("m5_pullback requires exec_fill_timestamp")
            sim = _replay_pullback_trail(p, p.panel, m5, act=P0["act"], dist=P0["dist"])
        else:
            pack5 = _pack_m5(p, m5)
            sim = _replay_m5_from_signal(p, pack5, act=P0["act"], dist=P0["dist"], fill="h1_close")
    else:
        if entry_mode in ("m5_pullback", "m15_pullback"):
            fill_col = "exec_fill_timestamp" if "exec_fill_timestamp" in panel.columns else (
                "m5_fill_timestamp" if entry_mode == "m5_pullback" else "m15_fill_timestamp"
            )
            if fill_col not in panel.columns:
                raise ValueError(f"{entry_mode} requires {fill_col}")
            sim = _replay_pullback_trail(p, p.panel, m15, act=P0["act"], dist=P0["dist"])
        else:
            pack15 = _pack_m15(p, m15)
            sim = _replay_m5_from_signal(p, pack15, act=P0["act"], dist=P0["dist"], fill="h1_close")
    if cost_mult != 1.0:
        sim = dict(sim)
        extra = COST * (cost_mult - 1.0)
        sim["net_return"] = sim["net_return"] - extra
        sim["r_multiple"] = sim["net_return"] / np.maximum(p.r_unit_pct, 1e-12)
    yrows = [_year_row(y, _port(p, sim, y), sim) for y in YEARS if y in set(panel["test_year"])]
    po = _pooled(yrows)
    po_oos = _pooled([r for r in yrows if r["year"] in TRUE_OOS])
    return {"yrows": yrows, "po": po, "po_oos": po_oos, "sim": sim, "paths": p}


def _empty_pooled() -> dict:
    return {
        "trades": 0, "pf": 0.0, "dd": 0.0, "ret": 0.0, "wr": 0.0,
        "payoff": 0.0, "avg_r": 0.0, "max_loss_streak": 0, "blown": False, "pnl": np.array([]),
    }


def monte_carlo_10k(pnl: np.ndarray, *, starting: float = STARTING, seed: int = 42) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    pnl = np.asarray(pnl, dtype=float)
    if len(pnl) < 5:
        return {k: 0.0 for k in ("p5", "p25", "p50", "p75", "p95", "prob_ruin", "median_dd", "worst_dd")}
    rets, dds, streaks = [], [], []
    ruin = 0
    for _ in range(10_000):
        sh = rng.permutation(pnl)
        eq = peak = starting
        mdd = 0.0
        ls = best = 0
        for x in sh:
            eq += x
            if eq <= 0:
                ruin += 1
                eq = 0.0
                break
            peak = max(peak, eq)
            mdd = max(mdd, (peak - eq) / peak)
            if x < 0:
                ls += 1
                best = max(best, ls)
            else:
                ls = 0
        rets.append((eq - starting) / starting)
        dds.append(mdd)
        streaks.append(best)
    q = lambda arr, p: float(np.quantile(arr, p))
    return {
        "p5": q(rets, 0.05),
        "p25": q(rets, 0.25),
        "p50": q(rets, 0.50),
        "p75": q(rets, 0.75),
        "p95": q(rets, 0.95),
        "prob_ruin": ruin / 10_000,
        "median_dd": q(dds, 0.50),
        "worst_dd": float(max(dds)),
        "p95_losing_streak": q(streaks, 0.95),
    }


def wf_fold_rows(exp: str, yrows: list[dict]) -> list[dict]:
    rows = []
    for yr in yrows:
        rows.append({
            "experiment": exp,
            "test_year": yr["year"],
            "trades": yr["trades"],
            "wr": yr["wr"],
            "avg_r": yr["avg_r"],
            "pf": yr["pf"],
            "dd": yr["dd"],
            "ret": yr["ret"],
            "oos": yr["year"] in TRUE_OOS,
        })
    return rows


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")


def run_m5_sweep(
    panel: pd.DataFrame,
    m5_feat: pd.DataFrame,
    m15: pd.DataFrame,
    strategies: tuple[str, ...],
) -> tuple[dict[str, dict], str]:
    results: dict[str, dict] = {}
    best_name, best_pf = "", -1.0
    for strat in strategies:
        executed, diag = apply_m5_execution(panel, m5_feat, strategy=strat)  # type: ignore
        sc = score_panel(executed, m15=m15)
        retention = len(executed) / max(len(panel), 1)
        results[strat] = {
            "retention": retention,
            "po_oos": sc["po_oos"],
            "yrows": sc["yrows"],
            "diag": diag,
            "executed": executed,
        }
        if sc["po_oos"]["pf"] > best_pf:
            best_pf = sc["po_oos"]["pf"]
            best_name = strat
    return results, best_name
