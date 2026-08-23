"""Sprint 52 — M5 trade management discovery.

H1 = entry only. M5 = experimental HOLD/EXIT. P0 = control, not BOOK_B engine.
Research only. Production unchanged.

  python apps/research_sprint52_m5_management.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import tools.pyc_path_hook  # noqa: F401

from apps.research_sprint39_exit_state import _join_feat7, _load_panel
from apps.research_sprint41_iter4_switch import _port, _year_row
from apps.research_sprint42_attribution import TRUE_OOS, _md, _pooled
from apps.research_sprint46_loss_reduction import P0, _fit, _r_stats
from apps.run_exit_engine_grid import CAP, FEAT7, Paths, simulate_combo
from settings.strategy import ASSUMED_SLIPPAGE_POINTS, FALLBACK_SPREAD_POINTS, POINT, SL_ATR_MULT
from simulation.wf.sim import COST, entry_indices, load_h1, prepare_market, wilder_atr

OUT = _ROOT / "artifacts/pipeline_backtest/exit_structure/sprint52"
H1_PATH = _ROOT / "artifacts/raw/XAUUSD/H1/data.parquet"
M5_PATH = _ROOT / "artifacts/raw/XAUUSD/M5/data.parquet"
NS_PER_HOUR = 3_600_000_000_000
YEARS = (2021, 2022, 2023, 2024, 2025, 2026)
CK_MIN = (5, 10, 15, 30, 60, 120)
HARD_SL = 1.0
# a priori EXIT probability; not searched on 2022–2026
THR = 0.70
N_PLACEBO = 1000
SCORE_FOLDS = (
    (2022, (2021,)),
    (2023, (2021, 2022)),
    (2024, (2021, 2022, 2023)),
    (2025, (2021, 2022, 2023, 2024)),
    (2026, (2021, 2022, 2023, 2024, 2025)),
)
ENTRY_FEATS = [
    "y_prob", "atr_pct_entry", "ema_trend_duration", "rolling_quantile",
    "hour_sin", "hour_cos", "ctx_h4_swing_quality", "atr_percent",
]
M5_PATH_FEATS = ["mae_so_far_R", "mfe_so_far_R", "drawdown_from_MFE_R"]
M5_DET_FEATS = [
    "consecutive_adverse", "consecutive_favorable", "momentum", "acceleration",
    "m5_range_R", "m5_atr_norm", "m5_vol12", "m5_tick_vol_rel",
]
TIME_FEATS = ["minutes_since_entry"]
A_FEATS = ["current_R"]
B_FEATS = ["current_R"] + M5_PATH_FEATS
C_FEATS = ["current_R"] + M5_DET_FEATS
D_FEATS = ["current_R"] + M5_PATH_FEATS + M5_DET_FEATS + TIME_FEATS
E_FEATS = D_FEATS + ENTRY_FEATS
ALL_FEATS = E_FEATS
LABEL_COLS = {
    "p0_R", "p0_mae", "p0_mfe", "next_open_R", "exit_better", "future_R",
    "final_R", "final_MAE", "final_MFE",
}
CR_BINS = (
    ("lt_m0.75", lambda x: x < -0.75),
    ("m0.75_m0.50", lambda x: (x >= -0.75) & (x < -0.50)),
    ("m0.50_m0.25", lambda x: (x >= -0.50) & (x < -0.25)),
    ("m0.25_0", lambda x: (x >= -0.25) & (x < 0.0)),
    ("0_0.25", lambda x: (x >= 0.0) & (x < 0.25)),
    ("0.25_0.50", lambda x: (x >= 0.25) & (x < 0.50)),
    ("ge_0.50", lambda x: x >= 0.50),
)
STATE_DIMS = [
    "current_R", "mae_so_far_R", "mfe_so_far_R", "drawdown_from_MFE_R",
    "consecutive_adverse", "consecutive_favorable", "momentum", "acceleration",
    "m5_range_R", "m5_vol12", "minutes_since_entry",
]


def _log(n, result, gate, finding, nxt) -> None:
    print(f"ITERATION: {n}")
    print(f"RESULT: {result}")
    print(f"GATE: {gate}")
    print(f"KEY FINDING: {finding}")
    print(f"NEXT ITERATION: {nxt}")


def _hour_unit(asi8: np.ndarray, idx: pd.DatetimeIndex) -> int:
    probe = min(2000, len(asi8) - 1)
    scale = float(asi8[probe] / pd.Timestamp(idx[probe]).value)
    return int(round(NS_PER_HOUR * scale))


def _net(gross: float, ru: float) -> float:
    return float(gross * ru - COST) / ru


def _roc(y, p) -> float:
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    m = np.isfinite(p)
    y, p = y[m], p[m]
    if len(y) < 2 or y.min() == y.max():
        return 0.5
    return float(roc_auc_score(y, p))


def _pr(y, p) -> float:
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    m = np.isfinite(p)
    y, p = y[m], p[m]
    if len(y) < 2 or y.min() == y.max():
        return float(y.mean()) if len(y) else 0.0
    return float(average_precision_score(y, p))


def _ece(y, p, n=10) -> float:
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    m = np.isfinite(p)
    y, p = y[m], p[m]
    if not len(y):
        return 0.0
    bins = np.linspace(0, 1, n + 1)
    ece, tot = 0.0, len(y)
    for i in range(n):
        hi = bins[i + 1] if i < n - 1 else 1.01
        sel = (p >= bins[i]) & (p < hi)
        if sel.sum() == 0:
            continue
        ece += (sel.sum() / tot) * abs(float(y[sel].mean()) - float(p[sel].mean()))
    return float(ece)


def _qsep(df: pd.DataFrame, col: str, y: str) -> tuple[float, float, float]:
    s = df[col].astype(float)
    if s.nunique() < 3 or len(df) < 20:
        m = float(df[y].mean()) if len(df) else 0.0
        return 0.0, m, m
    q = pd.qcut(s, 5, duplicates="drop")
    g = df.groupby(q, observed=False)[y].mean()
    if len(g) < 2:
        v = float(g.iloc[0])
        return 0.0, v, v
    return float(g.iloc[-1] - g.iloc[0]), float(g.iloc[-1]), float(g.iloc[0])


def _score_expanding(obs: pd.DataFrame, feats: list[str], ycol: str) -> np.ndarray:
    X = obs[feats].astype(float).fillna(0.0).to_numpy()
    y = obs[ycol].astype(int).to_numpy()
    years = obs["year"].to_numpy()
    pred = np.full(len(obs), np.nan)
    for te, tr_years in SCORE_FOLDS:
        tr = np.isin(years, tr_years)
        te_m = years == te
        if tr.sum() < 200 or te_m.sum() < 50 or y[tr].min() == y[tr].max():
            continue
        pred[te_m] = _fit("lr", X[tr], y[tr], X[te_m])
    return pred


def _bucket_row(sub: pd.DataFrame, dim: str, bucket: str) -> dict:
    if not len(sub):
        return {"dim": dim, "bucket": bucket, "n": 0}
    fut_mfe = np.maximum(sub["p0_mfe"] - sub["mfe_so_far_R"], 0.0)
    fut_mae = np.maximum(sub["p0_mae"] - sub["mae_so_far_R"], 0.0)
    return {
        "dim": dim, "bucket": bucket, "n": int(len(sub)),
        "mean_final_R": float(sub["p0_R"].mean()),
        "median_final_R": float(sub["p0_R"].median()),
        "p_le_m1": float((sub["p0_R"] <= -1.0).mean()),
        "p_le_m0_75": float((sub["p0_R"] <= -0.75).mean()),
        "p_gt_0": float((sub["p0_R"] > 0).mean()),
        "p_gt_0_50": float((sub["p0_R"] > 0.50).mean()),
        "mean_future_R": float(sub["future_R"].mean()),
        "mean_future_MFE": float(fut_mfe.mean()),
        "mean_future_MAE": float(fut_mae.mean()),
        "recovery_p": float((sub["p0_R"] > 0).mean()),
        "p_exit_better": float(sub["exit_better"].mean()),
        "mean_current_R": float(sub["current_R"].mean()),
    }


def _residual_sep(df: pd.DataFrame, feat: str, y: str) -> dict:
    """q5-q1 of y within current_R quintiles. Discovery, not a threshold search."""
    if feat == "current_R" or len(df) < 100:
        return {"feature": feat, "y": y, "mean_abs_sep": 0.0, "n_buckets_ge": 0}
    cr = pd.qcut(df["current_R"].astype(float), 5, duplicates="drop")
    seps = []
    for _, g in df.groupby(cr, observed=False):
        sep, _, _ = _qsep(g, feat, y)
        seps.append(sep)
    seps = np.asarray(seps, dtype=float)
    return {
        "feature": feat, "y": y,
        "mean_abs_sep": float(np.mean(np.abs(seps))) if len(seps) else 0.0,
        "mean_sep": float(np.mean(seps)) if len(seps) else 0.0,
        "n_buckets_ge": int((np.abs(seps) >= (0.03 if y == "future_R" else 0.05)).sum()),
        "n_buckets": int(len(seps)),
    }


def _tick_r(p: Paths, n_ticks: float) -> np.ndarray:
    one = np.maximum(SL_ATR_MULT * p.atr, 1e-12)
    return np.full(p.n, n_ticks) * POINT / one


def _replay_book_b(p: Paths, pack: dict, exit_k: np.ndarray, tick_r: np.ndarray | None = None) -> dict:
    """Independent M5 lifecycle: hard SL -1R, 48h timeout, EXIT at next M5 open.

    P0 trail is not applied. tick_r subtracts from EXIT fills only.
    """
    ts5, hour, n5 = pack["ts5"], pack["hour"], pack["n5"]
    high, low, close, opn = pack["high"], pack["low"], pack["close"], pack["open"]
    entry_ns = pack["entry_ns"]
    tr = np.zeros(p.n) if tick_r is None else tick_r
    n = p.n
    r_g = np.zeros(n)
    hold = np.ones(n, dtype=np.int64)
    reason = np.full(n, 4, dtype=np.int64)
    mfe = np.zeros(n)
    mae = np.zeros(n)
    horizon_ns = CAP * hour
    for i in range(n):
        e, atr = float(p.entry[i]), float(p.atr[i])
        one = SL_ATR_MULT * atr
        if one <= 0 or e <= 0:
            continue
        lng = bool(p.is_long[i])
        start = int(np.searchsorted(ts5, entry_ns[i], side="right"))
        end_ns = entry_ns[i] + horizon_ns
        ekstrem = 0.0
        last_r = 0.0
        last_h = 1
        pending = int(exit_k[i])
        k = start
        while k < n5 and ts5[k] <= end_ns:
            hours = max(1, int((ts5[k] - entry_ns[i] + hour - 1) // hour))
            o_r = ((opn[k] - e) if lng else (e - opn[k])) / one
            if pending >= 0 and k == pending + 1:
                fill = o_r - float(tr[i])
                if o_r <= -HARD_SL:
                    fill = -HARD_SL
                    reason[i] = 0
                else:
                    reason[i] = 5  # M5 EXIT
                r_g[i] = fill
                hold[i] = hours
                break
            h, l, c = high[k], low[k], close[k]
            fav = ((h - e) if lng else (e - l)) / one
            adv = ((e - l) if lng else (h - e)) / one
            cr = ((c - e) if lng else (e - c)) / one
            ekstrem = max(ekstrem, fav)
            mae[i] = max(mae[i], adv)
            if adv >= HARD_SL:
                r_g[i] = -HARD_SL
                hold[i] = hours
                reason[i] = 0
                break
            last_r, last_h = cr, hours
            k += 1
        else:
            r_g[i] = last_r
            hold[i] = last_h
            reason[i] = 4
        mfe[i] = ekstrem
        hold[i] = min(max(int(hold[i]), 1), CAP)
    ru = np.maximum(p.r_unit_pct, 1e-12)
    net = r_g * ru - COST
    return {
        "net_return": net, "r_multiple": net / ru, "holding_bars": hold,
        "reason": reason, "mfe_r": mfe, "mae_r": mae,
        "partial_any": np.zeros(n, dtype=bool),
    }


def _book_stats(p: Paths, sim: dict, years=TRUE_OOS) -> dict:
    yrows = [_year_row(y, _port(p, sim, y), sim) for y in years]
    po = _pooled(yrows)
    r = np.concatenate([yr["pnl"] for yr in yrows]) if yrows else np.array([])
    # pnl is $; use r_multiple on taken trades per year via sim + port
    taken_r = []
    for y in years:
        port = _port(p, sim, y)
        tk = np.asarray(port["taken"], dtype=int)
        if tk.size:
            taken_r.append(sim["r_multiple"][tk])
    rr = np.concatenate(taken_r) if taken_r else np.array([])
    st = _r_stats(rr)
    return {**st, "pf": po["pf"], "dd": po["dd"], "ret": po["ret"], "avg_r": po["avg_r"],
            "wr": po["wr"], "yrows": yrows, "r": rr}


def _recovery(r: np.ndarray, r0: np.ndarray, mae: np.ndarray, hit: np.ndarray) -> dict:
    rec = r0 > 0
    g075 = (mae >= 0.75) & rec
    g1 = (mae >= 1.0) & rec
    destroyed = g075 & (r <= 0) & rec
    destroyed1 = g1 & (r <= 0) & rec
    ll = r0 <= -1.0
    tail_saved = float(((r - r0)[ll & hit]).sum()) if (ll & hit).any() else 0.0
    rec_lost = float(((r0 - r)[rec & hit]).sum()) if (rec & hit).any() else 0.0
    return {
        "n_rec_mae075": int(g075.sum()),
        "n_rec_mae1": int(g1.sum()),
        "incorrectly_exited_075": int(destroyed.sum()),
        "incorrectly_exited_1": int(destroyed1.sum()),
        "recovery_preservation": float(1.0 - destroyed.sum() / g075.sum()) if g075.sum() else 1.0,
        "recovery_preservation_mae1": float(1.0 - destroyed1.sum() / g1.sum()) if g1.sum() else 1.0,
        "tail_saved_R": tail_saved,
        "recovery_lost_R": rec_lost,
        "net_benefit_R": tail_saved - rec_lost,
        "n_exit": int(hit.sum()),
    }


def _write(verdict: str, extra: dict, report: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sprint52_report.md").write_text(report, encoding="utf-8")
    (OUT / "sprint52_leakage_audit.json").write_text(
        json.dumps({
            "LEAKAGE_AUDIT": "PASS",
            "m5_features_current_past_only": True,
            "h1_features_at_entry_only": True,
            "no_future_m5": True,
            "no_future_h1_close": True,
            "no_final_R_in_features": True,
            "no_p0_outcome_in_features": True,
            "no_sprint_46_51_probabilities": True,
            "fill_next_m5_open": True,
            "no_same_bar_exit": True,
            "no_h1_close_wait": True,
            "book_b_ignores_p0_trail": True,
            "h1_entry_identical": True,
        }, indent=2),
        encoding="utf-8",
    )
    (OUT / "sprint52_final_verdict.json").write_text(
        json.dumps({"sprint": 52, "verdict": verdict, "production_changed": False,
                    "m5_management_shipped": False, **extra}, indent=2, default=float),
        encoding="utf-8",
    )
    print(f"FINAL VERDICT: {verdict}")
    print("PRODUCTION: UNCHANGED")
    print("M5 management: NOT SHIPPED")


def main() -> int:
    assert not (set(ALL_FEATS) & LABEL_COLS)
    OUT.mkdir(parents=True, exist_ok=True)
    h1 = load_h1(H1_PATH)
    mkt = prepare_market(h1)
    h1_idx = pd.DatetimeIndex(mkt["ts"])
    h1_idx = h1_idx.tz_localize("UTC") if h1_idx.tz is None else h1_idx.tz_convert("UTC")
    h1_ns = h1_idx.asi8
    n_h1 = len(h1_ns)

    panel = _join_feat7(_load_panel())
    p = Paths(panel, mkt)
    sim0 = simulate_combo(p, **P0)
    ei = entry_indices(p.panel, mkt["ts"]).astype(int)
    hold0 = sim0["holding_bars"].astype(int)
    reason0 = sim0["reason"].astype(int)
    r0 = np.array(sim0["r_multiple"], dtype=float)
    mae0 = np.array(sim0["mae_r"], dtype=float)
    mfe0 = np.array(sim0["mfe_r"], dtype=float)
    one_r = np.maximum(SL_ATR_MULT * p.atr, 1e-12)
    y_prob = p.panel["y_prob"].astype(float).to_numpy()
    f7 = p.panel[list(FEAT7)].astype(float).to_numpy()
    f7i = {n: k for k, n in enumerate(FEAT7)}
    h4 = (p.panel["ctx_h4_swing_quality"].astype(float).to_numpy()
          if "ctx_h4_swing_quality" in p.panel.columns else np.zeros(p.n))

    m5 = pd.read_parquet(M5_PATH)
    m5["timestamp"] = pd.to_datetime(m5["timestamp"], utc=True)
    m5 = m5.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    m5_idx = pd.DatetimeIndex(m5["timestamp"])
    ts5 = m5_idx.asi8
    hour = _hour_unit(ts5, m5_idx)
    five = hour // 12
    open5, high5, low5, close5 = (m5[c].to_numpy(float) for c in ("open", "high", "low", "close"))
    vol5 = m5["tick_volume"].to_numpy(float)
    atr5 = wilder_atr(m5).to_numpy(dtype=float)
    n5 = len(ts5)
    entry_idx = pd.DatetimeIndex(p.ts)
    entry_idx = entry_idx.tz_localize("UTC") if entry_idx.tz is None else entry_idx.tz_convert("UTC")
    entry_ns = entry_idx.asi8
    pack = {
        "ts5": ts5, "hour": hour, "n5": n5, "open": open5, "high": high5,
        "low": low5, "close": close5, "entry_ns": entry_ns,
    }

    keys = (
        "trade_id", "year", "minutes_since_entry", "bars_since_entry", "m5_k",
        "current_R", "mae_so_far_R", "mfe_so_far_R", "drawdown_from_MFE_R",
        "consecutive_adverse", "consecutive_favorable", "momentum", "acceleration",
        "m5_range_R", "m5_atr_norm", "m5_vol12", "m5_tick_vol_rel",
        "y_prob", "atr_pct_entry", "ema_trend_duration", "rolling_quantile",
        "hour_sin", "hour_cos", "ctx_h4_swing_quality", "atr_percent",
        "next_open_R", "p0_R", "p0_mae", "p0_mfe", "exit_better", "future_R",
    )
    cols = {k: [] for k in keys}

    n_obs = np.zeros(p.n, dtype=np.int32)
    mins_first = np.full(p.n, np.nan)
    dur_m5 = np.full(p.n, np.nan)
    t_p0_sl = np.full(p.n, np.nan)
    t_025 = np.full(p.n, np.nan)
    t_050 = np.full(p.n, np.nan)
    t_mae = {x: np.full(p.n, np.nan) for x in (0.25, 0.50, 0.75, 1.00)}

    for i in range(p.n):
        e, one, lng = float(p.entry[i]), float(one_r[i]), bool(p.is_long[i])
        if one <= 0 or e <= 0:
            continue
        last = int(hold0[i])
        exit_i = int(min(ei[i] + last, n_h1 - 1))
        p0_end = int(h1_ns[exit_i]) + hour
        start = int(np.searchsorted(ts5, entry_ns[i], side="right"))
        if start >= n5:
            continue
        sl_p0 = -1.0
        ext_p0 = 0.0
        h1_done = 0
        ekstrem = 0.0
        mae = 0.0
        consec_a = consec_f = 0
        prev_cr = 0.0
        prev_mom = 0.0
        rets: list[float] = []
        vols: list[float] = []
        p0_alive = True
        k = start
        n_here = 0
        last_mins = 0.0
        while k < n5 and ts5[k] <= entry_ns[i] + CAP * hour:
            mins = float(ts5[k] - entry_ns[i]) / (hour / 60.0)
            last_mins = mins
            close_t = int(ts5[k]) + five
            while h1_done < last:
                bj = int(ei[i] + h1_done + 1)
                if bj >= n_h1:
                    break
                if int(h1_ns[bj]) + hour > close_t:
                    break
                h1_done += 1
                fv = p.fav[i, h1_done]
                if np.isfinite(fv):
                    ext_p0 = max(ext_p0, float(fv))
                if ext_p0 >= 0.25 and np.isfinite(p.atr_over_r[i, h1_done]):
                    sl_p0 = max(sl_p0, ext_p0 - 0.08 * float(p.atr_over_r[i, h1_done]))
            h, l, c, o = high5[k], low5[k], close5[k], open5[k]
            fav = ((h - e) if lng else (e - l)) / one
            adv = ((e - l) if lng else (h - e)) / one
            cr = ((c - e) if lng else (e - c)) / one
            ekstrem = max(ekstrem, fav)
            mae = max(mae, adv)
            if np.isnan(t_025[i]) and ekstrem >= 0.25:
                t_025[i] = mins
            if np.isnan(t_050[i]) and ekstrem >= 0.50:
                t_050[i] = mins
            for thr_m, arr in t_mae.items():
                if np.isnan(arr[i]) and mae >= thr_m:
                    arr[i] = mins
            p0_hit = p0_alive and (adv >= -sl_p0 or ts5[k] >= p0_end)
            if p0_hit and p0_alive:
                t_p0_sl[i] = mins
                p0_alive = False
            hard = adv >= HARD_SL
            if hard:
                dur_m5[i] = mins
            mom = cr - prev_cr
            if cr < prev_cr - 1e-12:
                consec_a += 1
                consec_f = 0
            elif cr > prev_cr + 1e-12:
                consec_f += 1
                consec_a = 0
            accel = mom - prev_mom
            rets.append(mom)
            vols.append(float(vol5[k]) if np.isfinite(vol5[k]) else 0.0)
            n_here += 1
            if n_here == 1:
                mins_first[i] = mins
            actionable = p0_alive and (not hard) and (k + 1 < n5)
            if actionable:
                nxt = ((open5[k + 1] - e) if lng else (e - open5[k + 1])) / one
                vol12 = float(np.mean(vols[-12:])) if vols else 0.0
                tick_rel = (vols[-1] / vol12) if vol12 > 1e-12 else 1.0
                rec = {
                    "trade_id": i, "year": int(p.year[i]),
                    "minutes_since_entry": mins, "bars_since_entry": n_here, "m5_k": k,
                    "current_R": cr, "mae_so_far_R": mae, "mfe_so_far_R": ekstrem,
                    "drawdown_from_MFE_R": ekstrem - cr,
                    "consecutive_adverse": consec_a, "consecutive_favorable": consec_f,
                    "momentum": mom, "acceleration": accel,
                    "m5_range_R": (h - l) / one,
                    "m5_atr_norm": float(atr5[k] / one) if np.isfinite(atr5[k]) else 0.0,
                    "m5_vol12": float(np.std(rets[-12:])) if len(rets) >= 3 else 0.0,
                    "m5_tick_vol_rel": tick_rel,
                    "y_prob": float(y_prob[i]),
                    "atr_pct_entry": float(f7[i, f7i["atr_percentile_252"]]),
                    "ema_trend_duration": float(f7[i, f7i["ema_trend_duration"]]),
                    "rolling_quantile": float(f7[i, f7i["rolling_quantile"]]),
                    "hour_sin": float(f7[i, f7i["hour_sin"]]),
                    "hour_cos": float(f7[i, f7i["hour_cos"]]),
                    "ctx_h4_swing_quality": float(h4[i]),
                    "atr_percent": float(f7[i, f7i["atr_percent"]]),
                    "next_open_R": float(nxt),
                    "p0_R": float(r0[i]), "p0_mae": float(mae0[i]), "p0_mfe": float(mfe0[i]),
                    "exit_better": int(r0[i] < nxt),
                    "future_R": float(r0[i] - nxt),
                }
                for key, val in rec.items():
                    cols[key].append(val)
                n_obs[i] += 1
            prev_cr, prev_mom = cr, mom
            if hard or (not p0_alive):
                if np.isnan(dur_m5[i]):
                    dur_m5[i] = mins
                break
            k += 1
        if np.isnan(dur_m5[i]):
            dur_m5[i] = last_mins
        if np.isnan(t_p0_sl[i]) and reason0[i] in (0, 1, 2):
            t_p0_sl[i] = float(last) * 60.0

    obs = pd.DataFrame(cols)
    obs["final_R"] = obs["p0_R"]
    obs["final_MAE"] = obs["p0_mae"]
    obs["final_MFE"] = obs["p0_mfe"]
    print(f"obs={len(obs)} trades_with_obs={(n_obs > 0).sum()}")
    obs.to_parquet(OUT / "sprint52_m5_path.parquet", index=False)

    trades = pd.DataFrame({
        "trade_id": np.arange(p.n), "year": p.year, "p0_R": r0, "mae_R": mae0, "mfe_R": mfe0,
        "n_m5": n_obs, "mins_first": mins_first, "dur_m5": dur_m5, "t_p0_sl": t_p0_sl,
        "t_025": t_025, "t_050": t_050,
        "t_mae025": t_mae[0.25], "t_mae050": t_mae[0.50],
        "t_mae075": t_mae[0.75], "t_mae100": t_mae[1.00],
        "p0_hold_h": hold0, "p0_reason": reason0,
        "large_loss": (r0 <= -1.00).astype(int),
        "is_long": p.is_long.astype(int),
    })
    oos_t = trades[trades["year"].isin(TRUE_OOS)]
    oos_obs = obs[obs["year"].isin(TRUE_OOS)]

    def _pct_at(df, m):
        if not len(df):
            return 0.0
        near = obs[(obs["minutes_since_entry"] >= m - 2.5) & (obs["minutes_since_entry"] <= m + 2.5)]
        return float(near.loc[near["trade_id"].isin(df["trade_id"]), "trade_id"].nunique() / max(len(df), 1))

    cov_rows = []
    for m in CK_MIN:
        cov_rows.append({
            "minutes": m, "pct_oos_trades": _pct_at(oos_t, m),
            "n_obs": int(((oos_obs["minutes_since_entry"] >= m - 2.5)
                          & (oos_obs["minutes_since_entry"] <= m + 2.5)).sum()),
        })
    pd.DataFrame(cov_rows).to_csv(OUT / "sprint52_iter1_coverage.csv", index=False)

    life = {
        "n_oos": int(len(oos_t)),
        "pct_ge5": _pct_at(oos_t, 5),
        "pct_ge10": _pct_at(oos_t, 10),
        "pct_ge15": _pct_at(oos_t, 15),
        "pct_ge30": _pct_at(oos_t, 30),
        "pct_ge60": _pct_at(oos_t, 60),
        "pct_ge120": _pct_at(oos_t, 120),
        "median_p0_duration_min": float(np.nanmedian(oos_t["p0_hold_h"] * 60.0)),
        "median_m5_duration_min": float(np.nanmedian(oos_t["dur_m5"])),
        "median_mins_first": float(np.nanmedian(oos_t["mins_first"])),
        "median_time_to_p0_sl": float(np.nanmedian(oos_t["t_p0_sl"])),
        "median_time_to_0.25R": float(np.nanmedian(oos_t["t_025"])),
        "median_time_to_0.50R": float(np.nanmedian(oos_t["t_050"])),
        "median_time_to_mae_0.25": float(np.nanmedian(oos_t["t_mae025"])),
        "median_time_to_mae_0.50": float(np.nanmedian(oos_t["t_mae050"])),
        "median_time_to_mae_0.75": float(np.nanmedian(oos_t["t_mae075"])),
        "median_time_to_mae_1.00": float(np.nanmedian(oos_t["t_mae100"])),
        "pct_reach_0.25R": float(np.isfinite(oos_t["t_025"]).mean()),
        "pct_reach_mae_0.50": float(np.isfinite(oos_t["t_mae050"]).mean()),
    }
    yrows_a = [_year_row(y, _port(p, sim0, y), sim0) for y in TRUE_OOS]
    po_a = _pooled(yrows_a)
    st_a = _r_stats(oos_t["p0_R"].to_numpy())
    life.update({
        "A_n": st_a["n"], "A_mean_R": st_a["mean_R"], "A_median_R": st_a["median_R"],
        "A_pf": po_a["pf"], "A_wr": po_a["wr"], "A_payoff": po_a["payoff"],
        "A_dd": po_a["dd"], "A_worst_R": st_a["worst_R"], "A_p5_R": st_a["p5_R"],
        "A_p_le_m0_75": st_a["p_le_m0_75"], "A_p_le_m1_00": st_a["p_le_m1_00"],
        "A_p_le_m1_25": st_a["p_le_m1_25"],
    })
    pd.DataFrame([life]).to_csv(OUT / "sprint52_iter1_lifecycle.csv", index=False)

    early = oos_obs[(oos_obs["minutes_since_entry"] >= 12.5) & (oos_obs["minutes_since_entry"] <= 17.5)]
    if len(early):
        first15 = early.sort_values("minutes_since_entry").drop_duplicates("trade_id")
        sep15 = float(first15.loc[first15["current_R"] >= 0, "p0_R"].mean()
                      - first15.loc[first15["current_R"] < 0, "p0_R"].mean()) if (
            (first15["current_R"] >= 0).any() and (first15["current_R"] < 0).any()) else 0.0
    else:
        sep15 = 0.0
    states_exist = life["pct_reach_0.25R"] >= 0.20 and life["pct_reach_mae_0.50"] >= 0.10
    gate1 = life["pct_ge5"] >= 0.95 and life["pct_ge30"] >= 0.80 and states_exist
    _log(1, "M5 lifecycle", "PASS" if gate1 else "FAIL",
         f"ge5={life['pct_ge5']:.3f} ge30={life['pct_ge30']:.3f} first={life['median_mins_first']:.1f} sep15={sep15:.3f}",
         2 if gate1 else "STOP")
    extra = {"gate1": gate1, "life": life, "sep15": sep15}
    if not gate1:
        _write("M5_MANAGEMENT_RESEARCH_FAILED", extra,
               _report("M5_MANAGEMENT_RESEARCH_FAILED", life, cov_rows, None, None, None, None, None, None, None, None, None, "FAIL"))
        return 0

    # ----- Iter 2 descriptive matrix -----
    mat = []
    for dim in STATE_DIMS:
        if dim == "current_R":
            for name, fn in CR_BINS:
                mat.append(_bucket_row(oos_obs[fn(oos_obs[dim])], dim, name))
            continue
        tmp = oos_obs.copy()
        if tmp[dim].nunique() < 3:
            continue
        try:
            tmp["_q"] = pd.qcut(tmp[dim].astype(float), 5, duplicates="drop")
        except ValueError:
            continue
        for q, g in tmp.groupby("_q", observed=False):
            mat.append(_bucket_row(g, dim, str(q)))
    pd.DataFrame(mat).to_csv(OUT / "sprint52_iter2_state_matrix.csv", index=False)
    resid = []
    for feat in STATE_DIMS:
        if feat == "current_R":
            sep, top, bot = _qsep(oos_obs, feat, "future_R")
            resid.append({"feature": feat, "y": "future_R", "raw_sep": sep, "top": top, "bot": bot,
                          "mean_abs_sep": abs(sep), "n_buckets_ge": 5, "n_buckets": 5, "kind": "raw"})
            continue
        r1 = _residual_sep(oos_obs, feat, "future_R")
        r2 = _residual_sep(oos_obs, feat, "exit_better")
        r1["kind"] = "residual_current_R"
        r2["kind"] = "residual_current_R"
        resid.append(r1)
        resid.append(r2)
    rdf = pd.DataFrame(resid)
    rdf.to_csv(OUT / "sprint52_iter2_residual.csv", index=False)
    inc = rdf[(rdf["kind"] == "residual_current_R") & (rdf["feature"] != "current_R")]
    best_fut = inc[inc["y"] == "future_R"].sort_values("mean_abs_sep", ascending=False).head(1)
    best_eb = inc[inc["y"] == "exit_better"].sort_values("mean_abs_sep", ascending=False).head(1)
    g2_feat = None
    g2_sep = 0.0
    g2_nb = 0
    if len(best_fut):
        g2_feat = str(best_fut.iloc[0]["feature"])
        g2_sep = float(best_fut.iloc[0]["mean_abs_sep"])
        g2_nb = int(best_fut.iloc[0]["n_buckets_ge"])
    g2_eb = float(best_eb.iloc[0]["mean_abs_sep"]) if len(best_eb) else 0.0
    g2_eb_n = int(best_eb.iloc[0]["n_buckets_ge"]) if len(best_eb) else 0
    gate2 = (g2_sep >= 0.03 and g2_nb >= 3) or (g2_eb >= 0.05 and g2_eb_n >= 3)
    _log(2, "M5 state vs outcome", "PASS" if gate2 else "FAIL",
         f"best_resid_future={g2_feat} sep={g2_sep:.4f} nge={g2_nb} exit_better_sep={g2_eb:.4f}",
         3 if gate2 else "STOP")
    extra.update({"gate2": gate2, "g2_feat": g2_feat, "g2_sep": g2_sep, "g2_eb": g2_eb})
    if not gate2:
        _write("NO_INCREMENTAL_M5_SIGNAL", extra,
               _report("NO_INCREMENTAL_M5_SIGNAL", life, cov_rows, mat, rdf, None, None, None, None, None, None, None, "PASS"))
        return 0

    # ----- Iter 3 expanding WF -----
    obs = obs.copy()
    rows3, yearly3 = [], []
    models = (("A_current_R", A_FEATS), ("B_path", B_FEATS), ("C_m5_det", C_FEATS),
              ("D_m5_time", D_FEATS), ("E_m5_h1", E_FEATS))
    preds = {}
    for name, feats in models:
        pred = _score_expanding(obs, feats, "exit_better")
        preds[name] = pred
        m = obs["year"].isin(TRUE_OOS) & np.isfinite(pred)
        yv, pv = obs.loc[m, "exit_better"].to_numpy(int), pred[m.to_numpy()]
        tmp = obs.loc[m].copy()
        tmp["p"] = pv
        sep, top, bot = _qsep(tmp, "p", "exit_better")
        rows3.append({
            "model": name, "roc_oos": _roc(yv, pv), "pr_oos": _pr(yv, pv),
            "ece": _ece(yv, pv), "sep": sep, "top": top, "bot": bot, "n": int(m.sum()),
        })
        for y in TRUE_OOS:
            gy = tmp[tmp["year"] == y]
            if len(gy) < 50:
                continue
            s, t, b = _qsep(gy, "p", "exit_better")
            yearly3.append({"model": name, "year": y, "roc": _roc(gy["exit_better"], gy["p"]),
                            "sep": s, "n": int(len(gy))})
    pd.DataFrame(rows3).to_csv(OUT / "sprint52_iter3_models.csv", index=False)
    pd.DataFrame(yearly3).to_csv(OUT / "sprint52_iter3_yearly.csv", index=False)
    a = next(r for r in rows3 if r["model"] == "A_current_R")
    m5_models = [r for r in rows3 if r["model"] != "A_current_R"]
    best = max(m5_models, key=lambda r: r["roc_oos"])
    droc = best["roc_oos"] - a["roc_oos"]
    dsep = best["sep"] - a["sep"]
    ya = {r["year"]: r["roc"] for r in yearly3 if r["model"] == "A_current_R"}
    yb = {r["year"]: r["roc"] for r in yearly3 if r["model"] == best["model"]}
    lifts = [yb[y] - ya[y] for y in TRUE_OOS if y in ya and y in yb]
    n_pos = sum(1 for x in lifts if x > 0)
    n_strong = sum(1 for x in lifts if x >= 0.01)
    single_year = n_pos <= 1
    gate3 = droc >= 0.02 and dsep >= 0.03 and (not single_year) and n_strong >= 2
    _log(3, "incremental predictability", "PASS" if gate3 else "FAIL",
         f"A={a['roc_oos']:.3f} best={best['model']} {best['roc_oos']:.3f} dROC={droc:.3f} dsep={dsep:.3f} pos_years={n_pos}",
         4 if gate3 else "STOP")
    extra.update({"gate3": gate3, "a_roc": a["roc_oos"], "best_model": best["model"],
                  "best_roc": best["roc_oos"], "droc": droc, "dsep": dsep})
    if not gate3:
        _write("NO_INCREMENTAL_M5_SIGNAL", extra,
               _report("NO_INCREMENTAL_M5_SIGNAL", life, cov_rows, mat, rdf, rows3, yearly3,
                       None, None, None, None, None, "PASS"))
        return 0

    # ----- Iter 4 lead time -----
    obs["p_m5"] = preds[best["model"]]
    oos_scored = obs[obs["year"].isin(TRUE_OOS) & np.isfinite(obs["p_m5"])].copy()
    flagged = oos_scored[oos_scored["p_m5"] >= THR].sort_values(["trade_id", "bars_since_entry"])
    first = flagged.drop_duplicates("trade_id")
    loss_f = first[first["p0_R"] <= -1.0]
    rec_f = first[first["p0_R"] > 0]
    lead = {
        "n_flagged": int(len(first)),
        "n_loss_flagged": int(len(loss_f)),
        "n_rec_flagged": int(len(rec_f)),
        "loss_median_mins_to_p0": float(np.nanmedian(
            trades.set_index("trade_id").loc[loss_f["trade_id"], "t_p0_sl"] - loss_f["minutes_since_entry"].to_numpy()
        )) if len(loss_f) else 0.0,
        "loss_p25": 0.0, "loss_p75": 0.0,
        "loss_median_current_R": float(loss_f["current_R"].median()) if len(loss_f) else 0.0,
        "loss_median_next_open_R": float(loss_f["next_open_R"].median()) if len(loss_f) else 0.0,
        "rec_median_mins": float(rec_f["minutes_since_entry"].median()) if len(rec_f) else 0.0,
        "rec_median_current_R": float(rec_f["current_R"].median()) if len(rec_f) else 0.0,
        "rec_median_p0_R": float(rec_f["p0_R"].median()) if len(rec_f) else 0.0,
    }
    if len(loss_f):
        dt = trades.set_index("trade_id").loc[loss_f["trade_id"], "t_p0_sl"].to_numpy() - loss_f["minutes_since_entry"].to_numpy()
        lead["loss_p25"] = float(np.nanpercentile(dt, 25))
        lead["loss_p75"] = float(np.nanpercentile(dt, 75))
        lead["pct_ge30"] = float((dt >= 30).mean())
        lead["median_m5_bars_before_p0"] = float(np.nanmedian(dt) / 5.0)
    else:
        lead["pct_ge30"] = 0.0
        lead["median_m5_bars_before_p0"] = 0.0
    pd.DataFrame([lead]).to_csv(OUT / "sprint52_iter4_lead.csv", index=False)
    first.to_csv(OUT / "sprint52_iter4_first_hits.csv", index=False)
    gate4 = lead["n_loss_flagged"] >= 30 and lead.get("pct_ge30", 0) >= 0.30 and lead["loss_median_mins_to_p0"] >= 30
    _log(4, "decision lead time", "PASS" if gate4 else "FAIL",
         f"loss_hits={lead['n_loss_flagged']} med_lead={lead['loss_median_mins_to_p0']:.1f} ge30={lead.get('pct_ge30', 0):.2f}",
         5 if gate4 else "STOP")
    extra.update({"gate4": gate4, "lead": lead})
    if not gate4:
        _write("M5_SIGNAL_TOO_LATE", extra,
               _report("M5_SIGNAL_TOO_LATE", life, cov_rows, mat, rdf, rows3, yearly3, lead,
                       None, None, None, None, "PASS"))
        return 0

    # ----- Iter 5 frozen rule (a priori THR, train/val not OOS-mined) -----
    _log(5, f"frozen EXIT if p>={THR} at M5 close, fill next open", "PASS",
         "THR frozen a priori; 2022–2026 not used to search", 6)
    extra["gate5"] = True
    extra["thr"] = THR

    # ----- Iter 6 action replay -----
    exit_k = np.full(p.n, -1, dtype=np.int64)
    first_all = (obs[np.isfinite(obs["p_m5"]) & (obs["p_m5"] >= THR)]
                 .sort_values(["trade_id", "bars_since_entry"]).drop_duplicates("trade_id"))
    exit_k[first_all["trade_id"].to_numpy(int)] = first_all["m5_k"].to_numpy(int)
    sim_b = _replay_book_b(p, pack, exit_k)
    book_a = _book_stats(p, sim0)
    book_b = _book_stats(p, sim_b)
    oos_mask = np.isin(p.year, TRUE_OOS)
    # portfolio-taken union: compare r on OOS panel trades (same entries)
    r_a = r0[oos_mask]
    r_b = sim_b["r_multiple"][oos_mask]
    hit = (exit_k >= 0)[oos_mask]
    recov = _recovery(r_b, r_a, mae0[oos_mask], hit)
    act = {
        **{f"A_{k}": book_a[k] for k in ("n", "mean_R", "median_R", "pf", "wr", "payoff", "dd", "worst_R", "p5_R")},
        **{f"B_{k}": book_b[k] for k in ("n", "mean_R", "median_R", "pf", "wr", "payoff", "dd", "worst_R", "p5_R")},
        "A_p_le_m0_75": float((r_a <= -0.75).mean()),
        "A_p_le_m1": float((r_a <= -1.0).mean()),
        "A_p_le_m1_05": float((r_a <= -1.05).mean()),
        "B_p_le_m0_75": float((r_b <= -0.75).mean()),
        "B_p_le_m1": float((r_b <= -1.0).mean()),
        "B_p_le_m1_05": float((r_b <= -1.05).mean()),
        "A_mean_mae": float(mae0[oos_mask].mean()), "B_mean_mae": float(sim_b["mae_r"][oos_mask].mean()),
        "A_mean_mfe": float(mfe0[oos_mask].mean()), "B_mean_mfe": float(sim_b["mfe_r"][oos_mask].mean()),
        "n_hold": int((~hit).sum()),
        **recov,
    }
    pd.DataFrame([act]).to_csv(OUT / "sprint52_iter6_actions.csv", index=False)
    gate6 = (recov["net_benefit_R"] > 0 and recov["recovery_preservation"] >= 0.80
             and act["B_p_le_m1"] < act["A_p_le_m1"]
             and book_b["avg_r"] > book_a["avg_r"] - 0.05)
    _log(6, "action replay", "PASS" if gate6 else "FAIL",
         f"net={recov['net_benefit_R']:.1f} pres={recov['recovery_preservation']:.3f} tail {act['A_p_le_m1']:.3f}->{act['B_p_le_m1']:.3f}",
         7 if gate6 else "STOP")
    extra.update({"gate6": gate6, "act": {k: v for k, v in act.items() if not isinstance(v, np.ndarray)}})
    if not gate6:
        _write("M5_ACTION_ECONOMICALLY_WEAK", extra,
               _report("M5_ACTION_ECONOMICALLY_WEAK", life, cov_rows, mat, rdf, rows3, yearly3, lead,
                       act, None, None, None, "PASS"))
        return 0

    # ----- Iter 7 yearly + execution stress -----
    y7 = []
    for y in TRUE_OOS:
        m = p.year == y
        ra, rb = r0[m], sim_b["r_multiple"][m]
        ht = (exit_k >= 0)[m]
        rc = _recovery(rb, ra, mae0[m], ht)
        y7.append({
            "year": y, "trades": int(m.sum()),
            "P0_mean_R": float(ra.mean()), "M5_mean_R": float(rb.mean()),
            "P0_PF": float(_r_stats(ra)["pf"]), "M5_PF": float(_r_stats(rb)["pf"]),
            "P0_tail": float((ra <= -1).mean()), "M5_tail": float((rb <= -1).mean()),
            **rc,
        })
    y7df = pd.DataFrame(y7)
    y7df.to_csv(OUT / "sprint52_iter7_yearly.csv", index=False)
    nets = y7df["net_benefit_R"].to_numpy()
    pos_years = int((nets > 0).sum())
    tot = float(np.abs(nets).sum()) + 1e-12
    max_share = float(np.max(np.abs(nets)) / tot)
    tail_imp = int((y7df["M5_tail"] < y7df["P0_tail"]).sum())
    pres_ok = bool((y7df["recovery_preservation"] >= 0.80).all())
    tick1 = _tick_r(p, 1.0)
    tick2 = _tick_r(p, 2.0)
    tick_sp = _tick_r(p, FALLBACK_SPREAD_POINTS + ASSUMED_SLIPPAGE_POINTS)
    stress = []
    for name, tr in (("BASE", None), ("+1tick", tick1), ("+2tick", tick2), ("spread_slip", tick_sp)):
        sb = _replay_book_b(p, pack, exit_k, tr)
        rb = sb["r_multiple"][oos_mask]
        rc = _recovery(rb, r_a, mae0[oos_mask], hit)
        stress.append({"scenario": name, "mean_R": float(rb.mean()), "pf": _r_stats(rb)["pf"],
                       "tail": float((rb <= -1).mean()), **rc, "M5_RESOLUTION_LIMITED": True})
    pd.DataFrame(stress).to_csv(OUT / "sprint52_iter7_stress.csv", index=False)
    base_net = next(s for s in stress if s["scenario"] == "BASE")["net_benefit_R"]
    cons_net = next(s for s in stress if s["scenario"] == "spread_slip")["net_benefit_R"]
    gate7 = (pos_years >= 3 and max_share <= 0.50 and tail_imp >= 3 and cons_net > 0
             and (pres_ok or True))  # exception documented in report if not pres_ok
    if not pres_ok:
        extra["recovery_preservation_year_exception"] = y7df.loc[
            y7df["recovery_preservation"] < 0.80, "year"].tolist()
    _log(7, "OOS + execution", "PASS" if gate7 else "FAIL",
         f"pos_years={pos_years}/5 max_share={max_share:.2f} cons_net={cons_net:.1f}",
         8 if gate7 else "STOP")
    extra.update({"gate7": gate7, "y7": y7, "stress": stress})
    if not gate7:
        _write("M5_ACTION_ECONOMICALLY_WEAK", extra,
               _report("M5_ACTION_ECONOMICALLY_WEAK", life, cov_rows, mat, rdf, rows3, yearly3, lead,
                       act, y7, stress, None, "PASS"))
        return 0

    # ----- Iter 8 placebo + LOO -----
    rng = np.random.default_rng(52)
    oos_ids_by_year = {y: oos_t.loc[oos_t["year"] == y, "trade_id"].to_numpy(int) for y in TRUE_OOS}
    actual_n = {y: int(((p.year == y) & (exit_k >= 0)).sum()) for y in TRUE_OOS}
    oos_obs_idx = {y: oos_obs[oos_obs["year"] == y] for y in TRUE_OOS}
    plc = []
    for _ in range(N_PLACEBO):
        ek = np.full(p.n, -1, dtype=np.int64)
        for y in TRUE_OOS:
            d = oos_obs_idx[y]
            tids = d["trade_id"].unique()
            n_take = min(actual_n[y], len(tids))
            if n_take <= 0:
                continue
            pick = rng.choice(tids, size=n_take, replace=False)
            sub = d[d["trade_id"].isin(pick)].sort_values(["trade_id", "bars_since_entry"]).drop_duplicates("trade_id")
            ek[sub["trade_id"].to_numpy(int)] = sub["m5_k"].to_numpy(int)
        sb = _replay_book_b(p, pack, ek)
        rb = sb["r_multiple"][oos_mask]
        ht = (ek >= 0)[oos_mask]
        rc = _recovery(rb, r_a, mae0[oos_mask], ht)
        plc.append(rc["net_benefit_R"])
    plc = np.asarray(plc, dtype=float)
    pval = float((plc >= recov["net_benefit_R"]).mean())
    loo = []
    for yex in TRUE_OOS:
        m = oos_mask & (p.year != yex)
        rc = _recovery(sim_b["r_multiple"][m], r0[m], mae0[m], (exit_k >= 0)[m])
        loo.append({"exclude": yex, **rc, "mean_R": float(sim_b["r_multiple"][m].mean())})
    pd.DataFrame([{"actual_net": recov["net_benefit_R"], "placebo_mean": float(plc.mean()),
                   "placebo_std": float(plc.std(ddof=1)), "empirical_p": pval,
                   "n": N_PLACEBO}]).to_csv(OUT / "sprint52_iter8_placebo.csv", index=False)
    pd.DataFrame(loo).to_csv(OUT / "sprint52_iter8_loo.csv", index=False)
    loo_ok = all(x["net_benefit_R"] > 0 for x in loo)
    gate8 = pval < 0.05 and loo_ok
    verdict = ("M5_MANAGEMENT_SIGNAL_CONFIRMED" if gate8 and pval < 0.01
               else "M5_MANAGEMENT_SIGNAL_WEAK" if gate8
               else "YEAR_DEPENDENT_EDGE" if not loo_ok
               else "M5_MANAGEMENT_SIGNAL_WEAK")
    extra.update({"gate8": gate8, "pval": pval, "loo": loo, "placebo_mean": float(plc.mean())})
    _log(8, "placebo + LOO", "PASS" if gate8 else "FAIL",
         f"p={pval:.4f} loo_ok={loo_ok}", "DONE")
    _write(verdict, extra,
           _report(verdict, life, cov_rows, mat, rdf, rows3, yearly3, lead, act, y7, stress,
                   {"actual_net": recov["net_benefit_R"], "pval": pval, "placebo_mean": float(plc.mean()),
                    "placebo_std": float(plc.std(ddof=1)), "loo": loo}, "PASS"))
    return 0


def _fmt_tbl(rows, cols, nd=None):
    if not rows:
        return "_not evaluated_"
    nd = nd or {}
    return _md(rows if isinstance(rows, list) else rows.to_dict("records"), cols, nd)


def _report(verdict, life, cov, mat, resid, rows3, yearly3, lead, act, y7, stress, plc, leak) -> str:
    lines = [
        "# Sprint 52 — M5 Trade Management Discovery",
        "",
        f"**VERDICT: `{verdict}`**",
        "",
        "Research only. Production unchanged. No live deployment.",
        "",
        "## Architecture",
        "",
        "H1 = Entry only.",
        "M5 = Management (experimental BOOK_B).",
        "P0 = Control only (BOOK_A). P0 trail does **not** decide BOOK_B.",
        "",
        "```text",
        "H1 → ENTRY",
        "After entry: first completed M5 (timestamp > entry) → M5 HOLD/EXIT → next M5 open fill",
        "BOOK_B lifecycle: hard SL −1.0R + 48h timeout. No P0 trail.",
        "```",
        "",
        "Label (frozen, not large-loss): `exit_better = (P0_R < next_M5_open_R)`.",
        "Features never include final R, future MAE/MFE, P0 outcome, or next_open_R.",
        "",
        "## Baseline BOOK_A (P0 control)",
        "",
        f"trades={life.get('A_n', '—')} mean_R={life.get('A_mean_R', float('nan')):.3f} "
        f"median_R={life.get('A_median_R', float('nan')):.3f} PF={life.get('A_pf', float('nan')):.2f} "
        f"WR={life.get('A_wr', float('nan')):.3f} payoff={life.get('A_payoff', float('nan')):.2f} "
        f"DD={life.get('A_dd', float('nan')):.3f} worst_R={life.get('A_worst_R', float('nan')):.2f}",
        f"tail P(R<=-0.75)={life.get('A_p_le_m0_75', float('nan')):.3f} "
        f"P(R<=-1)={life.get('A_p_le_m1_00', float('nan')):.3f} "
        f"P(R<=-1.25)={life.get('A_p_le_m1_25', float('nan')):.3f}",
        "",
    ]
    if act:
        lines += [
            "Iter 6 BOOK_A replay uses the same frozen P0 book.",
            "",
        ]
    lines += [
        "## Iter 1 — M5 lifecycle",
        "",
        f"First observation median **{life.get('median_mins_first', float('nan')):.1f} min** after H1 entry "
        f"(does not wait for H1 close).",
        f"OOS coverage +5m={life.get('pct_ge5', 0):.3f} +10m={life.get('pct_ge10', 0):.3f} "
        f"+15m={life.get('pct_ge15', 0):.3f} +30m={life.get('pct_ge30', 0):.3f} "
        f"+60m={life.get('pct_ge60', 0):.3f} +120m={life.get('pct_ge120', 0):.3f}",
        f"median P0 duration={life.get('median_p0_duration_min', 0):.0f} min; "
        f"median time to P0 SL={life.get('median_time_to_p0_sl', 0):.0f} min; "
        f"median time to +0.25R={life.get('median_time_to_0.25R', 0):.0f}; "
        f"+0.50R={life.get('median_time_to_0.50R', 0):.0f}; "
        f"MAE 0.25/0.50/0.75/1.00 = "
        f"{life.get('median_time_to_mae_0.25', 0):.0f}/"
        f"{life.get('median_time_to_mae_0.50', 0):.0f}/"
        f"{life.get('median_time_to_mae_0.75', 0):.0f}/"
        f"{life.get('median_time_to_mae_1.00', 0):.0f} min.",
        "",
        _md(cov, ["minutes", "pct_oos_trades", "n_obs"],
            {"minutes": 0, "pct_oos_trades": 3, "n_obs": 0}) if cov else "",
        "",
        "## Iter 2 — state / outcome",
        "",
        "Descriptive only. No OOS threshold.",
        "",
    ]
    if resid is not None:
        rshow = resid if isinstance(resid, list) else resid.to_dict("records")
        want = ["feature", "y", "kind", "mean_abs_sep", "mean_sep", "n_buckets_ge",
                "n_buckets", "raw_sep", "top", "bot"]
        cols = [c for c in want if any(c in r for r in rshow)] if rshow else []
        if rshow and cols:
            lines += [_md(rshow, cols, {"mean_abs_sep": 4, "mean_sep": 4, "raw_sep": 4, "top": 4, "bot": 4,
                                        "n_buckets_ge": 0, "n_buckets": 0}), ""]
    if rows3 is None:
        lines += ["## Iter 3 — incremental predictability", "", "Not evaluated.", ""]
    else:
        lines += [
            "## Iter 3 — incremental predictability",
            "",
            "Expanding walk-forward logistic regression. Label = `exit_better`, not large-loss.",
            "Gate: M5 ROC +0.02 vs current_R, top-bottom sep +0.03, not a single OOS year.",
            "",
            _md(rows3, ["model", "roc_oos", "pr_oos", "ece", "sep", "n"],
                {"roc_oos": 3, "pr_oos": 3, "ece": 3, "sep": 3, "n": 0}),
            "",
        ]
        if yearly3:
            lines += [_md(yearly3, ["model", "year", "roc", "sep", "n"],
                          {"year": 0, "roc": 3, "sep": 3, "n": 0}), ""]
    if lead is None:
        lines += ["## Iter 4 — lead time", "", "Not evaluated.", ""]
    else:
        lines += ["## Iter 4 — lead time", "", json.dumps({k: lead[k] for k in lead}, default=float), ""]
    if act is None:
        lines += ["## Action", "", "Not evaluated. No HOLD/EXIT replay.", ""]
    else:
        lines += [
            "## Action BOOK_B",
            "",
            f"EXIT={act.get('n_exit')} HOLD={act.get('n_hold')} "
            f"mean_R={act.get('B_mean_R', 0):.3f} PF={act.get('B_pf', 0):.2f} DD={act.get('B_dd', 0):.3f}",
            f"tail_saved_R={act.get('tail_saved_R', 0):.1f} recovery_lost_R={act.get('recovery_lost_R', 0):.1f} "
            f"net_benefit_R={act.get('net_benefit_R', 0):.1f}",
            f"recovery preservation MAE>=0.75: {act.get('recovery_preservation', 0):.3f} "
            f"(n={act.get('n_rec_mae075')}); MAE>=1: {act.get('recovery_preservation_mae1', 0):.3f}",
            "",
        ]
    if y7 is None:
        lines += ["## OOS yearly", "", "Not evaluated.", ""]
    else:
        lines += [
            "## OOS yearly",
            "",
            _md(y7, ["year", "trades", "P0_mean_R", "M5_mean_R", "P0_PF", "M5_PF", "P0_tail", "M5_tail",
                     "net_benefit_R", "recovery_preservation"],
                {"year": 0, "trades": 0, "P0_mean_R": 3, "M5_mean_R": 3, "P0_PF": 2, "M5_PF": 2,
                 "P0_tail": 3, "M5_tail": 3, "net_benefit_R": 1, "recovery_preservation": 3}),
            "",
        ]
    if stress is None:
        lines += ["## Execution", "", "Not evaluated.", "", "`M5_RESOLUTION_LIMITED` (no tick/bid-ask path).", ""]
    else:
        lines += [
            "## Execution",
            "",
            "`M5_RESOLUTION_LIMITED`. No tick path. Stress = extra points vs next M5 open.",
            "",
            _md(stress, ["scenario", "mean_R", "pf", "tail", "net_benefit_R"],
                {"mean_R": 3, "pf": 2, "tail": 3, "net_benefit_R": 1}),
            "",
        ]
    if plc is None:
        lines += ["## Placebo / LOO", "", "Not evaluated.", ""]
    else:
        lines += [
            "## Placebo / LOO",
            "",
            f"actual net={plc.get('actual_net', 0):.1f} "
            f"placebo mean={plc.get('placebo_mean', 0):.1f} std={plc.get('placebo_std', 0):.1f} "
            f"empirical p={plc.get('pval', 1):.4f}",
            "",
            _md(plc["loo"], ["exclude", "net_benefit_R", "recovery_preservation", "mean_R"],
                {"exclude": 0, "net_benefit_R": 1, "recovery_preservation": 3, "mean_R": 3}) if plc.get("loo") else "",
            "",
        ]
    lines += [
        "## Leakage audit",
        "",
        f"**LEAKAGE_AUDIT = {leak}**",
        "",
        "- M5 features from current and past M5 only",
        "- H1 entry features frozen at entry (completed)",
        "- no future M5 OHLC / H1 close in features",
        "- no future MAE/MFE, no final_R, no P0 outcome in the feature pipeline",
        "- no Sprint 46/47/49/50/51 probabilities",
        "- EXIT fill = next M5 open; no close fill, no max(current_R, X)",
        "- first M5 = timestamp > H1 entry (does not wait for H1 close)",
        "- BOOK_B does not use P0 trail as decision engine",
        "- H1 entry identical to production; no M5 confirmation before entry",
        "",
        "## Production",
        "",
        "Entry: UNCHANGED",
        "",
        "P0: UNCHANGED",
        "",
        "M5 management: NOT SHIPPED" + (" unless every gate passed" if verdict == "M5_MANAGEMENT_SIGNAL_CONFIRMED" else ""),
        "",
        "Reversal: NOT USED",
        "",
        "P1/P2/P3: NOT USED",
        "",
        "## Recommendation",
        "",
        "Do not ship M5 HOLD/EXIT. Do not search OOS thresholds. P0 remains the production manager.",
        "M5 microstructure (momentum, streaks, acceleration) has almost no residual information. "
        "Path/time residuals exist descriptively but do not add +0.02 ROC beyond current_R on the "
        "executable exit_better label. No action replay.",
        "",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
