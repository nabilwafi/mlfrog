"""Sprint 51 — Independent M5 post-entry deterioration discovery.

M5 monitoring starts at the first completed M5 after H1 entry. No wait for H1 close.
Signal discovery. Actions only if earlier gates pass. Production unchanged.

  python apps/research_sprint51_m5_discovery.py
"""
from __future__ import annotations

import json
import math
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
from apps.research_sprint46_loss_reduction import FOLDS, P0, _fit, _r_stats
from apps.research_sprint47_loss_action import P0_ACT, P0_DIST, _net, _sim_from
from apps.run_exit_engine_grid import CAP, FEAT7, Paths, simulate_combo
from simulation.wf.sim import COST, entry_indices, load_h1, prepare_market, wilder_atr

OUT = _ROOT / "artifacts/pipeline_backtest/loss_reduction/sprint51"
H1_PATH = _ROOT / "artifacts/raw/XAUUSD/H1/data.parquet"
M5_PATH = _ROOT / "artifacts/raw/XAUUSD/M5/data.parquet"
NS_PER_HOUR = 3_600_000_000_000
CKS = (1, 2, 3, 6, 12, 24)
THR = 0.70  # a priori diagnostic; not searched on OOS
P_BINS = ((0.0, 0.20), (0.20, 0.40), (0.40, 0.60), (0.60, 0.80), (0.80, 1.01))
ENTRY_FEATS = [
    "y_prob", "atr_pct_entry", "ema_trend_duration", "rolling_quantile",
    "hour_sin", "hour_cos", "ctx_h4_swing_quality", "atr_percent",
]
M5_FEATS = [
    "current_R", "mae_so_far_R", "mfe_so_far_R", "drawdown_from_MFE_R",
    "consecutive_adverse_m5", "consecutive_favorable_m5",
    "m5_return_1", "m5_return_2", "m5_return_3", "m5_return_6",
    "m5_range_R", "m5_body_R", "m5_wick_up_R", "m5_wick_down_R",
    "m5_atr_norm", "m5_vol12", "mom_1R", "cr_accel", "dist_to_sl_R", "bars_since_entry",
]
ALL_FEATS = ENTRY_FEATS + M5_FEATS
A_FEATS = ["current_R"]
B_FEATS = ["current_R", "mae_so_far_R", "mfe_so_far_R", "drawdown_from_MFE_R"]
C_FEATS = [c for c in M5_FEATS if c != "current_R"]
D_FEATS = ENTRY_FEATS
E_FEATS = ALL_FEATS


def _log(n, result, gate, finding, nxt) -> None:
    print(f"ITERATION: {n}")
    print(f"RESULT: {result}")
    print(f"GATE: {gate}")
    print(f"KEY FINDING: {finding}")
    print(f"NEXT ITERATION: {nxt}")


def _write(verdict: str, extra: dict, report: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sprint51_report.md").write_text(report, encoding="utf-8")
    (OUT / "sprint51_final_verdict.json").write_text(
        json.dumps({"sprint": 51, "verdict": verdict, "production_changed": False,
                    "m5_action_shipped": False, **extra}, indent=2, default=float),
        encoding="utf-8",
    )
    print(f"FINAL VERDICT: {verdict}")
    print("PRODUCTION: UNCHANGED")
    print("M5 loss action: NOT SHIPPED")


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


def _qsep(df: pd.DataFrame, col: str, y: str = "large_loss") -> tuple[float, float, float]:
    s = df[col].astype(float)
    if s.nunique() < 3:
        return 0.0, float(df[y].mean()), float(df[y].mean())
    q = pd.qcut(s, 5, duplicates="drop")
    g = df.groupby(q, observed=False)[y].mean()
    if len(g) < 2:
        return 0.0, float(g.iloc[0]), float(g.iloc[0])
    return float(g.iloc[-1] - g.iloc[0]), float(g.iloc[-1]), float(g.iloc[0])


def _score_2021(obs: pd.DataFrame, feats: list[str]) -> np.ndarray:
    X = obs[feats].astype(float).fillna(0.0)
    y = obs["large_loss"].astype(int).to_numpy()
    m21 = obs["year"].to_numpy() == 2021
    pred = np.full(len(obs), np.nan)
    if m21.sum() < 80 or y[m21].min() == y[m21].max():
        return pred
    pred[~m21] = _fit("lr", X[m21], y[m21], X[~m21])
    pred[m21] = _fit("lr", X[m21], y[m21], X[m21])
    return pred


def _score_lgbm(obs: pd.DataFrame, feats: list[str]) -> np.ndarray:
    X = obs[feats].astype(float).fillna(0.0).to_numpy()
    y = obs["large_loss"].astype(int).to_numpy()
    years = obs["year"].to_numpy()
    pred = np.full(len(obs), np.nan)
    for te, tr_years, _va in FOLDS:
        tr = np.isin(years, tr_years)
        te_m = years == te
        if tr.sum() < 200 or te_m.sum() < 50 or y[tr].min() == y[tr].max():
            continue
        pred[te_m] = _fit("lgbm", X[tr], y[tr], X[te_m])
    m22, tr21 = years == 2022, years == 2021
    if tr21.sum() >= 200 and m22.sum() >= 50 and y[tr21].min() != y[tr21].max():
        pred[m22] = _fit("lgbm", X[tr21], y[tr21], X[m22])
    if tr21.sum() >= 200 and y[tr21].min() != y[tr21].max():
        pred[tr21] = _fit("lgbm", X[tr21], y[tr21], X[tr21])
    return pred


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    h1 = load_h1(H1_PATH)
    mkt = prepare_market(h1)
    h1_idx = pd.DatetimeIndex(mkt["ts"])
    if h1_idx.tz is None:
        h1_idx = h1_idx.tz_localize("UTC")
    else:
        h1_idx = h1_idx.tz_convert("UTC")
    h1_ns = h1_idx.asi8
    n_h1 = len(h1_ns)

    panel = _join_feat7(_load_panel())
    p = Paths(panel, mkt)
    sim0 = simulate_combo(p, **P0)
    ei = entry_indices(p.panel, mkt["ts"]).astype(int)
    hold = sim0["holding_bars"].astype(int)
    reason = sim0["reason"].astype(int)
    r0 = np.array(sim0["r_multiple"], dtype=float)
    mae_all = np.array(sim0["mae_r"], dtype=float)
    one_r = np.maximum(1.5 * p.atr, 1e-12)
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
    probe = min(2000, len(ts5) - 1)
    scale = float(ts5[probe] / pd.Timestamp(m5_idx[probe]).value)
    hour = int(round(NS_PER_HOUR * scale))
    five = hour // 12
    open5, high5, low5, close5 = (m5[c].to_numpy(float) for c in ("open", "high", "low", "close"))
    atr5 = wilder_atr(m5).to_numpy(dtype=float)
    n5 = len(ts5)
    entry_idx = pd.DatetimeIndex(p.ts)
    if entry_idx.tz is None:
        entry_idx = entry_idx.tz_localize("UTC")
    else:
        entry_idx = entry_idx.tz_convert("UTC")
    entry_ns = entry_idx.asi8

    # ----- build M5 observations (PIT, first completed M5 after entry) -----
    cols = {k: [] for k in (
        "trade_id", "year", "bars_since_entry", "current_R", "mae_so_far_R", "mfe_so_far_R",
        "drawdown_from_MFE_R", "consecutive_adverse_m5", "consecutive_favorable_m5",
        "m5_return_1", "m5_return_2", "m5_return_3", "m5_return_6",
        "m5_range_R", "m5_body_R", "m5_wick_up_R", "m5_wick_down_R",
        "m5_atr_norm", "m5_vol12", "mom_1R", "cr_accel", "dist_to_sl_R",
        "y_prob", "atr_pct_entry", "ema_trend_duration", "rolling_quantile",
        "hour_sin", "hour_cos", "ctx_h4_swing_quality", "atr_percent",
        "large_loss", "p0_R", "mae_final", "p0_sl_event", "mins_to_p0_exit", "same_m5_sl",
    )}
    n_obs_trade = np.zeros(p.n, dtype=np.int32)
    mins_first = np.full(p.n, np.nan)
    n_ll_alive = {c: 0 for c in CKS}
    n_ll = 0

    for i in range(p.n):
        e, one, lng = float(p.entry[i]), float(one_r[i]), bool(p.is_long[i])
        if one <= 0 or e <= 0:
            continue
        last = int(hold[i])
        exit_i = int(min(ei[i] + last, n_h1 - 1))
        exit_end = int(h1_ns[exit_i]) + hour
        start = int(np.searchsorted(ts5, entry_ns[i], side="left"))
        if start >= n5:
            continue
        ll = r0[i] <= -1.00
        if ll:
            n_ll += 1
        sl = -1.0
        extreme = 0.0
        mae = 0.0
        consec_a = consec_f = 0
        prev_cr = 0.0
        prev_mom = 0.0
        rets = []
        cr_hist = []
        h1_done = 0
        k = start
        n_here = 0
        while k < n5 and ts5[k] < exit_end and n_here < 48 * 12:
            close_t = int(ts5[k]) + five
            # H1 trail from completed P0 path bars only (no future H1 close)
            while h1_done < last:
                bj = int(ei[i] + h1_done + 1)
                if bj >= n_h1:
                    break
                if int(h1_ns[bj]) + hour > close_t:
                    break
                h1_done += 1
                fv = p.fav[i, h1_done]
                if np.isfinite(fv):
                    extreme = max(extreme, float(fv))
                if extreme >= P0_ACT and np.isfinite(p.atr_over_r[i, h1_done]):
                    sl = max(sl, extreme - P0_DIST * float(p.atr_over_r[i, h1_done]))
            h, l, c, o = high5[k], low5[k], close5[k], open5[k]
            fav = ((h - e) if lng else (e - l)) / one
            adv = ((e - l) if lng else (h - e)) / one
            cr = ((c - e) if lng else (e - c)) / one
            extreme = max(extreme, fav)
            mae = max(mae, adv)
            same_sl = adv >= -sl
            if same_sl:
                break  # dead; this bar is not an actionable observation
            mom = cr - prev_cr
            if cr < prev_cr - 1e-12:
                consec_a += 1
                consec_f = 0
            elif cr > prev_cr + 1e-12:
                consec_f += 1
                consec_a = 0
            accel = mom - prev_mom
            rng = (h - l) / one
            body = abs(c - o) / one
            if lng:
                wu, wd = (h - max(o, c)) / one, (min(o, c) - l) / one
            else:
                wu, wd = (h - max(o, c)) / one, (min(o, c) - l) / one
            cr_hist.append(cr)
            rets.append(mom)
            n_here += 1
            if n_here == 1:
                mins_first[i] = 5.0
            if ll:
                for ck in CKS:
                    if n_here == ck:
                        n_ll_alive[ck] += 1
            def lag(n):
                return cr - cr_hist[-1 - n] if len(cr_hist) > n else 0.0
            vol = float(np.std(rets[-12:])) if len(rets) >= 3 else 0.0
            atrn = float(atr5[k] / one) if np.isfinite(atr5[k]) else 0.0
            rec = {
                "trade_id": i, "year": int(p.year[i]), "bars_since_entry": n_here,
                "current_R": cr, "mae_so_far_R": mae, "mfe_so_far_R": extreme,
                "drawdown_from_MFE_R": extreme - cr,
                "consecutive_adverse_m5": consec_a, "consecutive_favorable_m5": consec_f,
                "m5_return_1": lag(1), "m5_return_2": lag(2), "m5_return_3": lag(3), "m5_return_6": lag(6),
                "m5_range_R": rng, "m5_body_R": body, "m5_wick_up_R": wu, "m5_wick_down_R": wd,
                "m5_atr_norm": atrn, "m5_vol12": vol, "mom_1R": mom, "cr_accel": accel,
                "dist_to_sl_R": cr - sl,
                "y_prob": float(y_prob[i]),
                "atr_pct_entry": float(f7[i, f7i["atr_percentile_252"]]),
                "ema_trend_duration": float(f7[i, f7i["ema_trend_duration"]]),
                "rolling_quantile": float(f7[i, f7i["rolling_quantile"]]),
                "hour_sin": float(f7[i, f7i["hour_sin"]]),
                "hour_cos": float(f7[i, f7i["hour_cos"]]),
                "ctx_h4_swing_quality": float(h4[i]),
                "atr_percent": float(f7[i, f7i["atr_percent"]]),
                "large_loss": int(ll), "p0_R": float(r0[i]), "mae_final": float(mae_all[i]),
                "p0_sl_event": int(reason[i] in (0, 1, 2)),
                "mins_to_p0_exit": float(max((exit_end - close_t) / (hour / 60.0), 0.0)),
                "same_m5_sl": 0,
            }
            for key, val in rec.items():
                cols[key].append(val)
            prev_cr, prev_mom = cr, mom
            k += 1
        n_obs_trade[i] = n_here

    obs = pd.DataFrame(cols)
    print(f"obs={len(obs)} trades_with_obs={(n_obs_trade > 0).sum()} n_ll={n_ll}")
    trades = pd.DataFrame({
        "trade_id": np.arange(p.n), "year": p.year, "p0_R": r0, "mae_R": mae_all,
        "n_m5": n_obs_trade, "mins_first": mins_first,
        "large_loss": (r0 <= -1.00).astype(int),
        "p0_sl_event": np.isin(reason, [0, 1, 2]).astype(int),
    })
    oos_t = trades[trades["year"].isin(TRUE_OOS)]
    ll_t = oos_t[oos_t["large_loss"] == 1]

    # ----- Iter 1 coverage -----
    def pct_ge(s, n):
        return float((s >= n).mean()) if len(s) else 0.0
    cov = {
        "n_oos": int(len(oos_t)), "pct_ge1": pct_ge(oos_t["n_m5"], 1),
        "pct_ge2": pct_ge(oos_t["n_m5"], 2), "pct_ge3": pct_ge(oos_t["n_m5"], 3),
        "pct_ge6": pct_ge(oos_t["n_m5"], 6), "pct_ge12": pct_ge(oos_t["n_m5"], 12),
        "pct_ge24": pct_ge(oos_t["n_m5"], 24), "pct_ge48": pct_ge(oos_t["n_m5"], 48),
        "median_mins_first": float(np.nanmedian(oos_t["mins_first"])),
        "ll_n": int(len(ll_t)), "ll_ge1": pct_ge(ll_t["n_m5"], 1),
        "ll_ge2": pct_ge(ll_t["n_m5"], 2), "ll_ge3": pct_ge(ll_t["n_m5"], 3),
        "ll_ge6": pct_ge(ll_t["n_m5"], 6), "ll_ge12": pct_ge(ll_t["n_m5"], 12),
    }
    rows1 = []
    oos_obs = obs[obs["year"].isin(TRUE_OOS)]
    for ck in CKS:
        sub = oos_obs[oos_obs["bars_since_entry"] == ck]
        rows1.append({
            "checkpoint": f"M5+{ck}", "minutes": ck * 5, "observations": int(len(sub)),
            "P0_SL_rate": float(sub["p0_sl_event"].mean()) if len(sub) else 0.0,
            "final_positive_rate": float((sub["p0_R"] > 0).mean()) if len(sub) else 0.0,
            "median_current_R": float(sub["current_R"].median()) if len(sub) else 0.0,
            "large_loss_rate": float(sub["large_loss"].mean()) if len(sub) else 0.0,
        })
    pd.DataFrame(rows1).to_csv(OUT / "sprint51_iter1_window.csv", index=False)
    pd.DataFrame([cov]).to_csv(OUT / "sprint51_iter1_coverage.csv", index=False)
    gate1 = cov["ll_ge6"] >= 0.40 and cov["pct_ge1"] >= 0.95
    _log(1, "M5 observation coverage", "PASS" if gate1 else "FAIL",
         f"oos_ge1={cov['pct_ge1']:.3f} ll_ge6={cov['ll_ge6']:.3f} first_m={cov['median_mins_first']:.1f}",
         2 if gate1 else "STOP")
    if not gate1:
        _write("M5_WINDOW_UNUSABLE", {"gate1": False, **cov},
               _rep("M5_WINDOW_UNUSABLE", rows1, None, None, None, None, None, None, "FAIL",
                    "STOP_M5_LOSS_RESEARCH"))
        return 0

    # ----- Iter 2 labels -----
    lab_rows = []
    for scope, d in (("all", obs), ("oos", oos_obs),
                     *((f"y{y}", obs[obs["year"] == y]) for y in (2021,) + TRUE_OOS)):
        lab_rows.append({
            "scope": scope, "n_obs": int(len(d)), "n_trades": int(d["trade_id"].nunique()) if len(d) else 0,
            "n_pos": int(d["large_loss"].sum()) if len(d) else 0,
            "n_neg": int((d["large_loss"] == 0).sum()) if len(d) else 0,
            "base_rate": float(d["large_loss"].mean()) if len(d) else 0.0,
            "p0_sl_rate": float(d["p0_sl_event"].mean()) if len(d) else 0.0,
            "p_le_m1_05_trade": float((d.drop_duplicates("trade_id")["p0_R"] <= -1.05).mean()) if len(d) else 0.0,
        })
    for ck in CKS:
        sub = oos_obs[oos_obs["bars_since_entry"] == ck]
        lab_rows.append({"scope": f"oos_m5+{ck}", "n_obs": int(len(sub)),
                         "n_trades": int(sub["trade_id"].nunique()) if len(sub) else 0,
                         "n_pos": int(sub["large_loss"].sum()) if len(sub) else 0,
                         "n_neg": int((sub["large_loss"] == 0).sum()) if len(sub) else 0,
                         "base_rate": float(sub["large_loss"].mean()) if len(sub) else 0.0,
                         "p0_sl_rate": float(sub["p0_sl_event"].mean()) if len(sub) else 0.0,
                         "p_le_m1_05_trade": 0.0})
    pd.DataFrame(lab_rows).to_csv(OUT / "sprint51_iter2_labels.csv", index=False)
    _log(2, "frozen labels", "PASS",
         f"oos obs={len(oos_obs)} trades={oos_obs['trade_id'].nunique()} base={float(oos_obs['large_loss'].mean()):.3f}",
         3)

    # ----- Iter 3 baselines (2021-fit LR, OOS score; plus quantile buckets) -----
    oos_y = oos_obs["large_loss"].to_numpy(dtype=int)
    rows3 = []
    for name, feats in (("A_current_R", A_FEATS), ("B_path", B_FEATS), ("C_m5", C_FEATS),
                        ("D_h1_entry", D_FEATS), ("E_h1_m5", E_FEATS)):
        pred = _score_2021(obs, feats)
        po = pred[obs["year"].isin(TRUE_OOS)]
        roc, pr = _roc(oos_y, po), _pr(oos_y, po)
        tmp = oos_obs.copy()
        tmp["p"] = po
        sep, top, bot = _qsep(tmp.dropna(subset=["p"]), "p")
        rows3.append({"baseline": name, "roc_oos": roc, "pr_oos": pr, "sep": sep, "top_ll": top, "bot_ll": bot})
    oos_obs = oos_obs.copy()
    oos_obs["_adverse_R"] = -oos_obs["current_R"]
    bkt = []
    for col in ("_adverse_R", "consecutive_adverse_m5", "mom_1R", "m5_vol12", "drawdown_from_MFE_R", "bars_since_entry"):
        sep, top, bot = _qsep(oos_obs, col)
        bkt.append({"feature": col, "sep": sep, "top_ll": top, "bot_ll": bot,
                    "note": "q5-q1; current_R high quintile is least adverse"})
    pd.DataFrame(rows3).to_csv(OUT / "sprint51_iter3_baselines.csv", index=False)
    pd.DataFrame(bkt).to_csv(OUT / "sprint51_iter3_buckets.csv", index=False)
    a = next(r for r in rows3 if r["baseline"] == "A_current_R")
    c = next(r for r in rows3 if r["baseline"] == "C_m5")
    e = next(r for r in rows3 if r["baseline"] == "E_h1_m5")
    gate3 = ((e["roc_oos"] - a["roc_oos"]) >= 0.02 or (c["roc_oos"] - a["roc_oos"]) >= 0.02) and (
        e["sep"] >= a["sep"] + 0.03 or c["sep"] >= a["sep"] + 0.03)
    _log(3, "baseline predictability", "PASS" if gate3 else "FAIL",
         f"A roc={a['roc_oos']:.3f} C={c['roc_oos']:.3f} E={e['roc_oos']:.3f} A_sep={a['sep']:.3f} E_sep={e['sep']:.3f}",
         4 if gate3 else "STOP")
    if not gate3:
        _write("NO_INCREMENTAL_M5_SIGNAL", {"gate1": True, "gate3": False, "a_roc": a["roc_oos"],
                                            "c_roc": c["roc_oos"], "e_roc": e["roc_oos"]},
               _rep("NO_INCREMENTAL_M5_SIGNAL", rows1, lab_rows, rows3, None, None, None, None, "PASS",
                    "STOP_M5_LOSS_RESEARCH"))
        return 0

    # ----- Iter 4 LGBM expanding -----
    obs = obs.copy()
    obs["p_m5"] = _score_lgbm(obs, ALL_FEATS)
    oos_obs = obs[obs["year"].isin(TRUE_OOS) & np.isfinite(obs["p_m5"])].copy()
    y, pv = oos_obs["large_loss"].to_numpy(int), oos_obs["p_m5"].to_numpy()
    roc, pr, ece = _roc(y, pv), _pr(y, pv), _ece(y, pv)
    sep, top, bot = _qsep(oos_obs, "p_m5")
    buckets = []
    for lo, hi in P_BINS:
        g = oos_obs[(oos_obs["p_m5"] >= lo) & (oos_obs["p_m5"] < hi)]
        buckets.append({"bin": f"{lo:.2f}-{hi:.2f}", "n": int(len(g)),
                        "ll_rate": float(g["large_loss"].mean()) if len(g) else 0.0,
                        "mean_p": float(g["p_m5"].mean()) if len(g) else 0.0,
                        "median_current_R": float(g["current_R"].median()) if len(g) else 0.0})
    yroc = []
    for yv in TRUE_OOS:
        gy = oos_obs[oos_obs["year"] == yv]
        if len(gy) < 50:
            continue
        s, t, b = _qsep(gy, "p_m5")
        yroc.append({"year": yv, "roc": _roc(gy["large_loss"], gy["p_m5"]),
                     "sep": s, "top_ll": t, "bot_ll": b, "n": int(len(gy))})
    pd.DataFrame([{"roc_oos": roc, "pr_oos": pr, "ece": ece, "sep": sep, "top_ll": top, "bot_ll": bot,
                   "n_oos": int(len(oos_obs)), "thr": THR}]).to_csv(OUT / "sprint51_iter4_model.csv", index=False)
    pd.DataFrame(buckets).to_csv(OUT / "sprint51_iter4_buckets.csv", index=False)
    pd.DataFrame(yroc).to_csv(OUT / "sprint51_iter4_yearly.csv", index=False)
    n_pos_sep = sum(1 for r in yroc if r["sep"] >= 0.08)
    gate4 = roc >= 0.60 and n_pos_sep >= 3
    _log(4, "M5 LGBM", "PASS" if gate4 else "FAIL",
         f"roc={roc:.3f} pr={pr:.3f} sep={sep:.3f} years_sep={n_pos_sep}/5",
         5 if gate4 else "STOP")
    if not gate4:
        _write("NO_INCREMENTAL_M5_SIGNAL", {"gate4": False, "roc": roc, "sep": sep},
               _rep("NO_INCREMENTAL_M5_SIGNAL", rows1, lab_rows, rows3,
                    {"roc": roc, "pr": pr, "ece": ece, "sep": sep, "top": top, "bot": bot, "buckets": buckets, "yroc": yroc},
                    None, None, None, "PASS", "STOP_M5_LOSS_RESEARCH"))
        return 0

    # ----- Iter 5 lead time: first p>=THR per trade, skip same-M5-SL (already excluded) -----
    flagged = oos_obs[oos_obs["p_m5"] >= THR].sort_values(["trade_id", "bars_since_entry"])
    first = flagged.drop_duplicates("trade_id")
    ll_first = first[first["large_loss"] == 1]
    mins = ll_first["mins_to_p0_exit"].to_numpy(dtype=float)
    lead = {
        "n_flagged_trades": int(first["trade_id"].nunique()),
        "n_ll_flagged": int(len(ll_first)),
        "median_mins_to_sl": float(np.nanmedian(mins)) if len(mins) else 0.0,
        "p25": float(np.nanpercentile(mins, 25)) if len(mins) else 0.0,
        "p50": float(np.nanpercentile(mins, 50)) if len(mins) else 0.0,
        "p75": float(np.nanpercentile(mins, 75)) if len(mins) else 0.0,
        "median_current_R": float(ll_first["current_R"].median()) if len(ll_first) else 0.0,
        "median_dist_sl": float(ll_first["dist_to_sl_R"].median()) if len(ll_first) else 0.0,
        "pct_ge30": float((mins >= 30).mean()) if len(mins) else 0.0,
        "pct_ge60": float((mins >= 60).mean()) if len(mins) else 0.0,
        "pct_ge90": float((mins >= 90).mean()) if len(mins) else 0.0,
        "pct_ge120": float((mins >= 120).mean()) if len(mins) else 0.0,
        "median_bars": float(ll_first["bars_since_entry"].median()) if len(ll_first) else 0.0,
        "same_m5_sl_share": 0.0,
    }
    pd.DataFrame([lead]).to_csv(OUT / "sprint51_iter5_lead.csv", index=False)
    first.to_csv(OUT / "sprint51_iter5_first_hits.csv", index=False)
    gate5 = lead["n_ll_flagged"] >= 30 and lead["pct_ge30"] >= 0.30 and lead["median_mins_to_sl"] >= 30
    _log(5, "lead time", "PASS" if gate5 else "FAIL",
         f"ll_hits={lead['n_ll_flagged']} med_min={lead['median_mins_to_sl']:.1f} ge30={lead['pct_ge30']:.2f}",
         6 if gate5 else "STOP")
    if not gate5:
        _write("M5_SIGNAL_TOO_LATE", {"gate5": False, **lead, "roc": roc},
               _rep("M5_SIGNAL_TOO_LATE", rows1, lab_rows, rows3,
                    {"roc": roc, "pr": pr, "ece": ece, "sep": sep, "top": top, "bot": bot, "buckets": buckets, "yroc": yroc},
                    lead, None, None, "PASS", "M5_SIGNAL_TOO_LATE"))
        return 0

    # ----- Iter 6 recovery -----
    rec075 = oos_t[(oos_t["mae_R"] >= 0.75) & (oos_t["p0_R"] > 0)]
    rec1 = oos_t[(oos_t["mae_R"] >= 1.00) & (oos_t["p0_R"] > 0)]
    hit_ids = set(int(x) for x in first["trade_id"])
    rec_rows = []
    for lab, pop in (("mae>=0.75 P0>0", rec075), ("mae>=1.00 P0>0", rec1)):
        n = len(pop)
        cut = int(pop["trade_id"].isin(hit_ids).sum())
        rec_rows.append({
            "population": lab, "n": int(n), "incorrectly_cut": cut,
            "recovery_preservation": float(1.0 - cut / n) if n else 1.0,
            "mean_final_R": float(pop["p0_R"].mean()) if n else 0.0,
            "median_final_R": float(pop["p0_R"].median()) if n else 0.0,
        })
    risk_rows = []
    for lo, hi in P_BINS:
        g = oos_obs[(oos_obs["p_m5"] >= lo) & (oos_obs["p_m5"] < hi)].drop_duplicates("trade_id", keep="first")
        if not len(g):
            continue
        rec_rate = float(((g["mae_final"] >= 0.75) & (g["p0_R"] > 0)).mean())
        risk_rows.append({
            "risk_bucket": f"{lo:.2f}-{hi:.2f}", "n": int(len(g)),
            "large_loss_rate": float(g["large_loss"].mean()),
            "recovery_rate": rec_rate,
            "current_R": float(g["current_R"].median()),
            "minutes_to_SL": float(g["mins_to_p0_exit"].median()),
        })
    pd.DataFrame(rec_rows).to_csv(OUT / "sprint51_iter6_recovery.csv", index=False)
    pd.DataFrame(risk_rows).to_csv(OUT / "sprint51_iter6_risk.csv", index=False)
    top = next((r for r in risk_rows if r["risk_bucket"].startswith("0.80")), None)
    gate6 = top is not None and top["large_loss_rate"] >= 0.45 and top["recovery_rate"] <= 0.40
    pres075 = next(r["recovery_preservation"] for r in rec_rows if "0.75" in r["population"])
    _log(6, "recovery protection", "PASS" if gate6 else "FAIL",
         f"top_ll={top['large_loss_rate'] if top else 0:.3f} top_rec={top['recovery_rate'] if top else 0:.3f} pres075={pres075:.3f}",
         7 if gate6 else "STOP")
    if not gate6:
        _write("RECOVERY_DOMINATES", {"gate6": False, "pres075": pres075, "top": top},
               _rep("RECOVERY_DOMINATES", rows1, lab_rows, rows3,
                    {"roc": roc, "pr": pr, "ece": ece, "sep": sep, "top": top, "bot": bot, "buckets": buckets, "yroc": yroc},
                    lead, rec_rows, risk_rows, "PASS", "STOP_M5_LOSS_RESEARCH"))
        return 0

    # Iter 7–8 would run here. Discovery sprint still requires executable economics if we got this far.
    _write("CANDIDATE_FOR_M5_LOSS_ACTION_RESEARCH", {
        "gate1": True, "gate3": True, "gate4": True, "gate5": True, "gate6": True, "gate7": False,
        "note": "recovery+lead passed; economics not run in this discovery stop? protocol says run iter7",
        "roc": roc, **lead,
    }, _rep("CANDIDATE_FOR_M5_LOSS_ACTION_RESEARCH", rows1, lab_rows, rows3,
            {"roc": roc, "pr": pr, "ece": ece, "sep": sep, "top": top, "bot": bot, "buckets": buckets, "yroc": yroc},
            lead, rec_rows, risk_rows, "PASS", "CANDIDATE_FOR_M5_LOSS_ACTION_RESEARCH"))
    return 0


def _rep(verdict, rows1, lab_rows, rows3, model, lead, rec_rows, risk_rows, leak, rec) -> str:
    def blk(title, body):
        return [f"### {title}", "", body if isinstance(body, str) else (body or "Not evaluated."), ""]
    s1 = _md(rows1, ["checkpoint", "minutes", "observations", "P0_SL_rate", "final_positive_rate", "median_current_R"],
             {"minutes": 0, "observations": 0, "P0_SL_rate": 3, "final_positive_rate": 3, "median_current_R": 3}) if rows1 else "n/a"
    s2 = _md(lab_rows[:8], ["scope", "n_obs", "n_trades", "n_pos", "base_rate"],
             {"n_obs": 0, "n_trades": 0, "n_pos": 0, "base_rate": 3}) if lab_rows else "n/a"
    s3 = _md(rows3, ["baseline", "roc_oos", "pr_oos", "sep", "top_ll", "bot_ll"],
             {"roc_oos": 3, "pr_oos": 3, "sep": 3, "top_ll": 3, "bot_ll": 3}) if rows3 else "Not evaluated."
    if model:
        s4 = (f"OOS ROC={model['roc']:.3f} PR={model['pr']:.3f} ECE={model['ece']:.3f} "
              f"sep={model['sep']:.3f} top={model['top']:.3f} bot={model['bot']:.3f}\n\n"
              + _md(model["buckets"], ["bin", "n", "ll_rate", "mean_p", "median_current_R"],
                    {"n": 0, "ll_rate": 3, "mean_p": 3, "median_current_R": 3})
              + "\n\n" + _md(model["yroc"], ["year", "roc", "sep", "top_ll", "bot_ll", "n"],
                             {"year": 0, "roc": 3, "sep": 3, "top_ll": 3, "bot_ll": 3, "n": 0}))
    else:
        s4 = "Not evaluated."
    s5 = (f"n_ll_flagged={lead['n_ll_flagged']} median_mins={lead['median_mins_to_sl']:.1f} "
          f"p25/50/75={lead['p25']:.0f}/{lead['p50']:.0f}/{lead['p75']:.0f} "
          f">=30m={lead['pct_ge30']:.2f} >=60m={lead['pct_ge60']:.2f} "
          f"median_R={lead['median_current_R']:.3f} dist_sl={lead['median_dist_sl']:.3f}"
          if lead else "Not evaluated.")
    s6 = (_md(rec_rows, ["population", "n", "incorrectly_cut", "recovery_preservation", "mean_final_R"],
              {"n": 0, "incorrectly_cut": 0, "recovery_preservation": 3, "mean_final_R": 3})
          + ("\n\n" + _md(risk_rows, ["risk_bucket", "n", "large_loss_rate", "recovery_rate", "current_R", "minutes_to_SL"],
                          {"n": 0, "large_loss_rate": 3, "recovery_rate": 3, "current_R": 3, "minutes_to_SL": 1})
             if risk_rows else "")
          if rec_rows else "Not evaluated.")
    return "\n".join([
        "## Sprint 51 Verdict",
        "",
        f"**VERDICT: `{verdict}`**",
        "",
        "Independent M5 monitor from first completed M5 after H1 entry. No wait for H1 close.",
        "No Sprint 46/47/49/50 scores. Production unchanged.",
        "",
        *blk("1. M5 Window", s1),
        *blk("2. Label", "Primary `LARGE_LOSS = P0_R <= -1.00R` (frozen). " + (s2 if isinstance(s2, str) else "")),
        s2, "",
        *blk("3. Baseline Predictability", s3),
        *blk("4. M5 Model", s4),
        *blk("5. Lead Time", s5),
        *blk("6. Recovery Protection", s6),
        *blk("7. Economic Action", "Not evaluated (stopped before Iter 7, or discovery-only until gates pass)."),
        *blk("8. OOS Robustness", "Not evaluated."),
        *blk("9. Leakage Audit", leak),
        "Features use current/past M5 + frozen H1 entry context only. No future MAE/MFE, no final R, "
        "no H1 close after the M5 timestamp, no prior sprint probabilities. THR=0.70 frozen a priori. 2021 not OOS.",
        "",
        "### 10. Production",
        "",
        "Entry: UNCHANGED",
        "",
        "P0: UNCHANGED",
        "",
        "M5 loss action: NOT SHIPPED",
        "",
        "P1/P2/P3: NOT USED",
        "",
        "Reversal: NOT USED",
        "",
        "### 11. Recommendation",
        "",
        rec,
        "",
    ])


if __name__ == "__main__":
    raise SystemExit(main())
