"""Sprint 45 — Reversal signal discovery (research only, no action).

Primary label frozen: first MAE>=0.50R, then recovery>=0.75R from that extreme.
No production change. No close+reverse. No threshold search.

  python apps/research_sprint45_reversal.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import tools.pyc_path_hook  # noqa: F401

from apps.research_sprint41_iter1_diff import STATE_FEATS
from apps.research_sprint42_attribution import TRUE_OOS, _md
from apps.run_exit_engine_grid import CAP, FEAT7, Paths, simulate_combo
from apps.research_sprint39_exit_state import _atr_pctile, _join_feat7, _load_panel
from simulation.wf.sim import entry_indices, load_h1, prepare_market

OUT = _ROOT / "artifacts/pipeline_backtest/reversal/sprint45"
ADV, REC = 0.50, 0.75  # primary frozen
SENS = ((0.25, 0.50), (0.50, 0.50), (0.50, 0.75), (0.75, 1.00))
MODEL_FEATS = STATE_FEATS + ["bars_in_trade", "adverse_excursion_R", "distance_from_entry_R"]
YEARS = (2021, 2022, 2023, 2024, 2025, 2026)
P0 = dict(act=0.25, dist=0.08, tp=None, partials=(), tmax=None, be=None)
GROUPS = {
    "geometry": ["current_R", "adverse_excursion_R", "distance_from_entry_R",
                 "mae_so_far_R", "mfe_so_far_R", "drawdown_from_MFE_R"],
    "momentum": ["mom_1R"],
    "trend": ["ema_dist_atr", "ema_trend_duration", "rolling_quantile"],
    "vol": ["atr_pct_entry", "atr_pct_now"],
    "session": ["hour_sin", "hour_cos", "hour_sin_now", "hour_cos_now"],
    "entry_prob": ["y_prob"],
}


def _spearman(a, b) -> float:
    s = pd.Series(a).corr(pd.Series(b), method="spearman")
    return float(s) if s == s else 0.0


def _roc(y, p) -> float:
    y = np.asarray(y, dtype=int)
    if y.min() == y.max():
        return 0.5
    return float(roc_auc_score(y, p))


def _pr(y, p) -> float:
    y = np.asarray(y, dtype=int)
    if y.min() == y.max():
        return float(y.mean())
    return float(average_precision_score(y, p))


def _fit(model: str, Xtr, ytr, Xte) -> np.ndarray:
    if model == "lr":
        sc = StandardScaler()
        clf = LogisticRegression(max_iter=200, class_weight="balanced")
        clf.fit(sc.fit_transform(Xtr), ytr)
        return clf.predict_proba(sc.transform(Xte))[:, 1]
    import lightgbm as lgb
    clf = lgb.LGBMClassifier(
        n_estimators=80, learning_rate=0.05, num_leaves=15,
        min_child_samples=40, subsample=0.8, colsample_bytree=0.8,
        verbosity=-1, class_weight="balanced",
    )
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]


def _ece(y, p, n_bins=10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    err = 0.0
    n = len(p)
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        err += m.mean() * abs(float(p[m].mean()) - float(y[m].mean()))
    return float(err)


def _build_events(p: Paths, mkt: dict, r_p0: np.ndarray) -> pd.DataFrame:
    ei = entry_indices(p.panel, mkt["ts"])
    close_s, atr_s = mkt["close"], mkt["atr"]
    ema = pd.Series(close_s).ewm(span=20, adjust=False).mean().to_numpy()
    atr_pct_bar = _atr_pctile(atr_s)
    hour0 = pd.DatetimeIndex(p.ts).hour.to_numpy()
    y_prob = p.panel["y_prob"].astype(float).to_numpy()
    f7 = p.panel[list(FEAT7)].astype(float).to_numpy()
    f7i = {name: k for k, name in enumerate(FEAT7)}
    ts0 = pd.DatetimeIndex(p.ts)
    rows = []
    for i in range(p.n):
        last = int(p.last_off[i])
        mae = -1.0
        j = -1
        for t in range(1, last + 1):
            a = p.adv[i, t]
            if np.isfinite(a):
                mae = max(mae, float(a))
            if mae >= ADV:
                j = t
                break
        if j < 0:
            continue
        recov = -np.inf
        t_rec = -1
        for k in range(j + 1, last + 1):
            f = p.fav[i, k]
            if not np.isfinite(f):
                continue
            rv = float(f) + mae
            if rv > recov:
                recov = rv
            if t_rec < 0 and rv >= REC:
                t_rec = k - j
        if recov == -np.inf:
            recov = float("nan")
        cr = p.close_r[i, j]
        if not np.isfinite(cr):
            continue
        mfe = float(np.nanmax(p.fav[i, 1 : j + 1]))
        idx = int(ei[i] + j)
        if idx >= len(close_s):
            continue
        prev = p.close_r[i, j - 1] if j > 1 else 0.0
        hr = int((hour0[i] + j) % 24)
        sign = 1.0 if p.is_long[i] else -1.0
        ema_dist = sign * (close_s[idx] - ema[idx]) / max(float(p.atr[i]), 1e-12)
        rows.append({
            "trade_id": i,
            "year": int(p.year[i]),
            "side": "long" if p.is_long[i] else "short",
            "entry_time": ts0[i],
            "event_time": ts0[i] + pd.Timedelta(hours=j),
            "hour_utc": hr,
            "bars_in_trade": j,
            "y_prob": float(y_prob[i]),
            "atr_pct_entry": float(f7[i, f7i["atr_percentile_252"]]),
            "atr_pct_now": float(atr_pct_bar[idx]) if np.isfinite(atr_pct_bar[idx]) else 0.5,
            "ema_dist_atr": float(ema_dist),
            "ema_trend_duration": float(f7[i, f7i["ema_trend_duration"]]),
            "rolling_quantile": float(f7[i, f7i["rolling_quantile"]]),
            "hour_sin": float(f7[i, f7i["hour_sin"]]),
            "hour_cos": float(f7[i, f7i["hour_cos"]]),
            "hour_sin_now": math.sin(2 * math.pi * hr / 24),
            "hour_cos_now": math.cos(2 * math.pi * hr / 24),
            "mfe_so_far_R": mfe,
            "mae_so_far_R": mae,
            "drawdown_from_MFE_R": mfe - float(cr),
            "current_R": float(cr),
            "mom_1R": float(cr - prev) if np.isfinite(prev) else 0.0,
            "adverse_excursion_R": mae,
            "distance_from_entry_R": float(cr),
            "max_recovery_from_extreme_R": float(recov) if recov == recov else 0.0,
            "bars_to_recovery": t_rec if t_rec > 0 else np.nan,
            "reversal": int(np.isfinite(recov) and recov >= REC),
            "p0_R": float(r_p0[i]),
        })
    return pd.DataFrame(rows)


def _sens_rates(p: Paths) -> pd.DataFrame:
    rows = []
    for adv, rec in SENS:
        n_ev = n_pos = 0
        by_y = {y: [0, 0] for y in YEARS}
        for i in range(p.n):
            last = int(p.last_off[i])
            mae = -1.0
            j = -1
            for t in range(1, last + 1):
                a = p.adv[i, t]
                if np.isfinite(a):
                    mae = max(mae, float(a))
                if mae >= adv:
                    j = t
                    break
            if j < 0:
                continue
            n_ev += 1
            recov = -np.inf
            for k in range(j + 1, last + 1):
                f = p.fav[i, k]
                if np.isfinite(f):
                    recov = max(recov, float(f) + mae)
            pos = recov >= rec
            n_pos += int(pos)
            y = int(p.year[i])
            if y in by_y:
                by_y[y][0] += 1
                by_y[y][1] += int(pos)
        rows.append({
            "adverse": adv, "recovery": rec, "n_events": n_ev, "n_pos": n_pos,
            "rate": n_pos / n_ev if n_ev else 0.0, "primary": adv == ADV and rec == REC,
            **{f"rate_{y}": (by_y[y][1] / by_y[y][0] if by_y[y][0] else 0.0) for y in YEARS},
        })
    return pd.DataFrame(rows)


def _write_stop(verdict: str, extra: dict) -> None:
    payload = {"sprint": 45, "verdict": verdict, "production_changed": False, **extra}
    (OUT / "sprint45_final_verdict.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = _join_feat7(_load_panel())
    mkt = prepare_market(load_h1(_ROOT / "artifacts/raw/XAUUSD/H1/data.parquet"))
    p = Paths(panel, mkt)
    sim0 = simulate_combo(p, **P0)
    ev = _build_events(p, mkt, sim0["r_multiple"])
    ev.to_csv(OUT / "reversal_event_dataset.csv", index=False)
    sens = _sens_rates(p)
    n_tr, n_ev = p.n, len(ev)
    n_pos = int(ev["reversal"].sum())
    rate = n_pos / n_ev if n_ev else 0.0
    oos = ev[ev["year"].isin(TRUE_OOS)]
    pos_years = [y for y in TRUE_OOS if ((oos["year"] == y) & (oos["reversal"] == 1)).any()]
    print(f"trades={n_tr} events={n_ev} pos={n_pos} rate={rate:.3f} oos_years_with_pos={pos_years}")

    leak_feats = [{
        "feature": f, "future_dependency": False,
        "notes": "PIT at first MAE>=0.50 bar; no future OHLC",
    } for f in MODEL_FEATS]
    leak = {
        "features": leak_feats,
        "labels": ["reversal", "max_recovery_from_extreme_R", "bars_to_recovery", "p0_R"],
        "label_future_dependency": True,
        "leakage": False,
    }
    (OUT / "leakage_audit.json").write_text(json.dumps(leak, indent=2), encoding="utf-8")

    pos, neg = ev[ev["reversal"] == 1], ev[ev["reversal"] == 0]
    diag = []
    for lab, sub in (("all", ev), ("positive", pos), ("negative", neg)):
        diag.append({
            "group": lab, "n": int(len(sub)),
            "mean_mae": float(sub["mae_so_far_R"].mean()) if len(sub) else 0,
            "mean_mfe": float(sub["mfe_so_far_R"].mean()) if len(sub) else 0,
            "mean_current_R": float(sub["current_R"].mean()) if len(sub) else 0,
            "mean_bars": float(sub["bars_in_trade"].mean()) if len(sub) else 0,
            "mean_adverse": float(sub["adverse_excursion_R"].mean()) if len(sub) else 0,
            "mean_recovery": float(sub["max_recovery_from_extreme_R"].mean()) if len(sub) else 0,
            "median_recovery": float(sub["max_recovery_from_extreme_R"].median()) if len(sub) else 0,
            "mean_bars_to_recovery": float(sub["bars_to_recovery"].mean()) if len(sub) else 0,
            "mean_p0_R": float(sub["p0_R"].mean()) if len(sub) else 0,
        })
    for y in YEARS:
        sub = ev[ev["year"] == y]
        diag.append({"group": f"year_{y}", "n": int(len(sub)),
                     "n_pos": int(sub["reversal"].sum()), "rate": float(sub["reversal"].mean()) if len(sub) else 0})
    for side in ("long", "short"):
        sub = ev[ev["side"] == side]
        diag.append({"group": side, "n": int(len(sub)), "rate": float(sub["reversal"].mean()) if len(sub) else 0})
    pd.concat([pd.DataFrame(diag), sens], axis=0, ignore_index=True).to_csv(
        OUT / "reversal_label_diagnostics.csv", index=False
    )
    (OUT / "dataset_manifest.json").write_text(json.dumps({
        "n_trades": n_tr, "n_events": n_ev, "n_pos": n_pos, "rate": rate,
        "primary_adverse": ADV, "primary_recovery": REC, "grain": "first_qualifying_MAE_per_trade",
        "years": {int(y): int((ev["year"] == y).sum()) for y in YEARS},
    }, indent=2), encoding="utf-8")

    enough = n_ev >= 200 and n_pos >= 40 and len(pos_years) >= 4 and 0.05 <= rate <= 0.90 and not leak["leakage"]
    if not enough:
        v = "NO_REVERSAL_SIGNAL" if n_ev else "NO_REVERSAL_SIGNAL"
        (OUT / "sprint45_report.md").write_text(
            f"# Sprint 45\n\n**Verdict: {v}**\n\nIter 1 gate fail. events={n_ev} pos={n_pos} rate={rate:.3f} years={pos_years}\n",
            encoding="utf-8",
        )
        _write_stop(v, {"iter": 1, "n_events": n_ev, "rate": rate})
        print(f"STOP {v} iter1")
        return 0

    # --- Iter 2 univariate ---
    stab = []
    q_edges = {f: np.nanquantile(ev.loc[ev["year"] == 2021, f].to_numpy(dtype=float), [0.2, 0.4, 0.6, 0.8])
               for f in MODEL_FEATS if f in ev.columns}
    for f in MODEL_FEATS:
        xall = ev[f].to_numpy(dtype=float)
        yall = ev["reversal"].to_numpy(dtype=int)
        signs = []
        for y in TRUE_OOS:
            sub = ev[ev["year"] == y]
            sp = _spearman(sub[f], sub["reversal"])
            au = _roc(sub["reversal"], sub[f].to_numpy(dtype=float))
            # if feature is inverse, AUC may be <0.5; keep raw spearman sign
            signs.append(np.sign(sp) if abs(sp) > 1e-12 else 0)
            stab.append({"feature": f, "year": y, "spearman": sp, "auc_raw": au, "n": int(len(sub))})
        oos_sp = [r["spearman"] for r in stab[-5:]]
        n_same = 0
        if oos_sp:
            maj = np.sign(np.nanmedian(oos_sp))
            n_same = int(sum(np.sign(s) == maj and abs(s) >= 0.10 for s in oos_sp))
        mean_sp = float(np.nanmean(oos_sp))
        stable = abs(mean_sp) >= 0.10 and n_same >= 4
        # quintiles on 2021 edges
        edges = q_edges.get(f)
        if edges is not None and np.isfinite(edges).all():
            q = np.digitize(oos[f].to_numpy(dtype=float), edges)
            rates = [float(oos.loc[q == k, "reversal"].mean()) if (q == k).any() else float("nan") for k in range(5)]
        else:
            rates = [float("nan")] * 5
        stab.append({
            "feature": f, "year": "oos_summary", "spearman": mean_sp,
            "n_years_abs_ge10_same_sign": n_same, "stable": stable,
            **{f"q{i+1}_rate": rates[i] for i in range(5)},
        })
    sdf = pd.DataFrame(stab)
    sdf.to_csv(OUT / "reversal_feature_stability.csv", index=False)
    n_stable = int(sdf.loc[sdf["year"] == "oos_summary", "stable"].sum())
    print(f"stable PIT features: {n_stable}")
    if n_stable == 0:
        (OUT / "sprint45_report.md").write_text(
            "# Sprint 45\n\n**Verdict: NO_REVERSAL_SIGNAL**\n\nIter 2: no stable PIT feature (|Spearman|>=0.10 in >=4/5 OOS years).\n",
            encoding="utf-8",
        )
        _write_stop("NO_REVERSAL_SIGNAL", {"iter": 2, "n_stable": 0, "n_events": n_ev, "rate": rate})
        print("STOP NO_REVERSAL_SIGNAL iter2")
        return 0

    # --- Iter 3 model ---
    X = ev[MODEL_FEATS].astype(float).fillna(0.0)
    y = ev["reversal"].astype(int)
    years = ev["year"].to_numpy()
    pred = np.full(len(ev), np.nan)
    val_rows = []
    for te_y in TRUE_OOS:
        tr = years < te_y
        te = years == te_y
        val_y = te_y - 1
        if tr.sum() < 80 or te.sum() < 30:
            continue
        for model in ("lr", "lgbm"):
            p_te = _fit(model, X[tr], y[tr], X[te])
            p_val = _fit(model, X[years < val_y], y[years < val_y], X[years == val_y]) if (years == val_y).sum() > 30 and (years < val_y).sum() > 80 else p_te
            val_rows.append({
                "model": model, "test_year": te_y, "val_year": val_y,
                "val_auc": _roc(y[years == val_y], p_val) if (years == val_y).sum() > 30 else float("nan"),
                "oos_auc": _roc(y[te], p_te), "oos_pr": _pr(y[te], p_te),
            })
            if model == "lr":
                pred[te] = p_te  # placeholder; overwrite with selected later
    vdf = pd.DataFrame(val_rows)
    val_mean = vdf.groupby("model")["val_auc"].mean()
    best = str(val_mean.idxmax()) if len(val_mean) else "lr"
    # refit selected on expanding train for each OOS year
    pred[:] = np.nan
    oos_rows = []
    for te_y in TRUE_OOS:
        tr, te = years < te_y, years == te_y
        if tr.sum() < 80 or te.sum() < 30:
            continue
        p_te = _fit(best, X[tr], y[tr], X[te])
        pred[te] = p_te
        yt, pt = y[te].to_numpy(), p_te
        oos_rows.append({
            "model": best, "year": te_y, "n": int(te.sum()), "rate": float(yt.mean()),
            "roc": _roc(yt, pt), "pr": _pr(yt, pt),
            "brier": float(brier_score_loss(yt, pt)),
            "logloss": float(log_loss(yt, np.clip(pt, 1e-6, 1 - 1e-6))),
            "ece": _ece(yt, pt),
            "prec_0.5": float(((pt >= 0.5) & (yt == 1)).sum() / max((pt >= 0.5).sum(), 1)),
            "rec_0.5": float(((pt >= 0.5) & (yt == 1)).sum() / max(yt.sum(), 1)),
        })
    ev["p_rev"] = pred
    oos_m = pd.DataFrame(oos_rows)
    oos_m.to_csv(OUT / "reversal_model_oos.csv", index=False)
    oos_ev = ev[np.isfinite(ev["p_rev"]) & ev["year"].isin(TRUE_OOS)].copy()
    oos_auc = _roc(oos_ev["reversal"], oos_ev["p_rev"])
    oos_pr = _pr(oos_ev["reversal"], oos_ev["p_rev"])
    lgbm_auc = float(vdf.loc[vdf["model"] == "lgbm", "oos_auc"].mean()) if (vdf["model"] == "lgbm").any() else 0
    lr_auc = float(vdf.loc[vdf["model"] == "lr", "oos_auc"].mean()) if (vdf["model"] == "lr").any() else 0
    if lgbm_auc > 0.95 and lr_auc < 0.75:
        print("WARN LGBM roc suspiciously high vs LR — treating as leakage risk")
        leak["suspected"] = True
        (OUT / "leakage_audit.json").write_text(json.dumps(leak, indent=2), encoding="utf-8")
        if lgbm_auc > 0.98:
            _write_stop("LEAKAGE_FAIL", {"lgbm_auc": lgbm_auc, "lr_auc": lr_auc})
            (OUT / "sprint45_report.md").write_text(
                f"# Sprint 45\n\n**Verdict: LEAKAGE_FAIL**\n\nLGBM OOS ROC={lgbm_auc:.3f} vs LR={lr_auc:.3f}\n",
                encoding="utf-8",
            )
            print("STOP LEAKAGE_FAIL")
            return 0
    (OUT / "model_config.json").write_text(json.dumps({
        "selected": best, "features": MODEL_FEATS, "val_mean_auc": val_mean.to_dict(),
        "lr": {"max_iter": 200, "class_weight": "balanced"},
        "lgbm": {"n_estimators": 80, "num_leaves": 15, "min_child_samples": 40},
        "selection": "mean val AUC, no OOS",
    }, indent=2), encoding="utf-8")
    print(f"best={best} oos_auc={oos_auc:.3f} oos_pr={oos_pr:.3f}")

    # --- Iter 4 buckets ---
    bins = [(-np.inf, 0.20), (0.20, 0.40), (0.40, 0.60), (0.60, 0.80), (0.80, np.inf)]
    blabs = ["lt_0.20", "0.20_0.40", "0.40_0.60", "0.60_0.80", "ge_0.80"]
    brows = []
    prb = oos_ev["p_rev"].to_numpy()
    for lab, (lo, hi) in zip(blabs, bins):
        m = (prb >= lo) & (prb < hi) if hi < np.inf else (prb >= lo)
        sub = oos_ev[m]
        brows.append({
            "bucket": lab, "n": int(len(sub)),
            "reversal_rate": float(sub["reversal"].mean()) if len(sub) else 0,
            "mean_p": float(sub["p_rev"].mean()) if len(sub) else 0,
            "mean_recovery": float(sub["max_recovery_from_extreme_R"].mean()) if len(sub) else 0,
            "median_recovery": float(sub["max_recovery_from_extreme_R"].median()) if len(sub) else 0,
            "mean_p0_R": float(sub["p0_R"].mean()) if len(sub) else 0,
            "median_p0_R": float(sub["p0_R"].median()) if len(sub) else 0,
        })
    bdf = pd.DataFrame(brows)
    bdf.to_csv(OUT / "reversal_probability_buckets.csv", index=False)
    recs = [r["mean_recovery"] for r in brows if r["n"] >= 10]
    mono = all(recs[i] <= recs[i + 1] + 0.05 for i in range(len(recs) - 1)) if len(recs) >= 3 else False
    top_bot_rec = (brows[-1]["mean_recovery"] - brows[0]["mean_recovery"]) if brows[0]["n"] and brows[-1]["n"] else 0.0
    top_bot_rate = (brows[-1]["reversal_rate"] - brows[0]["reversal_rate"]) if brows[0]["n"] and brows[-1]["n"] else 0.0

    # --- Iter 5 mechanism ---
    hi, lo = oos_ev[oos_ev["p_rev"] >= 0.60], oos_ev[oos_ev["p_rev"] < 0.40]
    mech_rows = []
    for f in MODEL_FEATS:
        mech_rows.append({
            "feature": f,
            "high_p_median": float(hi[f].median()) if len(hi) else float("nan"),
            "low_p_median": float(lo[f].median()) if len(lo) else float("nan"),
            "diff": float(hi[f].median() - lo[f].median()) if len(hi) and len(lo) else 0.0,
        })
    mdf = pd.DataFrame(mech_rows).sort_values("diff", key=np.abs, ascending=False)
    mdf.to_csv(OUT / "reversal_path_attribution.csv", index=False)
    topf = mdf.iloc[0]["feature"] if len(mdf) else ""
    if topf in GROUPS["geometry"]:
        mechanism = "E. simple distance-from-entry geometry"
    elif topf in GROUPS["momentum"]:
        mechanism = "A. momentum exhaustion"
    elif topf in GROUPS["trend"]:
        mechanism = "C. trend exhaustion"
    elif topf in GROUPS["vol"]:
        mechanism = "D. volatility shock"
    elif topf in GROUPS["session"]:
        mechanism = "F. session"
    else:
        mechanism = "G. no interpretable mechanism" if abs(mdf.iloc[0]["diff"]) < 0.05 else "F. other"

    # --- Iter 6 leave-group-out ---
    rob = []
    for gname, drop in list(GROUPS.items()) + [("none", [])]:
        cols = [c for c in MODEL_FEATS if c not in drop]
        if len(cols) < 3:
            continue
        pp = np.full(len(ev), np.nan)
        for te_y in TRUE_OOS:
            tr, te = years < te_y, years == te_y
            if tr.sum() < 80 or te.sum() < 30:
                continue
            pp[te] = _fit(best, X.loc[tr, cols], y[tr], X.loc[te, cols])
        m = np.isfinite(pp) & ev["year"].isin(TRUE_OOS)
        yt, pt = y[m].to_numpy(), pp[m]
        rec_hi = float(ev.loc[m & (pp >= 0.60), "max_recovery_from_extreme_R"].mean()) if (m & (pp >= 0.60)).any() else 0
        rec_lo = float(ev.loc[m & (pp < 0.40), "max_recovery_from_extreme_R"].mean()) if (m & (pp < 0.40)).any() else 0
        rob.append({
            "dropped": gname, "n_feats": len(cols), "oos_roc": _roc(yt, pt), "oos_pr": _pr(yt, pt),
            "top_minus_bot_recovery": rec_hi - rec_lo,
        })
    rdf = pd.DataFrame(rob)
    rdf.to_csv(OUT / "reversal_mechanism_robustness.csv", index=False)
    base_roc = float(rdf.loc[rdf["dropped"] == "none", "oos_roc"].iloc[0]) if (rdf["dropped"] == "none").any() else oos_auc
    geom_roc = float(rdf.loc[rdf["dropped"] == "geometry", "oos_roc"].iloc[0]) if (rdf["dropped"] == "geometry").any() else base_roc
    mechanical = (base_roc - geom_roc) > 0.05 and geom_roc < 0.58

    # --- Iter 7 yearly ---
    yrep = []
    edges_yr = []
    for r in oos_rows:
        y = r["year"]
        sub = oos_ev[oos_ev["year"] == y]
        hi_r = float(sub.loc[sub["p_rev"] >= 0.60, "reversal"].mean()) if (sub["p_rev"] >= 0.60).any() else float("nan")
        lo_r = float(sub.loc[sub["p_rev"] < 0.40, "reversal"].mean()) if (sub["p_rev"] < 0.40).any() else float("nan")
        hi_rec = float(sub.loc[sub["p_rev"] >= 0.60, "max_recovery_from_extreme_R"].mean()) if (sub["p_rev"] >= 0.60).any() else float("nan")
        lo_rec = float(sub.loc[sub["p_rev"] < 0.40, "max_recovery_from_extreme_R"].mean()) if (sub["p_rev"] < 0.40).any() else float("nan")
        sep = (hi_rec - lo_rec) if np.isfinite(hi_rec) and np.isfinite(lo_rec) else 0.0
        edges_yr.append(sep)
        yrep.append({
            **r, "top_rate": hi_r, "bot_rate": lo_r, "top_minus_bot_recovery": sep,
            "mean_recovery": float(sub["max_recovery_from_extreme_R"].mean()),
            "median_recovery": float(sub["max_recovery_from_extreme_R"].median()),
        })
    ydf = pd.DataFrame(yrep)
    ydf.to_csv(OUT / "reversal_yearly_oos.csv", index=False)
    seps = np.array(edges_yr, dtype=float)
    n_pos_sep = int((seps > 0).sum())
    one_year = bool(len(seps) and seps.max() > 0 and seps.max() / max(seps[seps > 0].sum(), 1e-9) > 0.50)
    sign_flip = n_pos_sep <= 3
    y2026_only = n_pos_sep == 1 and seps[-1] > 0

    # --- Iter 8 placebo ---
    rng = np.random.default_rng(42)
    actual_sep = float(np.nanmean(seps))
    pvals = []
    recov = oos_ev["max_recovery_from_extreme_R"].to_numpy()
    pr = oos_ev["p_rev"].to_numpy()
    yr = oos_ev["year"].to_numpy()
    ylab = oos_ev["reversal"].to_numpy()
    actual_auc = oos_auc
    for _ in range(1000):
        ysh = ylab.copy()
        for y in TRUE_OOS:
            m = yr == y
            if m.sum() < 5:
                continue
            ysh[m] = rng.permutation(ysh[m])
        pvals.append(_roc(ysh, pr))
    pva = np.asarray(pvals)
    p_emp = float((pva >= actual_auc).mean())
    pd.DataFrame([{"actual_auc": actual_auc, "placebo_mean_auc": float(pva.mean()),
                   "placebo_std": float(pva.std()), "empirical_p": p_emp, "actual_top_bot_recovery": actual_sep,
                   "n_rep": 1000}]).to_csv(OUT / "reversal_placebo.csv", index=False)

    # --- verdict ---
    if leak.get("suspected") and oos_auc > 0.95:
        verdict = "LEAKAGE_FAIL"
    elif n_pos_sep < 4 or sign_flip or one_year or y2026_only:
        verdict = "REGIME_SPECIFIC"
    elif oos_auc < 0.55 and abs(top_bot_rec) < 0.10:
        verdict = "NO_REVERSAL_SIGNAL"
    elif mechanical or (n_stable and geom_roc < 0.55 and base_roc < 0.65):
        verdict = "MECHANICAL_REVERSAL_SIGNAL"
    elif (n_pos_sep >= 4 and not one_year and p_emp <= 0.05 and not leak["leakage"]
          and mono and abs(top_bot_rec) >= 0.15 and not mechanical):
        verdict = "CANDIDATE_FOR_ACTION_RESEARCH"
    elif n_stable and oos_auc >= 0.55:
        verdict = "MECHANICAL_REVERSAL_SIGNAL" if mechanical else "NO_REVERSAL_SIGNAL"
    else:
        verdict = "NO_REVERSAL_SIGNAL"

    rec = {
        "NO_REVERSAL_SIGNAL": "Stop reversal research. No predictable PIT reversal state.",
        "LEAKAGE_FAIL": "Stop. Fix leakage before any reversal work.",
        "REGIME_SPECIFIC": "Stop reversal action. Signal is not OOS-stable.",
        "MECHANICAL_REVERSAL_SIGNAL": "Do not design close+reverse. Signal is geometry, not a new state. Stop or document only.",
        "CANDIDATE_FOR_ACTION_RESEARCH": "Sprint 46 may study reversal *action* design. Not production.",
    }[verdict]

    lines = [
        "# Sprint 45 — Reversal Signal Discovery",
        "",
        "Research only. No close+reverse. Production unchanged.",
        "",
        f"Primary label frozen: first MAE ≥ **{ADV}R**, then recovery ≥ **{REC}R** from that extreme.",
        f"Events {n_ev}/{n_tr} trades. Positives {n_pos} ({rate:.1%}).",
        "",
        "## Iter 1 — Base rate",
        "",
        _md(sens.to_dict("records"), ["adverse", "recovery", "n_events", "n_pos", "rate"] + [f"rate_{y}" for y in TRUE_OOS],
            {"adverse": 2, "recovery": 2, "n_events": 0, "n_pos": 0, "rate": 3,
             **{f"rate_{y}": 3 for y in TRUE_OOS}}),
        "",
        _md([d for d in diag if d["group"] in ("all", "positive", "negative")],
            ["group", "n", "mean_mae", "mean_mfe", "mean_current_R", "mean_recovery", "mean_p0_R"],
            {"n": 0, "mean_mae": 3, "mean_mfe": 3, "mean_current_R": 3, "mean_recovery": 3, "mean_p0_R": 3}),
        "",
        "## Iter 2 — Univariate",
        "",
        f"Stable PIT features: **{n_stable}**.",
        "",
        _md(sdf[sdf["year"] == "oos_summary"].sort_values("spearman", key=np.abs, ascending=False).head(8).to_dict("records"),
            ["feature", "spearman", "n_years_abs_ge10_same_sign", "stable"],
            {"spearman": 3, "n_years_abs_ge10_same_sign": 0}),
        "",
        "## Iter 3 — Model",
        "",
        f"Selected on val AUC: **{best}**. Pooled OOS ROC={oos_auc:.3f} PR={oos_pr:.3f}. LR mean OOS AUC={lr_auc:.3f} LGBM={lgbm_auc:.3f}.",
        "",
        _md(oos_rows, ["year", "n", "rate", "roc", "pr", "brier", "ece"],
            {"year": 0, "n": 0, "rate": 3, "roc": 3, "pr": 3, "brier": 3, "ece": 3}),
        "",
        "## Iter 4 — Probability buckets",
        "",
        _md(brows, ["bucket", "n", "reversal_rate", "mean_p", "mean_recovery", "median_recovery", "mean_p0_R"],
            {"n": 0, "reversal_rate": 3, "mean_p": 3, "mean_recovery": 3, "median_recovery": 3, "mean_p0_R": 3}),
        "",
        f"Monotonic recovery vs P: **{mono}**. Top−bottom recovery: {top_bot_rec:+.3f}R. Top−bottom rate: {top_bot_rate:+.3f}.",
        "",
        "## Iter 5 — Mechanism",
        "",
        f"**{mechanism}** (largest high-P vs low-P gap: `{topf}`).",
        "",
        _md(mdf.head(8).to_dict("records"), ["feature", "high_p_median", "low_p_median", "diff"],
            {"high_p_median": 3, "low_p_median": 3, "diff": 3}),
        "",
        "## Iter 6 — Leave-group-out",
        "",
        _md(rob, ["dropped", "n_feats", "oos_roc", "oos_pr", "top_minus_bot_recovery"],
            {"n_feats": 0, "oos_roc": 3, "oos_pr": 3, "top_minus_bot_recovery": 3}),
        "",
        f"Geometry-dependent: **{mechanical}** (full ROC {base_roc:.3f} vs drop-geometry {geom_roc:.3f}).",
        "",
        "## Iter 7 — Yearly OOS",
        "",
        _md(yrep, ["year", "n", "rate", "roc", "pr", "top_rate", "bot_rate", "top_minus_bot_recovery"],
            {"year": 0, "n": 0, "rate": 3, "roc": 3, "pr": 3, "top_rate": 3, "bot_rate": 3, "top_minus_bot_recovery": 3}),
        "",
        f"Years with positive top−bot recovery: {n_pos_sep}/5. One-year dominance: {one_year}.",
        "",
        "## Iter 8 — Placebo",
        "",
        f"Actual OOS AUC={actual_auc:.3f}. Placebo AUC mean={pva.mean():.3f} (sd {pva.std():.3f}). Empirical p={p_emp:.4f}.",
        "",
        "## Sprint 45 Verdict",
        "",
        "Question:",
        '"Is there a predictable reversal state after adverse movement?"',
        "",
        f"Verdict: **{verdict}**",
        "",
        "### Evidence",
        "",
        f"- OOS years: 2022–2026",
        f"- Events: {n_ev} (OOS {len(oos_ev)})",
        f"- Reversal rate: {rate:.1%}",
        f"- Best model: {best}",
        f"- OOS ROC: {oos_auc:.3f}",
        f"- OOS PR: {oos_pr:.3f}",
        f"- Top-vs-bottom separation: {top_bot_rec:+.3f}R recovery, {top_bot_rate:+.3f} rate",
        f"- Mechanism: {mechanism}",
        f"- Year stability: {n_pos_sep}/5 positive; one-year dominance={one_year}",
        f"- Placebo p: {p_emp:.4f}",
        f"- Leakage: {leak['leakage']}",
        "",
        "### Production",
        "",
        "Production entry: UNCHANGED",
        "",
        "Production exit: P0 a0.25/d0.08",
        "",
        "Conditional exit: NOT USED",
        "",
        "Reversal action: NOT USED",
        "",
        "### Recommendation",
        "",
        rec,
        "",
    ]
    (OUT / "sprint45_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_stop(verdict, {
        "n_events": n_ev, "n_pos": n_pos, "rate": rate, "best_model": best,
        "oos_roc": oos_auc, "oos_pr": oos_pr, "top_bot_recovery": top_bot_rec,
        "mechanism": mechanism, "placebo_p": p_emp, "n_stable_features": n_stable,
        "mechanical": mechanical, "yearly_pos_sep": n_pos_sep,
    })
    print(f"VERDICT={verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
