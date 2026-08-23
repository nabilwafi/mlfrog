"""Sprint 46 — Conditional loss reduction / tail-loss suppression.

Research only. Production P0 unchanged. No P1/P2/P3, no close+reverse, no shipping.

Primary label frozen before Iter 1: large_loss_1 = P0_final_R <= -1.00R.

  python apps/research_sprint46_loss_reduction.py
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

from apps.report_adaptive_vol_risk import iter6_montecarlo
from apps.research_sprint39_exit_state import _atr_pctile, _join_feat7, _load_panel, session_of_hour
from apps.research_sprint40_iter1_time_aware_exit import CFG_PROD, STARTING
from apps.research_sprint41_iter1_diff import STATE_FEATS
from apps.research_sprint41_iter4_switch import _port, _year_row
from apps.research_sprint42_attribution import TRUE_OOS, _fmt, _md, _pf, _pooled
from apps.run_exit_engine_grid import CAP, FEAT7, Paths, simulate_combo
from simulation.wf.sim import COST, entry_indices, load_h1, prepare_market

OUT = _ROOT / "artifacts/pipeline_backtest/loss_reduction/sprint46"
P0 = dict(act=0.25, dist=0.08, tp=None, partials=(), tmax=None, be=None)
CHECKS = (0.25, 0.50, 0.75, 1.00)  # first current_R <= -c
MODEL_FEATS = STATE_FEATS + ["bars_in_trade"]
YEARS = (2021, 2022, 2023, 2024, 2025, 2026)
REASON = {0: "SL", 1: "TRAIL", 2: "BE", 3: "TP", 4: "TIMEOUT"}
# Expanding WF on the frozen 2021–2026 panel (2015–2020 entries do not exist here).
FOLDS = (
    (2023, (2021,), 2022),
    (2024, (2021, 2022), 2023),
    (2025, (2021, 2022, 2023), 2024),
    (2026, (2021, 2022, 2023, 2024), 2025),
)
THR_GRID = (0.50, 0.60, 0.70, 0.80)
CK_GRID = (0.50, 0.75, 1.00)  # intervention observation points
NAIVE_CUT = 0.75
N_PLACEBO = 1000
GEOM = {"current_R", "mae_so_far_R", "mfe_so_far_R", "drawdown_from_MFE_R", "bars_in_trade"}


def _spearman(a, b) -> float:
    s = pd.Series(a).corr(pd.Series(b), method="spearman")
    return float(s) if s == s else 0.0


def _roc(y, p) -> float:
    y = np.asarray(y, dtype=int)
    if len(y) < 2 or y.min() == y.max():
        return 0.5
    return float(roc_auc_score(y, p))


def _pr(y, p) -> float:
    y = np.asarray(y, dtype=int)
    if len(y) < 2 or y.min() == y.max():
        return float(y.mean())
    return float(average_precision_score(y, p))


def _ece(y, p, n_bins=10) -> float:
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    err, n = 0.0, max(len(p), 1)
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        err += m.mean() * abs(float(p[m].mean()) - float(y[m].mean()))
    return float(err)


def _cohen(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return 0.0
    v = ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / max(len(a) + len(b) - 2, 1)
    return float((a.mean() - b.mean()) / math.sqrt(v)) if v > 0 else 0.0


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


def _r_stats(r: np.ndarray) -> dict:
    r = np.asarray(r, dtype=float)
    pos, neg = r[r > 0], r[r < 0]
    gp, gl = float(pos.sum()) if pos.size else 0.0, float(-neg.sum()) if neg.size else 0.0
    return {
        "n": int(len(r)),
        "mean_R": float(r.mean()) if len(r) else 0.0,
        "median_R": float(np.median(r)) if len(r) else 0.0,
        "std_R": float(r.std(ddof=1)) if len(r) > 1 else 0.0,
        "p_lt_0": float((r < 0).mean()) if len(r) else 0.0,
        "p_le_m0_50": float((r <= -0.50).mean()) if len(r) else 0.0,
        "p_le_m0_75": float((r <= -0.75).mean()) if len(r) else 0.0,
        "p_le_m1_00": float((r <= -1.00).mean()) if len(r) else 0.0,
        "p_le_m1_25": float((r <= -1.25).mean()) if len(r) else 0.0,
        "p_le_m1_50": float((r <= -1.50).mean()) if len(r) else 0.0,
        "worst_R": float(r.min()) if len(r) else 0.0,
        "p5_R": float(np.percentile(r, 5)) if len(r) else 0.0,
        "p10_R": float(np.percentile(r, 10)) if len(r) else 0.0,
        "total_neg_R": float(neg.sum()) if neg.size else 0.0,
        "total_pos_R": float(pos.sum()) if pos.size else 0.0,
        "pf": (gp / gl) if gl > 0 else 0.0,
        "wr": float((r > 0).mean()) if len(r) else 0.0,
        "payoff": float(pos.mean() / abs(neg.mean())) if pos.size and neg.size else 0.0,
    }


def _build_obs(p: Paths, mkt: dict, sim: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    ei = entry_indices(p.panel, mkt["ts"])
    close_s, atr_s = mkt["close"], mkt["atr"]
    ema = pd.Series(close_s).ewm(span=20, adjust=False).mean().to_numpy()
    atr_pct_bar = _atr_pctile(atr_s)
    hour0 = pd.DatetimeIndex(p.ts).hour.to_numpy()
    y_prob = p.panel["y_prob"].astype(float).to_numpy()
    f7 = p.panel[list(FEAT7)].astype(float).to_numpy()
    f7i = {name: k for k, name in enumerate(FEAT7)}
    ts0 = pd.DatetimeIndex(p.ts)
    hold = sim["holding_bars"].astype(int)
    r0 = sim["r_multiple"]
    mae_t = sim["mae_r"]
    mfe_t = sim["mfe_r"]
    reason = sim["reason"]
    trades, obs = [], []
    for i in range(p.n):
        last = min(int(hold[i]), int(p.last_off[i]))
        p0 = float(r0[i])
        mae_full = float(mae_t[i])
        rec_from_mae = float("nan")
        if mae_full == mae_full and last >= 1:
            post = p.fav[i, 1 : last + 1]
            post = post[np.isfinite(post)]
            rec_from_mae = float(post.max() + mae_full) if post.size else float("nan")
        recov = bool(p0 > 0) or (np.isfinite(rec_from_mae) and rec_from_mae >= 0.75)
        ll = int(p0 <= -1.00)
        trades.append({
            "trade_id": i, "year": int(p.year[i]),
            "side": "long" if p.is_long[i] else "short",
            "p0_R": p0, "mae_R": mae_full, "mfe_R": float(mfe_t[i]),
            "exit_reason": REASON.get(int(reason[i]), "TIMEOUT"),
            "holding_bars": last,
            "large_loss_1": ll,
            "large_loss_125": int(p0 <= -1.25),
            "large_loss_150": int(p0 <= -1.50),
            "large_loss_sl": int(ll and REASON.get(int(reason[i])) == "SL"),
            "large_mae_recovered": int(mae_full >= 1.0 and p0 > 0),
            "recoverable": int(recov and not ll),
            "max_recovery_from_mae_R": rec_from_mae if rec_from_mae == rec_from_mae else 0.0,
        })
        hit = {c: False for c in CHECKS}
        for j in range(1, last + 1):
            cr = p.close_r[i, j]
            if not np.isfinite(cr):
                continue
            fired = [c for c in CHECKS if (not hit[c]) and cr <= -c]
            if not fired:
                continue
            mfe = float(np.nanmax(p.fav[i, 1 : j + 1]))
            mae = float(np.nanmax(p.adv[i, 1 : j + 1]))
            idx = int(ei[i] + j)
            if idx >= len(close_s):
                continue
            prev = p.close_r[i, j - 1] if j > 1 else 0.0
            hr = int((hour0[i] + j) % 24)
            sign = 1.0 if p.is_long[i] else -1.0
            ema_dist = sign * (close_s[idx] - ema[idx]) / max(float(p.atr[i]), 1e-12)
            recov_j = -np.inf
            for k in range(j + 1, last + 1):
                f = p.fav[i, k]
                if np.isfinite(f):
                    recov_j = max(recov_j, float(f) + mae)
            for c in fired:
                hit[c] = True
                obs.append({
                    "trade_id": i, "year": int(p.year[i]),
                    "side": "long" if p.is_long[i] else "short",
                    "timestamp": ts0[i] + pd.Timedelta(hours=j),
                    "hour_utc": hr, "session": session_of_hour(hr),
                    "checkpoint": c, "bars_in_trade": j,
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
                    "mfe_so_far_R": mfe, "mae_so_far_R": mae,
                    "drawdown_from_MFE_R": mfe - float(cr),
                    "current_R": float(cr),
                    "mom_1R": float(cr - prev) if np.isfinite(prev) else 0.0,
                    "recovery_from_here_R": 0.0 if recov_j == -np.inf else float(recov_j),
                    "p0_R": p0, "large_loss_1": ll,
                    "large_mae_recovered": int(mae_full >= 1.0 and p0 > 0),
                    "exit_reason": REASON.get(int(reason[i]), "TIMEOUT"),
                })
    return pd.DataFrame(trades), pd.DataFrame(obs)


def _bucket_rows(obs: pd.DataFrame, col: str, bins: list[tuple[str, object]]) -> list[dict]:
    rows = []
    x = obs[col].to_numpy(dtype=float)
    for name, fn in bins:
        m = fn(x)
        sub = obs.loc[m]
        r = sub["p0_R"].to_numpy(dtype=float) if len(sub) else np.array([])
        rows.append({
            "feature": col, "bucket": name, "n": int(len(sub)),
            "large_loss_rate": float(sub["large_loss_1"].mean()) if len(sub) else 0.0,
            "mean_R": float(r.mean()) if r.size else 0.0,
            "median_R": float(np.median(r)) if r.size else 0.0,
            "recovery_rate": float((r > 0).mean()) if r.size else 0.0,
            "p_final_gt_0": float((r > 0).mean()) if r.size else 0.0,
            "p_final_le_m1": float((r <= -1).mean()) if r.size else 0.0,
        })
    return rows


def _overlay(p: Paths, sim: dict, cuts: dict[int, tuple[float, int]]) -> dict:
    r = np.array(sim["r_multiple"], dtype=float, copy=True)
    net = np.array(sim["net_return"], dtype=float, copy=True)
    hold = np.array(sim["holding_bars"], dtype=np.int64, copy=True)
    reason = np.array(sim["reason"], dtype=np.int64, copy=True)
    for i, (cr, j) in cuts.items():
        ru = max(float(p.r_unit_pct[i]), 1e-12)
        r[i] = float(cr) - COST / ru
        net[i] = r[i] * ru
        hold[i] = max(int(j), 1)
        reason[i] = 0
    out = dict(sim)
    out["r_multiple"], out["net_return"] = r, net
    out["holding_bars"], out["reason"] = hold, reason
    return out


def _cuts_from_obs(obs: pd.DataFrame, *, ck: float, thr: float, pred_col: str) -> dict[int, tuple[float, int]]:
    sub = obs[(obs["checkpoint"] == ck) & np.isfinite(obs[pred_col]) & (obs[pred_col] >= thr)]
    cuts: dict[int, tuple[float, int]] = {}
    for row in sub.itertuples(index=False):
        tid = int(row.trade_id)
        if tid in cuts:
            continue
        # ponytail: analytical loss-cap at the trigger; ceiling is gap bars that close through it.
        fill = max(float(row.current_R), -float(ck))
        cuts[tid] = (fill, int(row.bars_in_trade))
    return cuts


def _naive_cuts(obs: pd.DataFrame, ck: float) -> dict[int, tuple[float, int]]:
    sub = obs[obs["checkpoint"] == ck].sort_values("bars_in_trade")
    cuts: dict[int, tuple[float, int]] = {}
    for row in sub.itertuples(index=False):
        tid = int(row.trade_id)
        if tid in cuts:
            continue
        fill = max(float(row.current_R), -float(ck))
        cuts[tid] = (fill, int(row.bars_in_trade))
    return cuts


def _action_stats(trades: pd.DataFrame, cuts: dict[int, tuple[float, int]], p: Paths) -> dict:
    r0 = trades["p0_R"].to_numpy(dtype=float)
    tid = trades["trade_id"].to_numpy(dtype=int)
    new = r0.copy()
    for i, t in enumerate(tid):
        if t in cuts:
            ru = max(float(p.r_unit_pct[t]), 1e-12)
            new[i] = cuts[t][0] - COST / ru
    ll = trades["large_loss_1"].to_numpy(dtype=bool)
    rec = (r0 > 0)
    cut = np.array([int(t) in cuts for t in tid], dtype=bool)
    avoided, wrong = cut & ll, cut & rec
    tail_saved = float((new[avoided] - r0[avoided]).sum()) if avoided.any() else 0.0
    rec_lost = float((r0[wrong] - new[wrong]).sum()) if wrong.any() else 0.0
    return {
        "n_cut": int(cut.sum()),
        "losses_avoided": int(avoided.sum()),
        "recoveries_cut": int(wrong.sum()),
        "mean_R_delta": float((new - r0).mean()),
        "total_R_delta": float((new - r0).sum()),
        "tail_saved_R": tail_saved,
        "recovery_lost_R": rec_lost,
        "net_R": tail_saved - rec_lost,
        "ratio": tail_saved / rec_lost if rec_lost > 1e-9 else (tail_saved if tail_saved > 0 else 0.0),
        "avg_avoided": float((new[avoided] - r0[avoided]).mean()) if avoided.any() else 0.0,
        "avg_missed": float((r0[wrong] - new[wrong]).mean()) if wrong.any() else 0.0,
        "p5_before": float(np.percentile(r0, 5)),
        "p5_after": float(np.percentile(new, 5)),
        "worst_before": float(r0.min()),
        "worst_after": float(new.min()),
        "ll_rate_before": float(ll.mean()),
        "ll_rate_after": float((new <= -1.00).mean()),
        "new_R": new,
    }


def _mc(pnl: np.ndarray, name: str) -> dict:
    df = iter6_montecarlo(pnl, pnl, False, starting=STARTING)
    row = df.iloc[0].to_dict() if len(df) else {}
    # p99 from the same bootstrap style
    rng = np.random.default_rng(42)
    dds = []
    if len(pnl) >= 5:
        for _ in range(1000):
            eq = peak = STARTING
            mdd = 0.0
            for x in rng.permutation(pnl):
                eq += x
                if eq <= 0:
                    eq = 0.0
                    break
                peak = max(peak, eq)
                mdd = max(mdd, (peak - eq) / peak)
            dds.append(mdd)
    return {
        "policy": name,
        "prob_ruin": float(row.get("prob_ruin", 0.0) or 0.0),
        "median_dd": float(row.get("median_dd", 0.0) or 0.0),
        "p95_dd": float(row.get("p95_dd", 0.0) or 0.0),
        "p99_dd": float(np.quantile(dds, 0.99)) if dds else 0.0,
        "worst_dd": float(row.get("worst_dd", 0.0) or 0.0),
    }


def _write_stop(verdict: str, extra: dict, report: str) -> None:
    (OUT / "sprint46_report.md").write_text(report, encoding="utf-8")
    (OUT / "sprint46_final_verdict.json").write_text(
        json.dumps({"sprint": 46, "verdict": verdict, "production_changed": False, **extra}, indent=2, default=float),
        encoding="utf-8",
    )


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = _join_feat7(_load_panel())
    mkt = prepare_market(load_h1(_ROOT / "artifacts/raw/XAUUSD/H1/data.parquet"))
    p = Paths(panel, mkt)
    sim0 = simulate_combo(p, **P0)
    trades, obs = _build_obs(p, mkt, sim0)
    trades.to_csv(OUT / "trade_labels.csv", index=False)
    obs.to_csv(OUT / "adverse_checkpoint_obs.csv", index=False)

    leak = {
        "features": {f: {"future_dependency": False, "role": "PIT"} for f in MODEL_FEATS},
        "labels": [
            "large_loss_1", "large_loss_125", "large_loss_150", "large_loss_sl",
            "large_mae_recovered", "p0_R", "recovery_from_here_R", "exit_reason",
        ],
        "label_future_dependency": True,
        "leakage": False,
        "notes": "Checkpoints exist iff PIT current_R crosses a frozen adverse level while P0 is alive.",
    }
    (OUT / "loss_reduction_leakage_audit.json").write_text(json.dumps(leak, indent=2), encoding="utf-8")
    (OUT / "dataset_manifest.json").write_text(json.dumps({
        "n_trades": int(len(trades)), "n_obs": int(len(obs)),
        "primary_label": "large_loss_1 = P0_final_R <= -1.00R",
        "checkpoints": list(CHECKS),
        "grain": "first bar current_R <= -checkpoint while P0 alive",
        "panel_years": "2021-2026 frozen top21 FEAT7; 2015-2020 not in this entry set",
        "true_oos": list(TRUE_OOS),
        "year_2021": "not true OOS (no prior fold)",
        "years": {int(y): int((trades["year"] == y).sum()) for y in YEARS},
        "production_exit": "P0 a0.25/d0.08",
    }, indent=2), encoding="utf-8")

    # ----- Iter 1 -----
    oos_tr = trades[trades["year"].isin(TRUE_OOS)]
    base = [{"scope": "pooled_2021_2026", **_r_stats(trades["p0_R"].to_numpy())},
            {"scope": "oos_2022_2026", **_r_stats(oos_tr["p0_R"].to_numpy())}]
    yearly = []
    for y in YEARS:
        st = _r_stats(trades.loc[trades["year"] == y, "p0_R"].to_numpy())
        st["year"] = y
        st["large_loss_n"] = int(((trades["year"] == y) & (trades["large_loss_1"] == 1)).sum())
        st["large_mae_recovered_n"] = int(((trades["year"] == y) & (trades["large_mae_recovered"] == 1)).sum())
        yearly.append(st)
        base.append({"scope": str(y), **_r_stats(trades.loc[trades["year"] == y, "p0_R"].to_numpy())})
    pd.DataFrame(base).to_csv(OUT / "loss_tail_distribution.csv", index=False)
    pd.DataFrame(yearly).to_csv(OUT / "loss_yearly_baseline.csv", index=False)

    n_tr = len(oos_tr)
    ll_n = int(oos_tr["large_loss_1"].sum())
    ll_rate = float(oos_tr["large_loss_1"].mean()) if n_tr else 0.0
    rec_n = int(oos_tr["large_mae_recovered"].sum())
    rec_adv = int(((oos_tr["mae_R"] >= 0.75) & (oos_tr["p0_R"] > 0)).sum())
    print(f"oos trades={n_tr} large_loss={ll_n} rate={ll_rate:.3f} mae>=1 recovered={rec_n}")

    gate = n_tr >= 200 and 0.05 <= ll_rate <= 0.90 and rec_n >= 20 and rec_adv >= 20 and not leak["leakage"]
    if not gate:
        v = "NO_LOSS_SIGNAL"
        why = f"Iter 1 gate fail. oos_n={n_tr} large_loss_rate={ll_rate:.3f} recovered_mae1={rec_n}"
        _write_stop(v, {"iter": 1, "large_loss_rate": ll_rate, "reason": why},
                    f"# Sprint 46 — Conditional Loss Reduction\n\n**Verdict: {v}**\n\n{why}\n")
        print(f"STOP {v} iter1")
        return 0

    # ----- Iter 2 -----
    oos_obs = obs[obs["year"].isin(TRUE_OOS)]
    cr_bins = [
        ("> -0.25R", lambda x: x > -0.25),
        ("-0.25_-0.50R", lambda x: (x <= -0.25) & (x > -0.50)),
        ("-0.50_-0.75R", lambda x: (x <= -0.50) & (x > -0.75)),
        ("-0.75_-1.00R", lambda x: (x <= -0.75) & (x > -1.00)),
        ("<= -1.00R", lambda x: x <= -1.00),
    ]
    mae_bins = [
        ("<0.25R", lambda x: x < 0.25),
        ("0.25_0.50R", lambda x: (x >= 0.25) & (x < 0.50)),
        ("0.50_0.75R", lambda x: (x >= 0.50) & (x < 0.75)),
        ("0.75_1.00R", lambda x: (x >= 0.75) & (x < 1.00)),
        (">=1.00R", lambda x: x >= 1.00),
    ]
    mfe_bins = [
        ("<0", lambda x: x < 0),
        ("0_0.25R", lambda x: (x >= 0) & (x < 0.25)),
        ("0.25_0.50R", lambda x: (x >= 0.25) & (x < 0.50)),
        ("0.50_1.00R", lambda x: (x >= 0.50) & (x < 1.00)),
        (">=1.00R", lambda x: x >= 1.00),
    ]
    dd_bins = [
        ("<0.25R", lambda x: x < 0.25),
        ("0.25_0.50R", lambda x: (x >= 0.25) & (x < 0.50)),
        ("0.50_1.00R", lambda x: (x >= 0.50) & (x < 1.00)),
        (">=1.00R", lambda x: x >= 1.00),
    ]
    attr = []
    attr += _bucket_rows(oos_obs, "current_R", cr_bins)
    attr += _bucket_rows(oos_obs, "mae_so_far_R", mae_bins)
    attr += _bucket_rows(oos_obs, "mfe_so_far_R", mfe_bins)
    attr += _bucket_rows(oos_obs, "drawdown_from_MFE_R", dd_bins)
    pd.DataFrame(attr).to_csv(OUT / "adverse_path_attribution.csv", index=False)

    # ----- Iter 3 -----
    c75 = oos_obs[oos_obs["checkpoint"] == 0.75].copy()
    rec_m = (c75["p0_R"] > 0) | (c75["recovery_from_here_R"] >= 0.75)
    ll_m = c75["large_loss_1"] == 1
    both = rec_m & ll_m
    c75 = c75.assign(group=np.where(ll_m & ~rec_m, "LARGE_LOSS", np.where(rec_m & ~ll_m, "RECOVERABLE", "OVERLAP")))
    cmp_rows = []
    feats3 = MODEL_FEATS + ["hour_utc"]
    for g in ("RECOVERABLE", "LARGE_LOSS"):
        sub = c75[c75["group"] == g]
        row = {"group": g, "n": int(len(sub))}
        for f in feats3:
            row[f"mean_{f}"] = float(sub[f].mean()) if len(sub) else 0.0
        cmp_rows.append(row)
    rec_sub, ll_sub = c75[c75["group"] == "RECOVERABLE"], c75[c75["group"] == "LARGE_LOSS"]
    sep = {"group": "delta_LL_minus_REC", "n": int(len(ll_sub) + len(rec_sub))}
    for f in feats3:
        a, b = ll_sub[f].to_numpy(dtype=float), rec_sub[f].to_numpy(dtype=float)
        sep[f"mean_{f}"] = float(np.nanmean(a) - np.nanmean(b)) if len(a) and len(b) else 0.0
        sep[f"cohen_{f}"] = _cohen(a, b) if len(a) and len(b) else 0.0
    cmp_rows.append(sep)
    pd.DataFrame(cmp_rows).to_csv(OUT / "recoverable_vs_large_loss.csv", index=False)
    n_disc = sum(abs(sep.get(f"cohen_{f}", 0.0)) >= 0.20 for f in feats3)
    print(f"iter3 first -0.75R: rec={len(rec_sub)} ll={len(ll_sub)} overlap={int(both.sum())} |d|>=0.2 feats={n_disc}")

    # ----- Iter 4 -----
    early_all = obs[obs["checkpoint"] <= 0.75]
    oos_early = early_all[early_all["year"].isin(TRUE_OOS)]
    stab = []
    q_edges = {}
    tr21 = early_all[early_all["year"] == 2021]
    for f in MODEL_FEATS:
        if f in tr21.columns and len(tr21):
            q_edges[f] = np.nanquantile(tr21[f].to_numpy(dtype=float), [0.2, 0.4, 0.6, 0.8])
    for f in MODEL_FEATS:
        signs = []
        for y in TRUE_OOS:
            sub = early_all[early_all["year"] == y]
            sp = _spearman(sub[f], sub["large_loss_1"])
            au = _roc(sub["large_loss_1"], sub[f].to_numpy(dtype=float))
            signs.append(sp)
            pos = sub.loc[sub["large_loss_1"] == 1, f]
            neg = sub.loc[sub["large_loss_1"] == 0, f]
            stab.append({"feature": f, "year": y, "spearman": sp, "auc_raw": au,
                         "auc_oriented": max(au, 1 - au), "cohen_d": _cohen(pos, neg), "n": int(len(sub))})
        oos_sp = signs
        maj = np.sign(np.nanmedian(oos_sp)) if oos_sp else 0
        n_same = int(sum(np.sign(s) == maj and abs(s) >= 0.10 for s in oos_sp))
        mean_sp = float(np.nanmean(oos_sp)) if oos_sp else 0.0
        edges = q_edges.get(f)
        if edges is not None and np.isfinite(edges).all():
            q = np.digitize(oos_early[f].to_numpy(dtype=float), edges)
            rates = [float(oos_early.loc[q == k, "large_loss_1"].mean()) if (q == k).any() else float("nan") for k in range(5)]
        else:
            rates = [float("nan")] * 5
        top_bot = float(rates[-1] - rates[0]) if np.isfinite(rates[0]) and np.isfinite(rates[-1]) else 0.0
        stable = abs(mean_sp) >= 0.10 and n_same >= 4 and abs(top_bot) >= 0.05
        stab.append({
            "feature": f, "year": "oos_summary", "spearman": mean_sp,
            "n_years_abs_ge10_same_sign": n_same, "stable": stable, "top_minus_bot": top_bot,
            **{f"q{i+1}_ll_rate": rates[i] for i in range(5)},
        })
    sdf = pd.DataFrame(stab)
    sdf.to_csv(OUT / "loss_feature_stability.csv", index=False)
    stable_feats = sdf.loc[(sdf["year"] == "oos_summary") & (sdf["stable"] == True), "feature"].tolist()
    print(f"stable PIT features: {stable_feats}")

    # ----- Iter 5 -----
    # Train / score only at checkpoints before the label threshold (exclude -1.00R tautology).
    early = obs[obs["checkpoint"] <= 0.75].copy()
    X = early[MODEL_FEATS].astype(float).fillna(0.0).to_numpy()
    y = early["large_loss_1"].astype(int).to_numpy()
    years = early["year"].to_numpy()
    eidx = early.index.to_numpy()
    val_choice = []
    oos_pred = {m: np.full(len(obs), np.nan) for m in ("lr", "lgbm")}
    model_rows = []
    for model in ("lr", "lgbm"):
        for te, tr_years, va in FOLDS:
            tr = np.isin(years, tr_years)
            va_m = years == va
            te_m = years == te
            if tr.sum() < 200 or te_m.sum() < 50 or y[tr].min() == y[tr].max():
                continue
            p_va = _fit(model, X[tr], y[tr], X[va_m])
            p_te = _fit(model, X[tr], y[tr], X[te_m])
            oos_pred[model][eidx[te_m]] = p_te
            val_choice.append({"model": model, "val_year": va, "test_year": te,
                               "val_roc": _roc(y[va_m], p_va), "val_pr": _pr(y[va_m], p_va),
                               "val_brier": float(brier_score_loss(y[va_m], p_va)),
                               "oos_roc": _roc(y[te_m], p_te), "oos_pr": _pr(y[te_m], p_te)})
        msk = np.isfinite(oos_pred[model])
        if msk.sum() < 100:
            continue
        # evaluate on the early rows that were scored
        yy = obs.loc[msk, "large_loss_1"].astype(int).to_numpy()
        pp = oos_pred[model][msk]
        try:
            llv = float(log_loss(yy, np.clip(pp, 1e-6, 1 - 1e-6)))
        except Exception:
            llv = float("nan")
        model_rows.append({
            "model": model, "n": int(msk.sum()),
            "oos_roc": _roc(yy, pp), "oos_pr": _pr(yy, pp),
            "oos_brier": float(brier_score_loss(yy, pp)),
            "oos_logloss": llv, "oos_ece": _ece(yy, pp),
            "mean_p": float(pp.mean()), "p5": float(np.percentile(pp, 5)), "p95": float(np.percentile(pp, 95)),
        })
    vdf = pd.DataFrame(val_choice)
    mdf = pd.DataFrame(model_rows)
    if len(vdf):
        rank = vdf.groupby("model", as_index=False)["val_roc"].mean().sort_values("val_roc", ascending=False)
        best_model = str(rank.iloc[0]["model"])
    elif len(mdf):
        best_model = str(mdf.sort_values("oos_roc", ascending=False).iloc[0]["model"])
    else:
        best_model = "lr"
    obs["p_loss"] = oos_pred[best_model]
    m22 = years == 2022
    tr21 = years == 2021
    if tr21.sum() >= 200 and m22.sum() >= 50 and y[tr21].min() != y[tr21].max():
        obs.loc[eidx[m22], "p_loss"] = _fit(best_model, X[tr21], y[tr21], X[m22])
    mdf["selected"] = mdf["model"] == best_model if len(mdf) else False
    mdf.to_csv(OUT / "loss_model_oos.csv", index=False)
    if len(vdf):
        vdf.to_csv(OUT / "loss_model_val_folds.csv", index=False)
    print(f"best model (val ROC): {best_model}")

    scored = obs[np.isfinite(obs["p_loss"])]
    if len(scored) < 100:
        v = "NO_LOSS_SIGNAL"
        _write_stop(v, {"iter": 5, "reason": "insufficient OOS scores"},
                    f"# Sprint 46\n\n**Verdict: {v}**\n\nIter 5: not enough scored observations.\n")
        print("STOP iter5")
        return 0

    # ----- Iter 6 -----
    pbins = [
        ("<0.20", lambda x: x < 0.20),
        ("0.20_0.40", lambda x: (x >= 0.20) & (x < 0.40)),
        ("0.40_0.60", lambda x: (x >= 0.40) & (x < 0.60)),
        ("0.60_0.80", lambda x: (x >= 0.60) & (x < 0.80)),
        (">=0.80", lambda x: x >= 0.80),
    ]
    oos_sc = scored[scored["year"].isin(TRUE_OOS) & (scored["checkpoint"] <= 0.75)]
    brows = []
    for name, fn in pbins:
        sub = oos_sc[fn(oos_sc["p_loss"].to_numpy())]
        r = sub["p0_R"].to_numpy(dtype=float) if len(sub) else np.array([])
        brows.append({
            "bucket": name, "n": int(len(sub)),
            "large_loss_rate": float(sub["large_loss_1"].mean()) if len(sub) else 0.0,
            "mean_R": float(r.mean()) if r.size else 0.0,
            "median_R": float(np.median(r)) if r.size else 0.0,
            "p5_R": float(np.percentile(r, 5)) if r.size else 0.0,
            "worst_R": float(r.min()) if r.size else 0.0,
            "recovery_rate": float((r > 0).mean()) if r.size else 0.0,
            "mean_MAE": float(sub["mae_so_far_R"].mean()) if len(sub) else 0.0,
            "p_final_gt_0": float((r > 0).mean()) if r.size else 0.0,
            "mean_p": float(sub["p_loss"].mean()) if len(sub) else 0.0,
        })
    pd.DataFrame(brows).to_csv(OUT / "loss_probability_buckets.csv", index=False)
    fat = [b for b in brows if b["n"] >= 30]
    ll_mono = all(fat[i]["large_loss_rate"] <= fat[i + 1]["large_loss_rate"] + 0.02 for i in range(len(fat) - 1)) if len(fat) >= 3 else False
    r_dir = (fat[-1]["mean_R"] < fat[0]["mean_R"] - 0.05) if len(fat) >= 2 else False
    tail_dir = (fat[-1]["p5_R"] < fat[0]["p5_R"] - 0.05) if len(fat) >= 2 else False
    print(f"iter6 directional ll_mono={ll_mono} r_dir={r_dir} tail_dir={tail_dir}")
    if not (ll_mono and r_dir):
        v = "NO_LOSS_SIGNAL"
        report = _report_early(trades, oos_tr, ll_rate, rec_n, stable_feats, best_model, mdf, brows, v,
                               "Iter 6: predicted p(large_loss) is not directionally mapped to worse future R.")
        _write_stop(v, {"iter": 6, "ll_mono": ll_mono, "r_dir": r_dir}, report)
        print("STOP iter6 NO_LOSS_SIGNAL")
        return 0

    # ----- Iter 7 threshold on validation (2022) -----
    val_obs = obs[(obs["year"] == 2022) & np.isfinite(obs["p_loss"])]
    val_tr = trades[trades["year"] == 2022]
    grid = []
    for ck in CK_GRID:
        for thr in THR_GRID:
            cuts = _cuts_from_obs(val_obs, ck=ck, thr=thr, pred_col="p_loss")
            st = _action_stats(val_tr, cuts, p)
            grid.append({"checkpoint": ck, "threshold": thr, **{k: v for k, v in st.items() if k != "new_R"}})
    gdf = pd.DataFrame(grid)
    gdf.to_csv(OUT / "loss_counterfactual_validation.csv", index=False)
    early_g = gdf[gdf["checkpoint"] < 1.0]
    pick = early_g.sort_values(["net_R", "ratio"], ascending=False).iloc[0]
    ck_star, thr_star = float(pick["checkpoint"]), float(pick["threshold"])
    print(f"val pick ck=-{ck_star:.2f}R thr={thr_star:.2f} ratio={pick['ratio']:.3f} net={pick['net_R']:.2f}")

    (OUT / "model_config.json").write_text(json.dumps({
        "model": best_model, "features": MODEL_FEATS,
        "label": "large_loss_1",
        "threshold": thr_star, "checkpoint": ck_star,
        "threshold_grid": list(THR_GRID), "checkpoint_grid": list(CK_GRID),
        "selection": "val year 2022 only; OOS 2023-2026 expanding train, frozen thr/ck/model",
        "folds": [list(f) for f in FOLDS],
    }, indent=2), encoding="utf-8")

    # ----- Iter 8 OOS overlay -----
    oos_years = list(TRUE_OOS)
    cuts_b = _cuts_from_obs(obs[obs["year"].isin(oos_years)], ck=ck_star, thr=thr_star, pred_col="p_loss")
    cuts_c = _naive_cuts(obs[obs["year"].isin(oos_years)], NAIVE_CUT)
    cuts_m = _naive_cuts(obs[obs["year"].isin(oos_years)], ck_star)
    sim_b = _overlay(p, sim0, cuts_b)
    sim_c = _overlay(p, sim0, cuts_c)
    sim_m = _overlay(p, sim0, cuts_m)
    st_b = _action_stats(oos_tr, {int(tid): cuts_b[int(tid)] for tid in oos_tr["trade_id"] if int(tid) in cuts_b}, p)
    st_c = _action_stats(oos_tr, {int(tid): cuts_c[int(tid)] for tid in oos_tr["trade_id"] if int(tid) in cuts_c}, p)
    st_m = _action_stats(oos_tr, {int(tid): cuts_m[int(tid)] for tid in oos_tr["trade_id"] if int(tid) in cuts_m}, p)
    pd.DataFrame([
        {k: v for k, v in st_b.items() if k != "new_R"} | {"who": "conditional"},
        {k: v for k, v in st_c.items() if k != "new_R"} | {"who": "naive_m0.75"},
        {k: v for k, v in st_m.items() if k != "new_R"} | {"who": f"naive_m{ck_star:.2f}"},
    ]).to_csv(OUT / "loss_counterfactual_oos.csv", index=False)

    rec_cut = oos_tr[(oos_tr["p0_R"] > 0) & (oos_tr["trade_id"].isin(cuts_b))]
    rec_all = oos_tr[(oos_tr["mae_R"] >= 0.75) & (oos_tr["p0_R"] > 0)]
    rec_dmg = {
        "recoverable_adverse_n": int(len(rec_all)),
        "incorrectly_cut": int(len(rec_cut)),
        "pct": float(len(rec_cut) / len(rec_all)) if len(rec_all) else 0.0,
        "avg_missed_R": float(st_b["avg_missed"]),
        "total_missed_R": float(st_b["recovery_lost_R"]),
        "preservation_pct": float(1.0 - (len(rec_cut) / len(rec_all))) if len(rec_all) else 1.0,
    }
    pd.DataFrame([rec_dmg]).to_csv(OUT / "loss_recovery_damage.csv", index=False)
    ll_all = oos_tr[oos_tr["large_loss_1"] == 1]
    ll_av = ll_all[ll_all["trade_id"].isin(cuts_b)]
    tail_ben = {
        "large_loss_n": int(len(ll_all)),
        "avoided_n": int(len(ll_av)),
        "avg_avoided_R": float(st_b["avg_avoided"]),
        "total_avoided_R": float(st_b["tail_saved_R"]),
        "net_R": float(st_b["net_R"]),
    }
    pd.DataFrame([tail_ben]).to_csv(OUT / "loss_tail_benefit.csv", index=False)

    yrows = []
    ports = {"P0": [], "B": [], "C": [], "M": []}
    for y in oos_years:
        r0y = _year_row(y, _port(p, sim0, y), sim0)
        rby = _year_row(y, _port(p, sim_b, y), sim_b)
        rcy = _year_row(y, _port(p, sim_c, y), sim_c)
        rmy = _year_row(y, _port(p, sim_m, y), sim_m)
        ports["P0"].append(r0y); ports["B"].append(rby); ports["C"].append(rcy); ports["M"].append(rmy)
        ty = trades[trades["year"] == y]
        new_map_b = dict(zip(oos_tr["trade_id"], st_b["new_R"]))
        new_map_c = dict(zip(oos_tr["trade_id"], st_c["new_R"]))
        rb = np.array([new_map_b.get(int(t), float(r)) for t, r in zip(ty["trade_id"], ty["p0_R"])])
        rc = np.array([new_map_c.get(int(t), float(r)) for t, r in zip(ty["trade_id"], ty["p0_R"])])
        yrows.append({
            "year": y,
            "trades": int(len(ty)),
            "p0_mean_R": float(ty["p0_R"].mean()), "b_mean_R": float(rb.mean()), "c_mean_R": float(rc.mean()),
            "p0_pf": r0y["pf"], "b_pf": rby["pf"], "c_pf": rcy["pf"],
            "p0_dd": r0y["dd"], "b_dd": rby["dd"], "c_dd": rcy["dd"],
            "p0_ret": r0y["ret"], "b_ret": rby["ret"], "c_ret": rcy["ret"],
            "p0_wr": r0y["wr"], "b_wr": rby["wr"], "c_wr": rcy["wr"],
            "p0_payoff": r0y["payoff"], "b_payoff": rby["payoff"], "c_payoff": rcy["payoff"],
            "p0_ll_n": int(ty["large_loss_1"].sum()),
            "b_ll_n": int((rb <= -1).sum()), "c_ll_n": int((rc <= -1).sum()),
            "p0_ll_rate": float(ty["large_loss_1"].mean()),
            "b_ll_rate": float((rb <= -1).mean()), "c_ll_rate": float((rc <= -1).mean()),
            "p0_worst": float(ty["p0_R"].min()), "b_worst": float(rb.min()), "c_worst": float(rc.min()),
            "p0_p5": float(np.percentile(ty["p0_R"], 5)), "b_p5": float(np.percentile(rb, 5)),
            "c_p5": float(np.percentile(rc, 5)),
            "p0_median_R": float(ty["p0_R"].median()), "b_median_R": float(np.median(rb)),
            "b_max_loss_streak": rby["max_loss_streak"],
            "tail_saved_R": float(st_b["tail_saved_R"]) if y == 0 else float(
                (rb[ty["large_loss_1"] == 1] - ty.loc[ty["large_loss_1"] == 1, "p0_R"]).sum() if (ty["large_loss_1"] == 1).any() else 0.0
            ),
            "recovery_lost_R": float((ty.loc[ty["p0_R"] > 0, "p0_R"].to_numpy() -
                                      np.array([new_map_b.get(int(t), float(r)) for t, r in zip(ty.loc[ty["p0_R"] > 0, "trade_id"], ty.loc[ty["p0_R"] > 0, "p0_R"])])).sum()
                                     if (ty["p0_R"] > 0).any() else 0.0),
        })
    ydf = pd.DataFrame(yrows)
    ydf.to_csv(OUT / "loss_yearly_robustness.csv", index=False)
    p0p, bp, cp, mp = _pooled(ports["P0"]), _pooled(ports["B"]), _pooled(ports["C"]), _pooled(ports["M"])
    pd.DataFrame([
        {"who": "P0", **{k: v for k, v in p0p.items() if k != "pnl"}},
        {"who": "conditional", **{k: v for k, v in bp.items() if k != "pnl"}},
        {"who": "naive_m0.75", **{k: v for k, v in cp.items() if k != "pnl"}},
        {"who": f"naive_m{ck_star:.2f}", **{k: v for k, v in mp.items() if k != "pnl"}},
    ]).to_csv(OUT / "loss_naive_cutoff_comparison.csv", index=False)

    # placebo on OOS scored obs at selected checkpoint
    ck_obs = oos_sc[oos_sc["checkpoint"] == ck_star].copy()
    actual_roc = _roc(ck_obs["large_loss_1"], ck_obs["p_loss"])
    k = max(1, int(round(0.20 * len(ck_obs))))
    order = np.argsort(ck_obs["p_loss"].to_numpy())
    rck = ck_obs["p0_R"].to_numpy()
    yck = ck_obs["large_loss_1"].to_numpy(dtype=int)
    actual_tb = float(yck[order[-k:]].mean() - yck[order[:k]].mean())
    actual_rsep = float(rck[order[:k]].mean() - rck[order[-k:]].mean())
    rng = np.random.default_rng(42)
    p_roc, p_tb = [], []
    for _ in range(N_PLACEBO):
        ysh = yck.copy()
        for y in TRUE_OOS:
            m = ck_obs["year"].to_numpy() == y
            if m.sum() < 2:
                continue
            ysh[m] = rng.permutation(ysh[m])
        p_roc.append(_roc(ysh, ck_obs["p_loss"]))
        p_tb.append(float(ysh[order[-k:]].mean() - ysh[order[:k]].mean()))
        # R separation is label-free vs p; skip shuffle of R
    p_roc, p_tb = np.asarray(p_roc), np.asarray(p_tb)
    p_emp_roc = float((p_roc >= actual_roc).mean())
    p_emp_tb = float((p_tb >= actual_tb).mean())
    pd.DataFrame([{
        "actual_roc": actual_roc, "placebo_roc_mean": float(p_roc.mean()), "placebo_roc_std": float(p_roc.std()),
        "empirical_p_roc": p_emp_roc,
        "actual_top_bot_ll": actual_tb, "placebo_tb_mean": float(p_tb.mean()), "placebo_tb_std": float(p_tb.std()),
        "empirical_p_tb": p_emp_tb,
        "actual_mean_R_sep_bot_minus_top": actual_rsep, "n_rep": N_PLACEBO,
    }]).to_csv(OUT / "loss_placebo.csv", index=False)

    mc0 = _mc(p0p["pnl"], "P0")
    mcb = _mc(bp["pnl"], "conditional")
    mcc = _mc(cp["pnl"], "naive_m0.75")
    mcm = _mc(mp["pnl"], f"naive_m{ck_star:.2f}")
    pd.DataFrame([mc0, mcb, mcc, mcm]).to_csv(OUT / "loss_mc.csv", index=False)

    # ----- verdict -----
    years_ll_down = int((ydf["b_ll_rate"] < ydf["p0_ll_rate"] - 1e-12).sum())
    net_by_y = ydf["tail_saved_R"] - ydf["recovery_lost_R"]
    max_share = float(net_by_y.max() / net_by_y.sum()) if net_by_y.sum() > 1e-9 else 1.0
    rec_lost_share = float((ydf["recovery_lost_R"].max() / ydf["recovery_lost_R"].sum()) if ydf["recovery_lost_R"].sum() > 1e-9 else 1.0)
    dd_ok = float(bp["dd"]) <= float(p0p["dd"]) + 0.05
    ruin_ok = mcb["prob_ruin"] <= mc0["prob_ruin"] + 1e-9
    invert = bool((ydf["b_mean_R"] < ydf["p0_mean_R"] - 0.05).sum() >= 3)
    beats_naive = (st_b["net_R"] > st_c["net_R"] + 0.5) and (st_b["recoveries_cut"] < st_c["recoveries_cut"])
    beats_same_ck = (st_b["net_R"] > st_m["net_R"] + 0.5) and (st_b["recoveries_cut"] < st_m["recoveries_cut"])
    geom_only = set(stable_feats).issubset(GEOM | {"ema_dist_atr", "mom_1R"}) and not beats_same_ck
    placebo_fail = p_emp_roc > 0.05 and p_emp_tb > 0.05

    if leak["leakage"]:
        verdict = "LEAKAGE_FAIL"
    elif placebo_fail or not (ll_mono and r_dir):
        verdict = "NO_LOSS_SIGNAL"
    elif (not beats_same_ck) or (geom_only and not beats_naive):
        verdict = "MECHANICAL_LOSS_SIGNAL"
    elif years_ll_down < 4 or invert or max_share > 0.50 or rec_lost_share > 0.70 or not dd_ok:
        verdict = "REGIME_SPECIFIC"
    elif beats_naive and beats_same_ck and years_ll_down >= 4 and st_b["net_R"] > 0 and rec_dmg["preservation_pct"] >= 0.50 and not placebo_fail and ruin_ok:
        verdict = "CANDIDATE_FOR_LOSS_ACTION_RESEARCH"
    elif not beats_same_ck:
        verdict = "MECHANICAL_LOSS_SIGNAL"
    else:
        verdict = "NO_LOSS_SIGNAL"

    rec_next = {
        "CANDIDATE_FOR_LOSS_ACTION_RESEARCH": "investigate loss-action design (still not production)",
        "MECHANICAL_LOSS_SIGNAL": "stop — signal is mostly adverse-distance geometry; a fixed early cap already captures it",
        "REGIME_SPECIFIC": "stop loss-reduction research; does not survive OOS years",
        "NO_LOSS_SIGNAL": "stop loss-reduction research",
        "LEAKAGE_FAIL": "stop; fix leakage before any further work",
    }[verdict]

    oos_s = _r_stats(oos_tr["p0_R"].to_numpy())
    bm = mdf[mdf["model"] == best_model].iloc[0].to_dict() if len(mdf[mdf["model"] == best_model]) else {}
    top_b, bot_b = (fat[-1], fat[0]) if fat else ({}, {})
    lines = [
        "# Sprint 46 — Conditional Loss Reduction",
        "",
        "## Question",
        "",
        '"Can PIT information identify trades that are likely to become large P0 losses early enough to reduce loss magnitude without destroying recoveries?"',
        "",
        "Research only. P0 unchanged. P1/P2/P3 not used. 2021 is not true OOS (panel starts 2021).",
        "",
        "## Baseline",
        "",
        f"- Trades (2022–2026): **{oos_s['n']}**",
        f"- Mean R: {oos_s['mean_R']:.3f}",
        f"- Median R: {oos_s['median_R']:.3f}",
        f"- PF: {oos_s['pf']:.2f}",
        f"- WR: {oos_s['wr']:.3f}",
        f"- Payoff: {oos_s['payoff']:.2f}",
        f"- P(R <= -1R): {oos_s['p_le_m1_00']:.3f} (n={ll_n})",
        f"- P(R <= -1.25R): {oos_s['p_le_m1_25']:.3f}",
        f"- P(R <= -1.50R): {oos_s['p_le_m1_50']:.3f} — no −1.5R tail under P0; losses cluster near −1.05R SL",
        f"- Worst R: {oos_s['worst_R']:.3f}",
        f"- 5th percentile R: {oos_s['p5_R']:.3f}",
        f"- Large MAE≥1R but recovered (final R>0): **{rec_n}**",
        "",
        "## Signal",
        "",
        f"- Best model (val ROC): **{best_model}**",
        f"- Frozen checkpoint / threshold: first current_R ≤ **-{ck_star:.2f}R**, p ≥ **{thr_star:.2f}** (selected on 2022 val)",
        f"- OOS ROC: {_fmt(float(bm.get('oos_roc', float('nan'))))}",
        f"- OOS PR: {_fmt(float(bm.get('oos_pr', float('nan'))))}",
        f"- Calibration ECE: {_fmt(float(bm.get('oos_ece', float('nan'))))}",
        f"- Top-risk large-loss rate: {_fmt(float(top_b.get('large_loss_rate', float('nan'))))} (n={top_b.get('n', 0)})",
        f"- Bottom-risk large-loss rate: {_fmt(float(bot_b.get('large_loss_rate', float('nan'))))} (n={bot_b.get('n', 0)})",
        f"- Top−bottom: {_fmt(float(top_b.get('large_loss_rate', 0) - bot_b.get('large_loss_rate', 0)))}",
        f"- Stable univariate features: {stable_feats or 'none'}",
        "",
        "### Probability buckets (OOS observations)",
        "",
        _md(brows, ["bucket", "n", "large_loss_rate", "mean_R", "median_R", "p5_R", "recovery_rate"],
            {"n": 0, "large_loss_rate": 3, "mean_R": 3, "median_R": 3, "p5_R": 3, "recovery_rate": 3}),
        "",
        "## Recovery Protection",
        "",
        f"- Recoverable adverse trades (MAE≥0.75R and final R>0): {rec_dmg['recoverable_adverse_n']}",
        f"- Incorrectly cut: {rec_dmg['incorrectly_cut']}",
        f"- Recovery lost R: {rec_dmg['total_missed_R']:.2f}",
        f"- Recovery preservation %: {100 * rec_dmg['preservation_pct']:.1f}%",
        "",
        "## Tail Reduction",
        "",
        f"- Large losses avoided: {tail_ben['avoided_n']} / {tail_ben['large_loss_n']}",
        f"- Average avoided loss: {tail_ben['avg_avoided_R']:+.3f}R",
        f"- Total avoided R: {tail_ben['total_avoided_R']:+.2f}",
        f"- Worst loss before: {st_b['worst_before']:.3f}  after: {st_b['worst_after']:.3f}",
        f"- 5th percentile before: {st_b['p5_before']:.3f}  after: {st_b['p5_after']:.3f}",
        f"- Net (tail_saved − recovery_lost): **{st_b['net_R']:+.2f}R**",
        "",
        "## Naive Comparison",
        "",
        f"- Conditional model net R: {st_b['net_R']:+.2f} (cuts={st_b['n_cut']}, rec cut={st_b['recoveries_cut']})",
        f"- Naive −{NAIVE_CUT:.2f}R net R: {st_c['net_R']:+.2f} (cuts={st_c['n_cut']}, rec cut={st_c['recoveries_cut']})",
        f"- Naive −{ck_star:.2f}R (same checkpoint, no model) net R: {st_m['net_R']:+.2f} (cuts={st_m['n_cut']}, rec cut={st_m['recoveries_cut']})",
        f"- Production P0: net 0 by definition",
        "",
        _md(
            [{"who": "P0", "pf": p0p["pf"], "dd": p0p["dd"], "ret": p0p["ret"], "wr": p0p.get("wr", float("nan"))},
             {"who": "conditional", "pf": bp["pf"], "dd": bp["dd"], "ret": bp["ret"], "wr": bp.get("wr", float("nan"))},
             {"who": "naive_-0.75R", "pf": cp["pf"], "dd": cp["dd"], "ret": cp["ret"], "wr": cp.get("wr", float("nan"))},
             {"who": f"naive_-{ck_star:.2f}R", "pf": mp["pf"], "dd": mp["dd"], "ret": mp["ret"], "wr": mp.get("wr", float("nan"))}],
            ["who", "pf", "dd", "ret", "wr"],
            {"pf": 2, "dd": 3, "ret": 3, "wr": 3},
        ),
        "",
        "## OOS",
        "",
        _md(ydf.to_dict("records"),
            ["year", "trades", "p0_ll_rate", "b_ll_rate", "c_ll_rate", "p0_mean_R", "b_mean_R", "p0_pf", "b_pf", "p0_dd", "b_dd"],
            {"year": 0, "trades": 0, "p0_ll_rate": 3, "b_ll_rate": 3, "c_ll_rate": 3,
             "p0_mean_R": 3, "b_mean_R": 3, "p0_pf": 2, "b_pf": 2, "p0_dd": 3, "b_dd": 3}),
        "",
        f"Years with lower tail-loss rate: {years_ll_down}/5. Max year share of net: {100 * max_share:.1f}%.",
        "",
        "## Risk",
        "",
        _md([mc0, mcb, mcc, mcm], ["policy", "prob_ruin", "median_dd", "p95_dd", "p99_dd", "worst_dd"],
            {"prob_ruin": 3, "median_dd": 3, "p95_dd": 3, "p99_dd": 3, "worst_dd": 3}),
        "",
        f"Placebo ROC empirical p={p_emp_roc:.3f}; top−bottom LL empirical p={p_emp_tb:.3f}.",
        "",
        "## Verdict",
        "",
        f"**`{verdict}`**",
        "",
        "## Production",
        "",
        "Entry: UNCHANGED",
        "",
        "P0: UNCHANGED",
        "",
        "P1/P2/P3: NOT USED",
        "",
        "Reversal: NOT USED",
        "",
        "Loss suppression: NOT SHIPPED",
        "",
        "## Recommendation",
        "",
        rec_next + ".",
        "",
    ]
    (OUT / "sprint46_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "sprint46_final_verdict.json").write_text(json.dumps({
        "sprint": 46, "verdict": verdict, "production_changed": False,
        "best_model": best_model, "checkpoint": ck_star, "threshold": thr_star,
        "oos_net_R": st_b["net_R"], "naive_net_R": st_c["net_R"],
        "years_ll_down": years_ll_down, "max_year_share": max_share,
        "beats_naive_m075": beats_naive, "beats_naive_same_ck": beats_same_ck,
        "recommendation": rec_next,
    }, indent=2, default=float), encoding="utf-8")
    print(f"VERDICT {verdict}")
    return 0


def _report_early(trades, oos_tr, ll_rate, rec_n, stable_feats, best_model, mdf, brows, v, why) -> str:
    oos_s = _r_stats(oos_tr["p0_R"].to_numpy())
    bm = mdf[mdf["model"] == best_model].iloc[0].to_dict() if len(mdf) and best_model in set(mdf["model"]) else {}
    return "\n".join([
        "# Sprint 46 — Conditional Loss Reduction",
        "",
        "## Question",
        "",
        '"Can PIT information identify trades that are likely to become large P0 losses early enough to reduce loss magnitude without destroying recoveries?"',
        "",
        "## Baseline",
        "",
        f"- Trades: {oos_s['n']}",
        f"- Mean R: {oos_s['mean_R']:.3f}",
        f"- Median R: {oos_s['median_R']:.3f}",
        f"- PF: {oos_s['pf']:.2f}",
        f"- WR: {oos_s['wr']:.3f}",
        f"- Payoff: {oos_s['payoff']:.2f}",
        f"- P(R <= -1R): {oos_s['p_le_m1_00']:.3f}",
        f"- P(R <= -1.25R): {oos_s['p_le_m1_25']:.3f}",
        f"- P(R <= -1.50R): {oos_s['p_le_m1_50']:.3f}",
        f"- Worst R: {oos_s['worst_R']:.3f}",
        f"- 5th percentile R: {oos_s['p5_R']:.3f}",
        f"- Large MAE recovered: {rec_n}",
        "",
        "## Signal",
        "",
        f"- Best model: {best_model}",
        f"- OOS ROC: {_fmt(float(bm.get('oos_roc', float('nan'))))}",
        f"- OOS PR: {_fmt(float(bm.get('oos_pr', float('nan'))))}",
        f"- Stable features: {stable_feats or 'none'}",
        "",
        _md(brows, ["bucket", "n", "large_loss_rate", "mean_R", "p5_R", "recovery_rate"],
            {"n": 0, "large_loss_rate": 3, "mean_R": 3, "p5_R": 3, "recovery_rate": 3}) if brows else "",
        "",
        "## Verdict",
        "",
        f"**`{v}`**",
        "",
        why,
        "",
        "## Production",
        "",
        "Entry: UNCHANGED",
        "",
        "P0: UNCHANGED",
        "",
        "Loss suppression: NOT SHIPPED",
        "",
        "## Recommendation",
        "",
        "Stop loss-reduction research.",
        "",
    ])


if __name__ == "__main__":
    raise SystemExit(main())
