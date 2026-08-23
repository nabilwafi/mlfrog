"""Sprint 41 Iter 4–5 — policy switching replay + full OOS table.

LR p3_better (val-selected). Score each bar OOS. If P>=thr (val-chosen),
use P3 trail (a0.20/d0.10) this bar, else production a0.25/d0.08.

2021 stays production (no prior train year). 2022: train 2021, thr default 0.50.

  python apps/research_sprint41_iter4_switch.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import tools.pyc_path_hook  # noqa: F401

from apps.report_adaptive_vol_risk import iter6_montecarlo
from apps.report_multi_trade_engine import run_multi
from apps.research_sprint40_iter1_time_aware_exit import CFG_PROD, STARTING
from apps.research_sprint41_iter1_diff import STATE_FEATS
from apps.run_exit_engine_grid import CAP, Paths, simulate_combo
from apps.research_sprint39_exit_state import _join_feat7, _load_panel
from simulation.wf.sim import COST, load_h1, prepare_market

ITER1 = _ROOT / "artifacts/pipeline_backtest/exit_structure/sprint41/iter1_differential"
OUT = _ROOT / "artifacts/pipeline_backtest/exit_structure/sprint41/iter4_switch"
YEARS = (2021, 2022, 2023, 2024, 2025, 2026)
# P3 params
P3_ACT, P3_DIST = 0.20, 0.10
P0_ACT, P0_DIST = 0.25, 0.08


def _fit_lr(Xtr, ytr, Xte) -> np.ndarray:
    sc = StandardScaler()
    clf = LogisticRegression(max_iter=200, class_weight="balanced")
    clf.fit(sc.fit_transform(Xtr), ytr)
    return clf.predict_proba(sc.transform(Xte))[:, 1]


def _orders(p: Paths, year: int) -> np.ndarray:
    idx = np.where(p.year == year)[0]
    return idx[np.argsort(pd.DatetimeIndex(p.ts.iloc[idx]).as_unit("ns").asi8, kind="stable")]


def _port(p: Paths, sim: dict, year: int) -> dict:
    return run_multi(
        order=_orders(p, year),
        p=p,
        sim=sim,
        lot_mult=np.ones(p.n),
        atr_pct=np.ones(p.n),
        probs=np.full(p.n, 0.5),
        trend_ok=np.ones(p.n, dtype=bool),
        cfg=CFG_PROD,
        starting=STARTING,
        fixed_lot_01=True,
        daily_on_below=80.0,
    )


def _year_row(year: int, port: dict, sim: dict) -> dict:
    taken = np.asarray(port["taken"], dtype=int)
    r = sim["r_multiple"][taken] if taken.size else np.array([])
    wins, losses = r[r > 0], r[r < 0]
    payoff = float(wins.mean() / abs(losses.mean())) if losses.size else 0.0
    streak = best = 0
    for x in r:
        if x < 0:
            streak += 1
            best = max(best, streak)
        else:
            streak = 0
    return {
        "year": year,
        "trades": int(port["n_trades"]),
        "pf": float(port["profit_factor"]) if np.isfinite(port["profit_factor"]) else 0.0,
        "dd": float(port["max_drawdown"]),
        "ret": float(port["total_return"]),
        "wr": float(np.mean(r > 0)) if r.size else 0.0,
        "payoff": payoff,
        "avg_r": float(np.mean(r)) if r.size else 0.0,
        "median_r": float(np.median(r)) if r.size else 0.0,
        "max_loss_streak": int(best),
        "blown": bool(port["blown"]),
        "pnl": port["pnl"],
    }


def simulate_switch(p: Paths, use_p3: np.ndarray) -> dict:
    """Per-bar: if use_p3[i,j] then P3 act/dist else production. use_p3 shape (n, CAP+1)."""
    n = p.n
    sl = np.full(n, -1.0)
    extreme = np.zeros(n)
    mae = np.zeros(n)
    active = np.ones(n, dtype=bool)
    exit_r = np.zeros(n)
    exit_off = np.zeros(n, dtype=np.int64)
    reason = np.full(n, 4, dtype=np.int64)
    trail_moved = np.zeros(n, dtype=bool)
    for j in range(1, CAP + 1):
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
        sw = use_p3[np.where(m)[0], j]
        act = np.where(sw, P3_ACT, P0_ACT)
        dist = np.where(sw, P3_DIST, P0_DIST)
        act_ok = extreme[m] >= act
        if act_ok.any():
            idx = np.where(m)[0][act_ok]
            new_sl = extreme[idx] - dist[act_ok] * p.atr_over_r[idx, j]
            moved = new_sl > sl[idx]
            sl[idx[moved]] = new_sl[moved]
            trail_moved[idx[moved]] = True
        hit = m.copy()
        hit[m] = adv_j >= -sl[m]
        if hit.any():
            exit_r[hit] = sl[hit]
            exit_off[hit] = j
            reason[hit] = np.where(sl[hit] > 1e-9, 1, np.where(trail_moved[hit], 1, 0))
            active[hit] = False
    if active.any():
        lo = p.last_off[active]
        exit_r[active] = p.close_r[active, lo]
        exit_off[active] = lo
    net = exit_r * p.r_unit_pct - COST
    return {
        "net_return": net,
        "r_multiple": net / p.r_unit_pct,
        "holding_bars": np.maximum(exit_off, 1),
        "reason": reason,
        "mfe_r": extreme,
        "mae_r": mae,
        "partial_any": np.zeros(n, dtype=bool),
    }


def _bar_scores(ds: pd.DataFrame) -> np.ndarray:
    """OOS P(p3_better) per dataset row. 2021 = nan."""
    X = ds[STATE_FEATS].astype(float).fillna(0.0).to_numpy()
    y = ds["p3_better"].astype(int).to_numpy()
    years = ds["year"].to_numpy()
    pred = np.full(len(ds), np.nan)
    # 2022: train 2021
    tr = years == 2021
    te = years == 2022
    if tr.sum() > 200 and te.sum() > 50:
        pred[te] = _fit_lr(X[tr], y[tr], X[te])
    for te_y in (2023, 2024, 2025, 2026):
        tr = years < te_y
        te = years == te_y
        if tr.sum() < 200 or te.sum() < 50:
            continue
        pred[te] = _fit_lr(X[tr], y[tr], X[te])
    return pred


def _val_thr(ds: pd.DataFrame, pred: np.ndarray) -> float:
    """Pick thr on 2022–2025 val-style: year Y uses scores from expanding train, maximize mean diff on that year among grid."""
    # ponytail: grid on 2022 only (first scored year) — not OOS 2026
    grid = (0.30, 0.40, 0.50, 0.60, 0.70)
    m = (ds["year"].to_numpy() == 2022) & np.isfinite(pred)
    if m.sum() < 50:
        return 0.50
    d = ds.loc[m, "diff_p3"].to_numpy(dtype=float)
    p = pred[m]
    best_t, best = 0.50, -1e9
    for t in grid:
        sw = p >= t
        # mean diff if we took P3 on those obs (trade-level: use unique trades via last bar later)
        val = float(np.mean(d[sw])) if sw.any() else -1e9
        if val > best:
            best, best_t = val, t
    return float(best_t)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ds = pd.read_parquet(ITER1 / "exit_differential_dataset.parquet")
    pred = _bar_scores(ds)
    thr = _val_thr(ds, pred)
    print(f"val threshold (2022 grid)={thr:.2f}  scored_frac={float(np.mean(np.isfinite(pred))):.2f}")

    panel = _join_feat7(_load_panel())
    p = Paths(panel, prepare_market(load_h1(_ROOT / "artifacts/raw/XAUUSD/H1/data.parquet")))
    use = np.zeros((p.n, CAP + 1), dtype=bool)
    # map dataset rows -> (trade_id, bars_in_trade)
    tid = ds["trade_id"].to_numpy(dtype=int)
    j = ds["bars_in_trade"].to_numpy(dtype=int)
    ok = np.isfinite(pred) & (pred >= thr)
    for k in np.where(ok)[0]:
        ti, jj = int(tid[k]), int(j[k])
        if 0 <= ti < p.n and 1 <= jj <= CAP:
            use[ti, jj] = True
    print(f"switch bars={int(use.sum())} / {len(ds)}")

    sim0 = simulate_combo(p, act=P0_ACT, dist=P0_DIST, tp=None, partials=(), tmax=None, be=None)
    sims = simulate_switch(p, use)

    base_y, sw_y = [], []
    for y in YEARS:
        b = _year_row(y, _port(p, sim0, y), sim0)
        s = _year_row(y, _port(p, sims, y), sims)
        base_y.append(b)
        sw_y.append(s)
        print(f"  {y}: base PF={b['pf']:.2f} DD={b['dd']*100:.1f}%  sw PF={s['pf']:.2f} DD={s['dd']*100:.1f}% n={s['trades']}")

    base_pnl = np.concatenate([r["pnl"] for r in base_y if len(r["pnl"])])
    sw_pnl = np.concatenate([r["pnl"] for r in sw_y if len(r["pnl"])])
    mc = iter6_montecarlo(base_pnl, sw_pnl, any(r["blown"] for r in base_y), starting=STARTING)
    dyn = mc[mc["policy"] == "dynamic"]
    mc_ruin = float(dyn.iloc[0]["prob_ruin"]) if len(dyn) else 1.0
    mc_wdd = float(dyn.iloc[0]["worst_dd"]) if len(dyn) else 1.0
    mc0 = mc[mc["policy"] == "fixed"]
    mc0_ruin = float(mc0.iloc[0]["prob_ruin"]) if len(mc0) else 1.0
    mc0_wdd = float(mc0.iloc[0]["worst_dd"]) if len(mc0) else 1.0

    def _beat(b, s) -> bool:
        if s["trades"] < 0.90 * max(b["trades"], 1):
            return False
        if s["pf"] + 0.02 < b["pf"]:
            return False
        if s["dd"] > b["dd"] + 0.01:
            return False
        return True

    years_ok = {int(b["year"]): _beat(b, s) for b, s in zip(base_y, sw_y)}
    n_ok = sum(years_ok.values())
    early_fail = (not years_ok[2021]) or (not years_ok[2022])
    # pooled: max yearly DD, mean PF via concat pnl
    def _pf(pnl):
        gp, gl = float(pnl[pnl > 0].sum()), float(-pnl[pnl < 0].sum())
        return gp / gl if gl > 0 else 0.0

    pooled_ok = (
        _pf(sw_pnl) >= _pf(base_pnl) - 1e-9
        and max(s["dd"] for s in sw_y) <= max(b["dd"] for b in base_y) + 0.01
        and not any(s["blown"] for s in sw_y)
        and mc_ruin <= mc0_ruin + 1e-12
    )
    if early_fail and n_ok >= 4:
        verdict = "REGIME_SPECIFIC"
    elif n_ok >= 4 and pooled_ok:
        verdict = "CANDIDATE_FOR_REVIEW"
    elif n_ok == 0:
        verdict = "NO_SIGNAL"
    else:
        verdict = "NO_SIGNAL"

    lines = [
        "# Sprint 41 Iter 4–5 — Policy switching vs production",
        "",
        f"Model: LR `p3_better` (val-selected). Threshold **{thr:.2f}** from 2022 grid. P3 = a0.20/d0.10 else a0.25/d0.08.",
        "2021 unscored → production. Scores expanding-year OOS. Frozen portfolio $280 / lot 0.01 / mo5 / heat 3R.",
        "",
        "| Year | who | Trades | PF | DD | Return | WR | Payoff | AvgR | MedR | MaxLoss |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for b, s in zip(base_y, sw_y):
        for who, r in (("prod", b), ("switch", s)):
            lines.append(
                f"| {r['year']} | {who} | {r['trades']} | {r['pf']:.2f} | {r['dd']*100:.1f}% | "
                f"{r['ret']*100:.0f}% | {r['wr']:.3f} | {r['payoff']:.2f} | {r['avg_r']:.3f} | "
                f"{r['median_r']:.3f} | {r['max_loss_streak']} |"
            )
    lines += [
        "",
        f"MC ruin switch={mc_ruin:.3f} (prod {mc0_ruin:.3f})  worst DD {mc_wdd*100:.1f}% (prod {mc0_wdd*100:.1f}%)",
        "",
        "## Beat baseline per year",
        "",
        "| Year | beat |",
        "|---:|:---:|",
    ]
    for y in YEARS:
        lines.append(f"| {y} | {'yes' if years_ok[y] else 'no'} |")
    lines += [
        "",
        f"Years not worse: {n_ok}/6  pooled_ok={pooled_ok}  early_fail={early_fail}",
        "",
        f"**Verdict: {verdict}**",
        "",
        "Production exit unchanged (`a0.25/d0.08`). Not shipped.",
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    rows = []
    for b, s in zip(base_y, sw_y):
        for who, r in (("prod", b), ("switch", s)):
            rows.append({k: v for k, v in r.items() if k != "pnl"} | {"who": who})
    pd.DataFrame(rows).to_csv(OUT / "yearly_comparison.csv", index=False)
    (OUT / "decision.json").write_text(
        json.dumps(
            {
                "iter": 5,
                "verdict": verdict,
                "threshold": thr,
                "years_beat": years_ok,
                "n_years_ok": n_ok,
                "pooled_ok": pooled_ok,
                "mc_ruin": mc_ruin,
                "mc_worst_dd": mc_wdd,
                "production_exit": "a0.25_d0.08 unchanged",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"VERDICT={verdict} n_ok={n_ok}/6")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
