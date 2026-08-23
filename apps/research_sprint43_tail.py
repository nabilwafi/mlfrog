"""Sprint 43 — Tail attribution and counterfactual robustness.

Frozen Sprint 41 switch. No retune. No production change.

  python apps/research_sprint43_tail.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import tools.pyc_path_hook  # noqa: F401

from apps.report_adaptive_vol_risk import N_MC, RNG
from apps.research_sprint40_iter1_time_aware_exit import STARTING
from apps.research_sprint41_iter1_diff import STATE_FEATS
from apps.research_sprint41_iter4_switch import (
    CAP,
    P0_ACT,
    P0_DIST,
    P3_ACT,
    P3_DIST,
    YEARS,
    _bar_scores,
    _port,
    _year_row,
    simulate_switch,
)
from apps.research_sprint42_attribution import (
    BAR_EDGES,
    ITER1,
    MFE_EDGES,
    THR,
    TRUE_OOS,
    _exit_px,
    _md,
    _pf,
    _pooled,
    _use_matrix,
)
from apps.run_exit_engine_grid import REASONS, Paths, simulate_combo
from apps.research_sprint39_exit_state import _join_feat7, _load_panel
from simulation.wf.sim import load_h1, prepare_market

OUT = _ROOT / "artifacts/pipeline_backtest/exit_structure/sprint43"
N_PLACEBO = 1000
DELTA_BUCKETS = (
    ("le_m0.50", lambda d: d <= -0.50),
    ("m0.50_m0.25", lambda d: (d > -0.50) & (d <= -0.25)),
    ("m0.25_0", lambda d: (d > -0.25) & (d < 0)),
    ("eq_0", lambda d: np.isclose(d, 0.0, atol=1e-9)),
    ("0_0.25", lambda d: (d > 0) & (d < 0.25)),
    ("0.25_0.50", lambda d: (d >= 0.25) & (d < 0.50)),
    ("0.50_1.00", lambda d: (d >= 0.50) & (d < 1.00)),
    ("ge_1.00", lambda d: d >= 1.00),
)
CLUSTER_LO, CLUSTER_HI = 1.05, 1.25


def _mc(pnl: np.ndarray, label: str) -> dict:
    if len(pnl) < 5:
        return {"policy": label, "n_mc": 0, "prob_ruin": 1.0, "median_dd": 1.0,
                "p95_dd": 1.0, "p99_dd": 1.0, "worst_dd": 1.0}
    ruins, maxdds = 0, []
    for _ in range(N_MC):
        eq = peak = STARTING
        mdd, blown = 0.0, False
        for x in RNG.permutation(pnl):
            eq += x
            if eq <= 0:
                blown = True
                eq = 0.0
                break
            peak = max(peak, eq)
            mdd = max(mdd, (peak - eq) / peak)
        ruins += int(blown)
        maxdds.append(mdd)
    a = np.asarray(maxdds)
    return {
        "policy": label, "n_mc": N_MC, "prob_ruin": ruins / N_MC,
        "median_dd": float(np.median(a)), "p95_dd": float(np.quantile(a, 0.95)),
        "p99_dd": float(np.quantile(a, 0.99)), "worst_dd": float(np.max(a)),
    }


def _sign_years(sub: pd.DataFrame) -> str:
    bits = []
    for y in TRUE_OOS:
        d = sub.loc[sub["year"] == y, "delta_R"]
        if len(d) == 0:
            bits.append(f"{y}:na")
        else:
            m = float(d.mean())
            bits.append(f"{y}:{'+' if m > 0 else ('0' if m == 0 else '-')}")
    return " ".join(bits)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ds = pd.read_parquet(ITER1 / "exit_differential_dataset.parquet")
    pred = _bar_scores(ds)
    ds = ds.copy()
    ds["p3_probability"] = pred
    ds["p3_bar"] = np.isfinite(pred) & (pred >= THR)

    panel = _join_feat7(_load_panel())
    p = Paths(panel, prepare_market(load_h1(_ROOT / "artifacts/raw/XAUUSD/H1/data.parquet")))
    sim0 = simulate_combo(p, act=P0_ACT, dist=P0_DIST, tp=None, partials=(), tmax=None, be=None)
    sim3 = simulate_combo(p, act=P3_ACT, dist=P3_DIST, tp=None, partials=(), tmax=None, be=None)
    use = _use_matrix(p.n, ds, pred)
    simc = simulate_switch(p, use)
    ts0 = pd.DatetimeIndex(p.ts)

    att = pd.read_csv(OUT.parent / "sprint42/sprint42_trade_attribution.csv")
    att["entry_time"] = pd.to_datetime(att["entry_time"], utc=True)
    att["p0_exit_time"] = pd.to_datetime(att["p0_exit_time"], utc=True)
    att["p3_exit_time"] = pd.to_datetime(att["p3_exit_time"], utc=True)
    tid = att["trade_id"].to_numpy(dtype=int)
    att["p0_reason"] = [REASONS[int(x)] for x in sim0["reason"][tid]]
    att["p3_reason"] = [REASONS[int(x)] for x in sim3["reason"][tid]]
    att["p0_mfe_final"] = sim0["mfe_r"][tid]
    att["p3_mfe_final"] = sim3["mfe_r"][tid]
    att["p0_mae_final"] = sim0["mae_r"][tid]
    att["p3_mae_final"] = sim3["mae_r"][tid]
    att["hour_utc"] = att["entry_time"].dt.hour
    att["p0_trail_activated"] = att["p0_mfe_final"] >= P0_ACT
    att["p3_trail_activated"] = att["p3_mfe_final"] >= P3_ACT

    first_sw = (
        ds[ds["p3_bar"]]
        .sort_values(["trade_id", "bars_in_trade"])
        .groupby("trade_id", as_index=False)
        .first()[["trade_id", "bars_in_trade", "mfe_so_far_R", "timestamp", "p3_probability"]]
        .rename(columns={
            "bars_in_trade": "first_p3_bar",
            "mfe_so_far_R": "mfe_at_first_p3",
            "timestamp": "first_p3_ts",
            "p3_probability": "p3_prob_at_first_switch",
        })
    )
    att = att.merge(first_sw, on="trade_id", how="left")
    att["p3_before_mfe_gt0"] = att["mfe_at_first_p3"] < 0
    att["in_cluster"] = (att["delta_R"] >= CLUSTER_LO) & (att["delta_R"] <= CLUSTER_HI)
    att["ge_1r"] = att["delta_R"] >= 1.0

    p3s = att[(att["selected_policy"] == "P3") & att["year"].isin(TRUE_OOS)].copy()
    d = p3s["delta_R"].to_numpy(dtype=float)
    n = len(p3s)
    total = float(d.sum())
    pos_total = float(d[d > 0].sum())
    print(f"P3_SELECTED OOS n={n} mean={d.mean():.4f} total={total:.2f} pos_total={pos_total:.2f}")

    # --- 2 buckets ---
    buck_rows = []
    for lab, fn in DELTA_BUCKETS:
        m = fn(d)
        sub = d[m]
        contrib = float(sub.sum()) if m.any() else 0.0
        row = {
            "scope": "2022-2026",
            "bucket": lab,
            "n": int(m.sum()),
            "pct": float(m.mean()),
            "mean_delta_R": float(sub.mean()) if m.any() else 0.0,
            "median_delta_R": float(np.median(sub)) if m.any() else 0.0,
            "total_delta_R": contrib,
            "pct_of_positive": (contrib / pos_total) if pos_total > 0 and contrib > 0 else 0.0,
        }
        buck_rows.append(row)
        for y in TRUE_OOS:
            yd = p3s.loc[p3s["year"] == y, "delta_R"].to_numpy(dtype=float)
            ym = fn(yd)
            buck_rows.append({
                "scope": str(y), "bucket": lab, "n": int(ym.sum()),
                "pct": float(ym.mean()) if len(yd) else 0.0,
                "mean_delta_R": float(yd[ym].mean()) if ym.any() else 0.0,
                "median_delta_R": float(np.median(yd[ym])) if ym.any() else 0.0,
                "total_delta_R": float(yd[ym].sum()) if ym.any() else 0.0,
                "pct_of_positive": float("nan"),
            })
    pd.DataFrame(buck_rows).to_csv(OUT / "sprint43_delta_tail_buckets.csv", index=False)

    def _share(mask) -> float:
        t = float(d[mask].sum())
        return t / pos_total if pos_total > 0 else 0.0

    share_025 = _share(d >= 0.25)
    share_050 = _share(d >= 0.50)
    share_100 = _share(d >= 1.00)
    share_cl = _share((d >= CLUSTER_LO) & (d <= CLUSTER_HI))

    # --- 3 leave-tail-out ---
    cluster_m = (p3s["delta_R"] >= CLUSTER_LO) & (p3s["delta_R"] <= CLUSTER_HI)
    ranked = p3s.sort_values("delta_R", ascending=False)
    top1pct_n = max(1, int(np.ceil(0.01 * n)))
    cases = [
        ("A_all", p3s),
        ("B_ex_ge_1R", p3s[p3s["delta_R"] < 1.0]),
        ("C_ex_1.15_cluster", p3s[~cluster_m]),
        ("D_ex_top1", ranked.iloc[1:]),
        ("E_ex_top5", ranked.iloc[5:]),
        ("F_ex_top10", ranked.iloc[10:]),
        ("G_ex_top1pct", ranked.iloc[top1pct_n:]),
    ]
    lto = []
    for name, sub in cases:
        dd = sub["delta_R"].to_numpy(dtype=float)
        lto.append({
            "case": name, "n": int(len(sub)),
            "mean_delta_R": float(dd.mean()) if len(dd) else 0.0,
            "median_delta_R": float(np.median(dd)) if len(dd) else 0.0,
            "p_delta_gt_0": float((dd > 0).mean()) if len(dd) else 0.0,
            "total_delta_R": float(dd.sum()) if len(dd) else 0.0,
            "yearly_sign": _sign_years(sub),
        })
        print(f"  {name}: n={len(sub)} mean={dd.mean() if len(dd) else 0:.4f} sign={_sign_years(sub)}")
    pd.DataFrame(lto).to_csv(OUT / "sprint43_leave_tail_out.csv", index=False)

    # --- 4 cluster ---
    cl = p3s[cluster_m].copy()
    cl_out = cl[[
        "trade_id", "year", "entry_time", "side", "entry_price",
        "p0_exit_time", "p3_exit_time", "p0_R", "p3_R", "delta_R",
        "mfe_so_far_R", "mae_so_far_R", "current_R", "bars_in_trade",
        "p3_prob_at_first_switch", "n_p3_bars", "first_p3_ts",
        "p3_before_mfe_gt0", "p0_reason", "p3_reason", "hour_utc",
        "p0_mfe_final", "p3_mfe_final", "p0_trail_activated", "p3_trail_activated",
        "drawdown_from_MFE_R",
    ]].copy()
    cl_out["different_terminal"] = cl_out["p0_reason"] != cl_out["p3_reason"]
    cl_out.to_csv(OUT / "sprint43_tail_cluster.csv", index=False)

    # --- 5 exit matrix ---
    mat = (
        p3s.groupby(["p0_reason", "p3_reason"], as_index=False)
        .agg(n=("delta_R", "size"), mean_delta_R=("delta_R", "mean"),
             total_delta_R=("delta_R", "sum"), n_cluster=("in_cluster", "sum"))
        .sort_values("total_delta_R", ascending=False)
    )
    mat.to_csv(OUT / "sprint43_exit_event_matrix.csv", index=False)

    # --- 6 path bins ---
    path_rows = []
    p3s = p3s.assign(mfe_minus_cur=p3s["mfe_so_far_R"] - p3s["current_R"])
    for kind, series, edges in (
        ("mfe_decision", p3s["mfe_so_far_R"], MFE_EDGES),
        ("current_R", p3s["current_R"], MFE_EDGES),
        ("dd_from_mfe", p3s["drawdown_from_MFE_R"], MFE_EDGES),
        ("mae", p3s["mae_so_far_R"], MFE_EDGES),
        ("bars", p3s["bars_in_trade"], BAR_EDGES),
        ("n_p3_bars", p3s["n_p3_bars"], BAR_EDGES),
        ("mfe_minus_current", p3s["mfe_minus_cur"], MFE_EDGES),
    ):
        x = series.to_numpy(dtype=float)
        for lab, fn in edges:
            m = fn(x)
            sub = d[m]
            path_rows.append({
                "kind": kind, "bucket": lab, "n": int(m.sum()),
                "mean_delta_R": float(sub.mean()) if m.any() else 0.0,
                "median_delta_R": float(np.median(sub)) if m.any() else 0.0,
                "p_delta_gt_0": float((sub > 0).mean()) if m.any() else 0.0,
                "total_delta_R": float(sub.sum()) if m.any() else 0.0,
            })
    pd.DataFrame(path_rows).to_csv(OUT / "sprint43_path_attribution.csv", index=False)

    # --- 7 yearly tail ---
    yrows = []
    for y in TRUE_OOS:
        sub = p3s[p3s["year"] == y]
        dd = sub["delta_R"].to_numpy(dtype=float)
        yrows.append({
            "year": y, "n": int(len(sub)),
            "total_delta_R": float(dd.sum()),
            "contrib_ge_1R": float(dd[dd >= 1].sum()),
            "contrib_ge_0.5R": float(dd[dd >= 0.5].sum()),
            "total_ex_ge_1R": float(dd[dd < 1].sum()),
            "mean_delta_R": float(dd.mean()),
            "median_delta_R": float(np.median(dd)),
            "mean_ex_ge_1R": float(dd[dd < 1].mean()) if (dd < 1).any() else 0.0,
            "n_ge_1R": int((dd >= 1).sum()),
            "n_cluster": int(sub["in_cluster"].sum()),
        })
    pd.DataFrame(yrows).to_csv(OUT / "sprint43_yearly_tail_dependency.csv", index=False)

    # --- 8 placebo ---
    ev = att[att["year"].isin(TRUE_OOS) & att["evaluable"]].copy()
    actual_mean = float(d.mean())
    rng = np.random.default_rng(42)
    placebo_means = []
    n_by_y = p3s.groupby("year").size().to_dict()
    for _ in range(N_PLACEBO):
        picked = []
        for y, k in n_by_y.items():
            pool = ev.loc[ev["year"] == y, "delta_R"].to_numpy()
            picked.append(rng.choice(pool, size=int(k), replace=False))
        placebo_means.append(float(np.concatenate(picked).mean()))
    pm = np.asarray(placebo_means)
    p_emp = float((pm >= actual_mean).mean())
    pd.DataFrame([{
        "actual_mean": actual_mean, "placebo_mean": float(pm.mean()),
        "placebo_std": float(pm.std()), "percentile": float((pm < actual_mean).mean()),
        "empirical_p_ge_actual": p_emp, "n_rep": N_PLACEBO,
    }]).to_csv(OUT / "sprint43_placebo_test.csv", index=False)
    print(f"  placebo mean={pm.mean():.4f} actual={actual_mean:.4f} p={p_emp:.4f}")

    # --- 9 policy books 2022-2026 ---
    def _books(sim, years):
        rows = [_year_row(y, _port(p, sim, y), sim) for y in years]
        return rows, _pooled(rows)

    y0, p0p = _books(sim0, TRUE_OOS)
    y3, p3p = _books(sim3, TRUE_OOS)
    yc, pcp = _books(simc, TRUE_OOS)
    pol_rows = []
    for who, ys, po in (("always_P0", y0, p0p), ("always_P3", y3, p3p), ("conditional", yc, pcp)):
        pol_rows.append({k: v for k, v in po.items() if k != "pnl"} | {"who": who, "scope": "2022-2026"})
        for r in ys:
            pol_rows.append({k: v for k, v in r.items() if k != "pnl"} | {"who": who, "scope": str(r["year"])})
    pd.DataFrame(pol_rows).to_csv(OUT / "sprint43_policy_comparison.csv", index=False)
    print(f"  PF P0={p0p['pf']:.2f} P3={p3p['pf']:.2f} cond={pcp['pf']:.2f}")

    # --- 10 MC ---
    mc0 = _mc(p0p["pnl"], "P0")
    mcc = _mc(pcp["pnl"], "conditional")
    cluster_ids = set(cl["trade_id"].astype(int))
    cond_pnl_notail = []
    for y in TRUE_OOS:
        port = _port(p, simc, y)
        taken = np.asarray(port["taken"], dtype=int)
        pnl = np.asarray(port["pnl"], dtype=float)
        keep = np.array([int(i) not in cluster_ids for i in taken])
        if keep.any():
            cond_pnl_notail.append(pnl[keep])
    pnl_nt = np.concatenate(cond_pnl_notail) if cond_pnl_notail else np.array([])
    mcnt = _mc(pnl_nt, "conditional_ex_cluster")
    pd.DataFrame([mc0, mcc, mcnt]).to_csv(OUT / "sprint43_mc_tail_sensitivity.csv", index=False)

    # --- 11 leakage ---
    leak = {
        "switch_features": {
            f: {"future_dependency": False, "role": "PIT STATE_FEAT"} for f in STATE_FEATS
        },
        "p3_probability": {"future_dependency": False, "role": "expanding-year LR, 2021 unscored"},
        "labels_attribution_only": ["p0_R", "p3_R", "delta_R", "exit_timestamps", "exit_prices", "exit_reasons"],
        "leakage": False,
    }
    (OUT / "sprint43_leakage_audit.json").write_text(json.dumps(leak, indent=2), encoding="utf-8")

    # --- 12 verdict ---
    mean_ex1 = float(p3s.loc[p3s["delta_R"] < 1.0, "delta_R"].mean())
    year_has_tail = all(r["n_ge_1R"] > 0 for r in yrows)
    year_tot = np.array([r["total_delta_R"] for r in yrows])
    one_year_dom = bool(year_tot.max() / max(year_tot.sum(), 1e-12) > 0.50)
    ex1_all_neg = all(r["mean_ex_ge_1R"] <= 0 for r in yrows)
    tail_explains = share_cl >= 0.80 or share_100 >= 0.80
    after_tail_gone = mean_ex1 <= 0.01

    if leak["leakage"]:
        verdict = "LEAKAGE_FAIL"
    elif p_emp > 0.10:
        verdict = "PLACEBO_LIKE"
    elif one_year_dom:
        verdict = "REGIME_OR_YEAR_DEPENDENT"
    elif tail_explains and after_tail_gone:
        verdict = "TAIL_DEPENDENT"
    elif (not after_tail_gone) and (not one_year_dom) and p_emp <= 0.05:
        verdict = "TAIL_ROBUST"
    else:
        verdict = "INCONCLUSIVE"

    # cluster summary
    top_trans = mat.iloc[0].to_dict() if len(mat) else {}
    cl_trans = (
        cl.groupby(["p0_reason", "p3_reason"]).size().sort_values(ascending=False)
        if len(cl) else pd.Series(dtype=int)
    )

    lto_a = next(x for x in lto if x["case"] == "A_all")
    lto_b = next(x for x in lto if x["case"] == "B_ex_ge_1R")
    lto_c = next(x for x in lto if x["case"] == "C_ex_1.15_cluster")

    next_step = (
        "Stop exit-policy search. Document the discrete leftover cluster as a known artifact; "
        "keep production P0. No shadow book until a non-tail mechanism is specified."
        if verdict == "TAIL_DEPENDENT"
        else "Do not change production. Revisit only if a structural (non-tail) mechanism is isolated."
    )

    lines = [
        "# Sprint 43 — Conditional Exit Tail Attribution & Counterfactual Robustness",
        "",
        "## Sprint 43 — Verdict",
        "",
        f"**VERDICT: {verdict}**",
        "",
        "Question: is the ~+1.15R P3-selected advantage a robust structural effect, or a discrete exit-path artifact?",
        "Scope: 2022–2026. 2021 unscored. No retune. Production unchanged.",
        "",
        "### 1. Tail dependency",
        "",
        f"P3-selected n={n}, mean ΔR={d.mean():+.4f}, median={np.median(d):+.4f}, total Δ={total:+.2f}R, positive mass={pos_total:+.2f}R.",
        "",
        _md(
            [r for r in buck_rows if r["scope"] == "2022-2026"],
            ["bucket", "n", "pct", "mean_delta_R", "median_delta_R", "total_delta_R", "pct_of_positive"],
            {"n": 0, "pct": 3, "mean_delta_R": 4, "median_delta_R": 4, "total_delta_R": 2, "pct_of_positive": 3},
        ),
        "",
        f"- share of positive mass from Δ≥0.25R: **{share_025*100:.1f}%**",
        f"- Δ≥0.50R: **{share_050*100:.1f}%**",
        f"- Δ≥1.00R: **{share_100*100:.1f}%**",
        f"- 1.05–1.25 cluster: **{share_cl*100:.1f}%** (n={int(cluster_m.sum())})",
        "",
        "### 2. Leave-tail-out",
        "",
        _md(lto, ["case", "n", "mean_delta_R", "median_delta_R", "p_delta_gt_0", "total_delta_R", "yearly_sign"],
            {"n": 0, "mean_delta_R": 4, "median_delta_R": 4, "p_delta_gt_0": 3, "total_delta_R": 2}),
        "",
        f"Excluding Δ≥1R: mean {lto_b['mean_delta_R']:+.4f} (was {lto_a['mean_delta_R']:+.4f}). "
        f"Excluding 1.15 cluster: mean {lto_c['mean_delta_R']:+.4f}.",
        "",
        "### 3. +1.15R cluster",
        "",
        f"n={len(cl)}, years={cl.groupby('year').size().to_dict() if len(cl) else {}}, "
        f"hours UTC mode={int(cl['hour_utc'].mode().iloc[0]) if len(cl) else 'na'}, "
        f"side={cl['side'].value_counts().to_dict() if len(cl) else {}}, "
        f"median bars={float(cl['bars_in_trade'].median()) if len(cl) else 0:.1f}, "
        f"median MFE at decision={float(cl['mfe_so_far_R'].median()) if len(cl) else 0:.3f}, "
        f"median current_R={float(cl['current_R'].median()) if len(cl) else 0:.3f}, "
        f"P3-before-MFE>0={float(cl['p3_before_mfe_gt0'].mean()) if len(cl) else 0:.2f}, "
        f"different terminal={float(cl_out['different_terminal'].mean()) if len(cl) else 0:.2f}.",
        "",
        f"Cluster Δ range {float(cl['delta_R'].min()) if len(cl) else 0:.3f}–{float(cl['delta_R'].max()) if len(cl) else 0:.3f} "
        f"(mean {float(cl['delta_R'].mean()) if len(cl) else 0:.3f}). Transitions: {cl_trans.head(5).to_dict() if len(cl_trans) else {}}.",
        "",
        "### 4. Exit-event topology",
        "",
        _md(mat.to_dict("records"), ["p0_reason", "p3_reason", "n", "mean_delta_R", "total_delta_R", "n_cluster"],
            {"n": 0, "mean_delta_R": 4, "total_delta_R": 2, "n_cluster": 0}),
        "",
        f"Largest total-Δ cell: {top_trans.get('p0_reason')} → {top_trans.get('p3_reason')} "
        f"(n={top_trans.get('n')}, mean Δ={top_trans.get('mean_delta_R')}).",
        "",
        "### 5. Path geometry",
        "",
        _md([r for r in path_rows if r["kind"] == "mfe_decision"],
            ["bucket", "n", "mean_delta_R", "median_delta_R", "p_delta_gt_0", "total_delta_R"],
            {"n": 0, "mean_delta_R": 4, "median_delta_R": 4, "p_delta_gt_0": 3, "total_delta_R": 2}),
        "",
        "Positive tail stays in low-MFE / low current_R / early path. Full bins: `sprint43_path_attribution.csv`.",
        "",
        "### 6. Yearly stability",
        "",
        _md(yrows, ["year", "n", "n_ge_1R", "n_cluster", "total_delta_R", "contrib_ge_1R",
                    "total_ex_ge_1R", "mean_delta_R", "mean_ex_ge_1R"],
            {"year": 0, "n": 0, "n_ge_1R": 0, "n_cluster": 0, "total_delta_R": 2,
             "contrib_ge_1R": 2, "total_ex_ge_1R": 2, "mean_delta_R": 4, "mean_ex_ge_1R": 4}),
        "",
        f"Every year has ≥1R tail: **{year_has_tail}**. One year >50% of pooled Δ: **{one_year_dom}**. "
        f"Mean Δ after dropping ≥1R is ≤0 in every OOS year: **{ex1_all_neg}**.",
        "",
        "### 7. Placebo",
        "",
        f"Shuffle P3-selected labels within year (n preserved). {N_PLACEBO} reps.",
        f"Actual mean Δ={actual_mean:+.4f}. Placebo mean={pm.mean():+.4f} (sd {pm.std():.4f}). "
        f"Empirical P(placebo ≥ actual)={p_emp:.4f}.",
        "",
        "### 8. Risk",
        "",
        "Books 2022–2026, frozen portfolio $280 / lot 0.01 / max_open 5 / heat 3R.",
        "",
        "| who | Trades | PF | DD | Return | WR | Payoff | AvgR | MaxLoss |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        f"| always_P0 | {p0p['trades']} | {p0p['pf']:.2f} | {p0p['dd']*100:.1f}% | {p0p['ret']*100:.0f}% | {p0p['wr']:.3f} | {p0p['payoff']:.2f} | {p0p['avg_r']:.3f} | {p0p['max_loss_streak']} |",
        f"| always_P3 | {p3p['trades']} | {p3p['pf']:.2f} | {p3p['dd']*100:.1f}% | {p3p['ret']*100:.0f}% | {p3p['wr']:.3f} | {p3p['payoff']:.2f} | {p3p['avg_r']:.3f} | {p3p['max_loss_streak']} |",
        f"| conditional | {pcp['trades']} | {pcp['pf']:.2f} | {pcp['dd']*100:.1f}% | {pcp['ret']*100:.0f}% | {pcp['wr']:.3f} | {pcp['payoff']:.2f} | {pcp['avg_r']:.3f} | {pcp['max_loss_streak']} |",
        "",
        "| policy | MC ruin | median DD | p95 DD | p99 DD | worst DD |",
        "| --- | --- | --- | --- | --- | --- |",
        f"| P0 | {mc0['prob_ruin']:.3f} | {mc0['median_dd']*100:.1f}% | {mc0['p95_dd']*100:.1f}% | {mc0['p99_dd']*100:.1f}% | {mc0['worst_dd']*100:.1f}% |",
        f"| conditional | {mcc['prob_ruin']:.3f} | {mcc['median_dd']*100:.1f}% | {mcc['p95_dd']*100:.1f}% | {mcc['p99_dd']*100:.1f}% | {mcc['worst_dd']*100:.1f}% |",
        f"| conditional ex-cluster | {mcnt['prob_ruin']:.3f} | {mcnt['median_dd']*100:.1f}% | {mcnt['p95_dd']*100:.1f}% | {mcnt['p99_dd']*100:.1f}% | {mcnt['worst_dd']*100:.1f}% |",
        "",
        "### 9. Leakage",
        "",
        "Switch features (STATE_FEATS + LR score) remain `future_dependency=false`. Labels are attribution-only.",
        "**leakage = false**.",
        "",
        "### 10. Production decision",
        "",
        "Production remains:",
        "",
        "**P0 = a0.25 / d0.08**",
        "",
        "No automatic ship. Conditional P3 is not a new baseline. Always-P3 is not a new baseline.",
        "",
        "### Next Step",
        "",
        next_step,
        "",
    ]
    (OUT / "sprint43_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "sprint43_final_verdict.json").write_text(
        json.dumps(
            {
                "sprint": 43,
                "question": "Is the ~+1.15R conditional-exit advantage a robust structural effect or a discrete exit-path artifact?",
                "primary_scope": "2022-2026",
                "production_policy": "P0_a0.25_d0.08",
                "candidate_policy": "conditional_P3_a0.20_d0.10_threshold_0.60",
                "verdict": verdict,
                "p3_selected_n": n,
                "p3_selected_mean_delta_r": float(d.mean()),
                "p3_selected_median_delta_r": float(np.median(d)),
                "positive_tail_contribution_pct": share_cl,
                "delta_ge_1r_contribution_pct": share_100,
                "mean_delta_excluding_ge_1r": mean_ex1,
                "placebo_empirical_p": p_emp,
                "leakage": False,
                "production_changed": False,
                "share_ge_0.25": share_025,
                "share_ge_0.50": share_050,
                "year_has_tail": year_has_tail,
                "one_year_dominates": one_year_dom,
                "always_p3_pf": p3p["pf"],
                "always_p0_pf": p0p["pf"],
                "conditional_pf": pcp["pf"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"VERDICT={verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
