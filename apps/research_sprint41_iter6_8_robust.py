"""Sprint 41 Iter 6-8 — threshold neighborhood / switching stability / confirm.

Frozen: LR p3_better, P0 a0.25/d0.08, P3 a0.20/d0.10, portfolio $280 lot 0.01.
Threshold 0.60 is the pre-chosen candidate. Neighborhood is 0.55 / 0.60 / 0.65 only.
No extra grid. No production change.

  python apps/research_sprint41_iter6_8_robust.py
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

from apps.report_adaptive_vol_risk import iter6_montecarlo
from apps.research_sprint40_iter1_time_aware_exit import STARTING
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
from apps.run_exit_engine_grid import Paths, simulate_combo
from apps.research_sprint39_exit_state import _join_feat7, _load_panel
from simulation.wf.sim import load_h1, prepare_market

ITER1 = _ROOT / "artifacts/pipeline_backtest/exit_structure/sprint41/iter1_differential"
BASE = _ROOT / "artifacts/pipeline_backtest/exit_structure/sprint41"
OUT6 = BASE / "iter6_neighborhood"
OUT7 = BASE / "iter7_switching"
OUT8 = BASE / "iter8_confirm"
THRESHOLDS = (0.55, 0.60, 0.65)
CANDIDATE = 0.60
TRUE_OOS = (2022, 2023, 2024, 2025, 2026)
MFE_BUCKETS = (
    ("lt_0.25R", lambda x: x < 0.25),
    ("0.25_0.50R", lambda x: (x >= 0.25) & (x < 0.50)),
    ("0.50_1.00R", lambda x: (x >= 0.50) & (x < 1.00)),
    ("ge_1.00R", lambda x: x >= 1.00),
)


def _pf(pnl: np.ndarray) -> float:
    gp, gl = float(pnl[pnl > 0].sum()), float(-pnl[pnl < 0].sum())
    return gp / gl if gl > 0 else 0.0


def _pooled(rows: list[dict]) -> dict:
    pnl = np.concatenate([r["pnl"] for r in rows if len(r["pnl"])])
    w = [x["trades"] for x in rows]
    avg_r = float(np.average([x["avg_r"] for x in rows], weights=w))
    wr = float(np.average([x["wr"] for x in rows], weights=w))
    payoff = float(np.average([x["payoff"] for x in rows], weights=w))
    return {
        "trades": int(sum(x["trades"] for x in rows)),
        "pf": _pf(pnl),
        "dd": float(max(x["dd"] for x in rows)),
        "ret": float(np.mean([x["ret"] for x in rows])),
        "wr": wr,
        "payoff": payoff,
        "avg_r": avg_r,
        "median_r": float(np.mean([x["median_r"] for x in rows])),
        "max_loss_streak": int(max(x["max_loss_streak"] for x in rows)),
        "blown": any(x["blown"] for x in rows),
        "pnl": pnl,
    }


def _cell(b: dict, s: dict, year: int) -> str:
    if year == 2021:
        return "FLAT"
    dpf, ddd, dret = s["pf"] - b["pf"], s["dd"] - b["dd"], s["ret"] - b["ret"]
    if abs(dpf) < 0.02 and abs(ddd) < 0.005 and abs(dret) < 0.05:
        return "FLAT"
    if dpf >= -0.02 and ddd <= 0.01 and dret >= -0.05:
        return "WIN"
    return "LOSS"


def _use_matrix(n: int, ds: pd.DataFrame, pred: np.ndarray, thr: float) -> np.ndarray:
    use = np.zeros((n, CAP + 1), dtype=bool)
    tid = ds["trade_id"].to_numpy(dtype=int)
    j = ds["bars_in_trade"].to_numpy(dtype=int)
    ok = np.isfinite(pred) & (pred >= thr)
    for k in np.where(ok)[0]:
        ti, jj = int(tid[k]), int(j[k])
        if 0 <= ti < n and 1 <= jj <= CAP:
            use[ti, jj] = True
    return use


def _collapse(b: dict, s: dict) -> bool:
    if s["blown"]:
        return True
    if s["dd"] > 0.45 or s["dd"] > b["dd"] + 0.15:
        return True
    if s["wr"] < b["wr"] - 0.10:
        return True
    if s["trades"] < 0.80 * max(b["trades"], 1):
        return True
    if s["pf"] < 1.50 and b["pf"] >= 1.90:
        return True
    return False


def _material_ok(base_y: list, sw_y: list) -> bool:
    """Neighbor does not materially destroy 2022-2026 edge."""
    cells = [_cell(b, s, int(b["year"])) for b, s in zip(base_y, sw_y)]
    oos = [c for b, c in zip(base_y, cells) if int(b["year"]) in TRUE_OOS]
    if any(_collapse(b, s) for b, s in zip(base_y, sw_y) if int(b["year"]) in TRUE_OOS):
        return False
    if oos.count("LOSS") >= 3:
        return False
    bp, sp = _pooled(base_y), _pooled(sw_y)
    if sp["pf"] < bp["pf"] - 0.05:
        return False
    if sp["dd"] > bp["dd"] + 0.03:
        return False
    return True


def main() -> int:
    OUT6.mkdir(parents=True, exist_ok=True)
    OUT7.mkdir(parents=True, exist_ok=True)
    OUT8.mkdir(parents=True, exist_ok=True)

    ds = pd.read_parquet(ITER1 / "exit_differential_dataset.parquet")
    pred = _bar_scores(ds)
    assert not np.any(np.isfinite(pred) & (ds["year"].to_numpy() == 2021)), "2021 must stay unscored"

    panel = _join_feat7(_load_panel())
    p = Paths(panel, prepare_market(load_h1(_ROOT / "artifacts/raw/XAUUSD/H1/data.parquet")))
    sim0 = simulate_combo(p, act=P0_ACT, dist=P0_DIST, tp=None, partials=(), tmax=None, be=None)
    base_y = [_year_row(y, _port(p, sim0, y), sim0) for y in YEARS]
    base_p = _pooled(base_y)

    yearly_rows = []
    pooled_rows = []
    robustness = []
    by_thr: dict[float, dict] = {}

    for thr in THRESHOLDS:
        use = _use_matrix(p.n, ds, pred, thr)
        sim = simulate_switch(p, use)
        sw_y = [_year_row(y, _port(p, sim, y), sim) for y in YEARS]
        sw_p = _pooled(sw_y)
        cells = {int(b["year"]): _cell(b, s, int(b["year"])) for b, s in zip(base_y, sw_y)}
        n_sw_trade = int((use.any(axis=1)).sum())
        pct_sw = n_sw_trade / max(p.n, 1)
        by_thr[thr] = dict(use=use, sim=sim, sw_y=sw_y, sw_p=sw_p, cells=cells, n_sw_trade=n_sw_trade)
        print(
            f"thr={thr:.2f} bars={int(use.sum())} trades_sw={n_sw_trade} "
            f"PF={sw_p['pf']:.2f} DD={sw_p['dd']*100:.1f}% ret={sw_p['ret']*100:.0f}%"
        )
        for b, s in zip(base_y, sw_y):
            yearly_rows.append(
                {k: v for k, v in s.items() if k != "pnl"}
                | {
                    "who": f"p3_{thr:.2f}",
                    "threshold": thr,
                    "cell": cells[int(s["year"])],
                    "d_pf": s["pf"] - b["pf"],
                    "d_dd": s["dd"] - b["dd"],
                    "d_ret": s["ret"] - b["ret"],
                    "d_avg_r": s["avg_r"] - b["avg_r"],
                }
            )
        pooled_rows.append(
            {k: v for k, v in sw_p.items() if k != "pnl"}
            | {
                "who": f"p3_{thr:.2f}",
                "threshold": thr,
                "d_pf": sw_p["pf"] - base_p["pf"],
                "d_dd": sw_p["dd"] - base_p["dd"],
                "d_ret": sw_p["ret"] - base_p["ret"],
                "d_avg_r": sw_p["avg_r"] - base_p["avg_r"],
                "n_switched_trades": n_sw_trade,
                "pct_trades_switched": pct_sw,
            }
        )
        robustness.append(
            {
                "threshold": thr,
                "years_beating": sum(1 for y in TRUE_OOS if cells[y] == "WIN"),
                "years_flat": sum(1 for y in TRUE_OOS if cells[y] == "FLAT"),
                "years_loss": sum(1 for y in TRUE_OOS if cells[y] == "LOSS"),
                "pooled_delta_PF": sw_p["pf"] - base_p["pf"],
                "pooled_delta_DD": sw_p["dd"] - base_p["dd"],
                "pooled_delta_return": sw_p["ret"] - base_p["ret"],
                **{str(y): cells[y] for y in YEARS},
            }
        )

    # prod pooled/yearly once
    for b in base_y:
        yearly_rows.append({k: v for k, v in b.items() if k != "pnl"} | {"who": "prod", "threshold": None, "cell": "—"})
    pooled_rows.append(
        {k: v for k, v in base_p.items() if k != "pnl"}
        | {"who": "prod", "threshold": None, "d_pf": 0.0, "d_dd": 0.0, "d_ret": 0.0, "d_avg_r": 0.0,
           "n_switched_trades": 0, "pct_trades_switched": 0.0}
    )

    rob = pd.DataFrame(robustness)
    # drop duplicate 2021 prod rows from the accidental append in loop
    ydf = pd.DataFrame(yearly_rows)
    pdf = pd.DataFrame(pooled_rows)
    rob.to_csv(BASE / "sprint41_threshold_robustness.csv", index=False)
    ydf.to_csv(BASE / "sprint41_yearly_comparison.csv", index=False)
    pdf.to_csv(BASE / "sprint41_pooled_comparison.csv", index=False)

    c60 = by_thr[CANDIDATE]
    c60_ok = all(c60["cells"][y] != "LOSS" for y in TRUE_OOS) and (
        c60["sw_p"]["pf"] >= base_p["pf"] - 1e-9
        and c60["sw_p"]["dd"] <= base_p["dd"] + 0.01
        and c60["sw_p"]["ret"] >= base_p["ret"] - 1e-9
        and not c60["sw_p"]["blown"]
    )
    neighbors_ok = any(_material_ok(base_y, by_thr[t]["sw_y"]) for t in (0.55, 0.65))
    any_blow = any(by_thr[t]["sw_p"]["blown"] for t in THRESHOLDS)
    any_cat = any(_collapse(b, s) for t in THRESHOLDS for b, s in zip(base_y, by_thr[t]["sw_y"]))
    only_one_thr = c60_ok and not neighbors_ok

    if (not c60_ok) or only_one_thr or any_blow or any_cat:
        v6 = "FRAGILE_THRESHOLD"
    else:
        v6 = "THRESHOLD_ROBUST"

    lines6 = [
        "# Sprint 41 Iter 6 — Threshold neighborhood",
        "",
        "Frozen LR `p3_better`. Candidate threshold **0.60** (validation, not re-picked).",
        "Neighborhood only: 0.55 / 0.60 / 0.65. Production `a0.25/d0.08`. 2021 unscored = FLAT.",
        "",
        "| threshold | years_beating | pooled_delta_PF | pooled_delta_DD | pooled_delta_return |",
        "|---:|---:|---:|---:|---:|",
    ]
    for r in robustness:
        lines6.append(
            f"| {r['threshold']:.2f} | {r['years_beating']}/5 | {r['pooled_delta_PF']:+.3f} | "
            f"{r['pooled_delta_DD']*100:+.1f}pp | {r['pooled_delta_return']*100:+.0f}pp |"
        )
    lines6 += [
        "",
        "| threshold | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |",
        "|---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for r in robustness:
        lines6.append(
            f"| {r['threshold']:.2f} | {r['2021']} | {r['2022']} | {r['2023']} | {r['2024']} | {r['2025']} | {r['2026']} |"
        )
    lines6 += [
        "",
        "## Yearly vs production",
        "",
        "| thr | Year | Trades | WR | Payoff | PF | DD | Return | AvgR | MedR | MaxLoss | cell |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for b in base_y:
        lines6.append(
            f"| prod | {b['year']} | {b['trades']} | {b['wr']:.3f} | {b['payoff']:.2f} | {b['pf']:.2f} | "
            f"{b['dd']*100:.1f}% | {b['ret']*100:.0f}% | {b['avg_r']:.3f} | {b['median_r']:.3f} | {b['max_loss_streak']} | — |"
        )
    for thr in THRESHOLDS:
        for s in by_thr[thr]["sw_y"]:
            cell = by_thr[thr]["cells"][int(s["year"])]
            lines6.append(
                f"| {thr:.2f} | {s['year']} | {s['trades']} | {s['wr']:.3f} | {s['payoff']:.2f} | {s['pf']:.2f} | "
                f"{s['dd']*100:.1f}% | {s['ret']*100:.0f}% | {s['avg_r']:.3f} | {s['median_r']:.3f} | {s['max_loss_streak']} | {cell} |"
            )
    lines6 += [
        "",
        f"0.60 still beats production: {c60_ok}",
        f"neighbor 0.55 or 0.65 material-ok: {neighbors_ok}",
        f"blow-up: {any_blow}  collapse: {any_cat}  only-0.60: {only_one_thr}",
        "",
        f"**Verdict: {v6}**",
        "",
        "Production unchanged.",
    ]
    (OUT6 / "summary.md").write_text("\n".join(lines6) + "\n", encoding="utf-8")
    print(f"ITER6 {v6}")
    if v6 != "THRESHOLD_ROBUST":
        (BASE / "sprint41_final_verdict.json").write_text(
            json.dumps({"iter": 6, "verdict": v6, "production_exit": "a0.25_d0.08 unchanged"}, indent=2),
            encoding="utf-8",
        )
        print("STOP")
        return 0

    # --- Iter 7 ---
    stab_rows = []
    mfe = ds["mfe_so_far_R"].to_numpy(dtype=float)
    cur = ds["current_R"].to_numpy(dtype=float)
    mom = ds["mom_1R"].to_numpy(dtype=float)
    years_ds = ds["year"].to_numpy()
    tid = ds["trade_id"].to_numpy(dtype=int)
    pathological = False
    mono_ok = True
    for thr in THRESHOLDS:
        use = by_thr[thr]["use"]
        sw_bar = np.isfinite(pred) & (pred >= thr)
        n_bar = int(sw_bar.sum())
        per_trade = pd.Series(tid[sw_bar]).value_counts() if n_bar else pd.Series(dtype=int)
        ever = set(per_trade.index.tolist())
        n_ever = len(ever)
        scored = np.isfinite(pred)
        frac_scored = float(sw_bar[scored].mean()) if scored.any() else 0.0
        bucket_frac = []
        for name, fn in MFE_BUCKETS:
            m = fn(mfe) & scored
            fr = float(sw_bar[m].mean()) if m.any() else 0.0
            bucket_frac.append(fr)
            stab_rows.append(
                {
                    "threshold": thr,
                    "kind": "mfe_bucket",
                    "bucket": name,
                    "n_obs": int(m.sum()),
                    "p3_frac": fr,
                }
            )
        # monotonic: usage should fall as MFE rises
        if not all(bucket_frac[i] + 0.03 >= bucket_frac[i + 1] for i in range(len(bucket_frac) - 1)):
            if thr == CANDIDATE:
                mono_ok = False
        sp_mfe = float(pd.Series(mfe[scored]).corr(pd.Series(sw_bar[scored].astype(float)), method="spearman") or 0.0)
        sp_cur = float(pd.Series(cur[scored]).corr(pd.Series(sw_bar[scored].astype(float)), method="spearman") or 0.0)
        sp_mom = float(pd.Series(mom[scored]).corr(pd.Series(sw_bar[scored].astype(float)), method="spearman") or 0.0)
        if thr == CANDIDATE and (frac_scored > 0.50 or sp_mfe >= 0 or bucket_frac[-1] > bucket_frac[0]):
            pathological = True
        year_frac = {}
        for y in YEARS:
            m = (years_ds == y) & scored
            year_frac[y] = float(sw_bar[m].mean()) if m.any() else 0.0
            stab_rows.append(
                {
                    "threshold": thr,
                    "kind": "year",
                    "bucket": str(y),
                    "n_obs": int((years_ds == y).sum()),
                    "p3_frac": year_frac[y],
                }
            )
        stab_rows.append(
            {
                "threshold": thr,
                "kind": "summary",
                "bucket": "all",
                "n_obs": int(len(ds)),
                "p3_frac": float(sw_bar.mean()),
                "total_switch_bars": n_bar,
                "n_trades_ever_p3": n_ever,
                "pct_trades_ever_p3": n_ever / max(p.n, 1),
                "avg_p3_bars": float(per_trade.mean()) if len(per_trade) else 0.0,
                "median_p3_bars": float(per_trade.median()) if len(per_trade) else 0.0,
                "frac_scored_bars": frac_scored,
                "spearman_mfe": sp_mfe,
                "spearman_current_R": sp_cur,
                "spearman_mom_1R": sp_mom,
                "mfe_lt025": bucket_frac[0],
                "mfe_025_050": bucket_frac[1],
                "mfe_050_100": bucket_frac[2],
                "mfe_ge100": bucket_frac[3],
            }
        )
        print(
            f"  switch thr={thr:.2f} bars={n_bar} ever={n_ever} frac_scored={frac_scored:.2f} "
            f"sp_mfe={sp_mfe:.2f} buckets={[round(x, 2) for x in bucket_frac]}"
        )

    sdf = pd.DataFrame(stab_rows)
    sdf.to_csv(BASE / "sprint41_switching_stability.csv", index=False)
    # year P3 usage 2022-2026 shouldn't be one year 10x another
    cand_year = [r for r in stab_rows if r["threshold"] == CANDIDATE and r["kind"] == "year" and r["bucket"] != "2021"]
    frs = [r["p3_frac"] for r in cand_year if r["p3_frac"] > 0]
    year_wild = bool(frs) and (max(frs) > 8 * max(min(frs), 1e-6))
    v7 = "SWITCHING_UNSTABLE" if (pathological or not mono_ok or year_wild) else "SWITCHING_STABLE"

    def _sumrow(thr):
        return next(r for r in stab_rows if r["threshold"] == thr and r["kind"] == "summary")

    lines7 = [
        "# Sprint 41 Iter 7 — Switching intensity / stability",
        "",
        "Same frozen switch rule. No threshold change.",
        "",
        "| thr | switch bars | % trades ever P3 | avg P3 bars | median P3 bars | % scored bars | Spearman MFE |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for thr in THRESHOLDS:
        r = _sumrow(thr)
        lines7.append(
            f"| {thr:.2f} | {int(r['total_switch_bars'])} | {r['pct_trades_ever_p3']*100:.1f}% | "
            f"{r['avg_p3_bars']:.2f} | {r['median_p3_bars']:.1f} | {r['frac_scored_bars']*100:.1f}% | {r['spearman_mfe']:.2f} |"
        )
    lines7 += [
        "",
        "### P3 usage by MFE bucket (scored bars)",
        "",
        "| thr | <0.25R | 0.25-0.50R | 0.50-1.00R | >=1.00R |",
        "|---:|---:|---:|---:|---:|",
    ]
    for thr in THRESHOLDS:
        r = _sumrow(thr)
        lines7.append(
            f"| {thr:.2f} | {r['mfe_lt025']*100:.1f}% | {r['mfe_025_050']*100:.1f}% | "
            f"{r['mfe_050_100']*100:.1f}% | {r['mfe_ge100']*100:.1f}% |"
        )
    lines7 += [
        "",
        "### P3 usage by year (scored bars; 2021 = 0)",
        "",
        "| thr | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for thr in THRESHOLDS:
        fr = {r["bucket"]: r["p3_frac"] for r in stab_rows if r["threshold"] == thr and r["kind"] == "year"}
        lines7.append(
            f"| {thr:.2f} | {fr['2021']*100:.1f}% | {fr['2022']*100:.1f}% | {fr['2023']*100:.1f}% | "
            f"{fr['2024']*100:.1f}% | {fr['2025']*100:.1f}% | {fr['2026']*100:.1f}% |"
        )
    lines7 += [
        "",
        f"monotonic MFE (0.60): {mono_ok}  pathological: {pathological}  year-wild: {year_wild}",
        "",
        f"**Verdict: {v7}**",
        "",
        "Expected: LOW MFE / LOW current_R → P3; HIGH MFE → production.",
    ]
    (OUT7 / "summary.md").write_text("\n".join(lines7) + "\n", encoding="utf-8")
    print(f"ITER7 {v7}")
    if v7 != "SWITCHING_STABLE":
        (BASE / "sprint41_final_verdict.json").write_text(
            json.dumps({"iter": 7, "verdict": v7, "production_exit": "a0.25_d0.08 unchanged"}, indent=2),
            encoding="utf-8",
        )
        print("STOP")
        return 0

    # --- Iter 8 ---
    sw_y, sw_p = c60["sw_y"], c60["sw_p"]
    mc = iter6_montecarlo(base_p["pnl"], sw_p["pnl"], any(r["blown"] for r in base_y), starting=STARTING)
    dyn, fx = mc[mc["policy"] == "dynamic"], mc[mc["policy"] == "fixed"]
    mc_ruin = float(dyn.iloc[0]["prob_ruin"]) if len(dyn) else 1.0
    mc_wdd = float(dyn.iloc[0]["worst_dd"]) if len(dyn) else 1.0
    mc0_ruin = float(fx.iloc[0]["prob_ruin"]) if len(fx) else 1.0
    mc0_wdd = float(fx.iloc[0]["worst_dd"]) if len(fx) else 1.0

    cells = c60["cells"]
    oos_ok = sum(1 for y in TRUE_OOS if cells[y] != "LOSS")
    n_win = sum(1 for y in YEARS if cells[y] == "WIN")
    n_flat = sum(1 for y in YEARS if cells[y] == "FLAT")
    n_loss = sum(1 for y in YEARS if cells[y] == "LOSS")
    n_oos_win = sum(1 for y in TRUE_OOS if cells[y] == "WIN")
    one_year = n_oos_win <= 1
    bar = (
        sw_p["pf"] >= base_p["pf"] - 1e-9
        and sw_p["dd"] <= base_p["dd"] + 1e-12
        and sw_p["ret"] >= base_p["ret"] - 1e-9
        and mc_ruin <= mc0_ruin + 1e-12
        and not sw_p["blown"]
        and oos_ok >= 4
        and not any(_collapse(b, s) for b, s in zip(base_y, sw_y))
        and v6 == "THRESHOLD_ROBUST"
        and v7 == "SWITCHING_STABLE"
        and not one_year
    )
    if bar:
        v8 = "CANDIDATE_FOR_SHADOW"
    elif n_loss >= 2:
        v8 = "NO_ROBUST_EDGE"
    else:
        v8 = "KEEP_PRODUCTION"

    lines8 = [
        "# Sprint 41 Iter 8 — Final candidate confirmation",
        "",
        "Candidate = CONDITIONAL_P3 threshold **0.60** (validation-chosen, not Iter 6 best).",
        "vs PRODUCTION a0.25/d0.08. Frozen entry/risk/portfolio. No tuning.",
        "",
        "## Yearly",
        "",
        "| Year | who | Trades | PF | DD | Return | WR | Payoff | AvgR | MaxLoss | cell |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for b, s in zip(base_y, sw_y):
        for who, r in (("prod", b), ("switch_0.60", s)):
            cell = cells[int(r["year"])] if who != "prod" else "—"
            lines8.append(
                f"| {r['year']} | {who} | {r['trades']} | {r['pf']:.2f} | {r['dd']*100:.1f}% | "
                f"{r['ret']*100:.0f}% | {r['wr']:.3f} | {r['payoff']:.2f} | {r['avg_r']:.3f} | "
                f"{r['max_loss_streak']} | {cell} |"
            )
    lines8 += [
        "",
        "## Pooled (isolated $280/year; DD = max yearly; Return = mean yearly)",
        "",
        "| who | Trades | PF | DD | Return | WR | Payoff | AvgR | MaxLoss |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| prod | {base_p['trades']} | {base_p['pf']:.2f} | {base_p['dd']*100:.1f}% | {base_p['ret']*100:.0f}% | "
        f"{base_p['wr']:.3f} | {base_p['payoff']:.2f} | {base_p['avg_r']:.3f} | {base_p['max_loss_streak']} |",
        f"| switch_0.60 | {sw_p['trades']} | {sw_p['pf']:.2f} | {sw_p['dd']*100:.1f}% | {sw_p['ret']*100:.0f}% | "
        f"{sw_p['wr']:.3f} | {sw_p['payoff']:.2f} | {sw_p['avg_r']:.3f} | {sw_p['max_loss_streak']} |",
        "",
        "## Risk",
        "",
        f"- MC ruin: switch {mc_ruin:.3f} vs prod {mc0_ruin:.3f}",
        f"- MC worst DD: switch {mc_wdd*100:.1f}% vs prod {mc0_wdd*100:.1f}%",
        "- max drawdown duration: n/a (not in portfolio engine)",
        "",
        "## Consistency",
        "",
        f"- years better: {n_win}",
        f"- years equal: {n_flat}",
        f"- years worse: {n_loss}",
        f"- true OOS 2022-2026 not worse: {oos_ok}/5 (WIN {n_oos_win})",
        "",
        f"**Verdict: {v8}**",
        "",
        "Production exit unchanged (`a0.25/d0.08`). Not shipped. Not live.",
    ]
    (OUT8 / "summary.md").write_text("\n".join(lines8) + "\n", encoding="utf-8")
    verdict = {
        "iter": 8,
        "verdict": v8,
        "iter6": v6,
        "iter7": v7,
        "threshold_candidate": CANDIDATE,
        "years": cells,
        "years_better": n_win,
        "years_equal": n_flat,
        "years_worse": n_loss,
        "true_oos_not_worse": oos_ok,
        "pooled_prod": {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in base_p.items() if k != "pnl"},
        "pooled_switch": {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in sw_p.items() if k != "pnl"},
        "mc_ruin_switch": mc_ruin,
        "mc_ruin_prod": mc0_ruin,
        "mc_worst_dd_switch": mc_wdd,
        "mc_worst_dd_prod": mc0_wdd,
        "production_exit": "a0.25_d0.08 unchanged",
    }
    (BASE / "sprint41_final_verdict.json").write_text(json.dumps(verdict, indent=2, default=float), encoding="utf-8")
    print(f"ITER8 {v8} oos_ok={oos_ok}/5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
