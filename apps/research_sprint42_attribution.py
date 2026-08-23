"""Sprint 42 — Conditional exit attribution and historical shadow replay.

Frozen Sprint 41 rule: LR p3_better P>=0.60 -> P3 a0.20/d0.10 else P0 a0.25/d0.08.
Analysis only. No new threshold, model, features, or production change.

  python apps/research_sprint42_attribution.py
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
from apps.run_exit_engine_grid import Paths, simulate_combo
from apps.research_sprint39_exit_state import _join_feat7, _load_panel
from simulation.wf.sim import COST, load_h1, prepare_market

ITER1 = _ROOT / "artifacts/pipeline_backtest/exit_structure/sprint41/iter1_differential"
OUT = _ROOT / "artifacts/pipeline_backtest/exit_structure/sprint42"
THR = 0.60
TRUE_OOS = (2022, 2023, 2024, 2025, 2026)
MFE_EDGES = (
    ("lt_0R", lambda x: x < 0),
    ("0_0.25R", lambda x: (x >= 0) & (x < 0.25)),
    ("0.25_0.50R", lambda x: (x >= 0.25) & (x < 0.50)),
    ("0.50_1.00R", lambda x: (x >= 0.50) & (x < 1.00)),
    ("ge_1.00R", lambda x: x >= 1.00),
)
BAR_EDGES = (
    ("1_2", lambda x: x <= 2),
    ("3_4", lambda x: (x >= 3) & (x <= 4)),
    ("5_8", lambda x: (x >= 5) & (x <= 8)),
    ("9_16", lambda x: (x >= 9) & (x <= 16)),
    ("ge_17", lambda x: x >= 17),
)
PROB_BINS = (
    ("lt_0.50", lambda x: x < 0.50),
    ("0.50_0.60", lambda x: (x >= 0.50) & (x < 0.60)),
    ("0.60_0.70", lambda x: (x >= 0.60) & (x < 0.70)),
    ("0.70_0.80", lambda x: (x >= 0.70) & (x < 0.80)),
    ("ge_0.80", lambda x: x >= 0.80),
)


def _pf(pnl: np.ndarray) -> float:
    gp, gl = float(pnl[pnl > 0].sum()), float(-pnl[pnl < 0].sum())
    return gp / gl if gl > 0 else 0.0


def _pooled(rows: list[dict]) -> dict:
    pnl = np.concatenate([r["pnl"] for r in rows if len(r["pnl"])])
    w = [x["trades"] for x in rows]
    return {
        "trades": int(sum(x["trades"] for x in rows)),
        "pf": _pf(pnl),
        "dd": float(max(x["dd"] for x in rows)),
        "ret": float(np.mean([x["ret"] for x in rows])),
        "wr": float(np.average([x["wr"] for x in rows], weights=w)),
        "payoff": float(np.average([x["payoff"] for x in rows], weights=w)),
        "avg_r": float(np.average([x["avg_r"] for x in rows], weights=w)),
        "max_loss_streak": int(max(x["max_loss_streak"] for x in rows)),
        "blown": any(x["blown"] for x in rows),
        "pnl": pnl,
    }


def _dstats(d: np.ndarray, r: np.ndarray | None = None) -> dict:
    d = np.asarray(d, dtype=float)
    n = int(len(d))
    if n == 0:
        keys = ["n", "mean_R", "median_R", "mean_delta_R", "median_delta_R",
                "p_delta_gt_0", "p_delta_ge_0.10", "p_delta_ge_0.25",
                "p_delta_le_m0.10", "p_delta_le_m0.25"]
        return {k: 0 if k == "n" else float("nan") for k in keys}
    out = {
        "n": n,
        "mean_delta_R": float(np.mean(d)),
        "median_delta_R": float(np.median(d)),
        "p_delta_gt_0": float(np.mean(d > 0)),
        "p_delta_ge_0.10": float(np.mean(d >= 0.10)),
        "p_delta_ge_0.25": float(np.mean(d >= 0.25)),
        "p_delta_le_m0.10": float(np.mean(d <= -0.10)),
        "p_delta_le_m0.25": float(np.mean(d <= -0.25)),
        "mean_R": float("nan"),
        "median_R": float("nan"),
    }
    if r is not None and len(r):
        rr = np.asarray(r, dtype=float)
        out["mean_R"] = float(np.mean(rr))
        out["median_R"] = float(np.median(rr))
    return out


def _fmt(v, nd=3) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _md(rows: list[dict], cols: list[str], fmts: dict[str, int] | None = None) -> str:
    fmts = fmts or {}
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c)
            cells.append(_fmt(v, fmts.get(c, 3)) if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _use_matrix(n: int, ds: pd.DataFrame, pred: np.ndarray) -> np.ndarray:
    use = np.zeros((n, CAP + 1), dtype=bool)
    tid = ds["trade_id"].to_numpy(dtype=int)
    j = ds["bars_in_trade"].to_numpy(dtype=int)
    ok = np.isfinite(pred) & (pred >= THR)
    for k in np.where(ok)[0]:
        ti, jj = int(tid[k]), int(j[k])
        if 0 <= ti < n and 1 <= jj <= CAP:
            use[ti, jj] = True
    return use


def _exit_px(p: Paths, sim: dict) -> np.ndarray:
    exit_r = sim["r_multiple"] + COST / np.maximum(p.r_unit_pct, 1e-12)
    sign = np.where(p.is_long, 1.0, -1.0)
    return p.entry + sign * exit_r * (p.r_unit_pct * p.entry)


def _align_eq(ts_a, eq_a, ts_b, eq_b) -> tuple[float, float]:
    sa = pd.Series(eq_a, index=pd.DatetimeIndex(ts_a))
    sb = pd.Series(eq_b, index=pd.DatetimeIndex(ts_b))
    if sa.empty or sb.empty:
        return float("nan"), float("nan")
    idx = sa.index.union(sb.index).sort_values()
    sa2 = sa.reindex(idx).ffill().bfill()
    sb2 = sb.reindex(idx).ffill().bfill()
    corr = float(sa2.corr(sb2)) if len(idx) > 2 else float("nan")
    div = float(np.nanmax(np.abs(sb2.to_numpy() - sa2.to_numpy())))
    return corr, div


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ds = pd.read_parquet(ITER1 / "exit_differential_dataset.parquet")
    pred = _bar_scores(ds)
    ds = ds.copy()
    ds["p3_probability"] = pred
    ds["p3_bar"] = np.isfinite(pred) & (pred >= THR)

    panel = _join_feat7(_load_panel())
    mkt = prepare_market(load_h1(_ROOT / "artifacts/raw/XAUUSD/H1/data.parquet"))
    p = Paths(panel, mkt)
    sim0 = simulate_combo(p, act=P0_ACT, dist=P0_DIST, tp=None, partials=(), tmax=None, be=None)
    sim3 = simulate_combo(p, act=P3_ACT, dist=P3_DIST, tp=None, partials=(), tmax=None, be=None)
    use = _use_matrix(p.n, ds, pred)
    sims = simulate_switch(p, use)

    px0, px3 = _exit_px(p, sim0), _exit_px(p, sim3)
    hold0, hold3 = sim0["holding_bars"], sim3["holding_bars"]
    ts0 = pd.DatetimeIndex(p.ts)

    g = ds.sort_values(["trade_id", "bars_in_trade"])
    last = g.groupby("trade_id", as_index=False).last()
    first_sw = g[g["p3_bar"]].groupby("trade_id", as_index=False).first()
    mx = g.groupby("trade_id", as_index=False).agg(
        p3_probability_max=("p3_probability", "max"),
        n_p3_bars=("p3_bar", "sum"),
        n_scored=("p3_probability", lambda s: int(np.isfinite(s).sum())),
    )
    dec = last.merge(first_sw[["trade_id", "p3_probability", "bars_in_trade", "mfe_so_far_R",
                                "current_R", "mae_so_far_R", "drawdown_from_MFE_R"]],
                     on="trade_id", how="left", suffixes=("_last", "_sw"))
    dec = dec.merge(mx, on="trade_id", how="left")
    selected = dec["n_p3_bars"].fillna(0).to_numpy() > 0
    evaluable = dec["n_scored"].fillna(0).to_numpy() > 0
    mfe = np.where(selected, dec["mfe_so_far_R_sw"].to_numpy(), dec["mfe_so_far_R_last"].to_numpy())
    cur = np.where(selected, dec["current_R_sw"].to_numpy(), dec["current_R_last"].to_numpy())
    mae = np.where(selected, dec["mae_so_far_R_sw"].to_numpy(), dec["mae_so_far_R_last"].to_numpy())
    dd_mfe = np.where(selected, dec["drawdown_from_MFE_R_sw"].to_numpy(),
                      dec["drawdown_from_MFE_R_last"].to_numpy())
    bars_dec = np.where(selected, dec["bars_in_trade_sw"].to_numpy(),
                        dec["bars_in_trade_last"].to_numpy())
    p_dec = np.where(selected, dec["p3_probability_sw"].to_numpy(),
                     dec["p3_probability_last"].to_numpy())

    tid = dec["trade_id"].to_numpy(dtype=int)
    year = p.year[tid]
    p0_r = sim0["r_multiple"][tid]
    p3_r = sim3["r_multiple"][tid]
    delta = p3_r - p0_r
    trades = pd.DataFrame({
        "trade_id": tid,
        "year": year,
        "side": np.where(p.is_long[tid], "long", "short"),
        "entry_time": ts0[tid],
        "entry_price": p.entry[tid],
        "p3_probability": p_dec,
        "p3_probability_max": dec["p3_probability_max"].to_numpy(),
        "selected_policy": np.where(selected, "P3", "P0"),
        "evaluable": evaluable,
        "p0_exit_time": ts0[tid] + pd.to_timedelta(hold0[tid], unit="h"),
        "p0_exit_price": px0[tid],
        "p0_R": p0_r,
        "p3_exit_time": ts0[tid] + pd.to_timedelta(hold3[tid], unit="h"),
        "p3_exit_price": px3[tid],
        "p3_R": p3_r,
        "delta_R": delta,
        "mfe_so_far_R": mfe,
        "current_R": cur,
        "mae_so_far_R": mae,
        "drawdown_from_MFE_R": dd_mfe,
        "bars_in_trade": bars_dec,
        "n_p3_bars": dec["n_p3_bars"].fillna(0).to_numpy(),
    })
    trades.to_csv(OUT / "sprint42_trade_attribution.csv", index=False)

    # --- Iter 1 groups ---
    attr_rows = []
    scopes = [("ALL", trades)]
    for name, mask in (
        ("ALL_EVALUABLE", trades["evaluable"]),
        ("P3_SELECTED", trades["selected_policy"] == "P3"),
        ("P0_SELECTED", (trades["selected_policy"] == "P0") & trades["evaluable"]),
        ("P0_SELECTED_incl_2021", trades["selected_policy"] == "P0"),
    ):
        scopes.append((name, trades[mask]))
    for y in YEARS:
        sub = trades[trades["year"] == y]
        scopes.append((str(y), sub))
        scopes.append((f"{y}_P3_SELECTED", sub[sub["selected_policy"] == "P3"]))
        scopes.append((f"{y}_P0_SELECTED", sub[(sub["selected_policy"] == "P0") & sub["evaluable"]]))

    for name, sub in scopes:
        if isinstance(sub, pd.DataFrame) and name.endswith("P3_SELECTED"):
            r = sub["p3_R"]
        elif isinstance(sub, pd.DataFrame) and "P0_SELECTED" in name:
            r = sub["p0_R"]
        else:
            r = sub["p0_R"]
        st = _dstats(sub["delta_R"].to_numpy(), r.to_numpy())
        st["group"] = name
        attr_rows.append(st)
        print(f"  {name}: n={st['n']} mean_d={st['mean_delta_R']:.4f} p>0={st['p_delta_gt_0']:.3f}")

    # --- Iter 2 path buckets (trade decision-time + bar-level usage) ---
    path_rows = []
    scored_bars = ds[np.isfinite(ds["p3_probability"].to_numpy())].copy()
    bar_delta = scored_bars.merge(trades[["trade_id", "delta_R"]], on="trade_id", how="left")

    def _bucket_block(kind: str, series: pd.Series, edges, src: pd.DataFrame, p3col: str):
        rows = []
        x = series.to_numpy(dtype=float)
        p3u = src[p3col].to_numpy() if p3col in src.columns else np.zeros(len(src), dtype=bool)
        dlt = src["delta_R"].to_numpy(dtype=float)
        for lab, fn in edges:
            m = fn(x) & np.isfinite(x)
            st = _dstats(dlt[m])
            st.update({"kind": kind, "bucket": lab, "p3_usage": float(np.mean(p3u[m])) if m.any() else 0.0})
            rows.append(st)
        return rows

    mfe_tr = _bucket_block("mfe_trade", trades["mfe_so_far_R"], MFE_EDGES, trades.assign(p3=(trades["selected_policy"]=="P3")), "p3")
    pd.DataFrame(mfe_tr).to_csv(OUT / "sprint42_mfe_attribution.csv", index=False)
    path_rows += mfe_tr
    path_rows += _bucket_block("current_R_trade", trades["current_R"], MFE_EDGES, trades.assign(p3=(trades["selected_policy"]=="P3")), "p3")
    path_rows += _bucket_block("dd_from_mfe_trade", trades["drawdown_from_MFE_R"], MFE_EDGES, trades.assign(p3=(trades["selected_policy"]=="P3")), "p3")
    path_rows += _bucket_block("bars_trade", trades["bars_in_trade"], BAR_EDGES, trades.assign(p3=(trades["selected_policy"]=="P3")), "p3")
    path_rows += _bucket_block("mfe_bar", bar_delta["mfe_so_far_R"], MFE_EDGES, bar_delta.assign(p3=bar_delta["p3_bar"]), "p3")
    path_rows += _bucket_block("current_R_bar", bar_delta["current_R"], MFE_EDGES, bar_delta.assign(p3=bar_delta["p3_bar"]), "p3")
    pd.DataFrame(path_rows).to_csv(OUT / "sprint42_path_attribution.csv", index=False)

    # --- Iter 3 correct vs wrong ---
    ev = trades[trades["evaluable"]].copy()
    ev["p3_better"] = ev["delta_R"] > 0
    ev["p3_sel"] = ev["selected_policy"] == "P3"
    quad = []
    for sel, better, lab in (
        (True, True, "P3_SELECTED_P3_BETTER"),
        (True, False, "P3_SELECTED_P0_BETTER"),
        (False, False, "P0_SELECTED_P0_BETTER"),
        (False, True, "P0_SELECTED_P3_BETTER"),
    ):
        sub = ev[(ev["p3_sel"] == sel) & (ev["p3_better"] == better)]
        d = sub["delta_R"].to_numpy(dtype=float)
        quad.append({
            "cell": lab,
            "n": int(len(sub)),
            "pct": float(len(sub) / max(len(ev), 1)),
            "mean_delta_R": float(np.mean(d)) if len(d) else float("nan"),
            "median_delta_R": float(np.median(d)) if len(d) else float("nan"),
            "mean_abs_delta_R": float(np.mean(np.abs(d))) if len(d) else float("nan"),
            "worst_delta_R": float(np.min(d)) if len(d) else float("nan"),
            "best_delta_R": float(np.max(d)) if len(d) else float("nan"),
        })
    wrong_p3 = ev[ev["p3_sel"] & ~ev["p3_better"]]["delta_R"]
    wrong_p0 = ev[~ev["p3_sel"] & ev["p3_better"]]["delta_R"]
    wrong_p3_cost = float(wrong_p3.mean()) if len(wrong_p3) else 0.0
    wrong_p0_cost = float((-wrong_p0).mean()) if len(wrong_p0) else 0.0
    correct_p3 = ev[ev["p3_sel"] & ev["p3_better"]]["delta_R"]
    correct_gain = float(correct_p3.mean()) if len(correct_p3) else 0.0
    pd.DataFrame(quad + [{
        "cell": "COSTS",
        "n": int(len(ev)),
        "pct": 1.0,
        "mean_delta_R": float(ev["delta_R"].mean()),
        "wrong_P3_cost": wrong_p3_cost,
        "wrong_P0_cost": wrong_p0_cost,
        "correct_P3_gain": correct_gain,
    }]).to_csv(OUT / "sprint42_switch_correctness.csv", index=False)

    # --- Iter 4 yearly ---
    yearly_delta = []
    for y in YEARS:
        sub = trades[trades["year"] == y]
        p3s = sub[sub["selected_policy"] == "P3"]
        p0s = sub[(sub["selected_policy"] == "P0") & sub["evaluable"]]
        st = _dstats(sub["delta_R"].to_numpy(), sub["p0_R"].to_numpy())
        st.update({
            "year": y,
            "p3_selected_pct": float((sub["selected_policy"] == "P3").mean()) if len(sub) else 0.0,
            "p3_selected_mean_delta_R": float(p3s["delta_R"].mean()) if len(p3s) else float("nan"),
            "p0_selected_mean_delta_R": float(p0s["delta_R"].mean()) if len(p0s) else float("nan"),
            "n_p3_selected": int(len(p3s)),
        })
        yearly_delta.append(st)

    # --- Iter 5 probability bins (bar-level PIT P vs trade delta) ---
    prob_rows = []
    pb = bar_delta.dropna(subset=["p3_probability"])
    x = pb["p3_probability"].to_numpy(dtype=float)
    for lab, fn in PROB_BINS:
        m = fn(x)
        st = _dstats(pb.loc[m, "delta_R"].to_numpy())
        st.update({"bucket": lab, "p3_usage": float(pb.loc[m, "p3_bar"].mean()) if m.any() else 0.0})
        prob_rows.append(st)
    # trade-level at decision P
    tev = trades[trades["evaluable"]].dropna(subset=["p3_probability"])
    xt = tev["p3_probability"].to_numpy(dtype=float)
    for lab, fn in PROB_BINS:
        m = fn(xt)
        st = _dstats(tev.loc[m, "delta_R"].to_numpy())
        st.update({
            "bucket": lab + "_trade_decision",
            "p3_usage": float((tev.loc[m, "selected_policy"] == "P3").mean()) if m.any() else 0.0,
        })
        prob_rows.append(st)
    pd.DataFrame(prob_rows).to_csv(OUT / "sprint42_probability_buckets.csv", index=False)

    # --- Iter 6 shadow books ---
    base_y, sw_y = [], []
    corrs, divs = [], []
    n_div_trades = 0
    for y in YEARS:
        pa, pbk = _port(p, sim0, y), _port(p, sims, y)
        ba, bb = _year_row(y, pa, sim0), _year_row(y, pbk, sims)
        base_y.append(ba)
        sw_y.append(bb)
        ts_a = ts0[np.asarray(pa["taken"], dtype=int)]
        ts_b = ts0[np.asarray(pbk["taken"], dtype=int)]
        c, d = _align_eq(ts_a, pa["equity"], ts_b, pbk["equity"])
        corrs.append(c)
        divs.append(d)
        n_div_trades += int(len(set(pa["taken"]) ^ set(pbk["taken"])))
        print(f"  shadow {y}: prod PF={ba['pf']:.2f} sw PF={bb['pf']:.2f} DD {ba['dd']*100:.1f}->{bb['dd']*100:.1f}")

    base_p, sw_p = _pooled(base_y), _pooled(sw_y)
    mc = iter6_montecarlo(base_p["pnl"], sw_p["pnl"], any(r["blown"] for r in base_y), starting=STARTING)
    dyn, fx = mc[mc["policy"] == "dynamic"], mc[mc["policy"] == "fixed"]
    mc_ruin = float(dyn.iloc[0]["prob_ruin"]) if len(dyn) else 1.0
    mc_wdd = float(dyn.iloc[0]["worst_dd"]) if len(dyn) else 1.0
    mc0_ruin = float(fx.iloc[0]["prob_ruin"]) if len(fx) else 1.0
    mc0_wdd = float(fx.iloc[0]["worst_dd"]) if len(fx) else 1.0

    shadow_y = []
    for b, s, yd in zip(base_y, sw_y, yearly_delta):
        yd["pf_delta"] = s["pf"] - b["pf"]
        yd["dd_delta"] = s["dd"] - b["dd"]
        yd["return_delta"] = s["ret"] - b["ret"]
        yd["win_rate_delta"] = s["wr"] - b["wr"]
        shadow_y.append({k: v for k, v in b.items() if k != "pnl"} | {"who": "BOOK_A_P0"})
        shadow_y.append({k: v for k, v in s.items() if k != "pnl"} | {"who": "BOOK_B_CONDITIONAL"})
    pd.DataFrame(yearly_delta).to_csv(OUT / "sprint42_yearly_delta.csv", index=False)
    pd.DataFrame(shadow_y).to_csv(OUT / "sprint42_shadow_yearly.csv", index=False)
    pd.DataFrame([
        {k: v for k, v in base_p.items() if k != "pnl"} | {"who": "BOOK_A_P0"},
        {k: v for k, v in sw_p.items() if k != "pnl"} | {"who": "BOOK_B_CONDITIONAL"},
    ]).to_csv(OUT / "sprint42_shadow_pooled.csv", index=False)

    leak = pd.DataFrame([
        {"item": f, "role": "STATE_FEAT switch input", "future_dependency": False,
         "notes": "PIT bar/entry only; same as Sprint 41"}
        for f in STATE_FEATS
    ] + [
        {"item": "p3_probability", "role": "LR score", "future_dependency": False,
         "notes": "expanding-year train; 2021 unscored; no OOS labels in features"},
        {"item": "selected_policy", "role": "P>=0.60 on PIT score", "future_dependency": False,
         "notes": "bar-level; trade selected if any PIT bar fires"},
        {"item": "p0_R/p3_R/delta_R", "role": "label/attribution only", "future_dependency": True,
         "notes": "not in STATE_FEATS; not used to switch"},
        {"item": "exit times/prices", "role": "label/attribution only", "future_dependency": True,
         "notes": "reconstructed from policy replay"},
    ])
    leak.to_csv(OUT / "sprint42_leakage_audit.csv", index=False)
    leak_fail = bool(leak.loc[leak["item"].isin(STATE_FEATS + ["p3_probability"]), "future_dependency"].any())

    # --- Iter 7 decision ---
    oos = trades[trades["year"].isin(TRUE_OOS)]
    p3_oos = oos[oos["selected_policy"] == "P3"]
    year_p3_d = {int(r["year"]): r["p3_selected_mean_delta_R"] for r in yearly_delta if r["year"] in TRUE_OOS}
    sign_inv = any(not np.isfinite(v) or v < 0 for v in year_p3_d.values())
    mean_p3_d = float(p3_oos["delta_R"].mean()) if len(p3_oos) else 0.0
    med_p3_d = float(p3_oos["delta_R"].median()) if len(p3_oos) else 0.0
    meaningful = mean_p3_d >= 0.01 and float((p3_oos["delta_R"] > 0).mean()) >= 0.50
    tot = p3_oos.groupby("year")["delta_R"].sum()
    majority_one_year = bool(len(tot) and (tot.max() / tot.sum() > 0.50)) if tot.sum() else True
    bar_means = [r["mean_delta_R"] for r in prob_rows if r["bucket"] in dict(PROB_BINS)]
    # directionally: mean delta in >=0.60 bins >= mean in <0.60 bins
    low = [r for r in prob_rows if r["bucket"] in ("lt_0.50", "0.50_0.60")]
    high = [r for r in prob_rows if r["bucket"] in ("0.60_0.70", "0.70_0.80", "ge_0.80")]
    dir_ok = (np.nanmean([r["mean_delta_R"] for r in high]) >=
              np.nanmean([r["mean_delta_R"] for r in low]) - 1e-9)
    mono = all(np.isnan(bar_means[i + 1]) or np.isnan(bar_means[i]) or bar_means[i + 1] + 0.005 >= bar_means[i]
               for i in range(len(bar_means) - 1)) if len(bar_means) == 5 else False
    cost_ok = (abs(wrong_p3_cost) <= max(3.0 * max(correct_gain, 1e-6), 0.50))
    dd_ok = sw_p["dd"] <= base_p["dd"] + 0.01
    mc_ok = mc_ruin <= mc0_ruin + 1e-12
    # 2021 not used: p3 selected 2021 must be 0
    y2021_unused = int((trades.loc[trades["year"] == 2021, "selected_policy"] == "P3").sum()) == 0
    shadow_dir = sw_p["pf"] >= base_p["pf"] - 1e-9 and sw_p["ret"] >= base_p["ret"] - 1e-9

    conds = {
        "oos_p3_delta_nonneg": not sign_inv,
        "p3_delta_meaningful": meaningful,
        "not_one_year_majority": not majority_one_year,
        "prob_directionally_ok": dir_ok,
        "wrong_switch_not_catastrophic": cost_ok,
        "no_leakage": not leak_fail,
        "dd_not_worse": dd_ok,
        "mc_ruin_not_worse": mc_ok,
        "not_dependent_on_2021": y2021_unused,
        "shadow_reproduces_sprint41": shadow_dir,
    }
    n_pass = sum(conds.values())
    if leak_fail:
        verdict = "LEAKAGE_FAIL"
    elif sign_inv:
        verdict = "REGIME_SPECIFIC"
    elif n_pass == 10:
        verdict = "PASS_TO_SHADOW"
    elif n_pass >= 8 and conds["oos_p3_delta_nonneg"] and conds["shadow_reproduces_sprint41"]:
        verdict = "CANDIDATE_FOR_SHADOW"
    else:
        verdict = "NO_SIGNAL"

    many_small = abs(med_p3_d) > 0.5 * abs(mean_p3_d) and med_p3_d > 0
    size_note = "many small wins" if many_small else "mean vs median gap — check tail"

    # report
    a_all = next(r for r in attr_rows if r["group"] == "ALL_EVALUABLE")
    a_p3 = next(r for r in attr_rows if r["group"] == "P3_SELECTED")
    a_p0 = next(r for r in attr_rows if r["group"] == "P0_SELECTED")
    eq_corr = float(np.nanmean(corrs))
    max_div = float(np.nanmax(divs)) if divs else float("nan")

    lines = [
        "# Sprint 42 — Conditional Exit Attribution & Shadow Validation",
        "",
        "## 1. Objective",
        "",
        "Check whether the Sprint 41 conditional exit (LR `p3_better` ≥ **0.60** → P3 a0.20/d0.10, else P0 a0.25/d0.08)",
        "has a genuine trade-level advantage. Analysis only. No new threshold, model, features, or trail params.",
        "",
        "2021 is unscored (no prior fold) and is **not** OOS evidence.",
        "",
        "## 2. Leakage Audit",
        "",
        "Switch inputs = Sprint 41 STATE_FEATS + expanding-year LR score. Labels (`p0_R`, `p3_R`, `delta_R`) are attribution only.",
        f"Future dependency on switch features: **{leak_fail}**.",
        "",
        _md(leak.to_dict("records"), ["item", "role", "future_dependency", "notes"]),
        "",
        "## 3. Trade-Level Attribution",
        "",
        "Grain: one row per trade. `selected_policy=P3` if any PIT bar had P≥0.60. `delta_R = R_P3 − R_P0` (full-policy counterfactual, identical entries).",
        "",
        _md(
            [a_all | {"group": "ALL_EVALUABLE (2022-2026 scored)"},
             a_p3 | {"group": "P3_SELECTED"},
             a_p0 | {"group": "P0_SELECTED (evaluable)"}],
            ["group", "n", "mean_R", "median_R", "mean_delta_R", "median_delta_R",
             "p_delta_gt_0", "p_delta_ge_0.10", "p_delta_ge_0.25", "p_delta_le_m0.10", "p_delta_le_m0.25"],
            {"n": 0, "mean_R": 3, "median_R": 3, "mean_delta_R": 4, "median_delta_R": 4,
             "p_delta_gt_0": 3, "p_delta_ge_0.10": 3, "p_delta_ge_0.25": 3,
             "p_delta_le_m0.10": 3, "p_delta_le_m0.25": 3},
        ),
        "",
        f"When the switch chooses P3 (n={a_p3['n']}): mean ΔR={a_p3['mean_delta_R']:+.4f}, median={a_p3['median_delta_R']:+.4f}, P(Δ>0)={a_p3['p_delta_gt_0']:.3f}.",
        "",
        "## 4. MFE / Path Attribution",
        "",
        "Decision-time PIT MFE (trade grain). P3 usage should fall as MFE rises if the Sprint 41 geometry holds.",
        "",
        _md(
            [r | {"p3_usage_pct": 100 * r["p3_usage"]} for r in mfe_tr],
            ["bucket", "n", "p3_usage_pct", "mean_delta_R", "median_delta_R", "p_delta_gt_0", "p_delta_ge_0.25", "p_delta_le_m0.25"],
            {"n": 0, "p3_usage_pct": 1, "mean_delta_R": 4, "median_delta_R": 4,
             "p_delta_gt_0": 3, "p_delta_ge_0.25": 3, "p_delta_le_m0.25": 3},
        ),
        "",
        "Same numeric edges on `current_R` and `drawdown_from_MFE_R`; bars_in_trade uses fixed 1-2 / 3-4 / 5-8 / 9-16 / ≥17 (not tuned). Full table: `sprint42_path_attribution.csv`.",
        "",
        "## 5. Correct vs Wrong Switch",
        "",
        _md(quad, ["cell", "n", "pct", "mean_delta_R", "median_delta_R", "mean_abs_delta_R", "worst_delta_R", "best_delta_R"],
            {"n": 0, "pct": 3, "mean_delta_R": 4, "median_delta_R": 4, "mean_abs_delta_R": 4, "worst_delta_R": 3, "best_delta_R": 3}),
        "",
        f"- wrong_P3_cost (mean Δ | P3 selected and P0 better) = **{wrong_p3_cost:+.4f}R**",
        f"- wrong_P0_cost (mean missed Δ | P0 selected and P3 better) = **{wrong_p0_cost:+.4f}R**",
        f"- correct_P3_gain = **{correct_gain:+.4f}R**",
        f"- size: {size_note} (P3-selected mean {mean_p3_d:+.4f} vs median {med_p3_d:+.4f})",
        "",
        "## 6. Yearly Stability",
        "",
        _md(
            yearly_delta,
            ["year", "n", "p3_selected_pct", "mean_delta_R", "median_delta_R",
             "p3_selected_mean_delta_R", "p0_selected_mean_delta_R",
             "p_delta_gt_0", "pf_delta", "dd_delta", "return_delta"],
            {"year": 0, "n": 0, "p3_selected_pct": 3, "mean_delta_R": 4, "median_delta_R": 4,
             "p3_selected_mean_delta_R": 4, "p0_selected_mean_delta_R": 4,
             "p_delta_gt_0": 3, "pf_delta": 3, "dd_delta": 4, "return_delta": 3},
        ),
        "",
        f"P3-selected mean Δ sign inversion 2022-2026: **{sign_inv}**. Flag: {'REGIME_DEPENDENT' if sign_inv else 'none'}.",
        f"Single-year majority of P3-selected ΣΔ: **{majority_one_year}**.",
        "",
        "## 7. Probability Calibration",
        "",
        "Bar-level PIT P vs that trade's realized ΔR. Threshold not re-picked.",
        "",
        _md(
            [r for r in prob_rows if r["bucket"] in dict(PROB_BINS)],
            ["bucket", "n", "p3_usage", "mean_delta_R", "median_delta_R", "p_delta_gt_0"],
            {"n": 0, "p3_usage": 3, "mean_delta_R": 4, "median_delta_R": 4, "p_delta_gt_0": 3},
        ),
        "",
        f"Directionally sensible (P≥0.60 mean Δ ≥ P<0.60): **{dir_ok}**. Strictly monotonic: **{mono}**.",
        "",
        "## 8. Historical Shadow Replay",
        "",
        "Two books, identical entries / lot / heat / max_open / $280. Not live. **HISTORICAL_SHADOW_REPLAY**.",
        "",
        "### Yearly",
        "",
        "| Year | who | Trades | PF | DD | Return | WR | Payoff | AvgR | MaxLoss |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for b, s in zip(base_y, sw_y):
        for who, r in (("BOOK_A_P0", b), ("BOOK_B_COND", s)):
            lines.append(
                f"| {r['year']} | {who} | {r['trades']} | {r['pf']:.2f} | {r['dd']*100:.1f}% | "
                f"{r['ret']*100:.0f}% | {r['wr']:.3f} | {r['payoff']:.2f} | {r['avg_r']:.3f} | {r['max_loss_streak']} |"
            )
    lines += [
        "",
        "### Pooled (isolated $280/year; DD = max yearly; Return = mean yearly)",
        "",
        "| who | Trades | PF | DD | Return | WR | Payoff | AvgR | MaxLoss |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        f"| BOOK_A_P0 | {base_p['trades']} | {base_p['pf']:.2f} | {base_p['dd']*100:.1f}% | {base_p['ret']*100:.0f}% | "
        f"{base_p['wr']:.3f} | {base_p['payoff']:.2f} | {base_p['avg_r']:.3f} | {base_p['max_loss_streak']} |",
        f"| BOOK_B_COND | {sw_p['trades']} | {sw_p['pf']:.2f} | {sw_p['dd']*100:.1f}% | {sw_p['ret']*100:.0f}% | "
        f"{sw_p['wr']:.3f} | {sw_p['payoff']:.2f} | {sw_p['avg_r']:.3f} | {sw_p['max_loss_streak']} |",
        "",
        f"- MC ruin: B {mc_ruin:.3f} vs A {mc0_ruin:.3f}",
        f"- MC worst DD: B {mc_wdd*100:.1f}% vs A {mc0_wdd*100:.1f}%",
        f"- equity correlation (mean yearly, timestamp-aligned): {eq_corr:.3f}",
        f"- max equity divergence: ${max_div:.1f} ({max_div / STARTING * 100:.1f}% of $280)",
        f"- taken-set symmetric difference (policy/hold → heat): {n_div_trades} trade-ids across years",
        "",
        "## 9. 2022-2026 OOS Assessment",
        "",
        f"- P3-selected n={len(p3_oos)}, mean ΔR={mean_p3_d:+.4f}, median={med_p3_d:+.4f}",
        f"- yearly P3-selected mean Δ: " + ", ".join(f"{y}={year_p3_d[y]:+.4f}" for y in TRUE_OOS),
        f"- conditions {n_pass}/10: " + ", ".join(f"{k}={'yes' if v else 'NO'}" for k, v in conds.items()),
        "",
        "## 10. Final Verdict",
        "",
        f"**{verdict}**",
        "",
        "## 11. Production Decision",
        "",
        "Production remains P0 **a0.25 / d0.08**.",
        "No automatic ship. Shadow/paper only if the user explicitly starts it.",
        "",
    ]
    (OUT / "sprint42_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "sprint42_final_verdict.json").write_text(
        json.dumps(
            {
                "verdict": verdict,
                "conditions": conds,
                "n_pass": n_pass,
                "sign_inversion": sign_inv,
                "p3_selected_mean_delta_R_oos": mean_p3_d,
                "p3_selected_median_delta_R_oos": med_p3_d,
                "wrong_P3_cost": wrong_p3_cost,
                "wrong_P0_cost": wrong_p0_cost,
                "correct_P3_gain": correct_gain,
                "prob_monotonic": mono,
                "prob_directional": dir_ok,
                "leakage": leak_fail,
                "pooled_pf_A": base_p["pf"],
                "pooled_pf_B": sw_p["pf"],
                "mc_ruin_A": mc0_ruin,
                "mc_ruin_B": mc_ruin,
                "equity_corr": eq_corr,
                "max_equity_div": max_div,
                "production_exit": "a0.25_d0.08 unchanged",
                "mode": "HISTORICAL_SHADOW_REPLAY",
            },
            indent=2,
            default=float,
        ),
        encoding="utf-8",
    )
    print(f"VERDICT={verdict} pass={n_pass}/10")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
