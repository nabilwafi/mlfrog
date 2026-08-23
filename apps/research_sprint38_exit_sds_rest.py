"""Sprint 38 SDS remainder as a SEPARATE experiment (iter 7–12).

Not a continuation of the STOP'd Dynamic Exit claim. Frozen entry/risk/multi-trade.
$280. Does not change production.

  python apps/research_sprint38_exit_sds_rest.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import tools.pyc_path_hook  # noqa: F401

import numpy as np
import pandas as pd

from apps.report_adaptive_vol_risk import enrich_vol, iter6_montecarlo, mapping_from_edges
from apps.research_sprint38_iter3_exit import (
    ATR_MAP,
    CFG,
    EXIT_KW,
    STARTING,
    _load_panel,
    _port,
    _score_matrix,
    _yearly,
)
from apps.run_exit_engine_grid import CAP, REASONS, Paths, simulate_combo
from simulation.wf.sim import COST, load_h1, prepare_market

OUT = _ROOT / "artifacts" / "pipeline_backtest" / "rolling_wf" / "sprint38_exit_sds_rest"
ITER3 = _ROOT / "artifacts" / "pipeline_backtest" / "rolling_wf" / "sprint38_dynamic_exit"
LO, HI = 0.33, 0.66


def simulate_mapped(
    p: Paths,
    score: np.ndarray,
    *,
    trail_map: bool = False,
    be_map: bool = False,
    tp_map: bool = False,
    lo: float = LO,
    hi: float = HI,
    be_low: float = 0.5,
    be_mid: float = 1.0,
    tp_low: float = 1.5,
    tp_mid: float = 2.5,
) -> dict[str, np.ndarray]:
    """Baseline trail + optional score tertile maps for trail / BEP / TP."""
    n = p.n
    sl = np.full(n, -1.0)
    extreme = np.zeros(n)
    mae = np.zeros(n)
    active = np.ones(n, dtype=bool)
    exit_r = np.zeros(n)
    exit_off = np.zeros(n, dtype=np.int64)
    reason = np.full(n, 4, dtype=np.int64)
    trail_moved = np.zeros(n, dtype=bool)
    be_moved = np.zeros(n, dtype=bool)
    sc = np.where(np.isfinite(score), score, 0.5)

    for j in range(1, CAP + 1):
        ended = active & (p.last_off < j)
        if ended.any():
            loff = p.last_off[ended]
            exit_r[ended] = p.close_r[ended, loff]
            exit_off[ended] = loff
            reason[ended] = 4
            active[ended] = False
        m = active
        if not m.any():
            break
        fav_j = p.fav[m, j]
        adv_j = p.adv[m, j]
        extreme[m] = np.maximum(extreme[m], fav_j)
        mae[m] = np.maximum(mae[m], adv_j)
        sj = sc[np.where(m)[0], j]
        dist = np.full(int(m.sum()), 0.08)
        act = np.full(int(m.sum()), 0.25)
        if trail_map:
            dist[sj < lo] = 0.06
            act[sj < lo] = 0.20
            dist[sj >= hi] = 0.16
            act[sj >= hi] = 0.50
        if be_map:
            be_lvl = np.full(int(m.sum()), np.inf)
            be_lvl[sj < lo] = be_low
            be_lvl[(sj >= lo) & (sj < hi)] = be_mid
            raise_ = extreme[m] >= be_lvl
            if raise_.any():
                idx = np.where(m)[0][raise_]
                take = sl[idx] < 0.0
                sl[idx[take]] = 0.0
                be_moved[idx[take]] = True
        act_ok = extreme[m] >= act
        if act_ok.any():
            idx_m = np.where(m)[0][act_ok]
            new_sl = extreme[idx_m] - dist[act_ok] * p.atr_over_r[idx_m, j]
            moved = new_sl > sl[idx_m]
            sl[idx_m[moved]] = new_sl[moved]
            trail_moved[idx_m[moved]] = True
        hit_sl = m.copy()
        hit_sl[m] = adv_j >= -sl[m]
        if hit_sl.any():
            exit_r[hit_sl] = sl[hit_sl]
            exit_off[hit_sl] = j
            reason[hit_sl] = np.where(
                sl[hit_sl] > 1e-9,
                1,
                np.where(be_moved[hit_sl] & (sl[hit_sl] >= -1e-9), 2, np.where(trail_moved[hit_sl], 1, 0)),
            )
            active[hit_sl] = False
        m = active
        if not m.any():
            break
        fav_j = p.fav[m, j]
        if tp_map:
            sj2 = sc[np.where(m)[0], j]
            tp_lvl = np.full(int(m.sum()), np.inf)
            tp_lvl[sj2 < lo] = tp_low
            tp_lvl[(sj2 >= lo) & (sj2 < hi)] = tp_mid
            hit = fav_j >= tp_lvl
            if hit.any():
                idx = np.where(m)[0][hit]
                exit_r[idx] = tp_lvl[hit]
                exit_off[idx] = j
                reason[idx] = 3
                active[idx] = False
        if j == CAP:
            fin = active & (p.last_off >= j)
            exit_r[fin] = p.close_r[fin, j]
            exit_off[fin] = j
            reason[fin] = 4
            active[fin] = False

    if active.any():
        loff = p.last_off[active]
        exit_r[active] = p.close_r[active, loff]
        exit_off[active] = loff
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


def _row(name: str, family: str, port: dict, sim: dict, p: Paths) -> dict:
    taken = np.asarray(port["taken"], dtype=int)
    r = sim["r_multiple"][taken]
    reason = sim["reason"][taken]
    years = p.year[taken]
    y26 = years == 2026
    mfe = sim["mfe_r"][taken]
    mae = sim["mae_r"][taken]
    cap = np.where(mfe > 1e-9, r / mfe, np.nan)
    mix = {REASONS[i]: float((reason == i).mean()) if r.size else 0.0 for i in range(len(REASONS))}
    gp = float(r[r > 0].sum()) if r.size else 0.0
    gl = float(-r[r < 0].sum()) if r.size else 0.0
    return {
        "candidate": name,
        "family": family,
        "trades": int(port["n_trades"]),
        "pf": float(port["profit_factor"]),
        "dd": float(port["max_drawdown"]),
        "ret": float(port["total_return"]),
        "avg_R": float(r.mean()) if r.size else 0.0,
        "median_R": float(np.median(r)) if r.size else 0.0,
        "mfe_mean": float(mfe.mean()) if r.size else 0.0,
        "mae_mean": float(mae.mean()) if r.size else 0.0,
        "mfe_capture_med": float(np.nanmedian(cap)) if r.size else 0.0,
        "sl_pct": mix["SL"],
        "trail_pct": mix["TRAIL"],
        "bep_pct": mix["BREAKEVEN"],
        "tp_pct": mix["TP"],
        "timeout_pct": mix["TIMEOUT"],
        "n_2026": int(y26.sum()),
        "pf_2026": (
            float(r[y26][r[y26] > 0].sum() / max(-r[y26][r[y26] < 0].sum(), 1e-12)) if y26.any() else 0.0
        ),
        "ret_2026": float(r[y26].sum()) if y26.any() else 0.0,
        "blown": bool(port["blown"]),
        "win_rate": float(port["win_rate"]),
    }


def _eval(name, family, p, sim, lot_mult, atr_pct, probs, trend_ok, rows, yearly, sims):
    port = _port(p, sim, lot_mult, atr_pct, probs, trend_ok)
    rec = _row(name, family, port, sim, p)
    rows.append(rec)
    yearly.extend(_yearly(name, p, sim, lot_mult, atr_pct, probs, trend_ok))
    sims[name] = (sim, port)
    print(f"  {name}: PF={rec['pf']:.3f} DD={rec['dd']*100:.1f}% n={rec['trades']} avgR={rec['avg_R']:.3f} tp={rec['tp_pct']:.0%}")
    return rec


def _gates(rec, base) -> bool:
    return (
        rec["candidate"] != base["candidate"]
        and rec["pf"] > base["pf"]
        and rec["dd"] <= base["dd"] + 0.01
        and rec["trades"] >= 0.90 * base["trades"]
        and rec["blown"] is False
        and rec["pf_2026"] > 1.0
    )


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    h1 = load_h1(_ROOT / "artifacts/raw/XAUUSD/H1/data.parquet")
    mkt = prepare_market(h1)
    p = Paths(_load_panel(), mkt)
    d = enrich_vol(p.panel, mkt)
    atr_pct = d["atr_pct"].astype(float).fillna(0.5).to_numpy()
    lot_mult = mapping_from_edges(ATR_MAP["edges"], ATR_MAP["risks"])(atr_pct)
    probs = p.panel["y_prob"].astype(float).to_numpy()
    trend_ok = np.ones(p.n, dtype=bool)
    kw = dict(p=p, lot_mult=lot_mult, atr_pct=atr_pct, probs=probs, trend_ok=trend_ok)
    rows: list[dict] = []
    yearly: list[dict] = []
    sims: dict = {}

    print("baseline + OOS scores (separate SDS experiment)")
    base_sim = simulate_combo(p, **EXIT_KW)
    score = _score_matrix(p, base_sim)
    _eval("baseline_a0.25_d0.08", "baseline", sim=base_sim, rows=rows, yearly=yearly, sims=sims, **kw)
    base = rows[0]

    print("iter7 Policy D — BEP")
    for be in (0.50, 0.75, 1.00, 1.25):
        _eval(f"bep_{be:.2f}", "D_bep", sim=simulate_combo(p, act=0.25, dist=0.08, tp=None, partials=(), tmax=None, be=be), rows=rows, yearly=yearly, sims=sims, **kw)
    _eval("bep_from_score", "D_bep", sim=simulate_mapped(p, score, be_map=True), rows=rows, yearly=yearly, sims=sims, **kw)

    print("iter8 Policy E — TP")
    for tp in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
        _eval(f"tp_{tp:.1f}", "E_tp", sim=simulate_combo(p, act=0.25, dist=0.08, tp=tp, partials=(), tmax=None, be=None), rows=rows, yearly=yearly, sims=sims, **kw)
    _eval("tp_from_score", "E_tp", sim=simulate_mapped(p, score, tp_map=True), rows=rows, yearly=yearly, sims=sims, **kw)

    print("iter9 Policy F — hybrid + ablation")
    _eval("hybrid_trail+bep+tp", "F_hybrid", sim=simulate_mapped(p, score, trail_map=True, be_map=True, tp_map=True), rows=rows, yearly=yearly, sims=sims, **kw)
    _eval("ablate_no_bep", "F_ablation", sim=simulate_mapped(p, score, trail_map=True, tp_map=True), rows=rows, yearly=yearly, sims=sims, **kw)
    _eval("ablate_no_tp", "F_ablation", sim=simulate_mapped(p, score, trail_map=True, be_map=True), rows=rows, yearly=yearly, sims=sims, **kw)
    _eval("ablate_no_trail", "F_ablation", sim=simulate_mapped(p, score, be_map=True, tp_map=True), rows=rows, yearly=yearly, sims=sims, **kw)
    _eval("trail_map_only", "F_ablation", sim=simulate_mapped(p, score, trail_map=True), rows=rows, yearly=yearly, sims=sims, **kw)

    print("iter10 robustness (cut / TP-BEP levels)")
    _eval("hybrid_cuts_0.25_0.50", "robust", sim=simulate_mapped(p, score, trail_map=True, be_map=True, tp_map=True, lo=0.25, hi=0.50), rows=rows, yearly=yearly, sims=sims, **kw)
    _eval("hybrid_cuts_0.40_0.70", "robust", sim=simulate_mapped(p, score, trail_map=True, be_map=True, tp_map=True, lo=0.40, hi=0.70), rows=rows, yearly=yearly, sims=sims, **kw)
    _eval("hybrid_tp_1.0_2.0", "robust", sim=simulate_mapped(p, score, trail_map=True, be_map=True, tp_map=True, tp_low=1.0, tp_mid=2.0), rows=rows, yearly=yearly, sims=sims, **kw)
    _eval("hybrid_tp_2.0_3.0", "robust", sim=simulate_mapped(p, score, trail_map=True, be_map=True, tp_map=True, tp_low=2.0, tp_mid=3.0), rows=rows, yearly=yearly, sims=sims, **kw)

    df = pd.DataFrame(rows)
    ydf = pd.DataFrame(yearly)
    df.to_csv(OUT / "exit_policy_grid.csv", index=False)
    ydf.to_csv(OUT / "yearly_comparison.csv", index=False)

    print("iter11 MC + 2026")
    mc_names = ["baseline_a0.25_d0.08", "hybrid_trail+bep+tp", "trail_map_only", "bep_from_score", "tp_from_score"]
    # plus best PF in D and E
    for fam in ("D_bep", "E_tp"):
        sub = df[df["family"] == fam].sort_values("pf", ascending=False)
        if len(sub):
            mc_names.append(str(sub.iloc[0]["candidate"]))
    mc_names = list(dict.fromkeys(mc_names))
    mc_rows = []
    base_pnl = sims["baseline_a0.25_d0.08"][1]["pnl"]
    for name in mc_names:
        if name not in sims:
            continue
        port = sims[name][1]
        mc = iter6_montecarlo(base_pnl, port["pnl"], bool(port["blown"]), starting=STARTING)
        dyn = mc[mc["policy"] == "dynamic"]
        rec = {
            "candidate": name,
            "prob_ruin": float(dyn.iloc[0]["prob_ruin"]) if len(dyn) else 1.0,
            "mc_worst_dd": float(dyn.iloc[0]["worst_dd"]) if len(dyn) else 1.0,
            "mc_p95_dd": float(dyn.iloc[0]["p95_dd"]) if len(dyn) else 1.0,
        }
        mc_rows.append(rec)
        print(f"  MC {name}: ruin={rec['prob_ruin']:.3f} p95DD={rec['mc_p95_dd']*100:.1f}%")
    mcdf = pd.DataFrame(mc_rows)
    mcdf.to_csv(OUT / "montecarlo.csv", index=False)

    # years improved vs baseline (isolated, 2022–2026)
    ybase = ydf[ydf["candidate"] == "baseline_a0.25_d0.08"].set_index("year")
    rob_rows = []
    for name, g in ydf.groupby("candidate"):
        n_up = 0
        n_dd_ok = 0
        n_y = 0
        for _, r in g.iterrows():
            y = int(r["year"])
            if y == 2021 or y not in ybase.index:
                continue
            n_y += 1
            if r["pf"] >= ybase.loc[y, "pf"]:
                n_up += 1
            if r["dd"] <= ybase.loc[y, "dd"] + 0.01:
                n_dd_ok += 1
        rob_rows.append({"candidate": name, "years_pf_ge_base": n_up, "years_dd_ok": n_dd_ok, "n_years": n_y})
    rob = pd.DataFrame(rob_rows)
    rob.to_csv(OUT / "robustness_years.csv", index=False)

    # leaderboard: this run + iter3 policies
    lead = df.copy()
    if (ITER3 / "exit_policy_grid.csv").is_file():
        old = pd.read_csv(ITER3 / "exit_policy_grid.csv")
        old["family"] = old["candidate"].map(
            lambda c: "A_binary" if str(c).startswith("binary") else ("B_trail" if "trail" in str(c) or "hold" in str(c) else "prior")
        )
        keep = [c for c in lead.columns if c in old.columns]
        extra = old[[c for c in old.columns if c in lead.columns]].copy()
        extra["family"] = old["family"]
        for c in lead.columns:
            if c not in extra.columns:
                extra[c] = np.nan
        extra = extra[lead.columns]
        extra = extra[extra["candidate"] != "baseline_a0.25_d0.08"]
        lead = pd.concat([lead, extra], ignore_index=True)
    lead = lead.merge(mcdf, on="candidate", how="left")
    lead = lead.merge(rob, on="candidate", how="left")
    lead["n_vs_base"] = lead["trades"] / max(base["trades"], 1)
    lead["d_pf"] = lead["pf"] - base["pf"]
    lead["d_dd"] = lead["dd"] - base["dd"]
    lead["gate_pass"] = [
        _gates(r, base) if r["candidate"] != base["candidate"] else False for r in lead.to_dict("records")
    ]
    lead = lead.sort_values(["gate_pass", "pf"], ascending=[False, False])
    lead.to_csv(OUT / "dynamic_exit_leaderboard.csv", index=False)

    winners = lead[lead["gate_pass"] == True]
    hybrid = df[df["candidate"] == "hybrid_trail+bep+tp"].iloc[0]
    trail_only = df[df["candidate"] == "trail_map_only"].iloc[0]
    if len(winners):
        best = winners.iloc[0]
        y26_worse = float(best["pf_2026"]) + 0.10 < float(base["pf_2026"])
        ret_down = float(best["ret"]) < float(base["ret"])
        cut_crash = float(df[df["candidate"] == "hybrid_cuts_0.25_0.50"].iloc[0]["dd"]) > base["dd"] + 0.03
        if y26_worse or ret_down or cut_crash or int(best.get("years_pf_ge_base") or 0) < 4:
            verdict = "OVERFIT / FRAGILE"
        elif best["pf"] > base["pf"] + 0.05:
            verdict = "PERFORMANCE WINNER"
        else:
            verdict = "MARGINAL"
    else:
        best = df.sort_values("pf", ascending=False).iloc[0]
        verdict = "NO MATERIAL BENEFIT"
        if hybrid["dd"] > base["dd"] + 0.05 and hybrid["pf"] <= base["pf"]:
            verdict = "NO MATERIAL BENEFIT"

    # ablation attribution
    parts = {
        "trail": float(trail_only["pf"] - base["pf"]),
        "bep_map": float(df[df["candidate"] == "bep_from_score"].iloc[0]["pf"] - base["pf"]),
        "tp_map": float(df[df["candidate"] == "tp_from_score"].iloc[0]["pf"] - base["pf"]),
        "hybrid": float(hybrid["pf"] - base["pf"]),
    }
    n_stable = int((df[df["family"] == "robust"]["pf"] >= hybrid["pf"] - 0.05).sum())

    lines = [
        "# Sprint 38 SDS remainder — separate experiment (iter 7–12)",
        "",
        "Not a production candidate by default. Iter 2 ranking already failed OOS monotonicity;",
        "this run only completes SDS policies D–F + robustness/MC/ablation on **$280**, frozen stack.",
        "",
        f"Baseline PF **{base['pf']:.3f}** DD **{base['dd']*100:.1f}%** n={int(base['trades'])} avgR={base['avg_R']:.3f} 2026 PF={base['pf_2026']:.2f}",
        "",
        "## Leaderboard (this experiment)",
        "",
        "| Candidate | family | n | PF | DD | avg R | TP% | 2026 PF | dPF | dDD |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in df.to_dict("records"):
        lines.append(
            f"| {r['candidate']} | {r['family']} | {r['trades']} | {r['pf']:.3f} | {r['dd']*100:.1f}% | "
            f"{r['avg_R']:.3f} | {r['tp_pct']:.0%} | {r['pf_2026']:.2f} | {r['pf']-base['pf']:+.3f} | "
            f"{(r['dd']-base['dd'])*100:+.1f}pp |"
        )
    lines += [
        "",
        "## Yearly PF (isolated $280)",
        "",
        "| Year | baseline | bep_1.00 | tp_2.0 | hybrid | trail_map |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    show = ["baseline_a0.25_d0.08", "bep_1.00", "tp_2.0", "hybrid_trail+bep+tp", "trail_map_only"]
    for y in sorted(ydf["year"].unique()):
        cells = [str(int(y))]
        for c in show:
            hit = ydf[(ydf["year"] == y) & (ydf["candidate"] == c)]
            cells.append(f"{hit.iloc[0]['pf']:.2f}" if len(hit) else "—")
        lines.append("| " + " | ".join(cells) + " |")
    mc_base = mcdf[mcdf["candidate"] == "baseline_a0.25_d0.08"]
    mc_hy = mcdf[mcdf["candidate"] == "hybrid_trail+bep+tp"]
    lines += [
        "",
        "## Ablation (ΔPF vs baseline)",
        "",
        f"- trail map only: {parts['trail']:+.3f}",
        f"- BEP-from-score: {parts['bep_map']:+.3f}",
        f"- TP-from-score: {parts['tp_map']:+.3f}",
        f"- hybrid all three: {parts['hybrid']:+.3f}",
        "",
        "Hybrid is **not** claimed as a combo win unless it beats the best single component.",
        "",
        "## Monte Carlo (trade-order shuffle, $280)",
        "",
        f"- baseline ruin={float(mc_base.iloc[0]['prob_ruin']) if len(mc_base) else float('nan'):.3f} "
        f"p95DD={float(mc_base.iloc[0]['mc_p95_dd'])*100 if len(mc_base) else float('nan'):.1f}%",
        f"- hybrid ruin={float(mc_hy.iloc[0]['prob_ruin']) if len(mc_hy) else float('nan'):.3f} "
        f"p95DD={float(mc_hy.iloc[0]['mc_p95_dd'])*100 if len(mc_hy) else float('nan'):.1f}%",
        "",
        f"Robustness: {n_stable}/4 nearby hybrids within 0.05 PF of hybrid.",
        "",
        f"Best vs gates: **{best['candidate']}** PF={best['pf']:.3f}",
        f"Verdict: **{verdict}**",
        "",
        "Production recommendation: keep trail `a0.25_d0.08`. This folder is diagnostic SDS completion only.",
        "",
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "decision.json").write_text(
        json.dumps(
            {
                "experiment": "sds_rest_separate",
                "verdict": verdict,
                "best": str(best["candidate"]),
                "baseline_pf": float(base["pf"]),
                "ablation_dpf": parts,
                "n_gate_pass": int(len(winners)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"VERDICT {verdict} best={best['candidate']}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
