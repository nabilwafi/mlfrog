"""Sprint 38 — Dynamic Exit, iteration 1: baseline + leftover MFE.

Frozen: FEAT7 WF top 5%, trail a0.25_d0.08, ATR map_conservative, parallel_mo5 heat 3R.
Does NOT retrain entry. EXIT diagnostic only.

  python apps/research_sprint38_dynamic_exit.py
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

from apps.report_adaptive_vol_risk import enrich_vol, mapping_from_edges
from apps.report_multi_trade_engine import EngineCfg, run_multi
from apps.run_exit_engine_grid import (
    CAP,
    FEAT7,
    REASONS,
    Paths,
    build_entry_panel,
    simulate_combo,
)
from simulation.wf.sim import load_h1, prepare_market

OUT = _ROOT / "artifacts" / "pipeline_backtest" / "dynamic_exit" / "sprint38"
PANEL_CACHE = OUT / "entry_panel.parquet"
STARTING = 280.0  # locked: yearly isolated $280, never $80
EXIT_KW = dict(act=0.25, dist=0.08, tp=None, partials=(), tmax=None, be=None)
ATR_MAP = {"edges": [0.30, 0.60, 0.80, 0.90], "risks": [1.0, 0.70, 0.40, 0.25, 0.10]}
CFG = EngineCfg(
    name="parallel_mo5_d0.0_cd0_h3.0",
    family="parallel",
    max_positions=5,
    min_distance_atr=0.0,
    cooldown_bars=0,
    heat_budget_r=3.0,
)


def _capture(realized: np.ndarray, mfe: np.ndarray) -> np.ndarray:
    out = np.full(realized.shape, np.nan)
    ok = mfe > 1e-9
    out[ok] = realized[ok] / mfe[ok]
    return out


def _summ(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0, "mean": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0}
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "p75": float(np.percentile(x, 75)),
        "p90": float(np.percentile(x, 90)),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    h1 = load_h1(_ROOT / "artifacts/raw/XAUUSD/H1/data.parquet")
    mkt = prepare_market(h1)
    if PANEL_CACHE.is_file():
        print(f"load {PANEL_CACHE.name}")
        panel = pd.read_parquet(PANEL_CACHE)
    else:
        print("build FEAT7 WF top5% entry panel (frozen entry)")
        panel = build_entry_panel()
        panel.to_parquet(PANEL_CACHE, index=False)
    p = Paths(panel, mkt)
    d = enrich_vol(p.panel, mkt)
    atr_pct = d["atr_pct"].astype(float).fillna(0.5).to_numpy()
    lot_mult = mapping_from_edges(ATR_MAP["edges"], ATR_MAP["risks"])(atr_pct)
    probs = p.panel["y_prob"].astype(float).to_numpy()
    trend_ok = np.ones(p.n, dtype=bool)
    sim = simulate_combo(p, **EXIT_KW)
    order = np.argsort(pd.DatetimeIndex(p.ts).as_unit("ns").asi8, kind="stable")
    port = run_multi(
        order, p, sim, lot_mult, atr_pct, probs, trend_ok, CFG, starting=STARTING,
    )
    taken = np.asarray(port["taken"], dtype=int)
    realized = sim["r_multiple"][taken]
    mfe_exit = sim["mfe_r"][taken]
    mae_exit = sim["mae_r"][taken]
    reason = np.asarray(sim["reason"])[taken]
    hold = sim["holding_bars"][taken]
    mfe_48 = np.nanmax(p.fav[taken, 1:], axis=1)
    leftover_48 = np.maximum(0.0, mfe_48 - mfe_exit)
    # leftover after we already exited (continuation we did not harvest)
    leftover_after = np.zeros(len(taken))
    for k, i in enumerate(taken):
        j = int(hold[k])
        if j >= CAP:
            continue
        post = p.fav[i, j + 1 :]
        post = post[np.isfinite(post)]
        if post.size:
            leftover_after[k] = max(0.0, float(post.max()) - float(realized[k]))
    cap_exit = _capture(realized, mfe_exit)
    cap_48 = _capture(realized, mfe_48)
    years = pd.DatetimeIndex(p.ts).year.to_numpy()[taken]

    mix = {REASONS[i]: float((reason == i).mean()) for i in range(len(REASONS))}
    baseline = {
        "candidate": "baseline_a0.25_d0.08",
        "engine": CFG.name,
        "starting": STARTING,
        "top_pct": 0.05,
        "feat": list(FEAT7),
        "n_candidates": int(p.n),
        "trades": int(port["n_trades"]),
        "pf": float(port["profit_factor"]),
        "dd": float(port["max_drawdown"]),
        "ret": float(port["total_return"]),
        "blown": bool(port["blown"]),
        "equity_end": float(port["final_equity"]),
        "exit_mix": mix,
        "avg_R": float(realized.mean()),
        "median_R": float(np.median(realized)),
        "mfe_exit": _summ(mfe_exit),
        "mae_exit": _summ(mae_exit),
        "mfe_48": _summ(mfe_48),
        "capture_vs_exit_mfe": _summ(cap_exit),
        "capture_vs_48_mfe": _summ(cap_48),
        "leftover_48_R": _summ(leftover_48),
        "leftover_after_exit_R": _summ(leftover_after),
        "frac_leftover_after_gt_0.5R": float((leftover_after >= 0.5).mean()),
        "frac_leftover_after_gt_1R": float((leftover_after >= 1.0).mean()),
        "frac_leftover_after_gt_2R": float((leftover_after >= 2.0).mean()),
    }
    rows = []
    for y in sorted(set(int(v) for v in years)):
        m = years == y
        rr = realized[m]
        gp = float(rr[rr > 0].sum())
        gl = float(-rr[rr < 0].sum())
        pf = gp / gl if gl > 0 else (float("inf") if gp > 0 else 0.0)
        rows.append(
            {
                "year": y,
                "n": int(m.sum()),
                "pf": pf,
                "avg_R": float(rr.mean()),
                "mfe_exit": float(mfe_exit[m].mean()),
                "mfe_48": float(mfe_48[m].mean()),
                "capture_exit": float(np.nanmean(cap_exit[m])),
                "leftover_after": float(leftover_after[m].mean()),
                "trail_pct": float((reason[m] == 1).mean()),
            }
        )
    by_reason = []
    for i, name in enumerate(REASONS):
        m = reason == i
        if not m.any():
            continue
        by_reason.append(
            {
                "reason": name,
                "n": int(m.sum()),
                "pct": float(m.mean()),
                "avg_R": float(realized[m].mean()),
                "avg_mfe_exit": float(mfe_exit[m].mean()),
                "avg_mfe_48": float(mfe_48[m].mean()),
                "avg_capture_exit": float(np.nanmean(cap_exit[m])),
                "avg_leftover_after": float(leftover_after[m].mean()),
            }
        )
    trades = pd.DataFrame(
        {
            "year": years,
            "realized_R": realized,
            "mfe_exit": mfe_exit,
            "mae_exit": mae_exit,
            "mfe_48": mfe_48,
            "leftover_48": leftover_48,
            "leftover_after_exit": leftover_after,
            "capture_exit": cap_exit,
            "capture_48": cap_48,
            "reason": [REASONS[int(x)] for x in reason],
            "holding_bars": hold,
        }
    )
    trades.to_csv(OUT / "mfe_mae_analysis.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / "yearly_comparison.csv", index=False)
    pd.DataFrame(by_reason).to_csv(OUT / "exit_reason_comparison.csv", index=False)
    pd.DataFrame([baseline]).to_csv(OUT / "baseline.csv", index=False)
    (OUT / "baseline.json").write_text(json.dumps(baseline, indent=2), encoding="utf-8")

    leftover_mean = baseline["leftover_after_exit_R"]["mean"]
    substantial = leftover_mean >= 0.25 or baseline["frac_leftover_after_gt_1R"] >= 0.15
    decision = (
        "CONTINUE to label benchmark — trail leaves material post-exit MFE"
        if substantial
        else "STOP MODEL RESEARCH — leftover MFE too small to justify a continuation model"
    )
    lines = [
        "# Sprint 38 — Dynamic Exit Model Research Summary",
        "",
        "## Iteration 1 — Dataset + MFE/MAE",
        "",
        "Frozen: FEAT7, WF top 5%, trail `a0.25_d0.08`, ATR map_conservative, "
        f"`{CFG.name}`, start ${STARTING:.0f}, daily 1R ON.",
        "",
        "## 1. Baseline",
        "",
        f"PF: {baseline['pf']:.3f}",
        f"DD: {baseline['dd']*100:.1f}%",
        f"Trades: {baseline['trades']}",
        f"Return: {baseline['ret']*100:.0f}%",
        f"Avg R: {baseline['avg_R']:.3f}",
        f"Exit mix: { {k: round(v, 3) for k, v in mix.items()} }",
        "",
        "## 2. Does the current exit leave significant MFE unrealized?",
        "",
        f"Mean MFE at exit: {baseline['mfe_exit']['mean']:.3f}R  "
        f"(median {baseline['mfe_exit']['median']:.3f})",
        f"Mean MFE if held 48 bars: {baseline['mfe_48']['mean']:.3f}R  "
        f"(median {baseline['mfe_48']['median']:.3f})",
        f"Mean realized R: {baseline['avg_R']:.3f}",
        f"Capture vs MFE-at-exit: mean {baseline['capture_vs_exit_mfe']['mean']:.2f}  "
        f"median {baseline['capture_vs_exit_mfe']['median']:.2f}",
        f"Capture vs 48-bar MFE: mean {baseline['capture_vs_48_mfe']['mean']:.2f}  "
        f"median {baseline['capture_vs_48_mfe']['median']:.2f}",
        f"Leftover after exit (price kept running in favor): mean "
        f"{leftover_mean:.3f}R  median {baseline['leftover_after_exit_R']['median']:.3f}R",
        f"Share with leftover >= 0.5R / 1R / 2R: "
        f"{baseline['frac_leftover_after_gt_0.5R']:.1%} / "
        f"{baseline['frac_leftover_after_gt_1R']:.1%} / "
        f"{baseline['frac_leftover_after_gt_2R']:.1%}",
        "",
        f"**Answer:** {'YES' if substantial else 'NO — not enough leftover to chase.'}",
        "",
        f"Decision: {decision}",
        "",
        "## Yearly",
        "",
        "| Year | n | PF | avg R | MFE exit | MFE 48 | leftover after | TRAIL |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['year']} | {r['n']} | {r['pf']:.2f} | {r['avg_R']:.3f} | "
            f"{r['mfe_exit']:.3f} | {r['mfe_48']:.3f} | {r['leftover_after']:.3f} | "
            f"{r['trail_pct']:.1%} |"
        )
    lines += ["", "## By exit reason", ""]
    lines += [
        "| Reason | n | pct | avg R | MFE exit | leftover after |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in by_reason:
        lines.append(
            f"| {r['reason']} | {r['n']} | {r['pct']:.1%} | {r['avg_R']:.3f} | "
            f"{r['avg_mfe_exit']:.3f} | {r['avg_leftover_after']:.3f} |"
        )
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "decision.json").write_text(
        json.dumps({"iteration": 1, "substantial_leftover": substantial, "decision": decision}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({k: baseline[k] for k in ("trades", "pf", "dd", "avg_R", "exit_mix")}, indent=2, default=str))
    print(f"leftover_after mean={leftover_mean:.3f}R  gt1R={baseline['frac_leftover_after_gt_1R']:.1%}")
    print(f"DECISION: {decision}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
