"""Sprint 40 Iter 7 — full 2021-2026 OOS confirmation (no tuning).

Compare:
- baseline: a0.25/d0.08
- candidate: a0.20/d0.06/N4 delayed
- neighborhood: a0.20/d0.08/N4, a0.20/d0.10/N4

  python apps/research_sprint40_iter7_confirm.py
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import tools.pyc_path_hook  # noqa: F401

from apps.report_adaptive_vol_risk import iter6_montecarlo
from apps.report_multi_trade_engine import run_multi
from apps.research_sprint40_iter1_time_aware_exit import (
    CFG_PROD,
    STARTING,
    TimePolicy,
    simulate_timeaware_combo,
)
from apps.run_exit_engine_grid import Paths, build_entry_panel
from simulation.wf.sim import load_h1, prepare_market

OUT = _ROOT / "artifacts/pipeline_backtest/exit_structure/sprint40/iter7_confirm"
YEARS = (2021, 2022, 2023, 2024, 2025, 2026)


@dataclass(frozen=True)
class ExitCfg:
    name: str
    act: float
    dist: float
    n: int
    delayed: bool


CONFIGS = (
    ExitCfg("baseline_a0.25_d0.08", act=0.25, dist=0.08, n=0, delayed=False),
    ExitCfg("candidate_a0.20_d0.06_N4", act=0.20, dist=0.06, n=4, delayed=True),
    ExitCfg("neighbor_a0.20_d0.08_N4", act=0.20, dist=0.08, n=4, delayed=True),
    ExitCfg("neighbor_a0.20_d0.10_N4", act=0.20, dist=0.10, n=4, delayed=True),
)


def _orders(p: Paths, year: int) -> np.ndarray:
    idx = np.where(p.year == year)[0]
    return idx[np.argsort(pd.DatetimeIndex(p.ts.iloc[idx]).as_unit("ns").asi8, kind="stable")]


def _max_loss_streak(r: np.ndarray) -> int:
    best = streak = 0
    for x in r:
        if x < 0:
            streak += 1
            best = max(best, streak)
        else:
            streak = 0
    return int(best)


def _pf(r: np.ndarray) -> float:
    gp = float(r[r > 0].sum())
    gl = float(-r[r < 0].sum())
    if gl <= 0:
        return float("inf") if gp > 0 else 0.0
    return gp / gl


def _run_year(p: Paths, sim: dict, year: int) -> dict[str, Any]:
    port = run_multi(
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
    taken = np.asarray(port["taken"], dtype=int)
    r = sim["r_multiple"][taken] if taken.size else np.array([], dtype=float)
    wins = r[r > 0]
    losses = r[r < 0]
    payoff = float(wins.mean() / abs(losses.mean())) if losses.size else (float("inf") if wins.size else 0.0)
    return {
        "year": year,
        "trades": int(port["n_trades"]),
        "pf": float(port["profit_factor"]) if np.isfinite(port["profit_factor"]) else 0.0,
        "dd": float(port["max_drawdown"]),
        "ret": float(port["total_return"]),
        "wr": float(np.mean(r > 0)) if r.size else 0.0,
        "payoff": payoff,
        "avg_r": float(np.mean(r)) if r.size else 0.0,
        "max_loss_streak": _max_loss_streak(r),
        "blown": bool(port["blown"]),
        "pnl": port["pnl"],
        "r": r,
    }


def _pooled(yearly: list[dict]) -> dict[str, Any]:
    r = np.concatenate([y["r"] for y in yearly if y["r"].size]) if yearly else np.array([])
    pnl = np.concatenate([y["pnl"] for y in yearly if len(y["pnl"])]) if yearly else np.array([])
    wins = r[r > 0]
    losses = r[r < 0]
    payoff = float(wins.mean() / abs(losses.mean())) if losses.size else (float("inf") if wins.size else 0.0)
    return {
        "scope": "pooled",
        "trades": int(sum(y["trades"] for y in yearly)),
        "pf": _pf(r),
        "dd": float(max(y["dd"] for y in yearly)) if yearly else 0.0,
        "ret_mean": float(np.mean([y["ret"] for y in yearly])) if yearly else 0.0,
        "wr": float(np.mean(r > 0)) if r.size else 0.0,
        "payoff": payoff,
        "avg_r": float(np.mean(r)) if r.size else 0.0,
        "max_loss_streak": _max_loss_streak(r),
        "pnl": pnl,
    }


def _simulate(p: Paths, cfg: ExitCfg) -> dict[str, np.ndarray]:
    policy = TimePolicy(
        name=cfg.name,
        n=cfg.n if cfg.delayed else None,
        kind="baseline",
        dist_normal=cfg.dist,
        dist_wide=cfg.dist,
        dist_tight=cfg.dist,
        delayed=cfg.delayed,
        hybrid_mfe_thr=None,
    )
    return simulate_timeaware_combo(
        p,
        act=cfg.act,
        dist_normal=cfg.dist,
        dist_wide=cfg.dist,
        dist_tight=cfg.dist,
        policy=policy,
    )


def _beats(base: dict, cand: dict) -> bool:
    """Year-level: PF up, DD not worse, trades >= 90%."""
    if cand["trades"] < 0.90 * max(base["trades"], 1):
        return False
    if cand["pf"] < base["pf"] - 0.02:
        return False
    if cand["dd"] > base["dd"] + 0.01:
        return False
    return True


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("Build FEAT7 top21% panel...")
    p = Paths(build_entry_panel(top_pct=0.21), prepare_market(load_h1(_ROOT / "artifacts/raw/XAUUSD/H1/data.parquet")))

    results: dict[str, dict] = {}
    for cfg in CONFIGS:
        print(f"  {cfg.name}")
        sim = _simulate(p, cfg)
        yearly = [_run_year(p, sim, y) for y in YEARS]
        pooled = _pooled(yearly)
        results[cfg.name] = {"cfg": cfg, "yearly": yearly, "pooled": pooled}

    base = results[CONFIGS[0].name]
    cand = results[CONFIGS[1].name]

    # MC: dollar pnl streams (isolated years concatenated)
    base_pnl = base["pooled"]["pnl"]
    base_blown = any(y["blown"] for y in base["yearly"])
    mc_rows = []
    for cfg in CONFIGS:
        pnl = results[cfg.name]["pooled"]["pnl"]
        mc = iter6_montecarlo(base_pnl, pnl, base_blown, starting=STARTING)
        dyn = mc[mc["policy"] == "dynamic"]
        mc_rows.append({
            "name": cfg.name,
            "mc_ruin": float(dyn.iloc[0]["prob_ruin"]) if len(dyn) else 1.0,
            "mc_worst_dd": float(dyn.iloc[0]["worst_dd"]) if len(dyn) else 1.0,
        })
    for r in mc_rows:
        results[r["name"]]["pooled"]["mc_ruin"] = r["mc_ruin"]
        results[r["name"]]["pooled"]["mc_worst_dd"] = r["mc_worst_dd"]

    # Verdict
    years_beat = {y: _beats(base["yearly"][i], cand["yearly"][i]) for i, y in enumerate(YEARS)}
    early_fail = not years_beat[2021] and not years_beat[2022]
    early_win = years_beat[2021] and years_beat[2022]
    late_only = (
        all(years_beat[y] for y in (2023, 2024, 2025, 2026))
        and not early_win
    )
    n_beat = sum(years_beat.values())
    pooled_beat = (
        cand["pooled"]["pf"] >= base["pooled"]["pf"]
        and cand["pooled"]["dd"] <= base["pooled"]["dd"] + 0.01
        and cand["pooled"]["trades"] >= 0.90 * base["pooled"]["trades"]
    )
    nb_ok = all(
        _beats(base["yearly"][i], results[n.name]["yearly"][i])
        for n in CONFIGS[2:]
        for i in range(len(YEARS))
        if results[n.name]["yearly"][i]["pf"] >= cand["yearly"][i]["pf"] - 0.05
    )

    if late_only or early_fail:
        verdict = "REGIME_SPECIFIC"
    elif n_beat >= 5 and pooled_beat and early_win:
        verdict = "CANDIDATE_FOR_PRODUCTION"
    elif n_beat >= 4 and pooled_beat:
        verdict = "WEAK"
    else:
        verdict = "FAIL"

    def _md_row(scope: str, r: dict) -> str:
        ret = r.get("ret", r.get("ret_mean", 0.0))
        return (
            f"| {scope} | {r['trades']} | {r['pf']:.2f} | {r['dd']*100:.1f}% | {ret*100:.0f}% | "
            f"{r['wr']:.3f} | {r['payoff']:.2f} | {r['avg_r']:.3f} | {r['max_loss_streak']} | "
            f"{r.get('mc_ruin', float('nan')):.3f} | {r.get('mc_worst_dd', float('nan'))*100:.1f}% |"
        )

    lines = [
        "# Sprint 40 — Iter 7 Full OOS Confirmation (2021-2026)",
        "",
        "Frozen entry/risk/portfolio. Exit-only comparison. No tuning.",
        "",
        "| Config | scope | Trades | PF | DD | Return | WR | Payoff | AvgR | MaxLossStreak | MC ruin | MC worst DD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cfg in CONFIGS:
        res = results[cfg.name]
        pl = res["pooled"]
        lines.append(
            f"| {cfg.name} | pooled | {pl['trades']} | {pl['pf']:.2f} | {pl['dd']*100:.1f}% | "
            f"{pl['ret_mean']*100:.0f}% | {pl['wr']:.3f} | {pl['payoff']:.2f} | {pl['avg_r']:.3f} | "
            f"{pl['max_loss_streak']} | {pl.get('mc_ruin', 0):.3f} | {pl.get('mc_worst_dd', 0)*100:.1f}% |"
        )
        for yrow in res["yearly"]:
            lines.append(
                f"| {cfg.name} | {yrow['year']} | {yrow['trades']} | {yrow['pf']:.2f} | {yrow['dd']*100:.1f}% | "
                f"{yrow['ret']*100:.0f}% | {yrow['wr']:.3f} | {yrow['payoff']:.2f} | {yrow['avg_r']:.3f} | "
                f"{yrow['max_loss_streak']} | — | — |"
            )

    lines += [
        "",
        "## Candidate vs baseline (yearly beat?)",
        "",
        "| Year | beat baseline |",
        "|---:|:---:|",
    ]
    for y in YEARS:
        lines.append(f"| {y} | {'yes' if years_beat[y] else 'no'} |")

    lines += [
        "",
        f"**Verdict: {verdict}**",
        "",
        f"- Years beating baseline: {n_beat}/6",
        f"- Pooled beats baseline: {pooled_beat}",
        f"- 2021-2022 both beat: {early_win}",
        "",
        "Production exit unchanged (`a0.25/d0.08`). Do not ship without explicit approval.",
    ]

    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # CSV yearly
    rows = []
    for cfg in CONFIGS:
        for yrow in results[cfg.name]["yearly"]:
            rows.append({"config": cfg.name, **{k: v for k, v in yrow.items() if k not in ("pnl", "r")}})
        pl = results[cfg.name]["pooled"]
        rows.append({
            "config": cfg.name,
            "year": "pooled",
            "trades": pl["trades"],
            "pf": pl["pf"],
            "dd": pl["dd"],
            "ret": pl["ret_mean"],
            "wr": pl["wr"],
            "payoff": pl["payoff"],
            "avg_r": pl["avg_r"],
            "max_loss_streak": pl["max_loss_streak"],
            "mc_ruin": pl.get("mc_ruin"),
            "mc_worst_dd": pl.get("mc_worst_dd"),
            "blown": False,
        })
    pd.DataFrame(rows).to_csv(OUT / "comparison.csv", index=False)

    decision = {
        "iter": 7,
        "verdict": verdict,
        "years_beat_baseline": years_beat,
        "n_years_beat": n_beat,
        "pooled_beat": pooled_beat,
        "early_win": early_win,
        "late_only": late_only,
        "production_exit": "a0.25_d0.08 unchanged",
    }
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(f"VERDICT={verdict} years_beat={n_beat}/6 pooled_beat={pooled_beat}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
