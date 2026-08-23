"""H1 Reversal & Position Decision research runner.

Phases: labels | mae | baselines | decision_demo | report | all

Does NOT retrain LONG/SHORT. Does NOT ship production exits.
Reuses Sprint 52 path + H1 native entry panel when present.

  python apps/research_h1_reversal.py --phase all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import tools.pyc_path_hook  # noqa: F401

from apps.research_sprint42_attribution import TRUE_OOS
from research.h1_reversal.decision import (
    DecisionAction,
    DecisionConfig,
    MarketState,
    PositionState,
    evaluate,
)
from research.h1_reversal.labels import (
    BarrierReversalParams,
    ThesisFailureParams,
    label_opposite_barrier,
    label_thesis_failure,
)
from research.mtf_h4_h1_m5.backtest import (
    M15_PATH,
    apply_m15_execution,
    monte_carlo_10k,
    score_panel,
)
from research.mtf_h4_h1_m5.m5_engine import build_m15_bars
from simulation.wf.sim import load_h1

OUT = _ROOT / "results/h4_h1_m5/h1_reversal"
CHARTS = OUT / "charts"
DOCS = _ROOT / "docs/research/h1_reversal_report.md"
PATH52 = _ROOT / "artifacts/pipeline_backtest/exit_structure/sprint52/sprint52_m5_path.parquet"
PANEL = _ROOT / "results/h4_h1_m5/panels/entry_panel_h1fs_A_core_6_top21.parquet"
H1_PATH = _ROOT / "artifacts/raw/XAUUSD/H1/data.parquet"
SEED = 42

# A priori (not searched on OOS) — from Sprint 53/54 style gates
WEAK_R = -0.50
CONFIRM_BARS = 2  # consecutive M5 bars with current_R <= WEAK_R
IS_YEARS = {2021}


def phase_labels() -> dict[str, Any]:
    """Incidence of candidate A/B labels on H1 closes (diagnostic, not trading)."""
    OUT.mkdir(parents=True, exist_ok=True)
    h1 = load_h1(H1_PATH)
    # sample every H1 bar after warmup as a hypothetical LONG/SHORT thesis point
    idx = np.arange(50, len(h1) - 10, 5)
    ya_l = label_opposite_barrier(h1, side="long", decision_idx=idx, params=BarrierReversalParams())
    ya_s = label_opposite_barrier(h1, side="short", decision_idx=idx, params=BarrierReversalParams())
    yb_l = label_thesis_failure(h1, side="long", decision_idx=idx, params=ThesisFailureParams())
    yb_s = label_thesis_failure(h1, side="short", decision_idx=idx, params=ThesisFailureParams())
    # year split
    ts = pd.to_datetime(h1["timestamp"], utc=True).to_numpy()
    years = pd.Series(ts[idx]).dt.year.to_numpy()
    rows = []
    for name, y in (
        ("A_long", ya_l),
        ("A_short", ya_s),
        ("B_long", yb_l),
        ("B_short", yb_s),
    ):
        for yr in sorted(set(years.tolist())):
            m = years == yr
            yy = y[m]
            yy = yy[np.isfinite(yy)]
            rows.append(
                {
                    "label": name,
                    "year": int(yr),
                    "oos": int(yr) in TRUE_OOS,
                    "n": int(len(yy)),
                    "rate": float(np.mean(yy)) if len(yy) else 0.0,
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "label_incidence.csv", index=False)
    summary = {
        "A_long_oos_rate": float(df[(df.label == "A_long") & df.oos]["rate"].mean()),
        "A_short_oos_rate": float(df[(df.label == "A_short") & df.oos]["rate"].mean()),
        "B_long_oos_rate": float(df[(df.label == "B_long") & df.oos]["rate"].mean()),
        "B_short_oos_rate": float(df[(df.label == "B_short") & df.oos]["rate"].mean()),
        "definition_A": "opposite 1ATR before favor 1ATR within 8 H1",
        "definition_B": "adverse 1ATR with MFE<0.25ATR within 8 H1",
        "note": "Entry TB labels are NOT these; separate management labels.",
    }
    (OUT / "label_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("labels", summary)
    return summary


def phase_mae() -> dict[str, Any]:
    """MAE/MFE recovery vs deterioration using Sprint 52 path (OOS)."""
    if not PATH52.is_file():
        print("skip mae — no sprint52 path")
        return {}
    OUT.mkdir(parents=True, exist_ok=True)
    p = pd.read_parquet(PATH52)
    p = p[p["year"].isin(TRUE_OOS)].copy()
    # trade-level finals
    t = p.groupby("trade_id").agg(
        year=("year", "first"),
        final_R=("final_R", "first"),
        final_MAE=("final_MAE", "first"),
        final_MFE=("final_MFE", "first"),
        p0_R=("p0_R", "first"),
        max_mae=("mae_so_far_R", "max"),
        max_mfe=("mfe_so_far_R", "max"),
    ).reset_index()
    t["win"] = t["final_R"] > 0
    # recovery after MAE>=0.5: did final_R > 0?
    deep = p[p["mae_so_far_R"] >= 0.50]
    first_deep = deep.groupby("trade_id").head(1)
    merged = first_deep.merge(t[["trade_id", "final_R", "win"]], on="trade_id", how="left")
    recover_rate = float(merged["win"].mean()) if len(merged) else 0.0

    # IS MAE quantile of losers (2021 only if present)
    is_path = pd.read_parquet(PATH52)
    is_t = is_path[is_path["year"].isin(IS_YEARS)].groupby("trade_id").agg(
        final_R=("final_R", "first"), max_mae=("mae_so_far_R", "max")
    )
    losers = is_t[is_t["final_R"] <= 0]["max_mae"]
    mae_p50 = float(losers.quantile(0.50)) if len(losers) else 0.75
    mae_p75 = float(losers.quantile(0.75)) if len(losers) else 1.0

    summary = {
        "oos_trades": int(t["trade_id"].nunique()),
        "oos_winrate": float(t["win"].mean()),
        "median_final_MAE": float(t["final_MAE"].median()),
        "median_final_MFE": float(t["final_MFE"].median()),
        "recover_after_mae_ge_0.50": recover_rate,
        "n_first_deep_mae": int(len(merged)),
        "is_loser_mae_p50": mae_p50,
        "is_loser_mae_p75": mae_p75,
    }
    t.to_csv(OUT / "trade_mae_mfe_oos.csv", index=False)
    (OUT / "mae_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    CHARTS.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(t.loc[~t["win"], "final_MAE"], t.loc[~t["win"], "final_MFE"], s=8, alpha=0.3, label="loss")
    ax.scatter(t.loc[t["win"], "final_MAE"], t.loc[t["win"], "final_MFE"], s=8, alpha=0.3, label="win")
    ax.set_xlabel("final MAE (R)")
    ax.set_ylabel("final MFE (R)")
    ax.set_title("OOS MAE vs MFE (Sprint52 path / P0)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHARTS / "mae_mfe_recovery.png", dpi=120)
    plt.close(fig)
    print("mae", summary)
    return summary


def _path_early_exit_r(path: pd.DataFrame, *, weak_r: float, confirm: int) -> pd.DataFrame:
    """Per trade: R if exit after `confirm` consecutive bars with current_R<=weak_r, else final_R."""
    rows = []
    for tid, g in path.groupby("trade_id"):
        g = g.sort_values("bars_since_entry")
        streak = 0
        exit_r = None
        exit_bar = None
        for _, row in g.iterrows():
            if float(row["current_R"]) <= weak_r:
                streak += 1
            else:
                streak = 0
            if streak >= confirm:
                # exit at next open approx = current_R (path uses mark)
                exit_r = float(row["current_R"])
                exit_bar = int(row["bars_since_entry"])
                break
        final_r = float(g.iloc[-1]["final_R"])
        rows.append(
            {
                "trade_id": tid,
                "year": int(g.iloc[0]["year"]),
                "p0_R": final_r,
                "early_R": exit_r if exit_r is not None else final_r,
                "early_exit": exit_r is not None,
                "exit_bar": exit_bar if exit_bar is not None else int(g.iloc[-1]["bars_since_entry"]),
                "premature": bool(exit_r is not None and final_r > exit_r + 1e-9),
                "saved": bool(exit_r is not None and final_r < exit_r - 1e-9),
            }
        )
    return pd.DataFrame(rows)


def phase_baselines() -> dict[str, Any]:
    """Exit overlay vs P0 using path marks (a priori weak_r)."""
    if not PATH52.is_file():
        return {}
    OUT.mkdir(parents=True, exist_ok=True)
    path = pd.read_parquet(PATH52)
    oos = path[path["year"].isin(TRUE_OOS)]
    cmp = _path_early_exit_r(oos, weak_r=WEAK_R, confirm=CONFIRM_BARS)
    cmp.to_csv(OUT / "exit_overlay_weak_r.csv", index=False)

    def _pf(r: np.ndarray) -> float:
        r = np.asarray(r, dtype=float)
        gp = r[r > 0].sum()
        gl = -r[r < 0].sum()
        return float(gp / gl) if gl > 1e-12 else (999.0 if gp > 0 else 0.0)

    def _dd(r: np.ndarray) -> float:
        eq = np.cumsum(r)
        peak = np.maximum.accumulate(eq)
        dd = (peak - eq) / np.maximum(np.abs(peak), 1e-9)
        return float(np.nanmax(dd)) if len(dd) else 0.0

    p0 = cmp["p0_R"].to_numpy()
    early = cmp["early_R"].to_numpy()
    summary = {
        "weak_r": WEAK_R,
        "confirm_bars": CONFIRM_BARS,
        "n": int(len(cmp)),
        "early_exit_rate": float(cmp["early_exit"].mean()),
        "premature_exit_rate": float(cmp["premature"].mean()),
        "saved_rate": float(cmp["saved"].mean()),
        "p0_avg_r": float(np.mean(p0)),
        "early_avg_r": float(np.mean(early)),
        "p0_pf": _pf(p0),
        "early_pf": _pf(early),
        "delta_avg_r": float(np.mean(early) - np.mean(p0)),
        "delta_pf": float(_pf(early) - _pf(p0)),
        "note": "Path from Sprint52 (FEAT7-era entries). Overlay is diagnostic vs that P0, not prod v6 panel.",
    }
    mc0 = monte_carlo_10k(p0, seed=SEED)
    mc1 = monte_carlo_10k(early, seed=SEED)
    summary["p0_mc_median_dd"] = mc0["median_dd"]
    summary["early_mc_median_dd"] = mc1["median_dd"]
    summary["p0_mc_ruin"] = mc0["prob_ruin"]
    summary["early_mc_ruin"] = mc1["prob_ruin"]
    (OUT / "baseline_exit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    CHARTS.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["P0", "Early weak_r"], [summary["p0_avg_r"], summary["early_avg_r"]], color=["#3b82f6", "#ef4444"])
    ax.set_ylabel("Avg R (OOS)")
    ax.set_title("Exit comparison (path overlay)")
    fig.tight_layout()
    fig.savefig(CHARTS / "exit_comparison.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["premature", "saved", "no_early"], [
        summary["premature_exit_rate"],
        summary["saved_rate"],
        1.0 - summary["early_exit_rate"],
    ])
    ax.set_ylim(0, 1)
    ax.set_title("Early-exit quality rates")
    fig.tight_layout()
    fig.savefig(CHARTS / "premature_exit_rate.png", dpi=120)
    plt.close(fig)

    print("baselines", summary)
    return summary


def phase_decision_demo() -> None:
    """Sample trade event log from decision engine (synthetic probs)."""
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = DecisionConfig(confirm_bars=2)
    events = []
    # scenario 1: hold
    pos = PositionState(side="long", unrealized_r=0.72, mfe_r=0.8, mae_r=0.2)
    r = evaluate(
        market=MarketState(price=2650),
        p_long=0.68, p_short=0.22, reversal_prob=0.14,
        reversal_stage="normal", h4_context="supportive", m5_state="recovering",
        position=pos, cfg=cfg,
    )
    events.append({"scenario": "hold", "action": r.action.value, "reason": r.reason, **r.detail})
    # scenario 2: weaken then exit
    pos = PositionState(side="long", unrealized_r=-0.4, weakening_bars=0)
    r1 = evaluate(
        market=MarketState(price=2640),
        p_long=0.31, p_short=0.71, reversal_prob=0.6,
        reversal_stage="normal", h4_context="neutral", m5_state="deteriorating",
        position=pos, cfg=cfg,
    )
    pos.weakening_bars = 2
    pos.reversal_stage = "weakening"
    r2 = evaluate(
        market=MarketState(price=2635),
        p_long=0.28, p_short=0.74, reversal_prob=0.82,
        reversal_stage="weakening", h4_context="conflicting", m5_state="deteriorating",
        position=pos, cfg=cfg,
    )
    events.append({"scenario": "weaken", "action": r1.action.value, "reason": r1.reason})
    events.append({"scenario": "confirm_exit", "action": r2.action.value, "reason": r2.reason, **r2.detail})
    pd.DataFrame(events).to_csv(OUT / "decision_event_log_demo.csv", index=False)


def phase_prod_panel_score() -> dict[str, Any]:
    """Reference: current prod-aligned P0 on H1 native panel (no overlay)."""
    if not PANEL.is_file():
        return {}
    panel = pd.read_parquet(PANEL)
    m15_raw = pd.read_parquet(_ROOT / M15_PATH)
    m15_bars = build_m15_bars(m15_raw)
    ex, _ = apply_m15_execution(panel, m15_bars, strategy="pullback_recovery")
    sc = score_panel(ex, m15=m15_raw, exit_tf="M15", entry_mode="m15_pullback")
    po = sc["po_oos"]
    out = {
        "policy": "prod_v6_p0_trail",
        "pf": float(po["pf"]),
        "dd": float(po["dd"]),
        "avg_r": float(po["avg_r"]),
        "trades": int(po["trades"]),
    }
    (OUT / "prod_panel_p0.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    ydf = pd.DataFrame(sc["yrows"])
    if not ydf.empty:
        ydf.to_csv(OUT / "prod_panel_yearly.csv", index=False)
    print("prod_p0", out)
    return out


def phase_report(lab: dict, mae: dict, base: dict, prod: dict) -> None:
    CHARTS.mkdir(parents=True, exist_ok=True)
    for name, title in (
        ("reversal_probability_before_after.png", "Reversal ML not fitted this pass — baselines only"),
        ("long_reversal_probability.png", "See label_incidence A_long / B_long"),
        ("short_reversal_probability.png", "See label_incidence A_short / B_short"),
        ("time_to_reversal.png", "Deferred — needs bar-level lead-time study"),
        ("missed_reversal_rate.png", "Use premature/saved rates in exit overlay"),
        ("walk_forward_reversal_performance.png", "Prod P0 yearly in prod_panel_yearly.csv"),
        ("monte_carlo_exit_comparison.png", "MC DD/ruin in baseline_exit_summary.json"),
    ):
        if not (CHARTS / name).is_file():
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.axis("off")
            ax.text(0.5, 0.5, title, ha="center", va="center")
            fig.savefig(CHARTS / name, dpi=100)
            plt.close(fig)

    verdict = "KEEP_TRAIL_ONLY"
    if base:
        if base.get("delta_avg_r", -1) > 0.01 and base.get("premature_exit_rate", 1) < 0.35:
            verdict = "PROMOTE_DECISION_BASELINE"
        elif base.get("delta_avg_r", 0) >= -0.005 and base.get("premature_exit_rate", 1) < 0.25:
            verdict = "RESEARCH_MORE"

    text = "\n".join([
        "# H1 Reversal & Position Decision Report",
        "",
        f"**Verdict:** `{verdict}`",
        "",
        "## 1. Current architecture",
        "",
        "H1 native 6 → top 21% → M15 pullback → M15 ATR trail (P0). No HOLD/EXIT/WAIT engine in production.",
        "Audit: `docs/research/h1_reversal_audit.md`.",
        "",
        "## 2–3. LONG/SHORT labels & models",
        "",
        "Triple-barrier entry labels (TP/SL/TIMEOUT, horizon 8). **Not** reversal labels.",
        "Separate LONG/SHORT LGBM; H1 native 6. Untouched this sprint.",
        "",
        "## 4–5. Reversal definition & candidates",
        "",
        "- **A** Opposite 1 ATR before favor 1 ATR within 8 H1",
        "- **B** Adverse 1 ATR with MFE < 0.25 ATR within 8 H1",
        "- **C** IS loser MAE quantiles (see mae_summary)",
        "",
        f"```json\n{json.dumps(lab, indent=2)}\n```" if lab else "",
        "",
        "## 6. Feature research",
        "",
        "Sprints 45/53/54: do not ship bounce-based recovery exits; time/adverse velocity dominate.",
        "Keep reversal feature research separate from LONG/SHORT primary.",
        "",
        "## 7. Baseline rules",
        "",
        "DecisionEngine: NORMAL → WEAKENING → CONFIRMED; no direct reverse; cooldown; no pyramid.",
        f"Path overlay (a priori): {CONFIRM_BARS} bars with current_R ≤ {WEAK_R}.",
        "",
        f"```json\n{json.dumps(base, indent=2)}\n```" if base else "",
        "",
        "## 8–10. WF / directions",
        "",
        f"```json\n{json.dumps(prod, indent=2)}\n```" if prod else "Prod panel score skipped.",
        "",
        "Path overlay uses Sprint52 path (legacy FEAT7 entries) — directional evidence only.",
        "",
        "## 11–12. MAE/MFE",
        "",
        f"```json\n{json.dumps(mae, indent=2)}\n```" if mae else "",
        "",
        "## 13–15. Decision engine / exit / MC",
        "",
        "Code: `research/h1_reversal/decision.py`. Tests: `tests/test_h1_reversal.py`.",
        "Artifacts: `results/h4_h1_m5/h1_reversal/`.",
        "",
        "## 16. Final recommendation",
        "",
        f"**{verdict}**",
        "",
        "1–2. Reversal = A/B position-aware labels, not TB entry.",
        f"5. Premature early-exit rate ≈ {base.get('premature_exit_rate', 'n/a')}",
        f"6–8. ΔAvgR ≈ {base.get('delta_avg_r', 'n/a')}; simple weak_r overlay does not replace trail.",
        "12. Running trade → HOLD / WAIT_FOR_CONFIRMATION / EXIT_confirmed → FLAT (no flip).",
        "",
        "Diagnosis: missing **decision logic** layer. DecisionEngine research-ready; keep P0 trail until",
        "an overlay beats OOS on **prod v6** entries.",
        "",
        f"Seed={SEED}. Production unchanged.",
        "",
    ])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.md").write_text(text, encoding="utf-8")
    DOCS.write_text(text, encoding="utf-8")
    (OUT / "verdict.json").write_text(
        json.dumps({"verdict": verdict, "lab": lab, "mae": mae, "base": base, "prod": prod, "seed": SEED}, indent=2),
        encoding="utf-8",
    )
    print("verdict", verdict)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--phase",
        default="all",
        choices=("labels", "mae", "baselines", "decision_demo", "prod", "report", "all"),
    )
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    lab: dict = {}
    mae: dict = {}
    base: dict = {}
    prod: dict = {}
    if args.phase in ("labels", "all"):
        lab = phase_labels()
    if args.phase in ("mae", "all"):
        mae = phase_mae()
    if args.phase in ("baselines", "all"):
        base = phase_baselines()
    if args.phase in ("decision_demo", "all"):
        phase_decision_demo()
    if args.phase in ("prod", "all"):
        prod = phase_prod_panel_score()
    if args.phase in ("report", "all"):
        if not lab and (OUT / "label_summary.json").is_file():
            lab = json.loads((OUT / "label_summary.json").read_text(encoding="utf-8"))
        if not mae and (OUT / "mae_summary.json").is_file():
            mae = json.loads((OUT / "mae_summary.json").read_text(encoding="utf-8"))
        if not base and (OUT / "baseline_exit_summary.json").is_file():
            base = json.loads((OUT / "baseline_exit_summary.json").read_text(encoding="utf-8"))
        if not prod and (OUT / "prod_panel_p0.json").is_file():
            prod = json.loads((OUT / "prod_panel_p0.json").read_text(encoding="utf-8"))
        phase_report(lab, mae, base, prod)
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
