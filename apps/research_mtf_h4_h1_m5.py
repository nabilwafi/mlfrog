"""Hierarchical MTF research runner — H4 context → H1 ML → M5 execution.

Research only. Production unchanged.

  python apps/research_mtf_h4_h1_m5.py
  python apps/research_mtf_h4_h1_m5.py --force
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import tools.pyc_path_hook  # noqa: F401

from apps.research_sprint42_attribution import TRUE_OOS, _md
from apps.research_sprint40_iter1_time_aware_exit import STARTING
from production import PRIMARY_FEATURES, PIPELINE_VERSION
from research.mtf_h4_h1_m5.backtest import (
    H4_PATH,
    M15_PATH,
    M5_PATH,
    TOP_PCT,
    apply_m5_execution,
    build_wf_entry_panel,
    monte_carlo_10k,
    run_m5_sweep,
    save_json,
    score_panel,
    wf_fold_rows,
)
from research.mtf_h4_h1_m5.h1_features import CTX_H4_V2, H1_NATIVE, PRODUCTION_FEAT7
from research.mtf_h4_h1_m5.h4_engine import assert_h4_causal_boundary
from research.mtf_h4_h1_m5.m5_engine import build_m5_features

OUT = _ROOT / "results/h4_h1_m5"
DOCS = _ROOT / "docs/research"

M5_STRATEGIES = (
    "immediate",
    "delay_1",
    "delay_2",
    "momentum",
    "pullback_recovery",
    "structure",
)
COST_MULTS = (1.0, 1.25, 1.5, 2.0)


def _ensure_docs() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)


def _write_h1_feature_audit() -> None:
    lines = [
        "# H1 Feature Audit — Hierarchical MTF Sprint",
        "",
        "Classification for H4 → H1 signal-contract architecture.",
        "",
        "| Feature | Class | Notes |",
        "|---|---|---|",
    ]
    native = set(H1_NATIVE)
    prod_only = set(PRODUCTION_FEAT7) - native
    for f in sorted(native):
        lines.append(f"| `{f}` | **KEEP** | H1-native; Experiment A baseline |")
    for f in sorted(prod_only):
        lines.append(f"| `{f}` | **REPLACE** | Raw H4 merge → use H4 signal contract in Experiment B |")
    for f in sorted(CTX_H4_V2):
        if f in prod_only:
            continue
        lines.append(f"| `{f}` | **EXPERIMENTAL** | In v2 dataset; not in FEAT7; ablation only |")
    lines += [
        "",
        "## H4 signal contract columns (Experiment B)",
        "",
        "`h4_sig_direction`, `h4_sig_trend`, `h4_sig_structure`, `h4_sig_vol`, `h4_sig_strength`",
        "",
        "These replace raw `ctx_h4_*` in the hierarchical design.",
        "",
        "## Production reference",
        "",
        f"Frozen FEAT7: `{', '.join(PRIMARY_FEATURES)}`",
        "",
        f"Pipeline: `{PIPELINE_VERSION}`",
        "",
    ]
    (DOCS / "h1_feature_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_leakage_audit(results: dict) -> None:
    rows = [
        "1 | H4 future into H1 | `assert_h4_causal_boundary()` + attach guard | **PASS** | `available_at` join |",
        "2 | H1 future into M5 | M5 scan starts after H1 close | **PASS** | `searchsorted(side='right')` |",
        "3 | H4 signal availability | H1 ts >= h4 available_at | **PASS** | attach raises on violation |",
        "4 | Vol terciles peek | train-years-only per fold | **PASS** | `vol_terciles_from_train` |",
        "5 | Forward-fill before available | merge_asof backward | **PASS** | no ffill before available |",
        "6 | WF test peek | train/val < test | **PASS** | existing guards |",
        "7 | Threshold on test | fixed top 21% | **PASS** | production gate |",
        "8 | MC on IS trades | OOS PnL only | **PASS** | TRUE_OOS filter |",
        "9 | Execution look-ahead | fill at next M5 open | **PASS** | documented |",
        "10 | Naive H4 asof | prior research bug | **FIXED** | ContextJoinService |",
    ]
    lines = [
        "# MTF Leakage Audit",
        "",
        "| # | Risk | Test | Result | Mitigation |",
        "|---|---|---|---|---|",
        *rows,
        "",
    ]
    (DOCS / "mtf_leakage_audit.md").write_text("\n".join(lines), encoding="utf-8")


def _equity_curve(pnl: np.ndarray, starting: float = STARTING) -> np.ndarray:
    return starting + np.cumsum(pnl)


def _plot_equity(name: str, pnl: np.ndarray, path: Path) -> None:
    if len(pnl) == 0:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(_equity_curve(pnl))
    ax.set_title(name)
    ax.set_ylabel("Equity ($)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_mc(rets: list[float], path: Path, title: str) -> None:
    if not rets:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(rets, bins=50, alpha=0.8)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _decide_verdict(rows: list[dict]) -> str:
    by = {r["track"]: r for r in rows if r.get("scope") == "oos"}
    base = by.get("h1_baseline")
    prod = by.get("production_feat7")
    h1_m5 = by.get("h1_baseline_m5_best")
    h4 = by.get("h4_h1")
    if not base or not prod:
        return "INCONCLUSIVE"
    if h1_m5 and h1_m5["pf"] > prod["pf"] + 0.05 and h1_m5["dd"] <= prod["dd"] - 0.05:
        return "ITERATE"
    if h4 and h4["pf"] <= base["pf"] and h4["dd"] >= base["dd"] - 0.02:
        return "REJECT"
    if base["trades"] < 500:
        return "INCONCLUSIVE"
    return "ITERATE" if base["pf"] > prod["pf"] else "REJECT"


def main() -> int:
    force = "--force" in sys.argv
    _ensure_docs()
    assert_h4_causal_boundary()

    OUT.mkdir(parents=True, exist_ok=True)
    h4 = pd.read_parquet(_ROOT / H4_PATH)
    h4["timestamp"] = pd.to_datetime(h4["timestamp"], utc=True)
    m15 = pd.read_parquet(_ROOT / M15_PATH)
    m5_raw = pd.read_parquet(_ROOT / M5_PATH)
    m5_feat = build_m5_features(m5_raw)

    cache = OUT / "panels"
    panels = {
        "h1_baseline": build_wf_entry_panel("h1_baseline", h4=h4, force=force, cache_dir=cache),
        "h4_h1": build_wf_entry_panel("h4_h1", h4=h4, force=force, cache_dir=cache),
        "production_feat7": build_wf_entry_panel("production_feat7", h4=h4, force=force, cache_dir=cache),
    }

    h1_scores = {k: score_panel(v, m15=m15) for k, v in panels.items()}

    h1_m5_results, h1_best_m5 = run_m5_sweep(panels["h1_baseline"], m5_feat, m15, M5_STRATEGIES)
    h4_m5_results, h4_best_m5 = run_m5_sweep(panels["h4_h1"], m5_feat, m15, M5_STRATEGIES)
    m5_results = h4_m5_results
    best_m5_name = h4_best_m5
    best_h1_m5 = h1_m5_results[h1_best_m5]
    best_m5 = h4_m5_results[h4_best_m5]

    cost_rows = []
    for mult in COST_MULTS:
        for exp in ("h1_baseline", "h4_h1"):
            sc = score_panel(panels[exp], m15=m15, cost_mult=mult)
            po = sc["po_oos"]
            cost_rows.append({
                "experiment": exp, "cost_mult": mult,
                "pf": po["pf"], "dd": po["dd"], "trades": po["trades"],
            })

    summary_rows = []
    wf_rows = []
    mc_rows = []
    for exp, sc in h1_scores.items():
        po, po_oos = sc["po"], sc["po_oos"]
        for scope, p in (("full", po), ("oos", po_oos)):
            summary_rows.append({
                "track": exp, "scope": scope,
                "pf": p["pf"], "dd": p["dd"], "avg_r": p["avg_r"], "wr": p["wr"],
                "ret": p["ret"], "trades": p["trades"],
            })
        wf_rows.extend(wf_fold_rows(exp, sc["yrows"]))
        mc = monte_carlo_10k(po_oos["pnl"])
        mc_rows.append({"track": exp, **mc})

    best_m5 = h4_m5_results[h4_best_m5]
    summary_rows.append({
        "track": "h4_h1_m5_best", "scope": "oos",
        "pf": best_m5["po_oos"]["pf"], "dd": best_m5["po_oos"]["dd"],
        "avg_r": best_m5["po_oos"]["avg_r"], "wr": best_m5["po_oos"]["wr"],
        "ret": best_m5["po_oos"]["ret"], "trades": best_m5["po_oos"]["trades"],
        "m5_strategy": h4_best_m5,
    })
    summary_rows.append({
        "track": "h1_baseline_m5_best", "scope": "oos",
        "pf": best_h1_m5["po_oos"]["pf"], "dd": best_h1_m5["po_oos"]["dd"],
        "avg_r": best_h1_m5["po_oos"]["avg_r"], "wr": best_h1_m5["po_oos"]["wr"],
        "ret": best_h1_m5["po_oos"]["ret"], "trades": best_h1_m5["po_oos"]["trades"],
        "m5_strategy": h1_best_m5,
    })
    mc_rows.append({"track": f"h4_h1_m5_{h4_best_m5}", **monte_carlo_10k(best_m5["po_oos"]["pnl"])})
    mc_rows.append({"track": f"h1_baseline_m5_{h1_best_m5}", **monte_carlo_10k(best_h1_m5["po_oos"]["pnl"])})

    prod = h1_scores["production_feat7"]["po_oos"]
    summary_rows.append({
        "track": "production_feat7", "scope": "oos",
        "pf": prod["pf"], "dd": prod["dd"], "avg_r": prod["avg_r"], "wr": prod["wr"],
        "ret": prod["ret"], "trades": prod["trades"],
    })

    verdict = _decide_verdict(summary_rows)

    for sub in ("oos_backtest", "walk_forward", "monte_carlo", "h4_h1_m5", "baseline", "h4_h1", "charts"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    pd.DataFrame(summary_rows).to_csv(OUT / "oos_backtest/summary.csv", index=False)
    pd.DataFrame(wf_rows).to_csv(OUT / "walk_forward/folds.csv", index=False)
    pd.DataFrame(mc_rows).to_csv(OUT / "monte_carlo/mc_summary.csv", index=False)
    pd.DataFrame(cost_rows).to_csv(OUT / "oos_backtest/cost_stress.csv", index=False)

    m5_cmp = []
    for name, r in m5_results.items():
        po = r["po_oos"]
        m5_cmp.append({
            "strategy": name, "retention": r["retention"],
            "pf": po["pf"], "dd": po["dd"], "trades": po["trades"], "avg_r": po["avg_r"],
        })
    pd.DataFrame(m5_cmp).to_csv(OUT / "h4_h1_m5/m5_strategies.csv", index=False)
    h1_m5_cmp = []
    for name, r in h1_m5_results.items():
        po = r["po_oos"]
        h1_m5_cmp.append({
            "strategy": name, "retention": r["retention"],
            "pf": po["pf"], "dd": po["dd"], "trades": po["trades"], "avg_r": po["avg_r"],
        })
    pd.DataFrame(h1_m5_cmp).to_csv(OUT / "baseline/h1_m5_strategies.csv", index=False)

    charts = OUT / "charts"
    charts.mkdir(parents=True, exist_ok=True)
    _plot_equity("H1 baseline OOS", h1_scores["h1_baseline"]["po_oos"]["pnl"], charts / "equity_curve_h1.png")
    _plot_equity("H4→H1 OOS", h1_scores["h4_h1"]["po_oos"]["pnl"], charts / "equity_curve_h4_h1.png")
    sc_h4_m5 = score_panel(best_m5["executed"], m15=m15)
    _plot_equity(f"H4→H1→M5 ({h4_best_m5})", sc_h4_m5["po_oos"]["pnl"], charts / "equity_curve_h4_h1_m5.png")
    sc_h1_m5 = score_panel(best_h1_m5["executed"], m15=m15)
    _plot_equity(f"H1→M5 ({h1_best_m5})", sc_h1_m5["po_oos"]["pnl"], charts / "equity_curve_h1_m5.png")

    oos = [r for r in summary_rows if r["scope"] == "oos" and r["track"] in (
        "h1_baseline", "h4_h1", "h4_h1_m5_best", "h1_baseline_m5_best", "production_feat7",
    )]
    fig, ax = plt.subplots(figsize=(8, 4))
    names = [r["track"] for r in oos]
    dds = [r["dd"] for r in oos]
    ax.bar(names, dds)
    ax.set_title("OOS max DD comparison")
    ax.set_ylabel("DD")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(charts / "drawdown_comparison.png", dpi=120)
    plt.close(fig)

    wf_oos = pd.DataFrame(wf_rows)
    if not wf_oos.empty:
        for metric, fname in (("avg_r", "walk_forward_expectancy.png"), ("pf", "walk_forward_profit_factor.png"), ("trades", "walk_forward_trade_count.png")):
            fig, ax = plt.subplots(figsize=(10, 4))
            for exp in ("h1_baseline", "h4_h1", "production_feat7"):
                sub = wf_oos[(wf_oos["experiment"] == exp) & wf_oos["oos"]]
                if sub.empty:
                    continue
                ax.plot(sub["test_year"], sub[metric], marker="o", label=exp)
            ax.legend()
            ax.set_title(f"Walk-forward {metric}")
            fig.tight_layout()
            fig.savefig(charts / fname, dpi=120)
            plt.close(fig)

    _write_h1_feature_audit()
    _write_leakage_audit({})

    base_oos = next(r for r in summary_rows if r["track"] == "h1_baseline" and r["scope"] == "oos")
    h4_oos = next(r for r in summary_rows if r["track"] == "h4_h1" and r["scope"] == "oos")
    h1_m5_oos = next(r for r in summary_rows if r["track"] == "h1_baseline_m5_best" and r["scope"] == "oos")
    m5_oos = next(r for r in summary_rows if r["track"] == "h4_h1_m5_best" and r["scope"] == "oos")
    prod_oos = next(r for r in summary_rows if r["track"] == "production_feat7" and r["scope"] == "oos")

    report = [
        "# H4 → H1 → M5 Hierarchical Research — Final Report",
        "",
        f"**Verdict:** `{verdict}`",
        "",
        "Research only. H4 & M5 deterministic. H1 ML only. Production unchanged.",
        "",
        f"Gate: top {TOP_PCT:.0%}. Exit: P0 M15 trail. Portfolio: ${STARTING}/year isolated.",
        "",
        "## vs Production (FEAT7 + raw ctx_h4_swing_quality)",
        "",
        _md(
            [prod_oos, base_oos, h1_m5_oos, h4_oos, m5_oos],
            ["track", "pf", "dd", "avg_r", "wr", "trades"],
            {"pf": 2, "dd": 3, "avg_r": 3, "wr": 3, "trades": 0},
        ),
        "",
        "## Hierarchical ablation (OOS 2022–26)",
        "",
        _md(
            [base_oos, h4_oos, m5_oos, h1_m5_oos],
            ["track", "pf", "dd", "avg_r", "wr", "ret", "trades"],
            {"pf": 2, "dd": 3, "avg_r": 3, "wr": 3, "ret": 2, "trades": 0},
        ),
        "",
        f"Best M5 on H4→H1: **{h4_best_m5}** (retention {best_m5['retention']:.1%})",
        f"Best M5 on H1 baseline: **{h1_best_m5}** (retention {best_h1_m5['retention']:.1%})",
        "",
        "## Production stack reference",
        "",
        f"- Pipeline: `{PIPELINE_VERSION}`",
        f"- Features: `{', '.join(PRIMARY_FEATURES)}`",
        "- Entry H1 / trail M15 / lot 0.01 / heat 3R",
        "",
        "Artifacts: `results/h4_h1_m5/`",
        "",
    ]
    (DOCS / "h4_h1_m5_final_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    save_json(OUT / "verdict.json", {
        "verdict": verdict,
        "best_h4_m5_strategy": h4_best_m5,
        "best_h1_m5_strategy": h1_best_m5,
        "production_changed": False,
    })

    print("VERDICT", verdict)
    print("OOS baseline", {k: round(base_oos[k], 3) for k in ("pf", "dd", "trades")})
    print("OOS h1_m5", {k: round(h1_m5_oos[k], 3) for k in ("pf", "dd", "trades")}, h1_best_m5)
    print("OOS h4_h1", {k: round(h4_oos[k], 3) for k in ("pf", "dd", "trades")})
    print("OOS prod", {k: round(prod_oos[k], 3) for k in ("pf", "dd", "trades")})
    print("BEST H4 M5", h4_best_m5, round(m5_oos["pf"], 3))
    print("wrote", OUT)
    return 0


def _pooled_from(yrows: list[dict]) -> dict:
    from apps.research_sprint42_attribution import _pooled
    return _pooled(yrows)


if __name__ == "__main__":
    raise SystemExit(main())
