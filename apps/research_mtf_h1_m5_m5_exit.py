"""H1 + M5 pullback entry with M5 trailing exit (trail clock after H1 close).

Compare vs M15 exit (production clock) under same entry set.

  python apps/research_mtf_h1_m5_m5_exit.py

Outputs: results/h4_h1_m5/m5_exit/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import tools.pyc_path_hook  # noqa: F401

from apps.research_sprint42_attribution import TRUE_OOS, _md
from production import PIPELINE_VERSION
from research.mtf_h4_h1_m5.backtest import (
    M15_PATH,
    M5_PATH,
    apply_m5_execution,
    monte_carlo_10k,
    score_panel,
)
from research.mtf_h4_h1_m5.m5_engine import build_m5_features

OUT = _ROOT / "results/h4_h1_m5/m5_exit"
PANEL_DIR = _ROOT / "results/h4_h1_m5/panels"

POLICIES = (
    ("production_feat7_m15", "production_feat7", "immediate", "M15", "h1_close"),
    ("h1_immediate_m15", "h1_baseline", "immediate", "M15", "h1_close"),
    ("h1_pullback_m15_proper", "h1_baseline", "pullback_recovery", "M15", "m5_pullback"),
    ("h1_pullback_m5", "h1_baseline", "pullback_recovery", "M5", "m5_pullback"),
)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    m15 = pd.read_parquet(_ROOT / M15_PATH)
    m5_raw = pd.read_parquet(_ROOT / M5_PATH)
    m5_feat = build_m5_features(m5_raw)

    panels = {
        "production_feat7": pd.read_parquet(PANEL_DIR / "entry_panel_production_feat7_top21.parquet"),
        "h1_baseline": pd.read_parquet(PANEL_DIR / "entry_panel_h1_baseline_top21.parquet"),
    }
    for k in panels:
        panels[k]["timestamp"] = pd.to_datetime(panels[k]["timestamp"], utc=True)

    executed_cache: dict[tuple[str, str], pd.DataFrame] = {}
    rows = []
    for name, src, strat, exit_tf, entry_mode in POLICIES:
        key = (src, strat)
        if key not in executed_cache:
            if strat == "immediate":
                executed_cache[key] = panels[src].copy()
            else:
                ex, _ = apply_m5_execution(panels[src], m5_feat, strategy="pullback_recovery")
                executed_cache[key] = ex
        panel = executed_cache[key]
        sc = score_panel(
            panel, m15=m15, m5=m5_raw,
            exit_tf=exit_tf, entry_mode=entry_mode,
        )
        po = sc["po_oos"]
        mc = monte_carlo_10k(po["pnl"])
        rows.append({
            "policy": name,
            "entry": strat,
            "exit_tf": exit_tf,
            "entry_mode": entry_mode,
            "pf": po["pf"], "dd": po["dd"], "avg_r": po["avg_r"], "wr": po["wr"],
            "trades": po["trades"], "ret": po["ret"],
            "mc_prob_ruin": mc["prob_ruin"], "mc_median_dd": mc["median_dd"],
        })

    yearly = []
    for name, src, strat, exit_tf, entry_mode in POLICIES:
        panel = executed_cache[(src, strat)]
        sc = score_panel(panel, m15=m15, m5=m5_raw, exit_tf=exit_tf, entry_mode=entry_mode)
        for r in sc["yrows"]:
            yearly.append({
                "policy": name, "year": r["year"], "oos": r["year"] in TRUE_OOS,
                "pf": r["pf"], "dd": r["dd"], "trades": r["trades"], "avg_r": r["avg_r"],
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "oos_summary.csv", index=False)
    pd.DataFrame(yearly).to_csv(OUT / "yearly.csv", index=False)

    pb_m15 = df.loc[df["policy"] == "h1_pullback_m15_proper"].iloc[0]
    pb_m5 = df.loc[df["policy"] == "h1_pullback_m5"].iloc[0]
    prod = df.loc[df["policy"] == "production_feat7_m15"].iloc[0]

    if pb_m15["pf"] > prod["pf"] and pb_m15["dd"] <= prod["dd"]:
        verdict = "ITERATE"
    elif pb_m15["pf"] >= prod["pf"]:
        verdict = "ITERATE"
    else:
        verdict = "REJECT"

    lines = [
        "# H1 + M5 Pullback Entry × Trail Exit (M15 vs M5 clock)",
        "",
        f"**Verdict:** `{verdict}`",
        "",
        "**Target setup (M15 path):**",
        "- Entry: M5 `pullback_recovery` fill (after H1 close)",
        "- Exit: P0 a0.25/d0.08 trail looping each **M15 close**",
        "- Trail starts from first M15 bar at/after M5 fill (causal, after H1 close)",
        "- Horizon: 48 H1 bars from H1 close signal",
        "",
        f"Production ref: `{PIPELINE_VERSION}`",
        "",
        "## OOS comparison",
        "",
        _md(rows, ["policy", "entry", "exit_tf", "pf", "dd", "avg_r", "trades", "mc_prob_ruin"],
            {"pf": 2, "dd": 3, "avg_r": 3, "trades": 0, "mc_prob_ruin": 3}),
        "",
        "## Key delta: same pullback entry, exit clock",
        "",
        f"- Pullback + **M15 close** trail (proper): PF {pb_m15['pf']:.2f}, DD {pb_m15['dd']:.1%}, avgR {pb_m15['avg_r']:.3f}",
        f"- Pullback + **M5 close** trail: PF {pb_m5['pf']:.2f}, DD {pb_m5['dd']:.1%}, avgR {pb_m5['avg_r']:.3f}",
        f"- ΔPF (M5−M15 exit clock): {pb_m5['pf'] - pb_m15['pf']:+.2f}",
        "",
        "Production unchanged.",
    ]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "verdict.json").write_text(json.dumps({
        "verdict": verdict,
        "production_changed": False,
        "pullback_m15": pb_m15.to_dict(),
        "pullback_m5": pb_m5.to_dict(),
    }, indent=2), encoding="utf-8")

    print("VERDICT", verdict)
    print("pullback M15", pb_m15[["pf", "dd", "trades"]].to_dict())
    print("pullback M5 ", pb_m5[["pf", "dd", "trades"]].to_dict())
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
