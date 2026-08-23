"""D1 vs H4 context layer with M15 pullback entry + M15 trail exit.

  python apps/research_mtf_d1_vs_h4_pullback.py
  python apps/research_mtf_d1_vs_h4_pullback.py --force

Outputs: results/h4_h1_m5/d1_vs_h4/
"""
from __future__ import annotations

import argparse
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
    H4_PATH,
    M15_PATH,
    build_wf_entry_panel,
    monte_carlo_10k,
    score_panel,
)
from research.mtf_h4_h1_m5.d1_engine import assert_d1_causal_boundary, build_d1_from_h1
from research.mtf_h4_h1_m5.backtest import apply_m15_execution
from research.mtf_h4_h1_m5.m5_engine import build_m15_bars
from simulation.wf.sim import load_h1

OUT = _ROOT / "results/h4_h1_m5/d1_vs_h4"
PANEL_DIR = _ROOT / "results/h4_h1_m5/panels"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="Rebuild WF panels")
    args = ap.parse_args()

    assert_d1_causal_boundary()
    OUT.mkdir(parents=True, exist_ok=True)
    PANEL_DIR.mkdir(parents=True, exist_ok=True)

    h4 = pd.read_parquet(_ROOT / H4_PATH)
    h1 = load_h1(_ROOT / "artifacts/raw/XAUUSD/H1/data.parquet")
    d1 = build_d1_from_h1(h1)
    m15_raw = pd.read_parquet(_ROOT / M15_PATH)
    m15_bars = build_m15_bars(m15_raw)

    panels = {}
    for exp in ("h1_baseline", "h4_h1", "d1_h1"):
        panels[exp] = build_wf_entry_panel(
            exp, h4=h4, d1=d1, force=args.force, cache_dir=PANEL_DIR,
        )
        panels[exp]["timestamp"] = pd.to_datetime(panels[exp]["timestamp"], utc=True)

    rows = []
    yearly = []
    for exp, label in (
        ("h1_baseline", "h1_only"),
        ("h4_h1", "h4_context"),
        ("d1_h1", "d1_context"),
    ):
        ex, _ = apply_m15_execution(panels[exp], m15_bars, strategy="pullback_recovery")
        sc = score_panel(ex, m15=m15_raw, exit_tf="M15", entry_mode="m15_pullback")
        po = sc["po_oos"]
        mc = monte_carlo_10k(po["pnl"])
        rows.append({
            "context": label,
            "experiment": exp,
            "pf": po["pf"], "dd": po["dd"], "avg_r": po["avg_r"], "wr": po["wr"],
            "trades": po["trades"], "ret": po["ret"],
            "retention": len(ex) / len(panels[exp]),
            "mc_prob_ruin": mc["prob_ruin"], "mc_median_dd": mc["median_dd"],
        })
        for r in sc["yrows"]:
            yearly.append({
                "context": label, "year": r["year"], "oos": r["year"] in TRUE_OOS,
                "pf": r["pf"], "dd": r["dd"], "trades": r["trades"],
            })

    pd.DataFrame(rows).to_csv(OUT / "oos_summary.csv", index=False)
    pd.DataFrame(yearly).to_csv(OUT / "yearly.csv", index=False)

    h1 = next(r for r in rows if r["context"] == "h1_only")
    h4r = next(r for r in rows if r["context"] == "h4_context")
    d1r = next(r for r in rows if r["context"] == "d1_context")

    best = max(rows, key=lambda x: x["pf"])
    if best["context"] == "h1_only":
        verdict = "H1_ONLY_BEST"
    elif best["context"] == "d1_context" and d1r["pf"] > h1["pf"] and d1r["dd"] <= h1["dd"] + 0.05:
        verdict = "D1_ITERATE"
    elif best["context"] == "d1_context":
        verdict = "D1_MARGINAL"
    else:
        verdict = "REJECT_CONTEXT"

    lines = [
        "# D1 vs H4 Context + M15 Pullback Entry",
        "",
        f"**Verdict:** `{verdict}`",
        "",
        "Stack: context (optional) → H1 ML → **M15 pullback entry** → P0 **M15 trail** exit.",
        "D1 bars resampled from H1; same signal contract as H4 (`d1_sig_*`).",
        "",
        f"Ref: `{PIPELINE_VERSION}`",
        "",
        "## OOS",
        "",
        _md(rows, ["context", "pf", "dd", "avg_r", "trades", "retention", "mc_prob_ruin"],
            {"pf": 2, "dd": 3, "avg_r": 3, "trades": 0, "retention": 3, "mc_prob_ruin": 3}),
        "",
        "## vs H1-only baseline",
        "",
        f"- H4 context: ΔPF {h4r['pf']-h1['pf']:+.2f}, ΔDD {(h4r['dd']-h1['dd'])*100:+.1f}pp",
        f"- D1 context: ΔPF {d1r['pf']-h1['pf']:+.2f}, ΔDD {(d1r['dd']-h1['dd'])*100:+.1f}pp",
        "",
        "Production unchanged.",
    ]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "verdict.json").write_text(json.dumps({
        "verdict": verdict,
        "best": best["context"],
        "rows": rows,
    }, indent=2), encoding="utf-8")

    print("VERDICT", verdict)
    for r in rows:
        print(r["context"], {k: round(r[k], 3) for k in ("pf", "dd", "trades", "avg_r")})
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
