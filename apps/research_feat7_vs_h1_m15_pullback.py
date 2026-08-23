"""FEAT7 (production) vs H1-native FE under M15 pullback + M15 trail stack.

  python apps/research_feat7_vs_h1_m15_pullback.py

Outputs: results/h4_h1_m5/feat7_vs_h1/
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
from production import PIPELINE_VERSION, PRIMARY_FEATURES
from research.mtf_h4_h1_m5.backtest import (
    M15_PATH,
    apply_m15_execution,
    monte_carlo_10k,
    score_panel,
)
from research.mtf_h4_h1_m5.h1_features import H1_NATIVE, PRODUCTION_FEAT7
from research.mtf_h4_h1_m5.m5_engine import build_m15_bars

OUT = _ROOT / "results/h4_h1_m5/feat7_vs_h1"
PANEL_DIR = _ROOT / "results/h4_h1_m5/panels"
COST_MULTS = (1.0, 1.25, 1.5, 2.0)


def _run_panel(name: str, panel: pd.DataFrame, m15_raw: pd.DataFrame, m15_bars: pd.DataFrame) -> tuple[list, list, list]:
    panel = panel.copy()
    panel["timestamp"] = pd.to_datetime(panel["timestamp"], utc=True)
    n_src = len(panel)
    imm = panel.copy()
    pb_ex, _ = apply_m15_execution(panel, m15_bars, strategy="pullback_recovery")

    summary, yearly, cost_rows = [], [], []
    for pol, pnl, mode in (
        (f"{name}_immediate", imm, "h1_close"),
        (f"{name}_m15_pullback", pb_ex, "m15_pullback"),
    ):
        for mult in COST_MULTS:
            sc = score_panel(pnl, m15=m15_raw, exit_tf="M15", entry_mode=mode, cost_mult=mult)
            po = sc["po_oos"]
            row = {
                "policy": pol, "features": name, "entry": mode.replace("_", " "),
                "cost_mult": mult,
                "pf": po["pf"], "dd": po["dd"], "avg_r": po["avg_r"], "wr": po["wr"],
                "trades": po["trades"], "ret": po["ret"],
                "retention": len(pnl) / n_src if mode != "h1_close" else 1.0,
            }
            cost_rows.append(row)
            if mult == 1.0:
                mc = monte_carlo_10k(po["pnl"])
                row.update({f"mc_{k}": v for k, v in mc.items() if k in ("prob_ruin", "median_dd")})
                summary.append(row)
        sc1 = score_panel(pnl, m15=m15_raw, exit_tf="M15", entry_mode=mode, cost_mult=1.0)
        for r in sc1["yrows"]:
            yearly.append({
                "policy": pol, "features": name, "year": r["year"],
                "oos": r["year"] in TRUE_OOS, "pf": r["pf"], "dd": r["dd"], "trades": r["trades"],
            })
    return summary, yearly, cost_rows


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    m15_raw = pd.read_parquet(_ROOT / M15_PATH)
    m15_bars = build_m15_bars(m15_raw)

    feat7 = pd.read_parquet(PANEL_DIR / "entry_panel_production_feat7_top21.parquet")
    h1_fe = pd.read_parquet(PANEL_DIR / "entry_panel_h1_baseline_top21.parquet")

    rows = []
    yearly = []
    costs = []
    for name, panel in (("feat7", feat7), ("h1_native", h1_fe)):
        s, y, c = _run_panel(name, panel, m15_raw, m15_bars)
        rows.extend(s)
        yearly.extend(y)
        costs.extend(c)

    pd.DataFrame(rows).to_csv(OUT / "oos_summary.csv", index=False)
    pd.DataFrame(yearly).to_csv(OUT / "yearly.csv", index=False)
    pd.DataFrame(costs).to_csv(OUT / "cost_stress.csv", index=False)

    f7_pb = next(r for r in rows if r["policy"] == "feat7_m15_pullback")
    h1_pb = next(r for r in rows if r["policy"] == "h1_native_m15_pullback")
    f7_imm = next(r for r in rows if r["policy"] == "feat7_immediate")
    h1_imm = next(r for r in rows if r["policy"] == "h1_native_immediate")

    if h1_pb["pf"] > f7_pb["pf"] and h1_pb["dd"] < f7_pb["dd"]:
        verdict = "H1_NATIVE"
    elif h1_pb["pf"] >= f7_pb["pf"]:
        verdict = "H1_NATIVE_MARGINAL"
    else:
        verdict = "FEAT7"

    feat_diff = set(PRODUCTION_FEAT7) - set(H1_NATIVE)

    lines = [
        "# FEAT7 vs H1-Native FE — M15 Pullback Stack",
        "",
        f"**Verdict:** `{verdict}`",
        "",
        "Shared stack: WF top 21% → **M15 pullback entry** → P0 **M15 trail** exit.",
        "",
        f"Production: `{PIPELINE_VERSION}`",
        "",
        "## Feature sets",
        "",
        f"- **FEAT7** ({len(PRODUCTION_FEAT7)}): `{', '.join(PRODUCTION_FEAT7)}`",
        f"- **H1 native** ({len(H1_NATIVE)}): `{', '.join(H1_NATIVE)}`",
        f"- Diff: FEAT7 adds `{', '.join(feat_diff)}` (raw H4 merge)",
        "",
        "## OOS @ cost 1.0×",
        "",
        _md(rows, ["policy", "features", "entry", "pf", "dd", "avg_r", "trades", "retention", "mc_prob_ruin"],
            {"pf": 2, "dd": 3, "avg_r": 3, "trades": 0, "retention": 3, "mc_prob_ruin": 3}),
        "",
        "## Head-to-head (M15 pullback)",
        "",
        f"| | FEAT7 | H1 native | Δ |",
        f"|---|---:|---:|---:|",
        f"| PF | {f7_pb['pf']:.2f} | {h1_pb['pf']:.2f} | {h1_pb['pf']-f7_pb['pf']:+.2f} |",
        f"| Max DD | {f7_pb['dd']:.1%} | {h1_pb['dd']:.1%} | {(h1_pb['dd']-f7_pb['dd'])*100:+.1f}pp |",
        f"| Avg R | {f7_pb['avg_r']:.3f} | {h1_pb['avg_r']:.3f} | {h1_pb['avg_r']-f7_pb['avg_r']:+.3f} |",
        f"| Trades | {f7_pb['trades']} | {h1_pb['trades']} | {h1_pb['trades']-f7_pb['trades']:+d} |",
        "",
        "## Immediate entry (reference)",
        "",
        f"- FEAT7 immediate: PF {f7_imm['pf']:.2f}, DD {f7_imm['dd']:.1%}",
        f"- H1 native immediate: PF {h1_imm['pf']:.2f}, DD {h1_imm['dd']:.1%}",
        "",
        "## Cost stress (M15 pullback)",
        "",
        _md(
            [r for r in costs if r["entry"] == "m15 pullback" and r["features"] == "feat7"],
            ["cost_mult", "pf", "dd", "trades"],
            {"cost_mult": 2, "pf": 2, "dd": 3, "trades": 0},
        ),
        "",
        _md(
            [r for r in costs if r["entry"] == "m15 pullback" and r["features"] == "h1_native"],
            ["cost_mult", "pf", "dd", "trades"],
            {"cost_mult": 2, "pf": 2, "dd": 3, "trades": 0},
        ),
        "",
        "Production unchanged.",
    ]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "verdict.json").write_text(json.dumps({
        "verdict": verdict,
        "feat7_pullback": {k: f7_pb[k] for k in ("pf", "dd", "avg_r", "trades")},
        "h1_native_pullback": {k: h1_pb[k] for k in ("pf", "dd", "avg_r", "trades")},
        "production_features": list(PRIMARY_FEATURES),
        "h1_native_features": list(H1_NATIVE),
    }, indent=2), encoding="utf-8")

    print("VERDICT", verdict)
    print("FEAT7 pullback", {k: round(f7_pb[k], 3) for k in ("pf", "dd", "trades")})
    print("H1 FE pullback", {k: round(h1_pb[k], 3) for k in ("pf", "dd", "trades")})
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
