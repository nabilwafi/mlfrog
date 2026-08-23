"""Compare M15 vs M5 pullback entry timing (same exit: P0 trail on M15 close).

  python apps/research_mtf_m15_vs_m5_pullback.py

Outputs: results/h4_h1_m5/pullback_compare/
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
    apply_m15_execution,
    apply_m5_execution,
    monte_carlo_10k,
    score_panel,
)
from research.mtf_h4_h1_m5.m5_engine import build_m15_bars, build_m5_features

OUT = _ROOT / "results/h4_h1_m5/pullback_compare"
PANEL = _ROOT / "results/h4_h1_m5/panels/entry_panel_h1_baseline_top21.parquet"


def _delay_stats(diag: pd.DataFrame, col: str) -> dict:
    if diag.empty:
        return {}
    exe = diag.loc[diag["action"] == "EXECUTE"]
    d = exe["delay_m5_bars"].astype(float)
    return {
        f"median_delay_{col}": float(d.median()) if len(d) else 0.0,
        f"p90_delay_{col}": float(d.quantile(0.9)) if len(d) else 0.0,
        "invalidate_rate": float((diag["action"] == "INVALIDATE").mean()),
        "retention": float((diag["action"] == "EXECUTE").mean()),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    m15_raw = pd.read_parquet(_ROOT / M15_PATH)
    m15_bars = build_m15_bars(m15_raw)
    m5_feat = build_m5_features(pd.read_parquet(_ROOT / "artifacts/raw/XAUUSD/M5/data.parquet"))

    h1 = pd.read_parquet(PANEL)
    h1["timestamp"] = pd.to_datetime(h1["timestamp"], utc=True)
    n_src = len(h1)

    m5_ex, m5_diag = apply_m5_execution(h1, m5_feat, strategy="pullback_recovery")
    m15_ex, m15_diag = apply_m15_execution(h1, m15_bars, strategy="pullback_recovery")
    imm = h1.copy()

    rows = []
    yearly = []
    configs = (
        ("h1_immediate_m15", imm, "h1_close"),
        ("m15_pullback_m15_exit", m15_ex, "m15_pullback"),
        ("m5_pullback_m15_exit", m5_ex, "m5_pullback"),
    )
    for name, panel, entry_mode in configs:
        sc = score_panel(panel, m15=m15_raw, exit_tf="M15", entry_mode=entry_mode)
        po = sc["po_oos"]
        mc = monte_carlo_10k(po["pnl"])
        rows.append({
            "policy": name,
            "entry_tf": "H1" if name.startswith("h1_immediate") else ("M15" if "m15_" in name else "M5"),
            "pf": po["pf"], "dd": po["dd"], "avg_r": po["avg_r"], "wr": po["wr"],
            "trades": po["trades"], "ret": po["ret"],
            "retention": len(panel) / n_src,
            "mc_prob_ruin": mc["prob_ruin"], "mc_median_dd": mc["median_dd"],
        })
        for r in sc["yrows"]:
            yearly.append({
                "policy": name, "year": r["year"], "oos": r["year"] in TRUE_OOS,
                "pf": r["pf"], "dd": r["dd"], "trades": r["trades"], "avg_r": r["avg_r"],
            })

    pd.DataFrame(rows).to_csv(OUT / "oos_summary.csv", index=False)
    pd.DataFrame(yearly).to_csv(OUT / "yearly.csv", index=False)
    m5_diag.to_csv(OUT / "m5_entry_diagnostics.csv", index=False)
    m15_diag.to_csv(OUT / "m15_entry_diagnostics.csv", index=False)

    m15_row = next(r for r in rows if r["policy"] == "m15_pullback_m15_exit")
    m5_row = next(r for r in rows if r["policy"] == "m5_pullback_m15_exit")
    imm_row = next(r for r in rows if r["policy"] == "h1_immediate_m15")

    if m5_row["pf"] > m15_row["pf"] and m5_row["dd"] <= m15_row["dd"]:
        verdict = "M5_PULLBACK"
    elif m15_row["pf"] >= m5_row["pf"]:
        verdict = "M15_PULLBACK"
    else:
        verdict = "INCONCLUSIVE"

    m5_d = _delay_stats(m5_diag, "m5_bars")
    m15_d = _delay_stats(m15_diag, "m15_bars")

    lines = [
        "# M15 vs M5 Pullback Entry",
        "",
        f"**Winner:** `{verdict}`",
        "",
        "Shared exit: P0 a0.25/d0.08 trail on **M15 close**, from fill bar after H1 close.",
        "Same rule: 0.15 ATR pullback, 0.50 recovery, max wait 4 H1 bars.",
        "",
        f"Pipeline ref: `{PIPELINE_VERSION}`",
        "",
        "## OOS summary",
        "",
        _md(rows, ["policy", "entry_tf", "pf", "dd", "avg_r", "trades", "retention", "mc_prob_ruin"],
            {"pf": 2, "dd": 3, "avg_r": 3, "trades": 0, "retention": 3, "mc_prob_ruin": 3}),
        "",
        "## Head-to-head",
        "",
        f"| | M15 pullback | M5 pullback | Δ |",
        f"|---|---:|---:|---:|",
        f"| PF | {m15_row['pf']:.2f} | {m5_row['pf']:.2f} | {m5_row['pf']-m15_row['pf']:+.2f} |",
        f"| Max DD | {m15_row['dd']:.1%} | {m5_row['dd']:.1%} | {(m5_row['dd']-m15_row['dd'])*100:+.1f}pp |",
        f"| Avg R | {m15_row['avg_r']:.3f} | {m5_row['avg_r']:.3f} | {m5_row['avg_r']-m15_row['avg_r']:+.3f} |",
        f"| Trades | {m15_row['trades']} | {m5_row['trades']} | {m5_row['trades']-m15_row['trades']:+d} |",
        f"| Retention | {m15_row['retention']:.1%} | {m5_row['retention']:.1%} | |",
        "",
        "## Entry delay diagnostics",
        "",
        f"- M15 pullback: {m15_d}",
        f"- M5 pullback: {m5_d}",
        "",
        f"Baseline immediate H1: PF {imm_row['pf']:.2f}, DD {imm_row['dd']:.1%}",
        "",
        "Production unchanged.",
    ]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "verdict.json").write_text(json.dumps({
        "verdict": verdict,
        "m15_pullback": m15_row,
        "m5_pullback": m5_row,
        "delta_pf": m5_row["pf"] - m15_row["pf"],
        "m15_delay": m15_d,
        "m5_delay": m5_d,
    }, indent=2), encoding="utf-8")

    print("VERDICT", verdict)
    print("M15 pullback", {k: round(m15_row[k], 3) for k in ("pf", "dd", "trades", "avg_r")})
    print("M5 pullback ", {k: round(m5_row[k], 3) for k in ("pf", "dd", "trades", "avg_r")})
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
