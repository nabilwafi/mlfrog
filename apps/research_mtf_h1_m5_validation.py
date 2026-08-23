"""Validate M5 pullback timing on H1 baseline entries.

Research only. Production unchanged.

  python apps/research_mtf_h1_m5_validation.py

Outputs: results/h4_h1_m5/validation/
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
from production import PIPELINE_VERSION, PRIMARY_FEATURES
from research.mtf_h4_h1_m5.backtest import (
    M15_PATH,
    M5_PATH,
    apply_m5_execution,
    monte_carlo_10k,
    score_panel,
)
from research.mtf_h4_h1_m5.m5_engine import build_m5_features

OUT = _ROOT / "results/h4_h1_m5/validation"
PANEL_DIR = _ROOT / "results/h4_h1_m5/panels"
COST_MULTS = (1.0, 1.25, 1.5, 2.0)
POLICIES = (
    ("production_feat7", "immediate", {}),
    ("h1_baseline", "immediate", {}),
    ("h1_m5_pullback", "pullback_recovery", {}),
)
PULLBACK_GRID = (
    (0.10, 0.40),
    (0.10, 0.50),
    (0.15, 0.50),
    (0.15, 0.60),
    (0.20, 0.50),
    (0.25, 0.50),
)


def _side_yearly(panel: pd.DataFrame, m15: pd.DataFrame, side: str) -> list[dict]:
    sub = panel[panel["side"].astype(str).str.lower() == side]
    return score_panel(sub, m15=m15)["yrows"]


def _yearly_wide(rows_a: list[dict], rows_b: list[dict], name_a: str, name_b: str) -> list[dict]:
    by_a = {r["year"]: r for r in rows_a}
    by_b = {r["year"]: r for r in rows_b}
    out = []
    for y in sorted(set(by_a) | set(by_b)):
        a, b = by_a.get(y), by_b.get(y)
        out.append({
            "year": y,
            "oos": y in TRUE_OOS,
            f"{name_a}_pf": a["pf"] if a else float("nan"),
            f"{name_b}_pf": b["pf"] if b else float("nan"),
            f"{name_a}_dd": a["dd"] if a else float("nan"),
            f"{name_b}_dd": b["dd"] if b else float("nan"),
            f"{name_a}_trades": a["trades"] if a else 0,
            f"{name_b}_trades": b["trades"] if b else 0,
            "delta_pf": (b["pf"] - a["pf"]) if a and b else float("nan"),
        })
    return out


def _gates(prod_oos: dict, m5_oos: dict, cost50: dict, yearly: list[dict]) -> tuple[str, list[str]]:
    notes = []
    oos_years = [r for r in yearly if r["oos"]]
    wins = sum(1 for r in oos_years if r["delta_pf"] > 0)
    notes.append(f"OOS years M5 PF > prod: {wins}/{len(oos_years)}")
    ok_pf = m5_oos["pf"] > prod_oos["pf"] + 0.05
    ok_dd = m5_oos["dd"] <= prod_oos["dd"] - 0.05
    ok_cost = cost50["pf"] >= 1.0
    ok_trades = m5_oos["trades"] >= 4000
    ok_mc = m5_oos.get("_prob_ruin", 0) < 0.02
    ok_years = wins >= max(3, len(oos_years) - 1)
    if ok_pf and ok_dd and ok_cost and ok_trades and ok_mc and ok_years:
        return "ITERATE", notes + ["Passes validation gates — not PROMOTE until live paper test"]
    if ok_pf and ok_dd:
        return "ITERATE", notes + ["Core edge holds; some gates failed"]
    return "REJECT", notes + ["Does not beat production robustly"]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    m15 = pd.read_parquet(_ROOT / M15_PATH)
    m5_feat = build_m5_features(pd.read_parquet(_ROOT / M5_PATH))

    h1_panel = pd.read_parquet(PANEL_DIR / "entry_panel_h1_baseline_top21.parquet")
    prod_panel = pd.read_parquet(PANEL_DIR / "entry_panel_production_feat7_top21.parquet")
    h1_panel["timestamp"] = pd.to_datetime(h1_panel["timestamp"], utc=True)
    prod_panel["timestamp"] = pd.to_datetime(prod_panel["timestamp"], utc=True)

    executed, diag = apply_m5_execution(h1_panel, m5_feat, strategy="pullback_recovery")
    panels = {
        "production_feat7": prod_panel,
        "h1_baseline": h1_panel,
        "h1_m5_pullback": executed,
    }

    summary = []
    cost_rows = []
    for name, panel in panels.items():
        for mult in COST_MULTS:
            sc = score_panel(panel, m15=m15, cost_mult=mult)
            po = sc["po_oos"]
            row = {
                "policy": name, "cost_mult": mult,
                "pf": po["pf"], "dd": po["dd"], "avg_r": po["avg_r"], "wr": po["wr"],
                "trades": po["trades"], "ret": po["ret"],
            }
            cost_rows.append(row)
            if mult == 1.0:
                mc = monte_carlo_10k(po["pnl"])
                row.update({f"mc_{k}": v for k, v in mc.items()})
                summary.append({**row, "scope": "oos"})

    prod_sc = score_panel(prod_panel, m15=m15)
    h1_sc = score_panel(h1_panel, m15=m15)
    m5_sc = score_panel(executed, m15=m15)

    yearly = _yearly_wide(prod_sc["yrows"], m5_sc["yrows"], "prod", "h1_m5")
    side_rows = []
    for side in ("long", "short"):
        for pol, p in (("prod", prod_panel), ("h1_imm", h1_panel), ("h1_m5", executed)):
            yrows = _side_yearly(p, m15, side)
            po = [r for r in yrows if r["year"] in TRUE_OOS]
            if not po:
                continue
            pnl = np.concatenate([r["pnl"] for r in po if len(r["pnl"])])
            gp = float(pnl[pnl > 0].sum())
            gl = float(-pnl[pnl < 0].sum())
            side_rows.append({
                "side": side, "policy": pol,
                "trades": int(sum(r["trades"] for r in po)),
                "pf": gp / gl if gl > 0 else 0.0,
                "avg_r": float(np.mean([r["avg_r"] for r in po])),
            })

    param_rows = []
    for pb_atr, rec in PULLBACK_GRID:
        ex, _ = apply_m5_execution(
            h1_panel, m5_feat, strategy="pullback_recovery",
            m5_exec_kw={"pullback_atr": pb_atr, "recovery_frac": rec},
        )
        po = score_panel(ex, m15=m15)["po_oos"]
        param_rows.append({
            "pullback_atr": pb_atr, "recovery_frac": rec,
            "pf": po["pf"], "dd": po["dd"], "trades": po["trades"], "retention": len(ex) / len(h1_panel),
        })

    delay_stats = {}
    if not diag.empty and "delay_m5_bars" in diag.columns:
        d = diag.loc[diag["action"] == "EXECUTE", "delay_m5_bars"].astype(float)
        delay_stats = {
            "median_delay_m5": float(d.median()) if len(d) else 0.0,
            "p90_delay_m5": float(d.quantile(0.9)) if len(d) else 0.0,
            "invalidate_rate": float((diag["action"] == "INVALIDATE").mean()),
        }

    prod_oos = next(r for r in summary if r["policy"] == "production_feat7")
    m5_oos = next(r for r in summary if r["policy"] == "h1_m5_pullback")
    cost50 = next(r for r in cost_rows if r["policy"] == "h1_m5_pullback" and r["cost_mult"] == 1.5)
    m5_oos["_prob_ruin"] = m5_oos.get("mc_prob_ruin", 0)
    verdict, gate_notes = _gates(prod_oos, m5_oos, cost50, yearly)

    pd.DataFrame(summary).to_csv(OUT / "oos_summary.csv", index=False)
    pd.DataFrame(cost_rows).to_csv(OUT / "cost_stress.csv", index=False)
    pd.DataFrame(yearly).to_csv(OUT / "yearly_vs_production.csv", index=False)
    pd.DataFrame(side_rows).to_csv(OUT / "direction_breakdown.csv", index=False)
    pd.DataFrame(param_rows).to_csv(OUT / "pullback_param_grid.csv", index=False)
    if not diag.empty:
        diag.to_csv(OUT / "m5_execution_diagnostics.csv", index=False)

    charts = OUT / "charts"
    charts.mkdir(exist_ok=True)
    ydf = pd.DataFrame(yearly)
    if not ydf.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        oos = ydf[ydf["oos"]]
        ax.bar(oos["year"].astype(str), oos["delta_pf"])
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title("OOS ΔPF: H1+M5 pullback − Production FEAT7")
        fig.tight_layout()
        fig.savefig(charts / "yearly_delta_pf.png", dpi=120)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    sub = pd.DataFrame(cost_rows)
    for pol in ("production_feat7", "h1_m5_pullback"):
        s = sub[sub["policy"] == pol]
        ax.plot(s["cost_mult"], s["pf"], marker="o", label=pol)
    ax.axhline(1.0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("cost multiplier")
    ax.set_ylabel("OOS PF")
    ax.legend()
    fig.tight_layout()
    fig.savefig(charts / "cost_stress_pf.png", dpi=120)
    plt.close(fig)

    lines = [
        "# H1 + M5 Pullback Validation",
        "",
        f"**Verdict:** `{verdict}`",
        "",
        "Compare: Production FEAT7 immediate vs H1 baseline immediate vs H1+M5 pullback.",
        "",
        f"Production pipeline: `{PIPELINE_VERSION}`",
        "",
        "## OOS summary (cost 1.0×)",
        "",
        _md(
            summary,
            ["policy", "pf", "dd", "avg_r", "wr", "trades", "ret", "mc_prob_ruin", "mc_median_dd"],
            {"pf": 2, "dd": 3, "avg_r": 3, "wr": 3, "trades": 0, "ret": 2,
             "mc_prob_ruin": 3, "mc_median_dd": 3},
        ),
        "",
        "## Yearly vs production (H1+M5 − prod ΔPF)",
        "",
        _md(yearly, ["year", "oos", "prod_pf", "h1_m5_pf", "delta_pf", "prod_trades", "h1_m5_trades"],
            {"year": 0, "delta_pf": 2, "prod_pf": 2, "h1_m5_pf": 2}),
        "",
        "## Cost stress (H1+M5 pullback)",
        "",
        _md(
            [r for r in cost_rows if r["policy"] == "h1_m5_pullback"],
            ["cost_mult", "pf", "dd", "trades", "avg_r"],
            {"cost_mult": 2, "pf": 2, "dd": 3, "trades": 0, "avg_r": 3},
        ),
        "",
        "## LONG / SHORT (OOS pooled)",
        "",
        _md(side_rows, ["side", "policy", "pf", "trades", "avg_r"], {"pf": 2, "avg_r": 3, "trades": 0}),
        "",
        "## Pullback parameter grid",
        "",
        _md(param_rows, ["pullback_atr", "recovery_frac", "pf", "dd", "retention", "trades"],
            {"pullback_atr": 2, "recovery_frac": 2, "pf": 2, "dd": 3, "retention": 3, "trades": 0}),
        "",
        "## M5 delay diagnostics",
        "",
        str(delay_stats) if delay_stats else "n/a",
        "",
        "## Gate notes",
        "",
        *[f"- {n}" for n in gate_notes],
        "",
        "Production unchanged.",
        "",
    ]
    (OUT / "validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "verdict.json").write_text(json.dumps({
        "verdict": verdict,
        "production_changed": False,
        "policy": "h1_m5_pullback_recovery",
        "gates": gate_notes,
        "delay_stats": delay_stats,
    }, indent=2), encoding="utf-8")

    print("VERDICT", verdict)
    print("prod OOS", {k: round(prod_oos[k], 3) for k in ("pf", "dd", "trades")})
    print("h1_m5 OOS", {k: round(m5_oos[k], 3) for k in ("pf", "dd", "trades")})
    print("cost+50% PF", round(cost50["pf"], 3))
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
