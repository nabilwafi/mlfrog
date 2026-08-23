"""H1 Feature Selection & Robustness Research (38-feature universe).

Phases:
  audit | groups | logo | costs | mc | charts | report | all

Stack (frozen): WF top 21% → M15 pullback → P0 M15 trail.
Excludes: ctx_h4_*, d1_*, m5_*, session flags.

  python apps/research_h1_feature_selection.py --phase all
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

from apps.research_sprint42_attribution import TRUE_OOS, _md
from research.h1_feature_selection import (
    BASELINE_FEATURE_SET,
    CANDIDATE_BY_GROUP,
    FEATURE_GROUP,
    H1_ENGINE_38,
    all_minus_group,
    core_plus_group,
    feat_union,
)
from research.h1_feature_selection.audit import (
    FEATURE_DEFINITIONS,
    load_h1_matrix,
    quality_audit,
    redundancy_matrices,
    statistical_relationship_note,
)
from research.mtf_h4_h1_m5.backtest import (
    M15_PATH,
    apply_m15_execution,
    build_wf_entry_panel_feats,
    monte_carlo_10k,
    score_panel,
)
from research.mtf_h4_h1_m5.m5_engine import build_m15_bars

OUT = _ROOT / "results/h4_h1_m5/h1_feature_selection"
PANEL_DIR = _ROOT / "results/h4_h1_m5/panels"
CHARTS = OUT / "charts"
MATRIX = _ROOT / "artifacts/features/XAUUSD/H1/feature_matrix.parquet"
DOCS_REPORT = _ROOT / "docs/research/h1_feature_selection_report.md"

SEED = 42
_EXT_CACHE: pd.DataFrame | None = None


def _matrix_enrich() -> Any:
    """Attach any H1_ENGINE_38 cols missing from v2 (e.g. roc_3/12, momentum_acceleration)."""
    global _EXT_CACHE

    def _fn(ld: pd.DataFrame, sd: pd.DataFrame):
        global _EXT_CACHE
        if _EXT_CACHE is None:
            m = pd.read_parquet(MATRIX)
            m["timestamp"] = pd.to_datetime(m["timestamp"], utc=True)
            cols = [c for c in H1_ENGINE_38 if c in m.columns]
            _EXT_CACHE = m[["timestamp", *cols]]
        ext = _EXT_CACHE

        def _attach(df: pd.DataFrame) -> pd.DataFrame:
            out = df.copy()
            out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
            need = [c for c in H1_ENGINE_38 if c not in out.columns and c in ext.columns]
            if not need:
                return out
            return out.merge(ext[["timestamp", *need]], on="timestamp", how="left")

        return _attach(ld), _attach(sd)

    return _fn


def _experiments_group_add() -> list[tuple[str, list[str]]]:
    return [
        ("A_core_6", list(BASELINE_FEATURE_SET)),
        ("B_core_plus_trend", core_plus_group("trend")),
        ("C_core_plus_volatility", core_plus_group("volatility")),
        ("D_core_plus_momentum", core_plus_group("momentum")),
        ("E_core_plus_candle", core_plus_group("candle")),
        ("F_core_plus_session", core_plus_group("session")),
        ("G_core_plus_statistical", core_plus_group("statistical")),
        ("H_all_38", list(H1_ENGINE_38)),
    ]


def _experiments_logo() -> list[tuple[str, list[str]]]:
    return [
        (f"logo_minus_{g}", all_minus_group(g))
        for g in ("trend", "volatility", "momentum", "candle", "session", "statistical")
    ]


def phase_audit() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_h1_matrix(MATRIX)
    qa = quality_audit(df)
    qa.to_csv(OUT / "quality_audit.csv", index=False)
    pearson, spearman, pairs = redundancy_matrices(df)
    pearson.to_csv(OUT / "corr_pearson.csv")
    spearman.to_csv(OUT / "corr_spearman.csv")
    pairs.to_csv(OUT / "corr_high_pairs.csv", index=False)
    defs = pd.DataFrame(
        [{"feature": f, "group": FEATURE_GROUP[f], "definition": FEATURE_DEFINITIONS[f]} for f in H1_ENGINE_38]
    )
    defs.to_csv(OUT / "feature_definitions.csv", index=False)
    (OUT / "statistical_family.md").write_text(statistical_relationship_note(), encoding="utf-8")

    # Inventory table
    inv = []
    for f in H1_ENGINE_38:
        inv.append(
            {
                "feature": f,
                "group": FEATURE_GROUP[f],
                "baseline": f in BASELINE_FEATURE_SET,
                "candidate": f not in BASELINE_FEATURE_SET,
            }
        )
    pd.DataFrame(inv).to_csv(OUT / "feature_inventory.csv", index=False)
    summary = {
        "n_matrix_rows": len(df),
        "n_features": len(H1_ENGINE_38),
        "n_baseline": len(BASELINE_FEATURE_SET),
        "n_candidates": len(H1_ENGINE_38) - len(BASELINE_FEATURE_SET),
        "n_high_corr_pairs": len(pairs),
        "near_identical": ["rolling_zscore ~= rolling_mean_distance"],
        "deterministic_of_core": ["volatility_regime_score <- atr_percentile_252"],
        "quality_flags": qa.loc[qa["near_zero_var"] | (qa["pct_missing"] > 0.01), "feature"].tolist(),
    }
    (OUT / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("audit", summary)
    return summary


def _run_one(tag: str, feat: list[str], m15_bars: pd.DataFrame, m15_raw: pd.DataFrame) -> dict[str, Any]:
    print(f"Building panel {tag} n_feat={len(feat)}...")
    enrich = _matrix_enrich()
    panel = build_wf_entry_panel_feats(
        feat, cache_dir=PANEL_DIR, tag=f"h1fs_{tag}", enrich=enrich, force=False,
    )
    ex, _ = apply_m15_execution(panel, m15_bars, strategy="pullback_recovery")
    sc = score_panel(ex, m15=m15_raw, exit_tf="M15", entry_mode="m15_pullback")
    po = sc["po_oos"]
    ydf = pd.DataFrame(sc["yrows"])
    if not ydf.empty:
        ydf.insert(0, "policy", tag)
        ydf.to_csv(OUT / f"yearly_{tag}.csv", index=False)
    mc = monte_carlo_10k(po["pnl"], seed=SEED)
    # Long / short split on OOS trades if side column present
    side_rows = {}
    if "side" in ex.columns and "test_year" in ex.columns:
        for side in ("long", "short"):
            sub = ex[(ex["side"].str.lower() == side) & (ex["test_year"].isin(TRUE_OOS))]
            if sub.empty:
                continue
            sc_s = score_panel(sub, m15=m15_raw, exit_tf="M15", entry_mode="m15_pullback")
            po_s = sc_s["po_oos"]
            side_rows[side] = {
                "pf": po_s["pf"],
                "dd": po_s["dd"],
                "avg_r": po_s["avg_r"],
                "trades": po_s["trades"],
            }
    return {
        "policy": tag,
        "n_features": len(feat),
        "features": ",".join(feat),
        "pf": float(po["pf"]),
        "dd": float(po["dd"]),
        "avg_r": float(po["avg_r"]),
        "wr": float(po["wr"]),
        "trades": int(po["trades"]),
        "ret": float(po["ret"]),
        "mc_prob_ruin": float(mc["prob_ruin"]),
        "mc_p5": float(mc["p5"]),
        "mc_p50": float(mc["p50"]),
        "mc_p95": float(mc["p95"]),
        "mc_median_dd": float(mc["median_dd"]),
        "long_pf": side_rows.get("long", {}).get("pf"),
        "long_dd": side_rows.get("long", {}).get("dd"),
        "long_trades": side_rows.get("long", {}).get("trades"),
        "short_pf": side_rows.get("short", {}).get("pf"),
        "short_dd": side_rows.get("short", {}).get("dd"),
        "short_trades": side_rows.get("short", {}).get("trades"),
        "pnl": po["pnl"],
    }


def phase_score(exps: list[tuple[str, list[str]]], out_name: str) -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    m15_raw = pd.read_parquet(_ROOT / M15_PATH)
    m15_bars = build_m15_bars(m15_raw)
    rows = []
    for tag, feat in exps:
        r = _run_one(tag, feat, m15_bars, m15_raw)
        pnl = r.pop("pnl")
        np.save(OUT / f"pnl_{tag}.npy", np.asarray(pnl, dtype=float))
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / out_name, index=False)
    print(df[["policy", "n_features", "pf", "dd", "avg_r", "trades"]].to_string(index=False))
    return df


def phase_costs(policies: list[str] | None = None) -> pd.DataFrame:
    """Cost stress on CORE / best / ALL38."""
    m15_raw = pd.read_parquet(_ROOT / M15_PATH)
    m15_bars = build_m15_bars(m15_raw)
    catalog = {t: f for t, f in _experiments_group_add()}
    if policies is None:
        policies = ["A_core_6", "H_all_38"]
        # pick best non-core from oos if available
        sum_path = OUT / "group_ablation.csv"
        if sum_path.is_file():
            g = pd.read_csv(sum_path)
            best = g.loc[g["policy"] != "A_core_6"].sort_values("pf", ascending=False).iloc[0]["policy"]
            if best not in policies:
                policies.append(str(best))
    rows = []
    for tag in policies:
        feat = catalog.get(tag)
        if feat is None:
            continue
        panel = build_wf_entry_panel_feats(
            feat, cache_dir=PANEL_DIR, tag=f"h1fs_{tag}", enrich=_matrix_enrich(), force=False,
        )
        ex, _ = apply_m15_execution(panel, m15_bars, strategy="pullback_recovery")
        for cm in (1.0, 1.25, 1.5, 2.0):
            sc = score_panel(ex, m15=m15_raw, exit_tf="M15", entry_mode="m15_pullback", cost_mult=cm)
            po = sc["po_oos"]
            rows.append(
                {
                    "policy": tag,
                    "cost_mult": cm,
                    "pf": po["pf"],
                    "dd": po["dd"],
                    "avg_r": po["avg_r"],
                    "trades": po["trades"],
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "cost_stress.csv", index=False)
    return df


def phase_individual_from_best(group_csv: Path) -> pd.DataFrame | None:
    """If a core+group beats CORE on PF and DD not worse by >5pp, leave-one-out that group's candidates."""
    g = pd.read_csv(group_csv)
    core = g.loc[g["policy"] == "A_core_6"].iloc[0]
    cand = g.loc[g["policy"].str.startswith("B_") | g["policy"].str.startswith("C_")
                 | g["policy"].str.startswith("D_") | g["policy"].str.startswith("E_")
                 | g["policy"].str.startswith("F_") | g["policy"].str.startswith("G_")].copy()
    if cand.empty:
        return None
    # Prefer PF gain with DD not exploding
    cand["ok"] = (cand["pf"] >= float(core["pf"]) - 1e-9) & (cand["dd"] <= float(core["dd"]) + 0.05)
    good = cand.loc[cand["ok"]].sort_values("pf", ascending=False)
    if good.empty:
        # No group improves — skip expensive individual; document
        (OUT / "individual_ablation_skipped.json").write_text(
            json.dumps(
                {
                    "reason": "no core+group improves PF without DD +5pp vs CORE",
                    "core_pf": float(core["pf"]),
                    "core_dd": float(core["dd"]),
                    "best_group": cand.sort_values("pf", ascending=False).iloc[0].to_dict(),
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        return None
    best_pol = str(good.iloc[0]["policy"])
    group_key = {
        "B_core_plus_trend": "trend",
        "C_core_plus_volatility": "volatility",
        "D_core_plus_momentum": "momentum",
        "E_core_plus_candle": "candle",
        "F_core_plus_session": "session",
        "G_core_plus_statistical": "statistical",
    }[best_pol]
    base_feats = core_plus_group(group_key)
    m15_raw = pd.read_parquet(_ROOT / M15_PATH)
    m15_bars = build_m15_bars(m15_raw)
    rows = []
    for drop in CANDIDATE_BY_GROUP[group_key]:
        feat = [f for f in base_feats if f != drop]
        tag = f"ind_drop_{drop}"
        r = _run_one(tag, feat, m15_bars, m15_raw)
        r.pop("pnl")
        r["dropped"] = drop
        r["parent"] = best_pol
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "individual_ablation.csv", index=False)
    return df


def phase_charts(group_df: pd.DataFrame) -> None:
    CHARTS.mkdir(parents=True, exist_ok=True)

    # 1 group ablation
    fig, ax = plt.subplots(figsize=(10, 5))
    order = group_df.sort_values("pf")
    ax.barh(order["policy"], order["pf"], color="#3b82f6")
    ax.axvline(float(group_df.loc[group_df["policy"] == "A_core_6", "pf"].iloc[0]), color="red", ls="--", label="CORE")
    ax.set_xlabel("OOS Profit Factor")
    ax.set_title("Feature group ablation (CORE + group / ALL38)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHARTS / "feature_group_ablation.png", dpi=120)
    plt.close(fig)

    # 2 feature count vs performance
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(group_df["n_features"], group_df["pf"], s=60)
    for _, r in group_df.iterrows():
        ax.annotate(r["policy"].replace("core_plus_", "+")[:18], (r["n_features"], r["pf"]), fontsize=7)
    ax.set_xlabel("# features")
    ax.set_ylabel("OOS PF")
    ax.set_title("Feature count vs OOS PF")
    fig.tight_layout()
    fig.savefig(CHARTS / "feature_count_vs_performance.png", dpi=120)
    plt.close(fig)

    # 3 walk-forward by set (yearly for core / all / best)
    fig, ax = plt.subplots(figsize=(9, 5))
    for tag in ("A_core_6", "H_all_38"):
        yp = OUT / f"yearly_{tag}.csv"
        if not yp.is_file():
            continue
        y = pd.read_csv(yp)
        y = y[y["year"].isin(TRUE_OOS)] if "year" in y.columns else y
        ax.plot(y["year"], y["pf"], marker="o", label=tag)
    ax.axhline(1.0, color="gray", ls=":")
    ax.set_ylabel("PF")
    ax.set_title("Walk-forward yearly PF")
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHARTS / "walk_forward_performance_by_feature_set.png", dpi=120)
    plt.close(fig)

    # 4 correlation heatmap (subset)
    cp = OUT / "corr_pearson.csv"
    if cp.is_file():
        corr = pd.read_csv(cp, index_col=0)
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr)))
        ax.set_yticks(range(len(corr)))
        ax.set_xticklabels(corr.columns, rotation=90, fontsize=6)
        ax.set_yticklabels(corr.index, fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title("Pearson correlation (38 H1 features)")
        fig.tight_layout()
        fig.savefig(CHARTS / "correlation_matrix.png", dpi=120)
        plt.close(fig)

    # 5 long/short
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(group_df))
    w = 0.35
    lp = group_df["long_pf"].fillna(0)
    sp = group_df["short_pf"].fillna(0)
    ax.bar(x - w / 2, lp, w, label="long")
    ax.bar(x + w / 2, sp, w, label="short")
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace("core_plus_", "+")[:14] for p in group_df["policy"]], rotation=45, ha="right")
    ax.set_ylabel("OOS PF")
    ax.set_title("LONG vs SHORT")
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHARTS / "long_short_comparison.png", dpi=120)
    plt.close(fig)

    # 6 monte carlo comparison
    fig, ax = plt.subplots(figsize=(7, 5))
    for tag in ("A_core_6", "H_all_38"):
        row = group_df.loc[group_df["policy"] == tag]
        if row.empty:
            continue
        ax.errorbar(
            [tag],
            [row["mc_p50"].iloc[0]],
            yerr=[[row["mc_p50"].iloc[0] - row["mc_p5"].iloc[0]], [row["mc_p95"].iloc[0] - row["mc_p50"].iloc[0]]],
            fmt="o",
            capsize=6,
            label=tag,
        )
    ax.set_ylabel("MC return quantiles (p5–p50–p95)")
    ax.set_title("Monte Carlo comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHARTS / "monte_carlo_comparison.png", dpi=120)
    plt.close(fig)

    # placeholders for importance/stability/regime if not separately computed
    for name, title in (
        ("feature_importance.png", "Feature importance — see gain on CORE booster (diagnostic only)"),
        ("feature_importance_stability.png", "Importance stability — deferred; OOS PF is primary"),
        ("feature_stability.png", "Feature stability — yearly PF in walk_forward chart"),
        ("regime_performance.png", "Regime split — use existing d1_regime if present in future pass"),
    ):
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, title, ha="center", va="center", wrap=True)
        ax.axis("off")
        fig.savefig(CHARTS / name, dpi=100)
        plt.close(fig)


def _decision_table(group_df: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    core = group_df.loc[group_df["policy"] == "A_core_6"].iloc[0]
    high = set()
    if not pairs.empty:
        for _, r in pairs.iterrows():
            if abs(float(r["pearson"])) >= 0.95:
                high.add(r["feature_a"])
                high.add(r["feature_b"])
    rows = []
    for f in H1_ENGINE_38:
        in_base = f in BASELINE_FEATURE_SET
        # contribution: look at core+group containing f
        g = FEATURE_GROUP[f]
        pol = {
            "trend": "B_core_plus_trend",
            "volatility": "C_core_plus_volatility",
            "momentum": "D_core_plus_momentum",
            "candle": "E_core_plus_candle",
            "session": "F_core_plus_session",
            "statistical": "G_core_plus_statistical",
        }[g]
        row = group_df.loc[group_df["policy"] == pol]
        delta_pf = float(row["pf"].iloc[0] - core["pf"]) if len(row) else 0.0
        delta_dd = float(row["dd"].iloc[0] - core["dd"]) if len(row) else 0.0
        if in_base:
            decision = "KEEP"
        elif f == "volatility_regime_score":
            decision = "REDUNDANT"
        elif f == "rolling_mean_distance":
            decision = "REDUNDANT"
        elif delta_pf > 0.02 and delta_dd <= 0.02:
            decision = "KEEP_RESEARCH"
        elif delta_pf < -0.05 or delta_dd > 0.05:
            decision = "REMOVE"
        elif f in high and not in_base:
            decision = "REDUNDANT"
        else:
            decision = "UNSTABLE" if abs(delta_pf) < 0.02 else "REMOVE"
        rows.append(
            {
                "Feature": f,
                "Group": g,
                "Baseline": in_base,
                "Candidate": not in_base,
                "OOS_Contribution": round(delta_pf, 4),
                "Stability": "group_delta_pf",
                "Redundancy": "high_corr" if f in high else "",
                "Decision": decision,
            }
        )
    return pd.DataFrame(rows)


def phase_report(group_df: pd.DataFrame, logo_df: pd.DataFrame | None, audit: dict) -> None:
    core = group_df.loc[group_df["policy"] == "A_core_6"].iloc[0]
    all38 = group_df.loc[group_df["policy"] == "H_all_38"].iloc[0]
    best = group_df.sort_values(["pf", "dd"], ascending=[False, True]).iloc[0]

    pairs = pd.read_csv(OUT / "corr_high_pairs.csv") if (OUT / "corr_high_pairs.csv").is_file() else pd.DataFrame()
    dec = _decision_table(group_df, pairs)
    dec.to_csv(OUT / "final_decision_table.csv", index=False)

    # Final recommendation
    improves = (
        float(best["pf"]) > float(core["pf"]) + 0.02
        and float(best["dd"]) <= float(core["dd"]) + 0.03
        and best["policy"] != "A_core_6"
    )
    if improves:
        verdict = "EXPAND_FEATURE_SET"
        rec_set = str(best["policy"])
    elif float(all38["pf"]) < float(core["pf"]) - 0.02 or float(all38["dd"]) > float(core["dd"]) + 0.05:
        verdict = "KEEP_6"
        rec_set = "A_core_6"
    else:
        verdict = "KEEP_6"
        rec_set = "A_core_6"

    cost = pd.read_csv(OUT / "cost_stress.csv") if (OUT / "cost_stress.csv").is_file() else pd.DataFrame()

    lines = [
        "# H1 Feature Selection Report",
        "",
        "## 1. Executive Summary",
        "",
        f"**Verdict:** `{verdict}`",
        "",
        f"- Recommended set: `{rec_set}`",
        f"- CORE (6): PF {core['pf']:.3f}, DD {core['dd']:.1%}, trades {int(core['trades'])}, avg_r {core['avg_r']:.3f}",
        f"- ALL 38: PF {all38['pf']:.3f}, DD {all38['dd']:.1%}, trades {int(all38['trades'])}",
        f"- Best scored policy: `{best['policy']}` PF {best['pf']:.3f}, DD {best['dd']:.1%}",
        "",
        "Protocol: WF top 21% → M15 pullback → P0 M15 trail. H1-native features only.",
        "",
        "## 2. Current 38 Features",
        "",
        "See `results/h4_h1_m5/h1_feature_selection/feature_inventory.csv` and audit doc.",
        "",
        "## 3. Baseline 6 Features",
        "",
        ", ".join(BASELINE_FEATURE_SET),
        "",
        "## 4. Feature Quality Audit",
        "",
        f"- Matrix rows: {audit.get('n_matrix_rows')}",
        f"- Near-zero variance / high missing flags: {audit.get('quality_flags')}",
        f"- Near-identical by construction: {audit.get('near_identical')}",
        f"- Deterministic of CORE: {audit.get('deterministic_of_core')}",
        "",
        "Full table: `quality_audit.csv`. Causal: all builders use past windows only (no `center=True`).",
        "",
        "## 5. Redundancy Analysis",
        "",
        f"- High |ρ|≥0.90 pairs: {audit.get('n_high_corr_pairs')}",
        "",
        statistical_relationship_note().strip(),
        "",
        "High-corr pairs: `corr_high_pairs.csv`. Heatmap: `charts/correlation_matrix.png`.",
        "",
        "## 6. Group Ablation",
        "",
        _md(
            group_df.to_dict("records"),
            ["policy", "n_features", "pf", "dd", "avg_r", "trades", "mc_prob_ruin"],
            {"pf": 3, "dd": 3, "avg_r": 3, "trades": 0, "n_features": 0, "mc_prob_ruin": 4},
        ),
        "",
        "### vs CORE",
        "",
    ]
    for _, r in group_df.iterrows():
        if r["policy"] == "A_core_6":
            continue
        lines.append(
            f"- `{r['policy']}`: ΔPF {r['pf']-core['pf']:+.3f}, ΔDD {(r['dd']-core['dd'])*100:+.1f}pp, Δtrades {int(r['trades']-core['trades']):+d}"
        )

    if logo_df is not None and not logo_df.empty:
        lines += ["", "## 6b. Leave-one-group-out (from ALL 38)", "",
                  _md(logo_df.to_dict("records"), ["policy", "n_features", "pf", "dd", "avg_r", "trades"],
                      {"pf": 3, "dd": 3, "avg_r": 3, "trades": 0, "n_features": 0})]

    lines += [
        "",
        "## 7. Individual Ablation",
        "",
    ]
    if (OUT / "individual_ablation.csv").is_file():
        ind = pd.read_csv(OUT / "individual_ablation.csv")
        lines.append(_md(ind.to_dict("records"), ["dropped", "pf", "dd", "trades"], {"pf": 3, "dd": 3, "trades": 0}))
    elif (OUT / "individual_ablation_skipped.json").is_file():
        lines.append("Skipped — no CORE+group improved PF without material DD increase. See `individual_ablation_skipped.json`.")
    else:
        lines.append("Not run.")

    lines += [
        "",
        "## 8. Walk-Forward Stability",
        "",
        "Yearly CSVs: `yearly_*.csv`. Chart: `charts/walk_forward_performance_by_feature_set.png`.",
        "",
        "## 9. Feature Importance",
        "",
        "Diagnostic only — not used for final selection. Placeholder charts under `charts/`.",
        "Primary criterion remains OOS trading metrics under frozen protocol.",
        "",
        "## 10. Regime Analysis",
        "",
        "Deferred to existing regime tags in future pass; this run focuses on pooled OOS + LONG/SHORT.",
        "",
        "## 11. LONG vs SHORT",
        "",
        _md(
            group_df.to_dict("records"),
            ["policy", "long_pf", "long_dd", "long_trades", "short_pf", "short_dd", "short_trades"],
            {"long_pf": 3, "long_dd": 3, "short_pf": 3, "short_dd": 3, "long_trades": 0, "short_trades": 0},
        ),
        "",
        "## 12. Transaction Cost Robustness",
        "",
    ]
    if not cost.empty:
        lines.append(_md(cost.to_dict("records"), ["policy", "cost_mult", "pf", "dd", "trades"],
                         {"cost_mult": 2, "pf": 3, "dd": 3, "trades": 0}))
    else:
        lines.append("See `cost_stress.csv` when generated.")

    lines += [
        "",
        "## 13. Monte Carlo",
        "",
        _md(
            group_df.to_dict("records"),
            ["policy", "mc_p5", "mc_p50", "mc_p95", "mc_median_dd", "mc_prob_ruin"],
            {"mc_p5": 3, "mc_p50": 3, "mc_p95": 3, "mc_median_dd": 3, "mc_prob_ruin": 4},
        ),
        "",
        "## 14. Final Feature Set",
        "",
        f"- SET A — CORE (6): `{', '.join(BASELINE_FEATURE_SET)}`",
        f"- SET B — BEST RESEARCH: `{rec_set}`",
        f"- SET C — FULL (38): all H1 engine features",
        "",
        "## 15. Rejected Features and Reasons",
        "",
        "See `final_decision_table.csv` (Decision ∈ REMOVE / REDUNDANT / UNSTABLE).",
        "",
        "## 16. Final Recommendation",
        "",
        f"**{verdict}**",
        "",
        "Answers:",
        f"1. Are the current 6 sufficient? **{'Yes' if verdict == 'KEEP_6' else 'Possibly expand'}**",
        f"2. Trend add improves OOS? see Δ for `B_core_plus_trend`",
        f"3. Volatility add? `C_core_plus_volatility`",
        f"4. Momentum add? `D_core_plus_momentum`",
        f"5. Candle add? `E_core_plus_candle`",
        f"6. Session (day) add? `F_core_plus_session`",
        f"7. Statistical add? `G_core_plus_statistical`",
        "8–10. Individual / redundant / unstable — see decision table",
        f"11. Smallest robust set: **6 (CORE)** unless EXPAND verdict",
        f"12. Recommended production: **{rec_set}**",
        "",
        "Production unchanged until explicit PROMOTE.",
        "",
        f"Random seed: {SEED}. Config frozen: top_pct=0.21, M15 pullback, trail a0.25/d0.08.",
    ]

    text = "\n".join(lines) + "\n"
    (OUT / "report.md").write_text(text, encoding="utf-8")
    DOCS_REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOCS_REPORT.write_text(text, encoding="utf-8")
    (OUT / "verdict.json").write_text(
        json.dumps(
            {
                "verdict": verdict,
                "recommended": rec_set,
                "core": {k: (float(v) if isinstance(v, (float, np.floating)) else int(v) if isinstance(v, (int, np.integer)) else v)
                         for k, v in core.items() if k != "features"},
                "all38": {k: (float(v) if isinstance(v, (float, np.floating)) else int(v) if isinstance(v, (int, np.integer)) else v)
                          for k, v in all38.items() if k != "features"},
                "best": {k: (float(v) if isinstance(v, (float, np.floating)) else int(v) if isinstance(v, (int, np.integer)) else v)
                         for k, v in best.items() if k != "features"},
                "seed": SEED,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print("verdict", verdict, "recommended", rec_set)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--phase",
        default="all",
        choices=("audit", "groups", "logo", "costs", "individual", "charts", "report", "all"),
    )
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    audit: dict[str, Any] = {}
    if args.phase in ("audit", "all"):
        audit = phase_audit()
    elif (OUT / "audit_summary.json").is_file():
        audit = json.loads((OUT / "audit_summary.json").read_text(encoding="utf-8"))

    group_df = None
    if args.phase in ("groups", "all"):
        group_df = phase_score(_experiments_group_add(), "group_ablation.csv")
    elif (OUT / "group_ablation.csv").is_file():
        group_df = pd.read_csv(OUT / "group_ablation.csv")

    logo_df = None
    if args.phase in ("logo", "all"):
        logo_df = phase_score(_experiments_logo(), "leave_one_group_out.csv")
    elif (OUT / "leave_one_group_out.csv").is_file():
        logo_df = pd.read_csv(OUT / "leave_one_group_out.csv")

    if args.phase in ("costs", "all") and group_df is not None:
        phase_costs()

    if args.phase in ("individual", "all") and (OUT / "group_ablation.csv").is_file():
        phase_individual_from_best(OUT / "group_ablation.csv")

    if args.phase in ("charts", "all") and group_df is not None:
        phase_charts(group_df)

    if args.phase in ("report", "all") and group_df is not None:
        if not audit and (OUT / "audit_summary.json").is_file():
            audit = json.loads((OUT / "audit_summary.json").read_text(encoding="utf-8"))
        phase_report(group_df, logo_df, audit or {})

    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
