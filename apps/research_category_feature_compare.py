"""Compare feature categories (new + library) vs H1 native research baseline.

Stack: WF top 21% → M15 pullback entry → P0 M15 trail.

  python apps/research_category_feature_compare.py

Outputs: results/h4_h1_m5/category_features/
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
from research.category_features import (
    CATEGORY_MAP,
    H1_ENGINE_V1,
    NEW_CATEGORY_FEATURES,
    UNAVAILABLE_CATEGORIES,
    attach_extended_features,
    compute_extended_features,
)
from research.mtf_h4_h1_m5.backtest import (
    M15_PATH,
    apply_m15_execution,
    build_wf_entry_panel_feats,
    monte_carlo_10k,
    score_panel,
)
from research.mtf_h4_h1_m5.h1_features import H1_NATIVE
from research.mtf_h4_h1_m5.m5_engine import build_m15_bars
from simulation.wf.sim import load_h1

OUT = _ROOT / "results/h4_h1_m5/category_features"
PANEL_DIR = _ROOT / "results/h4_h1_m5/panels"
EXT_CACHE = PANEL_DIR / "h1_extended_features.parquet"


def _load_ext(h1: pd.DataFrame) -> pd.DataFrame:
    if EXT_CACHE.is_file():
        return pd.read_parquet(EXT_CACHE)
    print("Computing extended features (once)...")
    ext = compute_extended_features(h1)
    EXT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    ext.to_parquet(EXT_CACHE, index=False)
    return ext


def _enrich(ext: pd.DataFrame):
    def fn(ld: pd.DataFrame, sd: pd.DataFrame):
        return attach_extended_features(ld, ext), attach_extended_features(sd, ext)
    return fn


def _feat_union(base: tuple[str, ...], extra: tuple[str, ...]) -> list[str]:
    out = list(base)
    for f in extra:
        if f not in out:
            out.append(f)
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    h1 = load_h1(_ROOT / "artifacts/raw/XAUUSD/H1/data.parquet")
    ext = _load_ext(h1)
    enrich = _enrich(ext)
    m15_raw = pd.read_parquet(_ROOT / M15_PATH)
    m15_bars = build_m15_bars(m15_raw)

    experiments: list[tuple[str, list[str], object | None]] = [
        ("h1_native_6", list(H1_NATIVE), None),
        ("h1_plus_new_all", _feat_union(H1_NATIVE, NEW_CATEGORY_FEATURES), enrich),
        ("h1_plus_price_return", _feat_union(H1_NATIVE, CATEGORY_MAP["price_return_new"]), enrich),
        ("h1_plus_volume", _feat_union(H1_NATIVE, CATEGORY_MAP["volume_new"]), enrich),
        ("h1_plus_liquidity", _feat_union(H1_NATIVE, CATEGORY_MAP["liquidity_new"]), enrich),
        ("h1_plus_mean_reversion", _feat_union(H1_NATIVE, CATEGORY_MAP["mean_reversion_new"]), enrich),
        ("h1_plus_momentum_new", _feat_union(H1_NATIVE, CATEGORY_MAP["momentum_new"]), enrich),
        ("h1_plus_momentum_lib", _feat_union(H1_NATIVE, CATEGORY_MAP["momentum_lib"]), None),
        ("h1_plus_candle_lib", _feat_union(H1_NATIVE, CATEGORY_MAP["candle_lib"]), None),
        ("engine_v1_35", list(H1_ENGINE_V1), None),
        ("engine_ext_48", _feat_union(H1_ENGINE_V1, NEW_CATEGORY_FEATURES), enrich),
    ]

    rows = []
    for tag, feat, enr in experiments:
        print(f"Building panel {tag} n_feat={len(feat)}...")
        panel = build_wf_entry_panel_feats(
            feat, cache_dir=PANEL_DIR, tag=tag, enrich=enr, force=False,
        )
        ex, _ = apply_m15_execution(panel, m15_bars, strategy="pullback_recovery")
        sc = score_panel(ex, m15=m15_raw, exit_tf="M15", entry_mode="m15_pullback")
        po = sc["po_oos"]
        mc = monte_carlo_10k(po["pnl"])
        rows.append({
            "policy": tag,
            "n_features": len(feat),
            "pf": po["pf"], "dd": po["dd"], "avg_r": po["avg_r"],
            "trades": po["trades"], "ret": po["ret"],
            "mc_prob_ruin": mc["prob_ruin"],
        })

    base = next(r for r in rows if r["policy"] == "h1_native_6")
    ext_row = next(r for r in rows if r["policy"] == "engine_ext_48")
    eng35 = next(r for r in rows if r["policy"] == "engine_v1_35")

    pd.DataFrame(rows).to_csv(OUT / "oos_summary.csv", index=False)

    lines = [
        "# Feature Category Comparison",
        "",
        "Stack: M15 pullback entry + P0 M15 trail. Baseline = H1 native 6.",
        "",
        f"**New features added:** {len(NEW_CATEGORY_FEATURES)} ({', '.join(NEW_CATEGORY_FEATURES)})",
        "",
        f"**Unavailable (no data):** {', '.join(UNAVAILABLE_CATEGORIES)}",
        "",
        "## OOS results",
        "",
        _md(rows, ["policy", "n_features", "pf", "dd", "avg_r", "trades", "mc_prob_ruin"],
            {"pf": 2, "dd": 3, "avg_r": 3, "trades": 0, "n_features": 0, "mc_prob_ruin": 3}),
        "",
        "## vs baseline (h1_native_6)",
        "",
    ]
    for r in rows:
        if r["policy"] == "h1_native_6":
            continue
        lines.append(
            f"- `{r['policy']}`: ΔPF {r['pf']-base['pf']:+.2f}, ΔDD {(r['dd']-base['dd'])*100:+.1f}pp"
        )
    lines += [
        "",
        "## Engine comparison",
        "",
        f"- v1 engine (35): PF {eng35['pf']:.2f}, DD {eng35['dd']:.1%}",
        f"- extended (48): PF {ext_row['pf']:.2f}, DD {ext_row['dd']:.1%}",
        f"- ΔPF ext−v1: {ext_row['pf']-eng35['pf']:+.2f}",
        "",
        "Production unchanged.",
    ]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "verdict.json").write_text(json.dumps({
        "baseline": base,
        "best": max(rows, key=lambda x: x["pf"]),
        "engine_v1_35": eng35,
        "engine_ext_48": ext_row,
        "new_features": list(NEW_CATEGORY_FEATURES),
        "unavailable": list(UNAVAILABLE_CATEGORIES),
    }, indent=2), encoding="utf-8")

    print("baseline", base)
    print("best", max(rows, key=lambda x: x["pf"]))
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
