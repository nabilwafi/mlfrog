"""Old pipeline WITH meta + confidence gates — yearly table only."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from apps.run_old_meta_pipeline_yearly import isolated_years, run_old


def main() -> None:
    conf = pd.read_parquet(_ROOT / "artifacts/research/confidence_layer/confidence_panel.parquet")
    conf["timestamp"] = pd.to_datetime(conf["timestamp"], utc=True)

    print("Old pipeline: META>=0.45 + CONF>=40 + meta-edge sizing + heat -1R")
    print("Source: confidence_panel (2023-2026 only)\n")

    # funnel
    print("Funnel by year (candidates -> after conf gate):")
    for y, g in conf.groupby(conf["timestamp"].dt.year, sort=True):
        n = len(g)
        n_meta = int((g["meta_proba"] >= 0.45).sum())
        n_conf = int(((g["meta_proba"] >= 0.45) & (g["confidence"] >= 40)).sum())
        print(f"  {y}: panel={n}  meta>=0.45={n_meta}  +conf>=40={n_conf}")

    out = {}
    for mo in (1, 3):
        iso = isolated_years(conf, meta_gate=True, conf_gate=True, max_open=mo)
        _, ov = run_old(conf, meta_gate=True, conf_gate=True, max_open=mo)
        print(f"\n=== max_open={mo} | isolated $80/year ===")
        print(f"{'year':>6} {'trades':>7} {'ret':>8} {'dd':>7} {'wr':>7} {'pf':>6}")
        for r in iso:
            print(
                f"{r['year']:6d} {r['n_trades']:7d} {r['ret']:+8.1%} {r['dd']:7.1%} "
                f"{r['wr']:7.1%} {r['pf']:6.2f}"
            )
        print(
            f"{'ALL':>6} {ov['n_trades']:7d} {ov['total_return']:+8.1%} "
            f"{ov['max_drawdown']:7.1%} {ov['win_rate']:7.1%} {ov['profit_factor']:6.2f}"
            "  (compounded continuous)"
        )
        out[f"max_open_{mo}"] = {"isolated": iso, "continuous": ov}

    path = _ROOT / "artifacts/pipeline_backtest/old_meta_conf_yearly.json"
    path.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print("\nwrote", path)


if __name__ == "__main__":
    main()
