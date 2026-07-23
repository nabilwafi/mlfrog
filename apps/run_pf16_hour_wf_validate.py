"""Validate best hour+top_pct config on walk-forward 2015–2026."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from apps.run_primary_loosen_backtest import run_portfolio
from apps.run_yearly_walkforward_backtest import _load_side, build_year_panel
from production.paper.barrier_resim import load_h1
from research.portfolio_heat import STARTING_EQUITY


def main() -> None:
    long_df = _load_side("long")
    short_df = _load_side("short")
    h1 = load_h1(str(_ROOT / "artifacts/raw/XAUUSD/H1/data.parquet"))
    years = list(range(2015, 2027))

    configs = [
        dict(top_pct=0.15, lo=8, hi=16, mo=4, name="top15_LN_NY_8_16_mo4"),
        dict(top_pct=0.12, lo=7, hi=16, mo=4, name="top12_EU_US_7_16_mo4"),
        dict(top_pct=0.12, lo=6, hi=18, mo=4, name="top12_wide_6_18_mo4"),
        dict(top_pct=0.15, lo=0, hi=23, mo=4, name="top15_allhours_mo4"),  # control
    ]

    out_rows = []
    for cfg in configs:
        parts = []
        print(f"\n=== {cfg['name']} ===")
        print(f"{'year':>6} {'tr':>5} {'ret':>8} {'dd':>7} {'pf':>5}")
        for y in years:
            panel = build_year_panel(
                long_df, short_df, h1, year=y, lookback=7, top_pct=cfg["top_pct"]
            )
            if panel.empty:
                continue
            panel = panel.copy()
            panel["hour"] = pd.to_datetime(panel["timestamp"], utc=True).dt.hour
            panel = panel[(panel["hour"] >= cfg["lo"]) & (panel["hour"] <= cfg["hi"])]
            if panel.empty:
                continue
            traded, m = run_portfolio(panel, max_open=cfg["mo"], starting_equity=STARTING_EQUITY)
            by = (m.get("by_year") or [{}])[0]
            print(
                f"{y:6d} {m['n_trades']:5d} {by.get('year_return', m['total_return']):+8.1%} "
                f"{by.get('max_drawdown', m['max_drawdown']):7.1%} {m.get('profit_factor', float('nan')):5.2f}"
            )
            parts.append(panel)
        if not parts:
            continue
        all_p = pd.concat(parts, ignore_index=True).sort_values("timestamp")
        _, cont = run_portfolio(all_p, max_open=cfg["mo"], starting_equity=STARTING_EQUITY)
        full = [r for r in cont.get("by_year", []) if r["year"] < 2026]
        avg = sum(r["n_trades"] for r in full) / max(len(full), 1)
        print(
            f"CONTINUOUS avg/yr={avg:.0f} pf={cont['profit_factor']:.2f} "
            f"ret={cont['total_return']:+.1%} dd={cont['max_drawdown']:.1%}"
        )
        out_rows.append(
            {
                **cfg,
                "avg_trades_yr": avg,
                "pf": cont["profit_factor"],
                "total_return": cont["total_return"],
                "max_dd": cont["max_drawdown"],
                "by_year": cont.get("by_year", []),
            }
        )

    out = _ROOT / "artifacts/pipeline_backtest/pf16_hour_wf_validate.json"
    out.write_text(json.dumps(out_rows, indent=2, default=float), encoding="utf-8")
    print("\nwrote", out)


if __name__ == "__main__":
    main()
