"""Stress-test trail policy under harsher cost/slippage.

Same WF entries as winning config (top5% + sess 09-15 + trail 0.12),
then re-sim exits and subtract extra friction from net_return.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from apps.run_exit_engine_hunt import build_entries, metrics_for
from apps.run_yearly_walkforward_backtest import _load_side
from production.paper.barrier_resim import load_h1
from research.position_mgmt import COST as BASE_COST
from research.position_mgmt.services.paths import entry_indices, prepare_market
from research.position_mgmt.services.simulator import ManageConfig, simulate_panel

# Finex-ish: ~19pt spread + 5pt slip on XAU; point=0.01 → ~0.24 price units.
# Stress expresses extra friction as fraction of entry (like COST).


def main() -> None:
    years = list(range(2015, 2027))
    print("What this is:")
    print("  Re-run the WR~70%/PF~1.77 trail policy with fatter costs.")
    print("  If PF stays >1 and WR stays high -> edge not just lab artifact.")
    print("  If it collapses -> trail 0.12 is too tight for live spreads.\n")

    long_df = _load_side("long")
    short_df = _load_side("short")
    h1 = load_h1(str(_ROOT / "artifacts/raw/XAUUSD/H1/data.parquet"))
    mkt = prepare_market(h1)
    entries = build_entries(long_df, short_df, h1, top_pct=0.05, years=years)
    e = entries.copy()
    e["hour"] = pd.to_datetime(e["timestamp"], utc=True).dt.hour
    pool = e[(e["hour"] >= 9) & (e["hour"] <= 15)].drop(columns=["hour"])
    eis = entry_indices(pool, mkt["ts"])
    cfg = ManageConfig(
        name="trail_0.12", family="trail", trail_atr=0.12, horizon=16, max_bars=16
    )
    sim = simulate_panel(pool, h1, cfg, mkt=mkt, eis=eis)
    # simulate_panel already applied BASE_COST (~1.5e-4) once
    base_net = sim["net_return"].astype(float).copy()

    scenarios = [
        ("base_1.5bp", 0.0),
        ("+1.5bp_total3bp", 1.5e-4),
        ("+3bp_total4.5bp", 3.0e-4),
        ("+5bp_total6.5bp", 5.0e-4),
        ("+8bp_total9.5bp", 8.0e-4),
        ("+12bp_total13.5bp", 1.2e-3),
        # round-trip spread stress as fraction of price (~$0.24 on $2000 ≈ 1.2bp; scale up)
        ("spread_like_2bp_extra", 2.0e-4),
        ("spread_like_5bp_extra", 5.0e-4),
        ("brutal_20bp_extra", 2.0e-3),
    ]

    rows = []
    print(f"{'scenario':28} {'n':>5} {'avgWR':>7} {'contWR':>7} {'PF':>6} {'ret':>8} {'dd':>6}")
    for name, extra in scenarios:
        s = sim.copy()
        s["net_return"] = base_net - float(extra)
        s["atr_price"] = s["atr_entry"]
        s["holding_bars"] = s["holding_bars_sim"]
        m = metrics_for(s, max_open=1)
        m["scenario"] = name
        m["extra_cost"] = extra
        m["total_cost_approx"] = float(BASE_COST) + float(extra)
        rows.append(m)
        print(
            f"{name:28} {m['n_trades']:5d} {m['avg_iso_wr']:7.1%} {m['cont_wr']:7.1%} "
            f"{m['cont_pf']:6.2f} {m['cont_ret']:+8.1%} {m['cont_dd']:6.1%}"
        )

    out = _ROOT / "artifacts/pipeline_backtest/trail012_cost_stress.json"
    out.write_text(
        json.dumps(
            {
                "policy": "top5% + sess 09-15 + trail 0.12 + max_open 1",
                "base_cost": float(BASE_COST),
                "scenarios": rows,
            },
            indent=2,
            default=float,
        ),
        encoding="utf-8",
    )
    print("\nwrote", out)


if __name__ == "__main__":
    main()
