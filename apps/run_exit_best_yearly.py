"""Yearly breakdown for best exit config: top5% + sess 9-15 + trail 0.12 ATR."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from apps.run_exit_engine_hunt import build_entries, metrics_for
from apps.run_primary_loosen_backtest import run_portfolio
from apps.run_yearly_walkforward_backtest import _load_side
from production.paper.barrier_resim import load_h1
from research.portfolio_heat import STARTING_EQUITY
from research.position_mgmt.services.paths import entry_indices, prepare_market
from research.position_mgmt.services.simulator import ManageConfig, simulate_panel


def main() -> None:
    years = list(range(2015, 2027))
    long_df = _load_side("long")
    short_df = _load_side("short")
    h1 = load_h1(str(_ROOT / "artifacts/raw/XAUUSD/H1/data.parquet"))
    mkt = prepare_market(h1)
    entries = build_entries(long_df, short_df, h1, top_pct=0.05, years=years)
    e = entries.copy()
    e["hour"] = pd.to_datetime(e["timestamp"], utc=True).dt.hour
    pool = e[(e["hour"] >= 9) & (e["hour"] <= 15)].drop(columns=["hour"])
    cfg = ManageConfig(name="trail_0.12", family="trail", trail_atr=0.12, horizon=16, max_bars=16)
    sim = simulate_panel(pool, h1, cfg, mkt=mkt, eis=entry_indices(pool, mkt["ts"]))
    sim["atr_price"] = sim["atr_entry"]
    sim["holding_bars"] = sim["holding_bars_sim"]

    print("year  trades    ret     dd     wr     pf")
    iso = []
    for y, g in sim.groupby(pd.to_datetime(sim["timestamp"], utc=True).dt.year):
        keep = ["timestamp", "side", "y_prob", "entry_price", "atr_price", "net_return", "holding_bars"]
        _, m = run_portfolio(g[keep], max_open=1, starting_equity=STARTING_EQUITY)
        row = {
            "year": int(y),
            "n_trades": m["n_trades"],
            "ret": m["total_return"],
            "dd": m["max_drawdown"],
            "wr": m["win_rate"],
            "pf": m["profit_factor"],
        }
        iso.append(row)
        print(
            f"{y:4d}  {m['n_trades']:6d}  {m['total_return']:+6.1%}  {m['max_drawdown']:5.1%}  "
            f"{m['win_rate']:5.1%}  {m['profit_factor']:5.2f}"
        )

    overall = metrics_for(sim, max_open=1)
    print(
        f"\nOVERALL n={overall['n_trades']} avgWR={overall['avg_iso_wr']:.1%} "
        f"contWR={overall['cont_wr']:.1%} contPF={overall['cont_pf']:.2f} "
        f"ret={overall['cont_ret']:+.1%} dd={overall['cont_dd']:.1%}"
    )
    out = {
        "policy": {
            "entry": "WF primary top 5% unique bars",
            "session": "09-15 UTC",
            "exit": "ATR trail 0.12, horizon 16",
            "max_open": 1,
        },
        "overall": overall,
        "by_year": iso,
    }
    path = _ROOT / "artifacts/pipeline_backtest/exit_best_trail012_yearly.json"
    path.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print("wrote", path)


if __name__ == "__main__":
    main()
