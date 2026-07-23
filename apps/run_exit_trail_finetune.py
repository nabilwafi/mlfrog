"""Fine-tune ATR trail exits toward WR60 + PF1.7 on WF 2015–2026."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from apps.run_exit_engine_hunt import build_entries, metrics_for
from apps.run_yearly_walkforward_backtest import _load_side
from production.paper.barrier_resim import load_h1
from research.position_mgmt.services.paths import entry_indices, prepare_market
from research.position_mgmt.services.simulator import ManageConfig, simulate_panel


def main() -> None:
    years = list(range(2015, 2027))
    long_df = _load_side("long")
    short_df = _load_side("short")
    h1 = load_h1(str(_ROOT / "artifacts/raw/XAUUSD/H1/data.parquet"))
    mkt = prepare_market(h1)

    rows = []
    for top_pct in (0.02, 0.05, 0.08):
        print(f"\n=== entries top{top_pct:.0%} ===")
        entries = build_entries(long_df, short_df, h1, top_pct=top_pct, years=years)
        eis = entry_indices(entries, mkt["ts"])
        variants = list(entries.assign(tag="all").itertuples(index=False))  # unused
        pools = [("all", entries)]
        e2 = entries.copy()
        e2["hour"] = pd.to_datetime(e2["timestamp"], utc=True).dt.hour
        pools.append(("sess8-16", e2[(e2["hour"] >= 8) & (e2["hour"] <= 16)].drop(columns=["hour"])))

        for pool_name, pool in pools:
            if pool.empty:
                continue
            eis_p = entry_indices(pool, mkt["ts"])
            for trail in (0.25, 0.35, 0.45, 0.5, 0.55, 0.6, 0.7):
                for horizon in (8, 16, 24):
                    cfg = ManageConfig(
                        name=f"trail_{trail}",
                        family="trail",
                        trail_atr=trail,
                        horizon=horizon,
                        max_bars=horizon,
                    )
                    sim = simulate_panel(pool, h1, cfg, mkt=mkt, eis=eis_p)
                    sim["atr_price"] = sim["atr_entry"]
                    sim["holding_bars"] = sim["holding_bars_sim"]
                    m = metrics_for(sim, max_open=1)
                    m.update({"top_pct": top_pct, "pool": pool_name, "trail": trail, "horizon": horizon})
                    rows.append(m)
                    hit = m["avg_iso_wr"] >= 0.58 and m["cont_pf"] >= 1.5 and m["n_trades"] >= 30
                    if hit or (trail in (0.45, 0.5) and horizon == 8 and pool_name == "all"):
                        mark = " <<HIT" if hit else ""
                        print(
                            f"top{top_pct:.0%} {pool_name} trail{trail} h{horizon}: "
                            f"n={m['n_trades']} avgWR={m['avg_iso_wr']:.1%} "
                            f"PF={m['cont_pf']:.2f} ret={m['cont_ret']:+.1%}{mark}"
                        )

    df = pd.DataFrame(rows)
    print("\n=== HIT ===")
    hit = df[(df.avg_iso_wr >= 0.58) & (df.cont_pf >= 1.55) & (df.n_trades >= 30)]
    print(hit.sort_values("cont_pf", ascending=False).head(20).to_string(index=False) if not hit.empty else "(none)")

    print("\n=== best PF among WR>=55% ===")
    sub = df[(df.avg_iso_wr >= 0.55) & (df.n_trades >= 30)].sort_values("cont_pf", ascending=False)
    print(sub.head(20).to_string(index=False))

    print("\n=== best WR among PF>=1.2 ===")
    sub2 = df[(df.cont_pf >= 1.2) & (df.n_trades >= 30)].sort_values("avg_iso_wr", ascending=False)
    print(sub2.head(20).to_string(index=False) if not sub2.empty else "(none with PF>=1.2)")

    out = _ROOT / "artifacts/pipeline_backtest/exit_trail_finetune.json"
    out.write_text(json.dumps(rows, indent=2, default=float), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
