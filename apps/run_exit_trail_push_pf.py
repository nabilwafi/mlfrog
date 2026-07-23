"""Push trail+session toward PF 1.7 while keeping WR>=60%."""

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
from research.position_mgmt.services.paths import entry_indices, prepare_market
from research.position_mgmt.services.simulator import ManageConfig, simulate_panel


def main() -> None:
    years = list(range(2015, 2027))
    long_df = _load_side("long")
    short_df = _load_side("short")
    h1 = load_h1(str(_ROOT / "artifacts/raw/XAUUSD/H1/data.parquet"))
    mkt = prepare_market(h1)
    entries = build_entries(long_df, short_df, h1, top_pct=0.05, years=years)

    def sess(lo: int, hi: int) -> pd.DataFrame:
        e = entries.copy()
        e["hour"] = pd.to_datetime(e["timestamp"], utc=True).dt.hour
        return e[(e["hour"] >= lo) & (e["hour"] <= hi)].drop(columns=["hour"])

    e816 = sess(8, 16)
    pools = {
        "sess8-16": e816,
        "sess9-15": sess(9, 15),
        "sess8-16_top50": e816.assign(
            _r=e816.groupby(e816["timestamp"].dt.year)["y_prob"].rank(pct=True)
        )
        .query("_r >= 0.5")
        .drop(columns=["_r"]),
        "sess8-16_top30": e816.assign(
            _r=e816.groupby(e816["timestamp"].dt.year)["y_prob"].rank(pct=True)
        )
        .query("_r >= 0.7")
        .drop(columns=["_r"]),
    }

    rows = []
    for name, pool in pools.items():
        if pool.empty:
            continue
        eis = entry_indices(pool, mkt["ts"])
        for trail in (0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.28):
            for h in (8, 16, 24):
                cfg = ManageConfig(
                    name="t", family="trail", trail_atr=trail, horizon=h, max_bars=h
                )
                sim = simulate_panel(pool, h1, cfg, mkt=mkt, eis=eis)
                sim["atr_price"] = sim["atr_entry"]
                sim["holding_bars"] = sim["holding_bars_sim"]
                m = metrics_for(sim, max_open=1)
                m.update({"pool": name, "trail": trail, "horizon": h})
                rows.append(m)
                if m["cont_pf"] >= 1.45 or m["avg_iso_wr"] >= 0.65:
                    flag = ""
                    if m["avg_iso_wr"] >= 0.58 and m["cont_pf"] >= 1.65:
                        flag = " <<NEAR1.7"
                    if m["avg_iso_wr"] >= 0.58 and m["cont_pf"] >= 1.7:
                        flag = " <<HIT1.7"
                    print(
                        f"{name:16} trail{trail:.2f} h{h}: n={m['n_trades']:4d} "
                        f"avgWR={m['avg_iso_wr']:.1%} contPF={m['cont_pf']:.2f} "
                        f"avgPF={m['avg_iso_pf']:.2f} ret={m['cont_ret']:+.1%} "
                        f"dd={m['cont_dd']:.1%}{flag}"
                    )

    df = pd.DataFrame(rows)
    print("\n=== best by cont PF (WR>=58%) ===")
    sub = df[(df.avg_iso_wr >= 0.58) & (df.n_trades >= 30)].sort_values(
        "cont_pf", ascending=False
    )
    print(sub.head(25).to_string(index=False))

    out = _ROOT / "artifacts/pipeline_backtest/exit_trail_push_pf17.json"
    out.write_text(json.dumps(rows, indent=2, default=float), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
