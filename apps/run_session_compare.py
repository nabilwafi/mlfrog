"""Quick compare: ATR Trail 0.12 with session 09-15 vs all hours."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from apps.run_atr_trail_robustness import metrics_bundle, replay_trail
from apps.run_exit_engine_hunt import build_entries
from apps.run_yearly_walkforward_backtest import _load_side
from production.paper.barrier_resim import load_h1
from research.position_mgmt.services.paths import entry_indices, prepare_market


def run(pool: pd.DataFrame, mkt: dict, label: str) -> dict:
    pool = pool.reset_index(drop=True)
    eis = entry_indices(pool, mkt["ts"])
    rows = []
    for i in range(len(pool)):
        src = pool.iloc[i]
        r = replay_trail(
            side=str(src["side"]),
            entry=float(src["entry_price"]),
            atr=float(src["atr_entry"]),
            ei=int(eis[i]),
            mkt=mkt,
            trail=0.12,
        )
        if r is None:
            continue
        rows.append(
            {
                "timestamp": src["timestamp"],
                "side": src["side"],
                "y_prob": float(src["y_prob"]),
                "entry_price": float(src["entry_price"]),
                "atr_price": float(src["atr_entry"]),
                **r,
            }
        )
    panel = pd.DataFrame(rows)
    base = metrics_bundle(panel)
    by = base.get("by_year") or []
    pos = sum(1 for y in by if y["return"] > 0)
    worst = min(by, key=lambda y: y["return"]) if by else None
    print(f"=== {label} ===")
    print(
        f"candidates={len(pool)} trades={base['n_trades']} "
        f"ret={base['total_return']:+.1%} pf={base['profit_factor']:.2f} "
        f"wr={base['win_rate']:.1%} dd={base['max_drawdown']:.1%} "
        f"hold={base['avg_holding_bars']:.2f} years+={pos}/{len(by)}"
    )
    if worst:
        print(f"worst year {worst['year']}: {worst['return']:+.1%} pf={worst['pf']:.2f}")
    # hour attribution on candidate panel (pre max_open)
    g = panel.copy()
    g["hour"] = pd.to_datetime(g["timestamp"], utc=True).dt.hour
    hh = g.groupby("hour")["net_return"].agg(n="count", sum="sum", avg="mean")
    print("by UTC hour (candidate sum net_return):")
    for h, r in hh.iterrows():
        mark = " <<" if 9 <= int(h) <= 15 else ""
        print(f"  {int(h):02d}: n={int(r['n']):4d} sum={r['sum']:+.3f} avg={r['avg']:+.5f}{mark}")
    return base


def main() -> None:
    long_df = _load_side("long")
    short_df = _load_side("short")
    h1 = load_h1(str(_ROOT / "artifacts/raw/XAUUSD/H1/data.parquet"))
    mkt = prepare_market(h1)
    e = build_entries(long_df, short_df, h1, top_pct=0.05, years=list(range(2015, 2027)))
    e["timestamp"] = pd.to_datetime(e["timestamp"], utc=True)
    e["hour"] = e["timestamp"].dt.hour
    print(f"WF top5% entries total={len(e)}")
    a = run(e[(e["hour"] >= 9) & (e["hour"] <= 15)].copy(), mkt, "session 09-15 UTC")
    b = run(e.copy(), mkt, "ALL sessions")
    print("\n=== delta (all - sess) ===")
    print(
        f"ret {b['total_return']-a['total_return']:+.1%}  "
        f"pf {b['profit_factor']-a['profit_factor']:+.2f}  "
        f"dd {b['max_drawdown']-a['max_drawdown']:+.1%}  "
        f"trades {b['n_trades']-a['n_trades']:+d}"
    )


if __name__ == "__main__":
    main()
