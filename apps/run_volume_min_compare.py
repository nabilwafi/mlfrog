"""Compare ATR Trail 0.12 all-session: fractional lots vs VOLUME_MIN=0.01."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from apps.run_atr_trail_robustness import replay_trail
from apps.run_exit_engine_hunt import build_entries
from apps.run_primary_loosen_backtest import risk_pct_from_primary
from apps.run_yearly_walkforward_backtest import _load_side
from production import DAILY_LOSS_STOP_R, MAX_OPEN_POSITIONS, RISK_BASE
from production.paper.barrier_resim import load_h1
from research.portfolio_heat import H1_HOURS
from research.portfolio_backtest.services.engine import _lots_from_equity
from research.portfolio_backtest.services.metrics import max_dd_from_equity, profit_factor_pnl
from research.position_mgmt.services.paths import entry_indices, prepare_market
from settings.strategy import CONTRACT_SIZE

PAPER_EQ = 10_000.0
RESEARCH_EQ = 80.0


def _to_ts(v) -> pd.Timestamp:
    t = pd.Timestamp(v)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def run_pf(panel: pd.DataFrame, *, starting: float, enforce_min: bool) -> dict:
    t = panel.sort_values("timestamp").reset_index(drop=True).copy()
    t["exit_ts"] = pd.to_datetime(t["timestamp"], utc=True) + pd.to_timedelta(
        t["holding_bars"].astype(float) * H1_HOURS, unit="h"
    )
    equity = float(starting)
    day = None
    day_pnl = 0.0
    opens: list[tuple[pd.Timestamp, str]] = []
    rows: list[dict] = []
    skipped_vol = 0

    for _, row in t.iterrows():
        ts = _to_ts(row["timestamp"])
        d = ts.date()
        if day != d:
            day = d
            day_pnl = 0.0
        opens = [(e, s) for e, s in opens if e > ts]
        side = str(row["side"]).lower()
        if len(opens) >= int(MAX_OPEN_POSITIONS):
            continue
        if opens and side not in {s for _, s in opens}:
            continue
        r_unit = equity * RISK_BASE
        if r_unit <= 0 or day_pnl <= -DAILY_LOSS_STOP_R * r_unit:
            continue
        edge = float(row["y_prob"])
        risk_pct = risk_pct_from_primary(edge)
        lots = _lots_from_equity(
            equity,
            float(row["atr_price"]),
            mode="risk",
            fixed_lots=None,
            risk_pct=risk_pct,
            enforce_volume_min=enforce_min,
        )
        if lots <= 0:
            skipped_vol += 1
            continue
        pnl = lots * CONTRACT_SIZE * float(row["entry_price"]) * float(row["net_return"])
        equity += pnl
        day_pnl += pnl
        opens.append((_to_ts(row["exit_ts"]), side))
        rows.append({"timestamp": ts, "year": ts.year, "side": side, "lots": lots, "pnl": pnl, "equity": equity})

    traded = pd.DataFrame(rows)
    if traded.empty:
        return {
            "n_trades": 0,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "profit_factor": 0.0,
            "win_rate": 0.0,
            "final_equity": starting,
            "skipped_vol": skipped_vol,
            "avg_lot": 0.0,
            "min_lot": 0.0,
            "median_lot": 0.0,
        }
    pnl = traded["pnl"].to_numpy(dtype=float)
    return {
        "n_trades": int(len(traded)),
        "total_return": float(traded["equity"].iloc[-1] / starting - 1.0),
        "max_drawdown": float(max_dd_from_equity(traded["equity"].to_numpy(dtype=float))),
        "profit_factor": float(profit_factor_pnl(pnl)),
        "win_rate": float(np.mean(pnl > 0)),
        "final_equity": float(traded["equity"].iloc[-1]),
        "skipped_vol": skipped_vol,
        "avg_lot": float(traded["lots"].mean()),
        "min_lot": float(traded["lots"].min()),
        "median_lot": float(traded["lots"].median()),
    }


def main() -> None:
    long_df = _load_side("long")
    short_df = _load_side("short")
    h1 = load_h1(str(_ROOT / "artifacts/raw/XAUUSD/H1/data.parquet"))
    mkt = prepare_market(h1)
    e = build_entries(long_df, short_df, h1, top_pct=0.05, years=list(range(2015, 2027)))
    e["timestamp"] = pd.to_datetime(e["timestamp"], utc=True)
    eis = entry_indices(e, mkt["ts"])
    rows = []
    for i in range(len(e)):
        src = e.iloc[i]
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
                "net_return": r["net_return"],
                "holding_bars": r["holding_bars"],
            }
        )
    panel = pd.DataFrame(rows)
    print(f"all-session trail0.12 candidates={len(panel)} max_open={MAX_OPEN_POSITIONS}")
    print()
    out_rows = []
    for starting, label in ((PAPER_EQ, "paper_$10k"), (RESEARCH_EQ, "research_$80")):
        for enforce, tag in ((False, "fractional"), (True, "vol_min_0.01")):
            m = run_pf(panel, starting=starting, enforce_min=enforce)
            m["equity_label"] = label
            m["sizing"] = tag
            m["starting"] = starting
            out_rows.append(m)
            print(
                f"{label:12s} {tag:14s}  trades={m['n_trades']:4d}  "
                f"ret={m['total_return']:+7.1%}  pf={m['profit_factor']:.2f}  "
                f"wr={m['win_rate']:.1%}  dd={m['max_drawdown']:.1%}  "
                f"lot[min/med/avg]={m['min_lot']:.3f}/{m['median_lot']:.3f}/{m['avg_lot']:.3f}  "
                f"skip_vol={m['skipped_vol']}"
            )
    out = pd.DataFrame(out_rows)
    path = _ROOT / "artifacts/exit_research/volume_min_compare.csv"
    out.to_csv(path, index=False)
    print("\nwrote", path)


if __name__ == "__main__":
    main()
