"""Walk-forward yearly report 2015-2026 — current production policy @ $80.

Primary WF top 5% | all sessions | ATR Trail 0.12 | max_open=1
Modes: fractional | vol_min 0.01 | fixed 0.01
"""

from __future__ import annotations

import json
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
from research.portfolio_backtest.services.engine import _lots_from_equity
from research.portfolio_backtest.services.metrics import max_dd_from_equity, profit_factor_pnl
from research.portfolio_heat import H1_HOURS
from research.position_mgmt.services.paths import entry_indices, prepare_market
from settings.strategy import CONTRACT_SIZE

OUT = _ROOT / "artifacts/pipeline_backtest"
STARTING = 80.0
YEARS = list(range(2015, 2027))
MODES = ("fractional", "vol_min", "fixed_0.01")


def _to_ts(v) -> pd.Timestamp:
    t = pd.Timestamp(v)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def run_portfolio(panel: pd.DataFrame, *, starting: float, mode: str) -> tuple[pd.DataFrame, dict]:
    t = panel.sort_values("timestamp").reset_index(drop=True).copy()
    t["exit_ts"] = pd.to_datetime(t["timestamp"], utc=True) + pd.to_timedelta(
        t["holding_bars"].astype(float) * H1_HOURS, unit="h"
    )
    equity = float(starting)
    day = None
    day_pnl = 0.0
    opens: list[tuple[pd.Timestamp, str]] = []
    rows: list[dict] = []

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
        atr = float(row["atr_price"])
        risk_pct = risk_pct_from_primary(edge)
        if mode == "fixed_0.01":
            lots = 0.01
        elif mode == "vol_min":
            lots = _lots_from_equity(
                equity, atr, mode="risk", fixed_lots=None, risk_pct=risk_pct, enforce_volume_min=True
            )
        else:
            lots = _lots_from_equity(
                equity, atr, mode="risk", fixed_lots=None, risk_pct=risk_pct, enforce_volume_min=False
            )
        if lots <= 0:
            continue
        pnl = lots * CONTRACT_SIZE * float(row["entry_price"]) * float(row["net_return"])
        equity += pnl
        day_pnl += pnl
        opens.append((_to_ts(row["exit_ts"]), side))
        rows.append(
            {
                "timestamp": ts,
                "year": int(ts.year),
                "side": side,
                "lots": lots,
                "pnl": pnl,
                "equity": equity,
                "holding_bars": int(row["holding_bars"]),
            }
        )

    traded = pd.DataFrame(rows)
    if traded.empty:
        return traded, {
            "n_trades": 0,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "profit_factor": 0.0,
            "win_rate": 0.0,
            "final_equity": starting,
        }
    pnl = traded["pnl"].to_numpy(dtype=float)
    return traded, {
        "n_trades": int(len(traded)),
        "total_return": float(traded["equity"].iloc[-1] / starting - 1.0),
        "final_equity": float(traded["equity"].iloc[-1]),
        "max_drawdown": float(max_dd_from_equity(traded["equity"].to_numpy(dtype=float))),
        "profit_factor": float(profit_factor_pnl(pnl)),
        "win_rate": float(np.mean(pnl > 0)),
        "avg_holding_bars": float(traded["holding_bars"].mean()),
        "avg_lot": float(traded["lots"].mean()),
        "min_lot": float(traded["lots"].min()),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("Building WF Primary top5% entries 2015-2026 @ $80...")
    long_df = _load_side("long")
    short_df = _load_side("short")
    h1 = load_h1(str(_ROOT / "artifacts/raw/XAUUSD/H1/data.parquet"))
    mkt = prepare_market(h1)
    e = build_entries(long_df, short_df, h1, top_pct=0.05, years=YEARS)
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
    print(f"candidates={len(panel)}")

    payload = {"starting_equity": STARTING, "modes": {}}
    md = [
        "# Walk-Forward 2015-2026 @ $80",
        "",
        "Policy: Primary WF top 5% | all sessions | ATR Trail 0.12 | max_open=1",
        "",
    ]

    for mode in MODES:
        traded, overall = run_portfolio(panel, starting=STARTING, mode=mode)
        yearly = []
        for y in YEARS:
            g = panel[pd.to_datetime(panel["timestamp"], utc=True).dt.year == y]
            if g.empty:
                continue
            _, ym = run_portfolio(g, starting=STARTING, mode=mode)
            yearly.append({"year": y, **ym})
        payload["modes"][mode] = {"continuous": overall, "isolated_yearly": yearly}
        print(
            f"\n=== {mode} CONTINUOUS === trades={overall['n_trades']} "
            f"ret={overall['total_return']:+.1%} pf={overall['profit_factor']:.2f} "
            f"wr={overall['win_rate']:.1%} dd={overall['max_drawdown']:.1%} "
            f"final=${overall['final_equity']:.1f}"
        )
        md += [
            f"## {mode}",
            "",
            (
                f"Continuous: trades={overall['n_trades']} ret={overall['total_return']:+.1%} "
                f"pf={overall['profit_factor']:.2f} wr={overall['win_rate']:.1%} "
                f"dd={overall['max_drawdown']:.1%} final=${overall['final_equity']:.1f}"
            ),
            "",
            "| Year | Trades | Return | PF | WR | DD |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
        for r in yearly:
            print(
                f"  {r['year']}: n={r['n_trades']:4d} ret={r['total_return']:+7.1%} "
                f"pf={r['profit_factor']:.2f} wr={r['win_rate']:.1%} dd={r['max_drawdown']:.1%}"
            )
            md.append(
                f"| {r['year']} | {r['n_trades']} | {r['total_return']:+.1%} | "
                f"{r['profit_factor']:.2f} | {r['win_rate']:.1%} | {r['max_drawdown']:.1%} |"
            )
        md.append("")
        pd.DataFrame(yearly).to_csv(OUT / f"wf_2015_2026_eq80_{mode}_isolated.csv", index=False)

    (OUT / "wf_2015_2026_eq80.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    (OUT / "wf_2015_2026_eq80.md").write_text("\n".join(md), encoding="utf-8")
    print("\nwrote", OUT / "wf_2015_2026_eq80.md")


if __name__ == "__main__":
    main()
