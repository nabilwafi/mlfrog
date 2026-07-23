"""Backtest ATR Trail 0.12 all-session with Finex-style fixed 0.01 lot."""

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
from research.portfolio_backtest.services.engine import _lots_from_equity
from research.portfolio_backtest.services.metrics import max_dd_from_equity, profit_factor_pnl
from research.portfolio_heat import H1_HOURS
from research.position_mgmt.services.paths import entry_indices, prepare_market
from settings.strategy import CONTRACT_SIZE, SL_ATR_MULT

OUT = _ROOT / "artifacts/exit_research"
FIXED_LOT = 0.01


def _to_ts(v) -> pd.Timestamp:
    t = pd.Timestamp(v)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def run_pf(
    panel: pd.DataFrame,
    *,
    starting: float,
    mode: str,
) -> dict:
    """mode: fractional | vol_min | fixed_0.01"""
    t = panel.sort_values("timestamp").reset_index(drop=True).copy()
    t["exit_ts"] = pd.to_datetime(t["timestamp"], utc=True) + pd.to_timedelta(
        t["holding_bars"].astype(float) * H1_HOURS, unit="h"
    )
    equity = float(starting)
    peak = equity
    day = None
    day_pnl = 0.0
    opens: list[tuple[pd.Timestamp, str]] = []
    rows: list[dict] = []
    skipped = 0
    risk_at_entry = []

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

        atr = float(row["atr_price"])
        edge = float(row["y_prob"])
        risk_pct = risk_pct_from_primary(edge)
        if mode == "fixed_0.01":
            lots = FIXED_LOT
        elif mode == "vol_min":
            lots = _lots_from_equity(
                equity, atr, mode="risk", fixed_lots=None, risk_pct=risk_pct, enforce_volume_min=True
            )
        else:
            lots = _lots_from_equity(
                equity, atr, mode="risk", fixed_lots=None, risk_pct=risk_pct, enforce_volume_min=False
            )
        if lots <= 0:
            skipped += 1
            continue

        sl_dist = SL_ATR_MULT * atr
        dollar_risk = lots * CONTRACT_SIZE * sl_dist
        risk_at_entry.append(dollar_risk / equity if equity > 0 else float("nan"))

        pnl = lots * CONTRACT_SIZE * float(row["entry_price"]) * float(row["net_return"])
        equity += pnl
        peak = max(peak, equity)
        day_pnl += pnl
        opens.append((_to_ts(row["exit_ts"]), side))
        rows.append(
            {
                "timestamp": ts,
                "year": ts.year,
                "side": side,
                "lots": lots,
                "pnl": pnl,
                "equity": equity,
                "dollar_risk": dollar_risk,
            }
        )

    traded = pd.DataFrame(rows)
    if traded.empty:
        return {
            "mode": mode,
            "starting": starting,
            "n_trades": 0,
            "total_return": 0.0,
            "cagr": float("nan"),
            "max_drawdown": 0.0,
            "profit_factor": 0.0,
            "win_rate": 0.0,
            "final_equity": starting,
            "skipped": skipped,
            "avg_lot": 0.0,
            "avg_risk_pct_equity": float("nan"),
            "p95_risk_pct_equity": float("nan"),
            "blown": False,
        }

    pnl = traded["pnl"].to_numpy(dtype=float)
    years = (traded["timestamp"].max() - traded["timestamp"].min()).days / 365.25
    ret = float(traded["equity"].iloc[-1] / starting - 1.0)
    cagr = float((1.0 + ret) ** (1.0 / years) - 1.0) if years > 0 and (1.0 + ret) > 0 else float("nan")
    risk_arr = np.asarray(risk_at_entry, dtype=float)
    return {
        "mode": mode,
        "starting": starting,
        "n_trades": int(len(traded)),
        "total_return": ret,
        "cagr": cagr,
        "max_drawdown": float(max_dd_from_equity(traded["equity"].to_numpy(dtype=float))),
        "profit_factor": float(profit_factor_pnl(pnl)),
        "win_rate": float(np.mean(pnl > 0)),
        "final_equity": float(traded["equity"].iloc[-1]),
        "skipped": skipped,
        "avg_lot": float(traded["lots"].mean()),
        "avg_risk_pct_equity": float(np.nanmean(risk_arr)),
        "p95_risk_pct_equity": float(np.nanpercentile(risk_arr, 95)),
        "blown": bool(traded["equity"].iloc[-1] <= 0 or (peak > 0 and (peak - traded["equity"].min()) / peak > 0.5)),
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
    print(f"panel={len(panel)} trail=0.12 all-session max_open={MAX_OPEN_POSITIONS} fixed_lot={FIXED_LOT}")
    print()

    results = []
    for starting in (80.0, 10_000.0):
        for mode in ("fractional", "vol_min", "fixed_0.01"):
            m = run_pf(panel, starting=starting, mode=mode)
            results.append(m)
            print(
                f"${starting:>7.0f} {mode:12s}  trades={m['n_trades']:4d}  "
                f"ret={m['total_return']:+8.1%}  pf={m['profit_factor']:.2f}  "
                f"wr={m['win_rate']:.1%}  dd={m['max_drawdown']:.1%}  "
                f"avg_risk={m['avg_risk_pct_equity']:.1%}  "
                f"p95_risk={m['p95_risk_pct_equity']:.1%}  "
                f"final=${m['final_equity']:.1f}  skip={m['skipped']}"
            )

    df = pd.DataFrame(results)
    path = OUT / "fixed_lot_001_compare.csv"
    df.to_csv(path, index=False)

    # focus table for fixed 0.01
    md = [
        "# Fixed Lot 0.01 Backtest (Finex-style)",
        "",
        "Policy: Primary top 5% + ATR Trail 0.12 + max_open=1 + all sessions.",
        "",
        "| Starting | Mode | Trades | Return | PF | WR | DD | Avg risk/eq | P95 risk/eq | Final |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m in results:
        md.append(
            f"| ${m['starting']:.0f} | {m['mode']} | {m['n_trades']} | {m['total_return']:+.1%} | "
            f"{m['profit_factor']:.2f} | {m['win_rate']:.1%} | {m['max_drawdown']:.1%} | "
            f"{m['avg_risk_pct_equity']:.1%} | {m['p95_risk_pct_equity']:.1%} | ${m['final_equity']:.1f} |"
        )
    md += [
        "",
        "## Notes",
        "- `fixed_0.01`: always 0.01 lot (broker-style micro).",
        "- On $80 this oversizes vs RISK_MAX (~avg risk often >> 1.5% equity).",
        "- On $10k fixed 0.01 is often *undersized* vs risk% sizing.",
        "",
    ]
    (OUT / "fixed_lot_001_compare.md").write_text("\n".join(md), encoding="utf-8")
    print("\nwrote", path)
    print("wrote", OUT / "fixed_lot_001_compare.md")


if __name__ == "__main__":
    main()
