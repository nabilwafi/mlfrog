"""Backtest: ATR TP/SL (2.0/1.5) + multi-entry max_open sweep."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from production import DAILY_LOSS_STOP_R, RISK_BASE
from production.paper.edge_sizing import risk_pct_from_edge
from research.portfolio_backtest.services.engine import _lots_from_equity
from research.portfolio_backtest.services.metrics import max_dd_from_equity, profit_factor_pnl
from research.portfolio_heat import H1_HOURS, STARTING_EQUITY
from settings.strategy import CONTRACT_SIZE, SL_ATR_MULT, TP_ATR_MULT


def _to_ts(v) -> pd.Timestamp:
    t = pd.Timestamp(v)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def run(
    panel: pd.DataFrame,
    *,
    max_open: int = 3,
    block_opposite: bool = True,
    starting_equity: float = STARTING_EQUITY,
) -> tuple[pd.DataFrame, dict]:
    """
    TP/SL from ATR multipliers in settings (default 2.0 / 1.5) via panel net_return.
    Meta edge → risk_pct sizing. Multi-entry up to max_open.
    """
    t = panel.sort_values("timestamp").reset_index(drop=True).copy()
    if "atr_price" not in t.columns:
        t["atr_price"] = t["atr_entry"].astype(float)
    hold = t["holding_bars"].astype(float).fillna(4.0)
    t["exit_ts"] = pd.to_datetime(t["timestamp"], utc=True) + pd.to_timedelta(hold * H1_HOURS, unit="h")

    equity = float(starting_equity)
    day = None
    day_pnl = 0.0
    opens: list[tuple[pd.Timestamp, str]] = []  # (exit_ts, side)
    rows: list[dict] = []

    for _, row in t.iterrows():
        ts = _to_ts(row["timestamp"])
        d = ts.date()
        if day != d:
            day = d
            day_pnl = 0.0

        opens = [(e, s) for e, s in opens if e > ts]
        side = str(row.get("side", "long")).lower()

        if len(opens) >= int(max_open):
            continue
        if block_opposite and opens:
            open_sides = {s for _, s in opens}
            if side not in open_sides:
                continue

        r_unit = equity * RISK_BASE
        if r_unit <= 0 or day_pnl <= -DAILY_LOSS_STOP_R * r_unit:
            continue

        edge = float(row["meta_proba"])
        risk_pct = risk_pct_from_edge(edge)
        entry = float(row["entry_price"])
        atr = float(row["atr_price"])
        lots = _lots_from_equity(
            equity, atr, mode="risk", fixed_lots=None, risk_pct=risk_pct, enforce_volume_min=False
        )
        if lots <= 0:
            continue

        net_ret = float(row.get("net_return", 0.0) or 0.0)
        pnl = float(lots) * float(CONTRACT_SIZE) * entry * net_ret
        equity += pnl
        day_pnl += pnl
        opens.append((_to_ts(row["exit_ts"]), side))

        rows.append(
            {
                "timestamp": ts,
                "side": side,
                "meta_proba": edge,
                "risk_pct": risk_pct,
                "lot": lots,
                "pnl": pnl,
                "equity": equity,
                "n_open_after": len(opens),
            }
        )

    traded = pd.DataFrame(rows)
    if traded.empty:
        metrics: dict = {
            "n_trades": 0,
            "final_equity": starting_equity,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "profit_factor": float("nan"),
            "win_rate": float("nan"),
        }
    else:
        pnl = traded["pnl"].to_numpy(dtype=float)
        eq = traded["equity"].to_numpy(dtype=float)
        metrics = {
            "n_trades": int(len(traded)),
            "final_equity": float(eq[-1]),
            "total_return": float(eq[-1] / starting_equity - 1.0),
            "max_drawdown": float(max_dd_from_equity(eq)),
            "profit_factor": float(profit_factor_pnl(pnl)),
            "win_rate": float(np.mean(pnl > 0)),
            "risk_pct_mean": float(traded["risk_pct"].mean()),
            "max_concurrent": int(traded["n_open_after"].max()),
        }
    metrics.update(
        {
            "max_open": int(max_open),
            "block_opposite": bool(block_opposite),
            "sl_atr_mult": float(SL_ATR_MULT),
            "tp_atr_mult": float(TP_ATR_MULT),
            "starting_equity": float(starting_equity),
            "policy": f"atr_tp{TP_ATR_MULT}_sl{SL_ATR_MULT}_maxopen{max_open}",
        }
    )
    return traded, metrics


def yearly_breakdown(traded: pd.DataFrame, *, starting_equity: float) -> pd.DataFrame:
    """Per-year trade count + return within that year (equity path)."""
    if traded.empty:
        return pd.DataFrame()
    t = traded.copy()
    t["timestamp"] = pd.to_datetime(t["timestamp"], utc=True)
    t["year"] = t["timestamp"].dt.year
    rows = []
    eq_cursor = float(starting_equity)
    for year, g in t.groupby("year", sort=True):
        g = g.sort_values("timestamp")
        eq_start = float(eq_cursor)
        # equity column is already post-trade; year return from start→last in year
        eq_end = float(g["equity"].iloc[-1])
        pnl = g["pnl"].to_numpy(dtype=float)
        rows.append(
            {
                "year": int(year),
                "n_trades": int(len(g)),
                "equity_start": eq_start,
                "equity_end": eq_end,
                "year_return": eq_end / eq_start - 1.0 if eq_start > 0 else float("nan"),
                "pnl_sum": float(pnl.sum()),
                "win_rate": float(np.mean(pnl > 0)),
                "profit_factor": float(profit_factor_pnl(pnl)),
            }
        )
        eq_cursor = eq_end
    return pd.DataFrame(rows)


def main() -> None:
    panel = pd.read_parquet(_ROOT / "artifacts/research/confidence_layer/confidence_panel.parquet")
    panel["timestamp"] = pd.to_datetime(panel["timestamp"], utc=True)
    cand = panel.groupby(panel["timestamp"].dt.year).size()
    out = _ROOT / "artifacts/pipeline_backtest"
    out.mkdir(parents=True, exist_ok=True)

    print(f"TP/SL from ATR: {TP_ATR_MULT} / {SL_ATR_MULT} (settings.strategy)")
    print("Universe (confidence panel candidates/year) — hard ceiling before portfolio:")
    for y, n in cand.items():
        print(f"  {y}: {int(n)} candidates")
    print(f"  total: {len(panel)}")
    print("  -> 500 trades/year is NOT possible on this panel (max candidates/year ~163).\n")

    all_m = {}
    # default production knob
    for mo in (1, 2, 3, 5):
        traded, m = run(panel, max_open=mo)
        ydf = yearly_breakdown(traded, starting_equity=STARTING_EQUITY)
        m["by_year"] = ydf.to_dict(orient="records")
        all_m[str(mo)] = m
        print(
            f"=== max_open={mo} | total trades={m['n_trades']} "
            f"ret_all={m['total_return']:+.1%} dd={m['max_drawdown']:.1%} pf={m['profit_factor']:.3f} ==="
        )
        if not ydf.empty:
            print(ydf.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print()
        traded.to_parquet(out / f"multi_maxopen{mo}_trades.parquet", index=False)
        ydf.to_csv(out / f"multi_maxopen{mo}_by_year.csv", index=False)

    (out / "multi_entry_atr_tp_metrics.json").write_text(
        json.dumps(all_m, indent=2, default=float), encoding="utf-8"
    )
    print("wrote", out)


if __name__ == "__main__":
    main()
