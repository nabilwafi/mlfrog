"""Backtest paper policy steps 1–3 on confidence panel."""

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
from research.portfolio_backtest.services.metrics import max_dd_from_equity, profit_factor_pnl
from research.portfolio_heat import H1_HOURS, STARTING_EQUITY
from research.portfolio_backtest.services.engine import _lots_from_equity
from settings.strategy import CONTRACT_SIZE


def _to_ts(v) -> pd.Timestamp:
    t = pd.Timestamp(v)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def run(panel: pd.DataFrame, *, starting_equity: float = STARTING_EQUITY) -> tuple[pd.DataFrame, dict]:
    t = panel.sort_values("timestamp").reset_index(drop=True).copy()
    if "atr_price" not in t.columns:
        t["atr_price"] = t["atr_entry"].astype(float) if "atr_entry" in t.columns else t["atr"].astype(float)
    hold = t["holding_bars"].astype(float).fillna(4.0) if "holding_bars" in t.columns else 4.0
    t["exit_ts"] = pd.to_datetime(t["timestamp"], utc=True) + pd.to_timedelta(hold * H1_HOURS, unit="h")

    equity = float(starting_equity)
    day = None
    day_pnl = 0.0
    open_exit: pd.Timestamp | None = None
    rows: list[dict] = []

    for _, row in t.iterrows():
        ts = _to_ts(row["timestamp"])
        d = ts.date()
        if day != d:
            day = d
            day_pnl = 0.0

        # step 1: single position
        if open_exit is not None and ts < open_exit:
            continue
        open_exit = None

        # step 2–3: no confidence / no meta gate — heat only
        r_unit = equity * RISK_BASE
        if r_unit <= 0 or day_pnl <= -DAILY_LOSS_STOP_R * r_unit:
            continue

        edge = float(row["meta_proba"])
        risk_pct = risk_pct_from_edge(edge)
        side = str(row.get("side", "long")).lower()
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
        open_exit = _to_ts(row["exit_ts"])

        rows.append(
            {
                "timestamp": ts,
                "side": side,
                "meta_proba": edge,
                "risk_pct": risk_pct,
                "lot": lots,
                "pnl": pnl,
                "equity": equity,
            }
        )

    traded = pd.DataFrame(rows)
    if traded.empty:
        metrics = {
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
            "risk_pct_min": float(traded["risk_pct"].min()),
            "risk_pct_max": float(traded["risk_pct"].max()),
            "date_start": str(traded["timestamp"].min()),
            "date_end": str(traded["timestamp"].max()),
        }
    metrics["policy"] = "s1_single_s2_noconf_s3_meta_edge_heat1R"
    metrics["starting_equity"] = float(starting_equity)
    return traded, metrics


def main() -> None:
    panel = pd.read_parquet(_ROOT / "artifacts/research/confidence_layer/confidence_panel.parquet")
    traded, metrics = run(panel)
    print("--- Steps 1–3 backtest (confidence panel, $80) ---")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}" if abs(v) < 1e6 else f"{k}: {v}")
        else:
            print(f"{k}: {v}")

    out = _ROOT / "artifacts/pipeline_backtest"
    out.mkdir(parents=True, exist_ok=True)
    traded.to_parquet(out / "steps123_trades.parquet", index=False)
    (out / "steps123_metrics.json").write_text(json.dumps(metrics, indent=2, default=float), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
