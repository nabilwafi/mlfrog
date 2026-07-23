"""Backtest steps 4–5 on confidence panel + H1 OHLC resim."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from production import DAILY_LOSS_STOP_R, EXPECTED_R_REF, RISK_BASE, RISK_MAX, RISK_MIN
from production.paper.barrier_resim import attach_resim_returns, load_h1
from production.paper.edge_sizing import expected_r_from_edge, risk_pct_from_edge
from production.paper.market_state import infer_market_state
from research.portfolio_backtest.services.engine import _lots_from_equity
from research.portfolio_backtest.services.metrics import max_dd_from_equity, profit_factor_pnl
from research.portfolio_heat import H1_HOURS, STARTING_EQUITY
from settings.strategy import CONTRACT_SIZE, SL_ATR_MULT, TP_ATR_MULT

COST_HAIRCUT = 1.5e-4  # matches panel realized→net


def _to_ts(v) -> pd.Timestamp:
    t = pd.Timestamp(v)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def risk_pct_from_edge_rr(edge_score: float, *, rr: float) -> float:
    er = float(edge_score) * (float(rr) + 1.0) - 1.0
    if er <= 0:
        return float(RISK_MIN)
    raw = float(RISK_BASE) * er / float(EXPECTED_R_REF)
    return float(min(max(raw, float(RISK_MIN)), float(RISK_MAX)))


def run_portfolio(
    panel: pd.DataFrame,
    *,
    starting_equity: float = STARTING_EQUITY,
    rr: float | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Steps 1–3 portfolio rules + optional RR for edge sizing."""
    t = panel.sort_values("timestamp").reset_index(drop=True).copy()
    if "atr_price" not in t.columns:
        t["atr_price"] = t["atr_entry"].astype(float)
    hold = t["holding_bars"].astype(float).fillna(4.0)
    t["exit_ts"] = pd.to_datetime(t["timestamp"], utc=True) + pd.to_timedelta(hold * H1_HOURS, unit="h")

    use_rr = float(rr) if rr is not None else float(TP_ATR_MULT) / float(SL_ATR_MULT)
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

        if open_exit is not None and ts < open_exit:
            continue
        open_exit = None

        r_unit = equity * RISK_BASE
        if r_unit <= 0 or day_pnl <= -DAILY_LOSS_STOP_R * r_unit:
            continue

        edge = float(row["meta_proba"])
        risk_pct = risk_pct_from_edge_rr(edge, rr=use_rr)
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

        ms = infer_market_state(row)
        rows.append(
            {
                "timestamp": ts,
                "side": str(row.get("side", "long")).lower(),
                "meta_proba": edge,
                "risk_pct": risk_pct,
                "lot": lots,
                "pnl": pnl,
                "equity": equity,
                "trend": ms.trend,
                "volatility": ms.volatility,
                "session": ms.session,
                "structure": ms.structure,
                "regime_raw": ms.regime_raw,
                "exit_reason": row.get("exit_reason_resim", ""),
            }
        )

    traded = pd.DataFrame(rows)
    metrics = _metrics(traded, starting_equity)
    metrics["rr_used"] = use_rr
    return traded, metrics


def _metrics(traded: pd.DataFrame, starting_equity: float) -> dict:
    if traded.empty:
        return {
            "n_trades": 0,
            "final_equity": float(starting_equity),
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "profit_factor": float("nan"),
            "win_rate": float("nan"),
        }
    pnl = traded["pnl"].to_numpy(dtype=float)
    eq = traded["equity"].to_numpy(dtype=float)
    return {
        "n_trades": int(len(traded)),
        "final_equity": float(eq[-1]),
        "total_return": float(eq[-1] / starting_equity - 1.0),
        "max_drawdown": float(max_dd_from_equity(eq)),
        "profit_factor": float(profit_factor_pnl(pnl)),
        "win_rate": float(np.mean(pnl > 0)),
        "risk_pct_mean": float(traded["risk_pct"].mean()),
        "date_start": str(traded["timestamp"].min()),
        "date_end": str(traded["timestamp"].max()),
    }


def regime_breakdown(traded: pd.DataFrame) -> pd.DataFrame:
    if traded.empty:
        return pd.DataFrame()
    g = traded.groupby("regime_raw", dropna=False)
    return (
        g.agg(n=("pnl", "count"), pnl_sum=("pnl", "sum"), win_rate=("pnl", lambda s: float(np.mean(s > 0))))
        .reset_index()
        .sort_values("n", ascending=False)
    )


def main() -> None:
    panel = pd.read_parquet(_ROOT / "artifacts/research/confidence_layer/confidence_panel.parquet")
    out = _ROOT / "artifacts/pipeline_backtest"
    out.mkdir(parents=True, exist_ok=True)

    # --- Step 4: same PnL as 1–3 + market state labels ---
    t4, m4 = run_portfolio(panel)
    m4["policy"] = "step4_market_state_log_on_s123"
    br4 = regime_breakdown(t4)
    print("=== STEP 4 (Market State log-only, barriers unchanged) ===")
    for k, v in m4.items():
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")
    print("  regime breakdown:")
    print(br4.to_string(index=False) if not br4.empty else "  (empty)")
    t4.to_parquet(out / "step4_trades.parquet", index=False)
    br4.to_csv(out / "step4_regime_breakdown.csv", index=False)

    # --- Step 5: resim TP=3 / SL=1.5 / H=48 on H1 OHLC ---
    print("\n=== STEP 5 (resim TP3 / SL1.5 / H48 on H1 OHLC) ===")
    h1 = load_h1(str(_ROOT / "artifacts/raw/XAUUSD/H1/data.parquet"))
    resim = attach_resim_returns(panel, h1, tp_mult=3.0, sl_mult=1.5, horizon=48)
    resim["net_return"] = resim["net_return"].astype(float) - COST_HAIRCUT
    # drop rows with no path
    bad = (resim["exit_reason_resim"] == "NO_PATH").sum()
    resim = resim.loc[resim["exit_reason_resim"] != "NO_PATH"].copy()
    print(f"  dropped NO_PATH={bad} remaining={len(resim)}")
    print("  exit reasons:", resim["exit_reason_resim"].value_counts().to_dict())
    print(
        "  holding_bars mean/max:",
        float(resim["holding_bars"].mean()),
        float(resim["holding_bars"].max()),
    )

    t5, m5 = run_portfolio(resim, rr=3.0 / 1.5)
    m5["policy"] = "step5_tp3_h48_resim_s123_portfolio"
    m5["note"] = "Same Primary/Meta scores; exits resimulated. Not a full retrain."
    br5 = regime_breakdown(t5)
    for k, v in m5.items():
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")
    print("  regime breakdown:")
    print(br5.to_string(index=False) if not br5.empty else "  (empty)")

    t5.to_parquet(out / "step5_trades.parquet", index=False)
    br5.to_csv(out / "step5_regime_breakdown.csv", index=False)
    (out / "steps45_metrics.json").write_text(
        json.dumps({"step4": m4, "step5": m5}, indent=2, default=float),
        encoding="utf-8",
    )
    print("\nwrote", out)


if __name__ == "__main__":
    main()
