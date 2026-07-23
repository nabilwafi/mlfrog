"""Loosen Primary candidate density; yearly trades + DD sweep."""

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
from production.paper.barrier_resim import load_h1
from research.portfolio_backtest.services.engine import _lots_from_equity
from research.portfolio_backtest.services.metrics import max_dd_from_equity, profit_factor_pnl
from research.portfolio_heat import H1_HOURS, STARTING_EQUITY
from settings.strategy import CONTRACT_SIZE, SL_ATR_MULT, TP_ATR_MULT

COST = 1.5e-4


def _to_ts(v) -> pd.Timestamp:
    t = pd.Timestamp(v)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def risk_pct_from_primary(p: float) -> float:
    """Reuse option-C curve but edge = primary proba; RR from ATR TP/SL."""
    rr = float(TP_ATR_MULT) / float(SL_ATR_MULT)
    er = float(p) * (rr + 1.0) - 1.0
    if er <= 0:
        return float(RISK_MIN)
    raw = float(RISK_BASE) * er / float(EXPECTED_R_REF)
    return float(min(max(raw, float(RISK_MIN)), float(RISK_MAX)))


def wilder_atr(h1: pd.DataFrame, period: int = 14) -> pd.Series:
    c = h1["close"].astype(float)
    h = h1["high"].astype(float)
    l = h1["low"].astype(float)
    prev = c.shift(1)
    tr = pd.concat([(h - l), (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def build_universe(primary: pd.DataFrame, h1: pd.DataFrame, *, top_pct: float) -> pd.DataFrame:
    """Per year keep top_pct of unique bars by best-side y_prob; attach entry/ATR/net_return."""
    p = primary.copy()
    p["timestamp"] = pd.to_datetime(p["timestamp"], utc=True)
    # best side per bar first
    p = p.loc[p.groupby("timestamp")["y_prob"].idxmax()].copy()
    parts = []
    for y, g in p.groupby(p["timestamp"].dt.year):
        if y < 2023:
            continue
        k = max(1, int(round(len(g) * float(top_pct))))
        parts.append(g.nlargest(k, "y_prob"))
    u = pd.concat(parts).sort_values("timestamp").reset_index(drop=True)

    h = h1.copy()
    h["timestamp"] = pd.to_datetime(h["timestamp"], utc=True)
    h["atr"] = wilder_atr(h)
    m = u.merge(h[["timestamp", "close", "atr"]], on="timestamp", how="left")
    m["entry_price"] = m["close"].astype(float)
    m["atr_price"] = m["atr"].astype(float)
    # label realized_return already on primary panel
    m["net_return"] = m["realized_return"].astype(float) - COST
    # ponytail: holding unknown on this panel — use median conf-panel hold ~4
    m["holding_bars"] = 4.0
    return m.dropna(subset=["entry_price", "atr_price", "net_return"])


def run_portfolio(
    panel: pd.DataFrame,
    *,
    max_open: int = 3,
    block_opposite: bool = True,
    starting_equity: float = STARTING_EQUITY,
    use_primary_edge: bool = True,
) -> tuple[pd.DataFrame, dict]:
    t = panel.sort_values("timestamp").reset_index(drop=True).copy()
    t["exit_ts"] = pd.to_datetime(t["timestamp"], utc=True) + pd.to_timedelta(
        t["holding_bars"].astype(float) * H1_HOURS, unit="h"
    )
    equity = float(starting_equity)
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
        if len(opens) >= max_open:
            continue
        if block_opposite and opens and side not in {s for _, s in opens}:
            continue
        r_unit = equity * RISK_BASE
        if r_unit <= 0 or day_pnl <= -DAILY_LOSS_STOP_R * r_unit:
            continue

        edge = float(row["y_prob"] if use_primary_edge else row.get("meta_proba", row["y_prob"]))
        risk_pct = risk_pct_from_primary(edge)
        lots = _lots_from_equity(
            equity,
            float(row["atr_price"]),
            mode="risk",
            fixed_lots=None,
            risk_pct=risk_pct,
            enforce_volume_min=False,
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
                "year": ts.year,
                "side": side,
                "y_prob": edge,
                "risk_pct": risk_pct,
                "pnl": pnl,
                "equity": equity,
            }
        )

    traded = pd.DataFrame(rows)
    if traded.empty:
        return traded, {"n_trades": 0, "final_equity": starting_equity, "total_return": 0.0, "max_drawdown": 0.0}

    pnl = traded["pnl"].to_numpy(dtype=float)
    eq = traded["equity"].to_numpy(dtype=float)
    by = []
    eq_c = float(starting_equity)
    for year, g in traded.groupby("year", sort=True):
        eq_s = eq_c
        eq_e = float(g["equity"].iloc[-1])
        pp = g["pnl"].to_numpy(dtype=float)
        # year equity path: start + after each trade
        eq_path = np.concatenate([[eq_s], g["equity"].to_numpy(dtype=float)])
        by.append(
            {
                "year": int(year),
                "n_trades": int(len(g)),
                "year_return": eq_e / eq_s - 1.0,
                "max_drawdown": float(max_dd_from_equity(eq_path)),
                "win_rate": float(np.mean(pp > 0)),
            }
        )
        eq_c = eq_e
    metrics = {
        "n_trades": int(len(traded)),
        "final_equity": float(eq[-1]),
        "total_return": float(eq[-1] / starting_equity - 1.0),
        "max_drawdown": float(max_dd_from_equity(eq)),
        "profit_factor": float(profit_factor_pnl(pnl)),
        "win_rate": float(np.mean(pnl > 0)),
        "by_year": by,
        "max_open": max_open,
    }
    return traded, metrics


def main() -> None:
    primary = pd.read_parquet(_ROOT / "artifacts/research/probability_quality/predictions.parquet")
    h1 = load_h1(str(_ROOT / "artifacts/raw/XAUUSD/H1/data.parquet"))
    out = _ROOT / "artifacts/pipeline_backtest"
    out.mkdir(parents=True, exist_ok=True)

    print("Primary density (unique bar, best side) top_pct -> candidates/year")
    dens = {}
    for pct in (0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25):
        u = build_universe(primary, h1, top_pct=pct)
        by = u.groupby(u["timestamp"].dt.year).size().to_dict()
        dens[str(pct)] = {str(k): int(v) for k, v in by.items()}
        print(f"  top {pct:.0%}: { {int(k): int(v) for k,v in by.items()} }")

    print("\nBacktest target: ~500 trades/year, show ret + maxDD overall and per year")
    print("Rules: ATR TP/SL 2.0/1.5, primary-edge sizing, heat -1R\n")
    results = {"density": dens, "runs": {}}
    for pct in (0.12, 0.15, 0.18, 0.20, 0.25):
        for mo in (3, 5, 8):
            u = build_universe(primary, h1, top_pct=pct)
            traded, m = run_portfolio(u, max_open=mo)
            key = f"top{int(pct*100)}_maxopen{mo}"
            m["top_pct"] = pct
            results["runs"][key] = m
            yrs = ", ".join(
                f"{r['year']}:{r['n_trades']}(ret={r['year_return']:+.1%},dd={r['max_drawdown']:.1%})"
                for r in m.get("by_year", [])
            )
            flag = "OK" if m["max_drawdown"] <= 0.15 else "DD>15%"
            print(
                f"{key}: trades={m['n_trades']:4d} ret_all={m['total_return']:+.1%} "
                f"maxDD={m['max_drawdown']:.1%} pf={m.get('profit_factor', float('nan')):.2f} [{flag}]"
            )
            print(f"         by_year {yrs}")

    (out / "primary_loosen_500_sweep.json").write_text(
        json.dumps(results, indent=2, default=float), encoding="utf-8"
    )
    print("\nwrote", out / "primary_loosen_500_sweep.json")


if __name__ == "__main__":
    main()
