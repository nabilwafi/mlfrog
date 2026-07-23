"""Old pipeline (meta label/gate) backtest — yearly like the trail table.

Panel only covers 2023–2026 (no meta OOF for 2015–2022).
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

from production import DAILY_LOSS_STOP_R, RISK_BASE
from production.paper.edge_sizing import risk_pct_from_edge
from research.portfolio_backtest.services.engine import _lots_from_equity
from research.portfolio_backtest.services.metrics import max_dd_from_equity, profit_factor_pnl
from research.portfolio_heat import H1_HOURS, STARTING_EQUITY
from settings.strategy import CONTRACT_SIZE

META_THR = 0.45
CONF_THR = 40.0


def _to_ts(v) -> pd.Timestamp:
    t = pd.Timestamp(v)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def run_old(
    panel: pd.DataFrame,
    *,
    meta_gate: bool = True,
    conf_gate: bool = True,
    max_open: int = 1,
    starting_equity: float = STARTING_EQUITY,
) -> tuple[pd.DataFrame, dict]:
    t = panel.sort_values("timestamp").reset_index(drop=True).copy()
    t["timestamp"] = pd.to_datetime(t["timestamp"], utc=True)
    if meta_gate:
        t = t[t["meta_proba"].astype(float) >= META_THR]
    if conf_gate and "confidence" in t.columns:
        t = t[t["confidence"].astype(float) >= CONF_THR]
    if t.empty:
        return pd.DataFrame(), {"n_trades": 0, "by_year": []}

    t["atr_price"] = t["atr_entry"].astype(float) if "atr_entry" in t.columns else t["atr_price"].astype(float)
    hold = t["holding_bars"].astype(float).fillna(4.0)
    t["exit_ts"] = t["timestamp"] + pd.to_timedelta(hold * H1_HOURS, unit="h")

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
        if len(opens) >= max_open:
            continue
        side = str(row["side"]).lower()
        r_unit = equity * RISK_BASE
        if r_unit <= 0 or day_pnl <= -DAILY_LOSS_STOP_R * r_unit:
            continue

        edge = float(row["meta_proba"])
        risk_pct = risk_pct_from_edge(edge)
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
        pnl = float(lots) * float(CONTRACT_SIZE) * float(row["entry_price"]) * float(row["net_return"])
        equity += pnl
        day_pnl += pnl
        opens.append((_to_ts(row["exit_ts"]), side))
        rows.append(
            {
                "timestamp": ts,
                "year": ts.year,
                "side": side,
                "meta_proba": edge,
                "confidence": float(row["confidence"]) if "confidence" in row.index and pd.notna(row.get("confidence")) else float("nan"),
                "pnl": pnl,
                "equity": equity,
            }
        )

    traded = pd.DataFrame(rows)
    if traded.empty:
        return traded, {"n_trades": 0, "by_year": [], "total_return": 0.0}

    pnl = traded["pnl"].to_numpy(dtype=float)
    eq = traded["equity"].to_numpy(dtype=float)
    by = []
    eq_c = float(starting_equity)
    for year, g in traded.groupby("year", sort=True):
        eq_s = eq_c
        eq_e = float(g["equity"].iloc[-1])
        pp = g["pnl"].to_numpy(dtype=float)
        eq_path = np.concatenate([[eq_s], g["equity"].to_numpy(dtype=float)])
        by.append(
            {
                "year": int(year),
                "n_trades": int(len(g)),
                "year_return": eq_e / eq_s - 1.0,
                "max_drawdown": float(max_dd_from_equity(eq_path)),
                "win_rate": float(np.mean(pp > 0)),
                "profit_factor": float(profit_factor_pnl(pp)),
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
        "meta_gate": meta_gate,
        "conf_gate": conf_gate,
        "max_open": max_open,
        "meta_thr": META_THR,
        "conf_thr": CONF_THR,
    }
    return traded, metrics


def isolated_years(panel: pd.DataFrame, **kwargs) -> list[dict]:
    """Reset equity each year."""
    p = panel.copy()
    p["timestamp"] = pd.to_datetime(p["timestamp"], utc=True)
    rows = []
    for y, g in p.groupby(p["timestamp"].dt.year, sort=True):
        _, m = run_old(g, starting_equity=STARTING_EQUITY, **kwargs)
        if m["n_trades"] == 0:
            rows.append({"year": int(y), "n_trades": 0})
            continue
        by = m["by_year"][0]
        rows.append(
            {
                "year": int(y),
                "n_trades": by["n_trades"],
                "ret": by["year_return"],
                "dd": by["max_drawdown"],
                "wr": by["win_rate"],
                "pf": by["profit_factor"],
            }
        )
    return rows


def print_table(title: str, rows: list[dict], overall: dict) -> None:
    print(f"\n=== {title} ===")
    print(f"{'year':>6} {'trades':>7} {'ret':>8} {'dd':>7} {'wr':>7} {'pf':>6}")
    for r in rows:
        if r.get("n_trades", 0) == 0:
            print(f"{r['year']:6d} {'—':>7}")
            continue
        print(
            f"{r['year']:6d} {r['n_trades']:7d} {r['ret']:+8.1%} {r['dd']:7.1%} "
            f"{r['wr']:7.1%} {r['pf']:6.2f}"
        )
    print(
        f"{'ALL':>6} {overall['n_trades']:7d} {overall['total_return']:+8.1%} "
        f"{overall['max_drawdown']:7.1%} {overall['win_rate']:7.1%} {overall['profit_factor']:6.2f}"
    )


def main() -> None:
    conf = pd.read_parquet(_ROOT / "artifacts/research/confidence_layer/confidence_panel.parquet")
    meta = pd.read_parquet(_ROOT / "artifacts/research/meta_model/oof_predictions.parquet")

    print("NOTE: meta/confidence panels only exist for 2023–2026 (not 2015–2022).")
    print("Old pipeline = Primary -> Meta label/proba -> (optional conf) -> single/multi -> meta-edge sizing -> heat -1R")

    results = {}

    # 1) Classic freeze on confidence panel
    kwargs = dict(meta_gate=True, conf_gate=True, max_open=1)
    traded, overall = run_old(conf, **kwargs)
    iso = isolated_years(conf, **kwargs)
    print_table("OLD: confidence_panel | meta>=0.45 | conf>=40 | max_open=1 | meta-edge (isolated $80/yr)", iso, {
        **overall,
        # for ALL line use continuous from full run
    })
    # also show continuous by_year
    print("\n(continuous equity compounding)")
    print(f"{'year':>6} {'trades':>7} {'ret':>8} {'dd':>7} {'wr':>7} {'pf':>6}")
    for r in overall.get("by_year", []):
        print(
            f"{r['year']:6d} {r['n_trades']:7d} {r['year_return']:+8.1%} {r['max_drawdown']:7.1%} "
            f"{r['win_rate']:7.1%} {r['profit_factor']:6.2f}"
        )
    print(
        f"{'ALL':>6} {overall['n_trades']:7d} {overall['total_return']:+8.1%} "
        f"{overall['max_drawdown']:7.1%} {overall['win_rate']:7.1%} {overall['profit_factor']:6.2f}"
    )
    results["confidence_meta_conf_single"] = {"isolated": iso, "continuous": overall}

    # 2) Meta OOF gate only (no confidence column)
    kwargs2 = dict(meta_gate=True, conf_gate=False, max_open=1)
    traded2, overall2 = run_old(meta, **kwargs2)
    iso2 = isolated_years(meta, **kwargs2)
    print_table("OLD-ish: meta_oof | meta>=0.45 | NO conf | max_open=1 (isolated)", iso2, overall2)
    results["meta_oof_gate_single"] = {"isolated": iso2, "continuous": overall2}

    # 3) Confidence panel, meta+conf, max_open=3 (later experiment on old panel)
    kwargs3 = dict(meta_gate=True, conf_gate=True, max_open=3)
    traded3, overall3 = run_old(conf, **kwargs3)
    iso3 = isolated_years(conf, **kwargs3)
    print_table("OLD panel + multi max_open=3 | meta+conf (isolated)", iso3, overall3)
    results["confidence_meta_conf_maxopen3"] = {"isolated": iso3, "continuous": overall3}

    out = _ROOT / "artifacts/pipeline_backtest"
    out.mkdir(parents=True, exist_ok=True)
    (out / "old_meta_pipeline_yearly.json").write_text(
        json.dumps(results, indent=2, default=float), encoding="utf-8"
    )
    if not traded.empty:
        traded.to_parquet(out / "old_meta_pipeline_trades.parquet", index=False)
    print("\nwrote", out / "old_meta_pipeline_yearly.json")


if __name__ == "__main__":
    main()
