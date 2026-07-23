"""Point 2: exit-engine hunt for WR~60% + PF~1.7 on WF entries 2015–2026.

Fixed walk-forward Primary entries (top 5% unique bars); only exits change.
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

from apps.run_primary_loosen_backtest import COST, run_portfolio
from apps.run_yearly_walkforward_backtest import _load_side, build_year_panel
from production.paper.barrier_resim import attach_resim_returns, load_h1
from research.portfolio_heat import STARTING_EQUITY
from research.position_mgmt.services.paths import prepare_market, entry_indices
from research.position_mgmt.services.simulator import ManageConfig, simulate_panel


def build_entries(
    long_df: pd.DataFrame,
    short_df: pd.DataFrame,
    h1: pd.DataFrame,
    *,
    top_pct: float,
    years: list[int],
) -> pd.DataFrame:
    parts = []
    for y in years:
        p = build_year_panel(long_df, short_df, h1, year=y, lookback=7, top_pct=top_pct)
        if not p.empty:
            parts.append(p)
    if not parts:
        return pd.DataFrame()
    e = pd.concat(parts, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    e["atr_entry"] = e["atr_price"].astype(float)
    return e


def metrics_for(panel: pd.DataFrame, *, max_open: int = 1) -> dict:
    keep = [
        "timestamp",
        "side",
        "y_prob",
        "entry_price",
        "atr_price",
        "net_return",
        "holding_bars",
    ]
    p = panel[keep].dropna(subset=["net_return", "entry_price", "atr_price"]).copy()
    if p.empty:
        return {"n_trades": 0}
    _, cont = run_portfolio(p, max_open=max_open, starting_equity=STARTING_EQUITY)
    iso_wr, iso_pf = [], []
    for _, g in p.groupby(pd.to_datetime(p["timestamp"], utc=True).dt.year):
        _, m = run_portfolio(g, max_open=max_open, starting_equity=STARTING_EQUITY)
        if m["n_trades"] >= 3:
            iso_wr.append(m["win_rate"])
            iso_pf.append(m["profit_factor"])
    return {
        "n_trades": cont["n_trades"],
        "cont_wr": float(cont["win_rate"]),
        "cont_pf": float(cont["profit_factor"]),
        "cont_ret": float(cont["total_return"]),
        "cont_dd": float(cont["max_drawdown"]),
        "avg_iso_wr": float(np.mean(iso_wr)) if iso_wr else float("nan"),
        "avg_iso_pf": float(np.mean(iso_pf)) if iso_pf else float("nan"),
        "years_ge3": int(len(iso_wr)),
        "years_wr_ge_60": int(sum(1 for w in iso_wr if w >= 0.60)),
    }


def main() -> None:
    years = list(range(2015, 2027))
    top_pct = 0.05  # selective entries; trade count free
    print("Loading + building WF entries top5%…")
    long_df = _load_side("long")
    short_df = _load_side("short")
    h1 = load_h1(str(_ROOT / "artifacts/raw/XAUUSD/H1/data.parquet"))
    entries = build_entries(long_df, short_df, h1, top_pct=top_pct, years=years)
    print(f"entries={len(entries)} years={entries['timestamp'].dt.year.nunique()}")

    rows: list[dict] = []

    def report(name: str, panel: pd.DataFrame) -> None:
        m = metrics_for(panel, max_open=1)
        m["name"] = name
        rows.append(m)
        mark = ""
        if m.get("avg_iso_wr", 0) >= 0.58 and m.get("cont_pf", 0) >= 1.5 and m.get("n_trades", 0) >= 30:
            mark = " <<HIT"
        print(
            f"{name:28} n={m.get('n_trades', 0):4d} "
            f"avgWR={m.get('avg_iso_wr', float('nan')):.1%} "
            f"contWR={m.get('cont_wr', float('nan')):.1%} "
            f"contPF={m.get('cont_pf', float('nan')):.2f} "
            f"ret={m.get('cont_ret', float('nan')):+.1%} "
            f"dd={m.get('cont_dd', float('nan')):.1%}{mark}"
        )

    # --- A: barrier geometry sweep ---
    print("\n=== Barrier TP/SL/H sweep ===")
    report("baseline_label_exit", entries)  # original label exits
    geometries = [
        (1.0, 1.0, 8),
        (1.0, 1.5, 8),
        (1.5, 1.0, 8),
        (1.5, 1.5, 8),
        (2.0, 1.0, 8),
        (2.0, 1.5, 8),
        (2.0, 2.0, 8),
        (2.5, 1.5, 8),
        (3.0, 1.5, 8),
        (1.0, 2.0, 8),  # tight TP wide SL -> higher WR?
        (0.75, 2.0, 8),
        (0.5, 1.5, 8),
        (1.5, 1.0, 16),
        (2.0, 1.5, 16),
        (2.0, 1.5, 24),
        (3.0, 1.5, 24),
        (3.0, 1.5, 48),
        (1.0, 1.0, 4),
        (0.75, 1.0, 4),
        (1.5, 0.75, 8),  # tight SL
        (2.0, 0.75, 8),
        (2.0, 1.5, 4),
    ]
    for tp, sl, h in geometries:
        sim = attach_resim_returns(entries, h1, tp_mult=tp, sl_mult=sl, horizon=h)
        sim["net_return"] = sim["net_return"].astype(float) - COST
        sim["atr_price"] = sim["atr_entry"]
        report(f"barrier_tp{tp}_sl{sl}_h{h}", sim)

    # --- B: position management policies ---
    print("\n=== BE / trail / time / partial ===")
    mkt = prepare_market(h1)
    eis = entry_indices(entries, mkt["ts"])
    policies = [
        ManageConfig(name="pm_baseline", family="baseline"),
        ManageConfig(name="be_0.3R", family="be", be_trigger_r=0.3),
        ManageConfig(name="be_0.5R", family="be", be_trigger_r=0.5),
        ManageConfig(name="be_0.7R", family="be", be_trigger_r=0.7),
        ManageConfig(name="be_1.0R", family="be", be_trigger_r=1.0),
        ManageConfig(name="trail_0.5atr", family="trail", trail_atr=0.5),
        ManageConfig(name="trail_0.8atr", family="trail", trail_atr=0.8),
        ManageConfig(name="trail_1.0atr", family="trail", trail_atr=1.0),
        ManageConfig(name="trail_1.5atr", family="trail", trail_atr=1.5),
        ManageConfig(name="time_4b", family="time", time_exit_bars=4, max_bars=8),
        ManageConfig(name="time_6b", family="time", time_exit_bars=6, max_bars=8),
        ManageConfig(name="time_12b", family="time", time_exit_bars=12, max_bars=24),
        ManageConfig(
            name="partial50_0.5R", family="partial", partial_frac=0.5, partial_at_r=0.5
        ),
        ManageConfig(
            name="partial50_1.0R", family="partial", partial_frac=0.5, partial_at_r=1.0
        ),
        # combos that often raise WR: tight TP barrier + BE
        ManageConfig(
            name="be0.5_h16", family="be", be_trigger_r=0.5, horizon=16, max_bars=16
        ),
        ManageConfig(
            name="trail0.8_h16", family="trail", trail_atr=0.8, horizon=16, max_bars=16
        ),
    ]
    for cfg in policies:
        sim = simulate_panel(entries, h1, cfg, mkt=mkt, eis=eis)
        # simulate_panel already net of COST; map atr
        sim["atr_price"] = sim["atr_entry"]
        sim["holding_bars"] = sim["holding_bars_sim"]
        report(cfg.name, sim)

    # --- C: high-WR barrier + session (exit+entry hour, still exit focus) ---
    print("\n=== Best barriers + session 08-16 ===")
    for tp, sl, h in ((0.75, 2.0, 8), (1.0, 2.0, 8), (0.5, 1.5, 8), (1.0, 1.0, 4)):
        sim = attach_resim_returns(entries, h1, tp_mult=tp, sl_mult=sl, horizon=h)
        sim["net_return"] = sim["net_return"].astype(float) - COST
        sim["atr_price"] = sim["atr_entry"]
        sim["hour"] = pd.to_datetime(sim["timestamp"], utc=True).dt.hour
        sim = sim[(sim["hour"] >= 8) & (sim["hour"] <= 16)]
        report(f"sess+tp{tp}_sl{sl}_h{h}", sim)

    df = pd.DataFrame(rows)
    hit = df[
        (df["avg_iso_wr"] >= 0.58)
        & (df["cont_pf"] >= 1.55)
        & (df["n_trades"] >= 30)
        & (df["years_ge3"] >= 8)
    ].sort_values(["avg_iso_wr", "cont_pf"], ascending=False)

    print("\n=== HIT WR>=58% & PF>=1.55 ===")
    print(hit.to_string(index=False) if not hit.empty else "(none)")

    print("\n=== top by avg WR ===")
    print(df.sort_values("avg_iso_wr", ascending=False).head(15).to_string(index=False))

    print("\n=== top by cont PF ===")
    print(df.sort_values("cont_pf", ascending=False).head(15).to_string(index=False))

    # distance to target
    ok = df[df["n_trades"] >= 30].copy()
    ok["dist"] = np.sqrt((ok["avg_iso_wr"] - 0.60) ** 2 + ((ok["cont_pf"] - 1.7) / 1.7) ** 2)
    print("\n=== closest to WR60/PF1.7 ===")
    print(ok.sort_values("dist").head(12).to_string(index=False))

    out = _ROOT / "artifacts/pipeline_backtest/exit_engine_wr60_pf17.json"
    out.write_text(json.dumps(rows, indent=2, default=float), encoding="utf-8")
    print("\nwrote", out)


if __name__ == "__main__":
    main()
