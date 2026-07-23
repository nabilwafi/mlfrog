"""Sweep for ~300 trades/year with PF ~1.6–1.7.

A) OOF probability_quality panel 2023–2026 (same as prior loosen sweep)
B) Walk-forward 2015–2026 with tighter top_pct / min_prob / max_open
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from apps.run_primary_loosen_backtest import build_universe, run_portfolio
from apps.run_yearly_walkforward_backtest import _load_side, build_year_panel
from production.paper.barrier_resim import load_h1
from research.portfolio_heat import STARTING_EQUITY


def _filter_min_prob(panel: pd.DataFrame, min_prob: float) -> pd.DataFrame:
    if min_prob <= 0:
        return panel
    return panel.loc[panel["y_prob"] >= float(min_prob)].copy()


def _avg_trades_per_full_year(by_year: list[dict]) -> float:
    full = [r for r in by_year if int(r["year"]) < 2026]
    if not full:
        return float("nan")
    return float(sum(r["n_trades"] for r in full) / len(full))


def sweep_oof() -> list[dict]:
    primary = pd.read_parquet(_ROOT / "artifacts/research/probability_quality/predictions.parquet")
    h1 = load_h1(str(_ROOT / "artifacts/raw/XAUUSD/H1/data.parquet"))
    rows = []
    for top_pct in (0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12):
        base = build_universe(primary, h1, top_pct=top_pct)
        for min_prob in (0.0, 0.50, 0.52, 0.55, 0.58):
            u = _filter_min_prob(base, min_prob)
            if u.empty:
                continue
            for mo in (1, 2, 3):
                traded, m = run_portfolio(u, max_open=mo, starting_equity=STARTING_EQUITY)
                avg = _avg_trades_per_full_year(m.get("by_year", []))
                rows.append(
                    {
                        "panel": "oof_2023_2026",
                        "top_pct": top_pct,
                        "min_prob": min_prob,
                        "max_open": mo,
                        "n_trades": m["n_trades"],
                        "avg_trades_yr": avg,
                        "total_return": m["total_return"],
                        "max_dd": m["max_drawdown"],
                        "pf": m.get("profit_factor", float("nan")),
                        "win_rate": m.get("win_rate", float("nan")),
                    }
                )
    return rows


def sweep_walkforward() -> list[dict]:
    """Fewer configs — WF train is slow-ish but 7y*configs still ok."""
    long_df = _load_side("long")
    short_df = _load_side("short")
    h1 = load_h1(str(_ROOT / "artifacts/raw/XAUUSD/H1/data.parquet"))
    years = list(range(2015, 2027))
    rows = []
    configs = [
        (0.05, 0.0, 2),
        (0.05, 0.0, 3),
        (0.06, 0.0, 2),
        (0.06, 0.0, 3),
        (0.08, 0.0, 2),
        (0.08, 0.0, 3),
        (0.08, 0.52, 2),
        (0.08, 0.55, 2),
        (0.10, 0.0, 2),
        (0.10, 0.55, 2),
    ]
    for top_pct, min_prob, mo in configs:
        parts = []
        iso_pf = []
        for y in years:
            panel = build_year_panel(
                long_df, short_df, h1, year=y, lookback=7, top_pct=top_pct
            )
            panel = _filter_min_prob(panel, min_prob)
            if panel.empty:
                continue
            traded, m = run_portfolio(panel, max_open=mo, starting_equity=STARTING_EQUITY)
            iso_pf.append(m.get("profit_factor", float("nan")))
            parts.append(panel)
        if not parts:
            continue
        all_panel = pd.concat(parts, ignore_index=True).sort_values("timestamp")
        _, cont = run_portfolio(all_panel, max_open=mo, starting_equity=STARTING_EQUITY)
        avg = _avg_trades_per_full_year(cont.get("by_year", []))
        rows.append(
            {
                "panel": "wf_2015_2026",
                "top_pct": top_pct,
                "min_prob": min_prob,
                "max_open": mo,
                "n_trades": cont["n_trades"],
                "avg_trades_yr": avg,
                "total_return": cont["total_return"],
                "max_dd": cont["max_drawdown"],
                "pf": cont.get("profit_factor", float("nan")),
                "win_rate": cont.get("win_rate", float("nan")),
                "median_iso_pf": float(pd.Series(iso_pf).median()),
            }
        )
        print(
            f"WF top{top_pct:.0%} minp={min_prob} mo={mo}: "
            f"avg/yr={avg:.0f} pf={cont.get('profit_factor', float('nan')):.2f} "
            f"ret={cont['total_return']:+.1%} dd={cont['max_drawdown']:.1%}"
        )
    return rows


def main() -> None:
    print("=== OOF panel 2023–2026 (target ~250–350 trades/yr, PF>=1.55) ===")
    oof = sweep_oof()
    df = pd.DataFrame(oof)
    hit = df[(df["avg_trades_yr"] >= 250) & (df["avg_trades_yr"] <= 380) & (df["pf"] >= 1.55)]
    near = df[(df["avg_trades_yr"] >= 200) & (df["avg_trades_yr"] <= 400)].sort_values(
        "pf", ascending=False
    )
    print("\nHits (250–380 trades/yr & PF>=1.55):")
    if hit.empty:
        print("  (none)")
    else:
        print(hit.sort_values("pf", ascending=False).to_string(index=False))
    print("\nTop 15 by PF in 200–400 trades/yr band:")
    print(near.head(15).to_string(index=False))

    print("\n=== Walk-forward 2015–2026 ===")
    wf = sweep_walkforward()
    wdf = pd.DataFrame(wf)
    if not wdf.empty:
        print("\nAll WF configs:")
        print(wdf.sort_values("pf", ascending=False).to_string(index=False))

    out = _ROOT / "artifacts/pipeline_backtest"
    out.mkdir(parents=True, exist_ok=True)
    payload = {"oof": oof, "walkforward": wf}
    (out / "pf16_300trades_sweep.json").write_text(
        json.dumps(payload, indent=2, default=float), encoding="utf-8"
    )
    print("\nwrote", out / "pf16_300trades_sweep.json")


if __name__ == "__main__":
    main()
