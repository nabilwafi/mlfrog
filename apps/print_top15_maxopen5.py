"""Print top15% max_open5 including 2026 sealed holdout."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from apps.run_primary_loosen_backtest import build_universe, load_h1, run_portfolio


def main() -> None:
    primary = pd.read_parquet(_ROOT / "artifacts/research/probability_quality/predictions.parquet")
    h1 = load_h1(str(_ROOT / "artifacts/raw/XAUUSD/H1/data.parquet"))
    u = build_universe(primary, h1, top_pct=0.15)
    traded, m = run_portfolio(u, max_open=5)

    print("=== top 15% + max_open 5 ===")
    print(
        f"ret_all={m['total_return']:+.1%}  maxDD={m['max_drawdown']:.1%}  "
        f"pf={m['profit_factor']:.2f}  trades={m['n_trades']}"
    )
    print()
    print(f"{'year':<8} {'trades':>7} {'year_ret':>10} {'maxDD':>8} {'winrate':>8}  note")
    for r in m["by_year"]:
        note = ""
        if r["year"] == 2026:
            note = "SEALED (2026-01-02 .. 2026-07-10)"
        print(
            f"{r['year']:<8} {r['n_trades']:>7} {r['year_return']:>+9.1%} "
            f"{r['max_drawdown']:>7.1%} {r['win_rate']:>7.1%}  {note}"
        )

    t = traded.copy()
    t["timestamp"] = pd.to_datetime(t["timestamp"], utc=True)
    s = t.loc[t["timestamp"].dt.year == 2026]
    y26 = next(r for r in m["by_year"] if r["year"] == 2026)
    months = (s["timestamp"].max() - s["timestamp"].min()).days / 30.44
    ann = (1.0 + y26["year_return"]) ** (12.0 / max(months, 1e-6)) - 1.0
    print()
    print("=== 2026 SEALED detail ===")
    print(f"trades={len(s)}")
    print(f"period={s['timestamp'].min()} -> {s['timestamp'].max()}")
    print(f"year_return={y26['year_return']:+.1%}")
    print(f"maxDD={y26['max_drawdown']:.1%}")
    print(f"win_rate={y26['win_rate']:.1%}")
    print(f"window_months~{months:.1f}")
    print(f"approx_annualized={ann:+.1%}  (from sealed window only; not full calendar year)")


if __name__ == "__main__":
    main()
