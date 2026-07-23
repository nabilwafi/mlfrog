"""Find hour-window + top_pct hitting ~300 trades/yr and PF>=1.6."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from apps.run_primary_loosen_backtest import build_universe, run_portfolio
from apps.run_pf16_probe import with_holds
from production.paper.barrier_resim import load_h1
from research.portfolio_heat import STARTING_EQUITY

WINDOWS = [
    (7, 17, "EU_US_7_17"),
    (8, 16, "LN_NY_8_16"),
    (8, 17, "LN_NY_8_17"),
    (7, 16, "EU_US_7_16"),
    (6, 18, "wide_6_18"),
    (9, 16, "core_9_16"),
]


def main() -> None:
    primary = pd.read_parquet(_ROOT / "artifacts/research/probability_quality/predictions.parquet")
    h1 = load_h1(str(_ROOT / "artifacts/raw/XAUUSD/H1/data.parquet"))
    rows = []
    print(f"{'cfg':40} {'avg':>5} {'pf':>5} {'ret':>8} {'dd':>6} {'wr':>6}")
    for pct in (0.08, 0.10, 0.12, 0.15, 0.18, 0.20):
        base = with_holds(build_universe(primary, h1, top_pct=pct))
        base["hour"] = pd.to_datetime(base["timestamp"], utc=True).dt.hour
        for lo, hi, wname in WINDOWS:
            u = base[(base["hour"] >= lo) & (base["hour"] <= hi)]
            for mo in (2, 3, 4):
                if u.empty:
                    continue
                _, m = run_portfolio(u, max_open=mo, starting_equity=STARTING_EQUITY)
                full = [r for r in m.get("by_year", []) if r["year"] < 2026]
                avg = sum(r["n_trades"] for r in full) / max(len(full), 1)
                row = {
                    "top_pct": pct,
                    "window": wname,
                    "hour_lo": lo,
                    "hour_hi": hi,
                    "max_open": mo,
                    "n_trades": m["n_trades"],
                    "avg_trades_yr": avg,
                    "pf": m["profit_factor"],
                    "total_return": m["total_return"],
                    "max_dd": m["max_drawdown"],
                    "win_rate": m["win_rate"],
                    "by_year": m.get("by_year", []),
                }
                rows.append(row)
                tag = f"top{pct:.0%} {wname} mo={mo}"
                flag = ""
                if 270 <= avg <= 340 and m["profit_factor"] >= 1.55:
                    flag = " << HIT"
                elif 250 <= avg <= 360 and m["profit_factor"] >= 1.50:
                    flag = " << near"
                if flag or (m["profit_factor"] >= 1.55 and 200 <= avg <= 400):
                    print(
                        f"{tag:40} {avg:5.0f} {m['profit_factor']:5.2f} "
                        f"{m['total_return']:+8.1%} {m['max_drawdown']:6.1%} "
                        f"{m['win_rate']:6.1%}{flag}"
                    )

    df = pd.DataFrame([{k: v for k, v in r.items() if k != "by_year"} for r in rows])
    hits = df[(df.avg_trades_yr.between(270, 340)) & (df.pf >= 1.55)].sort_values("pf", ascending=False)
    near = df[(df.avg_trades_yr.between(250, 360)) & (df.pf >= 1.50)].sort_values("pf", ascending=False)
    print("\n=== HITS 270-340 trades/yr & PF>=1.55 ===")
    print(hits.to_string(index=False) if not hits.empty else "(none)")
    print("\n=== NEAR 250-360 & PF>=1.50 (top 20) ===")
    print(near.head(20).to_string(index=False) if not near.empty else "(none)")

    out = _ROOT / "artifacts/pipeline_backtest"
    (out / "pf16_hour_sweep.json").write_text(json.dumps(rows, indent=2, default=float), encoding="utf-8")
    print("\nwrote", out / "pf16_hour_sweep.json")


if __name__ == "__main__":
    main()
