"""Probe filters aiming for ~300 trades/yr and PF ~1.6."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from apps.run_primary_loosen_backtest import COST, build_universe, run_portfolio, wilder_atr
from production.paper.barrier_resim import load_h1
from research.portfolio_heat import STARTING_EQUITY


def with_holds(u: pd.DataFrame) -> pd.DataFrame:
    longs = pd.read_parquet(
        _ROOT / "artifacts/labels/XAUUSD/H1/long/triple_barrier_v1.parquet",
        columns=["timestamp", "holding_bars", "realized_return"],
    )
    shorts = pd.read_parquet(
        _ROOT / "artifacts/labels/XAUUSD/H1/short/triple_barrier_v1.parquet",
        columns=["timestamp", "holding_bars", "realized_return"],
    )
    longs["timestamp"] = pd.to_datetime(longs["timestamp"], utc=True)
    longs["side"] = "long"
    shorts["timestamp"] = pd.to_datetime(shorts["timestamp"], utc=True)
    shorts["side"] = "short"
    lab = pd.concat([longs, shorts])
    m = u.drop(columns=["holding_bars", "net_return", "realized_return"], errors="ignore").merge(
        lab, on=["timestamp", "side"], how="left"
    )
    m["holding_bars"] = m["holding_bars"].astype(float).fillna(4)
    m["net_return"] = m["realized_return"].astype(float) - COST
    return m.dropna(subset=["net_return", "atr_price", "entry_price"])


def summarize(name: str, panel: pd.DataFrame, mo: int) -> None:
    if panel.empty:
        print(f"{name:42} EMPTY")
        return
    _, m = run_portfolio(panel, max_open=mo, starting_equity=STARTING_EQUITY)
    by = m.get("by_year", [])
    full = [r for r in by if r["year"] < 2026]
    avg = sum(r["n_trades"] for r in full) / max(len(full), 1)
    print(
        f"{name:42} mo={mo} trades={m['n_trades']:4d} avg/yr={avg:6.0f} "
        f"pf={m['profit_factor']:.2f} ret={m['total_return']:+.1%} "
        f"dd={m['max_drawdown']:.1%} wr={m['win_rate']:.1%}"
    )


def main() -> None:
    primary = pd.read_parquet(_ROOT / "artifacts/research/probability_quality/predictions.parquet")
    h1 = load_h1(str(_ROOT / "artifacts/raw/XAUUSD/H1/data.parquet"))

    print("--- top_pct + real holding ---")
    for pct in (0.05, 0.06, 0.08, 0.10, 0.12):
        u = with_holds(build_universe(primary, h1, top_pct=pct))
        for mo in (2, 3):
            summarize(f"top{pct:.0%}+hold", u, mo)

    print("\n--- absolute top-N / year ---")
    p = primary.copy()
    p["timestamp"] = pd.to_datetime(p["timestamp"], utc=True)
    p = p.loc[p.groupby("timestamp")["y_prob"].idxmax()].copy()
    h = h1.copy()
    h["timestamp"] = pd.to_datetime(h["timestamp"], utc=True)
    h["atr"] = wilder_atr(h)
    for n in (250, 300, 350, 400):
        parts = []
        for y, g in p.groupby(p["timestamp"].dt.year):
            if y < 2023:
                continue
            parts.append(g.nlargest(n, "y_prob"))
        u = pd.concat(parts)
        u = u.merge(h[["timestamp", "close", "atr"]], on="timestamp", how="left")
        u["entry_price"] = u["close"].astype(float)
        u["atr_price"] = u["atr"].astype(float)
        u["net_return"] = u["realized_return"].astype(float) - COST
        u["holding_bars"] = 4.0
        u = with_holds(u.dropna(subset=["entry_price", "atr_price", "net_return"]))
        for mo in (2, 3):
            summarize(f"topN={n}/yr", u, mo)

    print("\n--- hour filter on top10% ---")
    u = with_holds(build_universe(primary, h1, top_pct=0.10))
    u["hour"] = pd.to_datetime(u["timestamp"], utc=True).dt.hour
    for lo, hi, name in ((7, 17, "EU_US"), (8, 16, "LondonNY"), (12, 20, "NY")):
        f = u[(u["hour"] >= lo) & (u["hour"] <= hi)]
        summarize(f"top10% hour {name}", f, 3)

    print("\n--- side / min_prob on top15% ---")
    u = with_holds(build_universe(primary, h1, top_pct=0.15))
    summarize("top15% long-only", u[u.side == "long"], 3)
    summarize("top15% short-only", u[u.side == "short"], 3)
    for q in (0.50, 0.55, 0.60, 0.65):
        summarize(f"top15% & p>={q}", u[u["y_prob"] >= q], 3)

    # earlier sweet spot without hold fix
    print("\n--- prior sweet-spot rebuild ---")
    for pct, mo in ((0.12, 3), (0.08, 3), (0.10, 3)):
        summarize(f"raw top{pct:.0%}", build_universe(primary, h1, top_pct=pct), mo)


if __name__ == "__main__":
    main()
