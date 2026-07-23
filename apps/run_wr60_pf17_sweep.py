"""Hunt WR~60% + PF~1.7 on walk-forward 2015–2026 (trade count free).

Caches yearly panels per top_pct so filters don't retrain.
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

from apps.run_primary_loosen_backtest import run_portfolio
from apps.run_yearly_walkforward_backtest import _load_side, build_year_panel
from production.paper.barrier_resim import load_h1
from research.portfolio_heat import STARTING_EQUITY


def main() -> None:
    print("Loading…")
    long_df = _load_side("long")
    short_df = _load_side("short")
    h1 = load_h1(str(_ROOT / "artifacts/raw/XAUUSD/H1/data.parquet"))
    years = list(range(2015, 2027))
    top_list = (0.01, 0.02, 0.03, 0.05, 0.08)

    # cache panels
    cache: dict[float, dict[int, pd.DataFrame]] = {}
    for top_pct in top_list:
        print(f"building panels top{top_pct:.0%}…")
        cache[top_pct] = {}
        for y in years:
            panel = build_year_panel(
                long_df, short_df, h1, year=y, lookback=7, top_pct=top_pct
            )
            if not panel.empty:
                panel = panel.copy()
                panel["hour"] = pd.to_datetime(panel["timestamp"], utc=True).dt.hour
            cache[top_pct][y] = panel

    filters = []
    for min_prob in (0.0, 0.50, 0.52, 0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70):
        for lo, hi in ((0, 23), (8, 16), (9, 15), (10, 14), (7, 17)):
            for mo in (1, 2):
                filters.append((min_prob, lo, hi, mo))

    rows = []
    total = len(top_list) * len(filters)
    n = 0
    for top_pct in top_list:
        for min_prob, lo, hi, mo in filters:
            n += 1
            parts = []
            iso = []
            for y in years:
                panel = cache[top_pct].get(y, pd.DataFrame())
                if panel.empty:
                    iso.append({"year": y, "n_trades": 0, "wr": float("nan"), "pf": float("nan")})
                    continue
                p = panel[(panel["hour"] >= lo) & (panel["hour"] <= hi)]
                if min_prob > 0:
                    p = p[p["y_prob"] >= float(min_prob)]
                if p.empty:
                    iso.append({"year": y, "n_trades": 0, "wr": float("nan"), "pf": float("nan")})
                    continue
                _, m = run_portfolio(p, max_open=mo, starting_equity=STARTING_EQUITY)
                iso.append(
                    {
                        "year": y,
                        "n_trades": int(m["n_trades"]),
                        "wr": float(m.get("win_rate", float("nan"))),
                        "pf": float(m.get("profit_factor", float("nan"))),
                        "ret": float(m.get("total_return", float("nan"))),
                        "dd": float(m.get("max_drawdown", float("nan"))),
                    }
                )
                parts.append(p)
            if not parts:
                continue
            all_p = pd.concat(parts, ignore_index=True).sort_values("timestamp")
            _, cont = run_portfolio(all_p, max_open=mo, starting_equity=STARTING_EQUITY)
            wrs = [r["wr"] for r in iso if r["n_trades"] >= 3 and np.isfinite(r["wr"])]
            pfs = [r["pf"] for r in iso if r["n_trades"] >= 3 and np.isfinite(r["pf"])]
            row = {
                "top_pct": top_pct,
                "hour": f"{lo}-{hi}",
                "min_prob": min_prob,
                "max_open": mo,
                "n_trades": cont["n_trades"],
                "cont_wr": float(cont.get("win_rate", float("nan"))),
                "cont_pf": float(cont.get("profit_factor", float("nan"))),
                "cont_ret": float(cont["total_return"]),
                "cont_dd": float(cont["max_drawdown"]),
                "avg_iso_wr": float(np.mean(wrs)) if wrs else float("nan"),
                "avg_iso_pf": float(np.mean(pfs)) if pfs else float("nan"),
                "years_traded": int(sum(1 for r in iso if r["n_trades"] > 0)),
                "years_ge3": int(sum(1 for r in iso if r["n_trades"] >= 3)),
                "years_wr_ge_60": int(sum(1 for r in iso if r["n_trades"] >= 3 and r["wr"] >= 0.60)),
                "by_year": iso,
            }
            rows.append(row)
            if (
                row["avg_iso_wr"] >= 0.58
                and row["cont_pf"] >= 1.5
                and row["n_trades"] >= 30
                and row["years_ge3"] >= 8
            ) or n % 40 == 0:
                mark = " <<CAND" if row["avg_iso_wr"] >= 0.58 and row["cont_pf"] >= 1.5 else ""
                print(
                    f"[{n}/{total}] top{top_pct:.0%} p>={min_prob} h{lo}-{hi} mo={mo} "
                    f"n={row['n_trades']:4d} avgWR={row['avg_iso_wr']:.1%} "
                    f"contPF={row['cont_pf']:.2f} contWR={row['cont_wr']:.1%} "
                    f"ret={row['cont_ret']:+.1%}{mark}"
                )

    df = pd.DataFrame([{k: v for k, v in r.items() if k != "by_year"} for r in rows])
    cols = [
        "top_pct",
        "hour",
        "min_prob",
        "max_open",
        "n_trades",
        "avg_iso_wr",
        "cont_wr",
        "avg_iso_pf",
        "cont_pf",
        "cont_ret",
        "cont_dd",
        "years_ge3",
        "years_wr_ge_60",
    ]

    hit = df[
        (df["avg_iso_wr"] >= 0.58)
        & (df["cont_pf"] >= 1.55)
        & (df["n_trades"] >= 30)
        & (df["years_ge3"] >= 8)
    ].sort_values(["avg_iso_wr", "cont_pf"], ascending=False)

    print("\n=== HIT avgWR>=58% & contPF>=1.55 & 8+ years ===")
    print(hit[cols].head(25).to_string(index=False) if not hit.empty else "(none)")

    print("\n=== top by avg isolated WR ===")
    print(
        df[(df.n_trades >= 30) & (df.years_ge3 >= 8)]
        .sort_values("avg_iso_wr", ascending=False)[cols]
        .head(20)
        .to_string(index=False)
    )

    print("\n=== top by continuous PF ===")
    print(
        df[(df.n_trades >= 30) & (df.years_ge3 >= 8)]
        .sort_values("cont_pf", ascending=False)[cols]
        .head(20)
        .to_string(index=False)
    )

    # also: best joint distance to (0.60 WR, 1.7 PF)
    ok = df[(df.n_trades >= 30) & (df.years_ge3 >= 8)].copy()
    if not ok.empty:
        ok["dist"] = np.sqrt((ok["avg_iso_wr"] - 0.60) ** 2 + ((ok["cont_pf"] - 1.7) / 1.7) ** 2)
        print("\n=== closest to (WR60%, PF1.7) ===")
        print(ok.sort_values("dist")[cols].head(15).to_string(index=False))

    out = _ROOT / "artifacts/pipeline_backtest/wr60_pf17_wf_sweep.json"
    out.write_text(json.dumps(rows, indent=2, default=float), encoding="utf-8")
    print("\nwrote", out)


if __name__ == "__main__":
    main()
