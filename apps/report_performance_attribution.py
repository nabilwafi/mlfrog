"""Sprint 30 — Performance Attribution Report (DB-only).

Source of truth: research.wf_* (trades now include regime + MFE/MAE from the WF run).
No retrain / no re-inference here — run apps/run_rolling_walkforward.py first if columns empty.

Example:
  python apps/report_performance_attribution.py
  python apps/report_performance_attribution.py --run-id <id>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import tools.pyc_path_hook  # noqa: F401

import numpy as np
import pandas as pd
import yaml

OUT = _ROOT / "artifacts" / "pipeline_backtest" / "rolling_wf" / "sprint30_attribution"
STARTING = 80.0


def profit_factor(pnl: np.ndarray) -> float:
    gp = float(pnl[pnl > 0].sum())
    gl = float(-pnl[pnl < 0].sum())
    if gl <= 0:
        return float("inf") if gp > 0 else 0.0
    return gp / gl


def _fmt_pf(v: float) -> str:
    return "inf" if not np.isfinite(v) else f"{v:.2f}"


def group_stats(df: pd.DataFrame, key: str | list[str]) -> pd.DataFrame:
    rows = []
    for name, g in df.groupby(key, dropna=False):
        pnl = g["pnl"].to_numpy(dtype=float)
        keys = name if isinstance(name, tuple) else (name,)
        cols = key if isinstance(key, list) else [key]
        row: dict[str, Any] = dict(zip(cols, keys))
        row.update(
            {
                "trades": int(len(g)),
                "win_rate": float(np.mean(pnl > 0)),
                "profit_factor": profit_factor(pnl),
                "net_pnl": float(pnl.sum()),
                "return_vs_80": float(pnl.sum() / STARTING),
                "expectancy_usd": float(pnl.mean()),
                "avg_r": float(g["r_multiple"].mean()),
                "avg_holding_bars": float(g["holding_bars"].mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(key if isinstance(key, str) else key[0]).reset_index(drop=True)


def dist_table(s: pd.Series, bins: list[float], labels: list[str]) -> pd.DataFrame:
    cut = pd.cut(s.dropna(), bins=bins, labels=labels, include_lowest=True)
    d = cut.value_counts().reindex(labels).fillna(0).astype(int)
    return pd.DataFrame({"bucket": labels, "count": d.to_numpy(), "share": d.to_numpy() / max(len(s.dropna()), 1)})


def md_table(df: pd.DataFrame, cols: list[tuple[str, str, str]]) -> list[str]:
    head = "| " + " | ".join(h for _, h, _ in cols) + " |"
    sep = "|" + "|".join("---:" for _ in cols) + "|"
    out = [head, sep]
    for _, r in df.iterrows():
        cells = []
        for c, _h, f in cols:
            v = r[c]
            if f == "pf":
                cells.append(_fmt_pf(float(v)))
            elif f == "s":
                cells.append(str(v))
            elif f.endswith("d"):
                cells.append(format(int(v), f))
            else:
                cells.append(format(float(v), f))
        out.append("| " + " | ".join(cells) + " |")
    return out


def _connect(dsn: str):
    import psycopg2

    return psycopg2.connect(dsn, connect_timeout=10)


def _query(conn, sql: str, params: tuple = ()) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)


def main() -> None:
    p = argparse.ArgumentParser(description="Sprint 30 performance attribution (DB read-only)")
    p.add_argument("--config", type=Path, default=_ROOT / "configs" / "config.yaml")
    p.add_argument("--run-id", default=None)
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    dsn = (cfg.get("paper_trading") or {}).get("postgres_dsn")
    if not dsn:
        raise SystemExit("configs/config.yaml: paper_trading.postgres_dsn required")

    conn = _connect(str(dsn))
    try:
        run_id = args.run_id
        if not run_id:
            runs = _query(conn, "SELECT run_id FROM research.wf_runs ORDER BY created_at DESC LIMIT 1")
            if runs.empty:
                raise SystemExit("research.wf_runs empty — run apps/run_rolling_walkforward.py first")
            run_id = str(runs.iloc[0]["run_id"])
        runs = _query(conn, "SELECT * FROM research.wf_runs WHERE run_id = %s", (run_id,))
        trades = _query(
            conn,
            "SELECT test_year, window_id, trade_seq, timestamp, side, entry_price, lots, y_prob, "
            "net_return, pnl, equity, holding_bars, exit_reason, "
            "regime, trend_state, vol_state, mfe_pct, mae_pct, mfe_r, mae_r, r_multiple "
            "FROM research.wf_trades WHERE run_id = %s ORDER BY timestamp",
            (run_id,),
        )
        backtests = _query(
            conn,
            "SELECT window_id, test_year, scope, n_trades, total_return, final_equity, max_drawdown, "
            "profit_factor, win_rate, avg_holding_bars, starting_equity "
            "FROM research.wf_backtests WHERE run_id = %s ORDER BY test_year, scope",
            (run_id,),
        )
    finally:
        conn.close()

    if trades.empty:
        raise SystemExit(f"no trades for run_id={run_id}")
    if trades["regime"].isna().all() or trades["mfe_r"].isna().all():
        raise SystemExit(
            f"run_id={run_id} missing regime/MFE columns — re-run:\n"
            "  python apps/run_rolling_walkforward.py --apply-schema --debug-files"
        )

    print(f"run_id={run_id} trades={len(trades)}")
    trades["timestamp"] = pd.to_datetime(trades["timestamp"], utc=True)
    for c in ("entry_price", "net_return", "pnl", "equity", "y_prob", "mfe_pct", "mae_pct", "mfe_r", "mae_r", "r_multiple"):
        trades[c] = trades[c].astype(float)
    trades["holding_bars"] = trades["holding_bars"].fillna(0).astype(int)
    trades["regime"] = trades["regime"].fillna("UNKNOWN")
    trades["year"] = trades["timestamp"].dt.year
    trades["month"] = trades["timestamp"].dt.month
    trades["hour"] = trades["timestamp"].dt.hour
    trades["win"] = trades["pnl"] > 0

    pnl = trades["pnl"].to_numpy(dtype=float)
    yearly_db = backtests[backtests["scope"] == "year"].copy()
    combined = backtests[backtests["scope"] == "combined"]
    combined_row = combined.iloc[0] if not combined.empty else None

    ls = group_stats(trades, "side")
    yearly = group_stats(trades, "year").merge(
        yearly_db[["test_year", "total_return", "max_drawdown", "final_equity"]].rename(columns={"test_year": "year"}),
        on="year",
        how="left",
    )
    monthly = group_stats(trades, "month")
    monthly["month_name"] = pd.to_datetime(monthly["month"], format="%m").dt.strftime("%b")
    regime = group_stats(trades, "regime").sort_values("net_pnl", ascending=False).reset_index(drop=True)
    hourly = group_stats(trades, "hour")
    exits = group_stats(trades, "exit_reason")
    exit_extra = []
    for reason, g in trades.groupby("exit_reason"):
        wins = g.loc[g["win"], "pnl"]
        loss = g.loc[~g["win"], "pnl"]
        exit_extra.append(
            {
                "exit_reason": reason,
                "avg_profit_usd": float(wins.mean()) if len(wins) else 0.0,
                "avg_loss_usd": float(loss.mean()) if len(loss) else 0.0,
                "avg_mfe_r": float(g["mfe_r"].mean()),
                "avg_mae_r": float(g["mae_r"].mean()),
            }
        )
    exits = exits.merge(pd.DataFrame(exit_extra), on="exit_reason", how="left")

    mm_rows = []
    for label, g in (("ALL", trades), ("WIN", trades[trades["win"]]), ("LOSS", trades[~trades["win"]])):
        mm_rows.append(
            {
                "outcome": label,
                "trades": int(len(g)),
                "mean_mfe_r": float(g["mfe_r"].mean()),
                "median_mfe_r": float(g["mfe_r"].median()),
                "p95_mfe_r": float(g["mfe_r"].quantile(0.95)),
                "mean_mae_r": float(g["mae_r"].mean()),
                "median_mae_r": float(g["mae_r"].median()),
                "p95_mae_r": float(g["mae_r"].quantile(0.95)),
                "mean_mfe_pct": float(g["mfe_pct"].mean()),
                "mean_mae_pct": float(g["mae_pct"].mean()),
            }
        )
    mfe_mae = pd.DataFrame(mm_rows)

    hold_dist = dist_table(
        trades["holding_bars"].astype(float), [0, 1, 2, 3, 4, 8, 16, 1e9], ["1", "2", "3", "4", "5-8", "9-16", ">16"]
    )
    r_dist = dist_table(
        trades["r_multiple"], [-1e9, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 1e9],
        ["<-1R", "-1..-0.5R", "-0.5..0R", "0..0.5R", "0.5..1R", "1..2R", ">2R"],
    )
    mfe_dist = dist_table(trades["mfe_r"], [-1e9, 0.25, 0.5, 1.0, 2.0, 1e9], ["<0.25R", "0.25-0.5R", "0.5-1R", "1-2R", ">2R"])
    mae_dist = dist_table(trades["mae_r"], [-1e9, 0.25, 0.5, 1.0, 1e9], ["<0.25R", "0.25-0.5R", "0.5-1R", ">1R"])

    OUT.mkdir(parents=True, exist_ok=True)
    yearly.to_csv(OUT / "yearly_summary.csv", index=False)
    monthly.to_csv(OUT / "monthly_summary.csv", index=False)
    regime.to_csv(OUT / "regime_summary.csv", index=False)
    hourly.to_csv(OUT / "hour_summary.csv", index=False)
    exits.to_csv(OUT / "exit_summary.csv", index=False)
    ls.to_csv(OUT / "long_short_summary.csv", index=False)
    mfe_mae.to_csv(OUT / "mfe_mae_summary.csv", index=False)
    trades.to_csv(OUT / "trades_enriched.csv", index=False)

    best_month = monthly.loc[monthly["net_pnl"].idxmax()]
    worst_month = monthly.loc[monthly["net_pnl"].idxmin()]
    best_regime, worst_regime = regime.iloc[0], regime.iloc[-1]
    hour_pool = hourly[hourly["trades"] >= 20]
    if hour_pool.empty:
        hour_pool = hourly
    best_hour = hour_pool.sort_values("net_pnl", ascending=False).iloc[0]
    best_hour_pf = hour_pool.sort_values("profit_factor", ascending=False).iloc[0]
    worst_hour = hour_pool.sort_values("profit_factor").iloc[0]
    long_row = ls[ls["side"] == "long"].iloc[0]
    short_row = ls[ls["side"] == "short"].iloc[0]
    pos_years = int((yearly["net_pnl"] > 0).sum())
    trail_row = exits[exits["exit_reason"] == "TRAIL"].iloc[0] if (exits["exit_reason"] == "TRAIL").any() else None
    sl_row = exits[exits["exit_reason"] == "SL"].iloc[0] if (exits["exit_reason"] == "SL").any() else None
    win_mfe = mfe_mae[mfe_mae["outcome"] == "WIN"].iloc[0]
    loss_mfe = mfe_mae[mfe_mae["outcome"] == "LOSS"].iloc[0]
    dd_worst = yearly.loc[yearly["max_drawdown"].idxmax()] if yearly["max_drawdown"].notna().any() else None
    avg_win = float(trades.loc[trades["win"], "pnl"].mean())
    avg_loss = float(-trades.loc[~trades["win"], "pnl"].mean())
    breakeven_wr = avg_loss / (avg_win + avg_loss)

    md: list[str] = [
        "# Sprint 30 — Performance Attribution Report",
        "",
        f"run_id: `{run_id}`  ",
        "source: `research.wf_trades` (regime / MFE / MAE / R written by rolling WF — no post-hoc derivation)  ",
        f"policy: {runs.iloc[0]['policy'] if not runs.empty else 'n/a'}",
        "",
        "**Data provenance.** Per-test-year trades from `research.wf_trades` (Method A: $80 reset each year). "
        "Yearly Return/MaxDD from `research.wf_backtests`. Regime, MFE, MAE and R-multiple were computed "
        "inside `apps/run_rolling_walkforward.py` during the backtest and stored on each trade row.",
        "",
        "## 1. Overall Performance",
        "",
    ]
    overall = [
        ("Total trades", f"{len(trades)}"),
        ("Win rate", f"{np.mean(pnl > 0):.1%}"),
        ("Profit factor", _fmt_pf(profit_factor(pnl))),
        ("Expectancy", f"${pnl.mean():.2f} / trade ({trades['r_multiple'].mean():+.2f}R)"),
        ("Total net PnL", f"${pnl.sum():.2f}"),
        ("Average holding bars", f"{trades['holding_bars'].mean():.2f}"),
    ]
    if combined_row is not None:
        overall += [
            ("Total return (combined, Method B)", f"{float(combined_row['total_return']):+.1%}"),
            ("Final equity (combined)", f"${float(combined_row['final_equity']):.2f}"),
            ("Max drawdown (combined)", f"{float(combined_row['max_drawdown']):.1%}"),
        ]
    md += ["| Metric | Value |", "|---|---:|"] + [f"| {k} | {v} |" for k, v in overall]
    md += [
        "",
        f"Per-test-year returns average {yearly['total_return'].mean():+.1%}, median "
        f"{yearly['total_return'].median():+.1%}. Average win ${avg_win:.2f} vs average loss ${avg_loss:.2f} "
        f"→ break-even WR **{breakeven_wr:.1%}**; strategy runs at {np.mean(pnl > 0):.1%} "
        f"(+{np.mean(pnl > 0) - breakeven_wr:.1%} cushion).",
        "",
        "## 2. Long vs Short",
        "",
    ]
    md += md_table(
        ls,
        [
            ("side", "Side", "s"),
            ("trades", "Trades", "d"),
            ("win_rate", "Win Rate", ".1%"),
            ("profit_factor", "PF", "pf"),
            ("avg_r", "Avg R", "+.3f"),
            ("net_pnl", "Total PnL ($)", "+.2f"),
            ("expectancy_usd", "Expectancy ($)", "+.3f"),
        ],
    )
    md += [
        "",
        f"**Read:** long PF {_fmt_pf(long_row['profit_factor'])} / {long_row['win_rate']:.1%} WR vs short PF "
        f"{_fmt_pf(short_row['profit_factor'])} / {short_row['win_rate']:.1%} WR. "
        + (
            "Both sides carry edge."
            if min(long_row["profit_factor"], short_row["profit_factor"]) > 1.1
            else "Edge is concentrated on one side."
        ),
        "",
        "## 3. Performance by Year",
        "",
    ]
    md += md_table(
        yearly,
        [
            ("year", "Year", "d"),
            ("trades", "Trades", "d"),
            ("win_rate", "Win Rate", ".1%"),
            ("profit_factor", "PF", "pf"),
            ("total_return", "Return", "+.1%"),
            ("max_drawdown", "MaxDD", ".1%"),
        ],
    )
    md += [
        "",
        f"**Read:** {pos_years}/{len(yearly)} years positive; PF "
        f"{yearly['profit_factor'].min():.2f}–{yearly['profit_factor'].max():.2f}. "
        + (
            f"Worst year {int(dd_worst['year'])}: PF {dd_worst['profit_factor']:.2f}, MaxDD "
            f"{dd_worst['max_drawdown']:.1%} on {int(dd_worst['trades'])} trades (ruin-stop sample)."
            if dd_worst is not None
            else ""
        ),
        "",
        "## 4. Performance by Month",
        "",
    ]
    md += md_table(
        monthly,
        [
            ("month_name", "Month", "s"),
            ("trades", "Trades", "d"),
            ("win_rate", "Win Rate", ".1%"),
            ("profit_factor", "PF", "pf"),
            ("net_pnl", "Net PnL ($)", "+.2f"),
            ("return_vs_80", "Return vs $80", "+.1%"),
        ],
    )
    md += [
        "",
        f"**Best:** {best_month['month_name']} (PF {_fmt_pf(best_month['profit_factor'])}, "
        f"{best_month['net_pnl']:+.2f}). **Worst:** {worst_month['month_name']} "
        f"(PF {_fmt_pf(worst_month['profit_factor'])}, {worst_month['net_pnl']:+.2f}).",
        "",
        "## 5. Performance by Market Regime",
        "",
        "Regime stored on each trade at entry (H4 trend × vol tercile).",
        "",
    ]
    md += md_table(
        regime,
        [
            ("regime", "Regime", "s"),
            ("trades", "Trades", "d"),
            ("win_rate", "Win Rate", ".1%"),
            ("profit_factor", "PF", "pf"),
            ("net_pnl", "Net PnL ($)", "+.2f"),
            ("return_vs_80", "Return vs $80", "+.1%"),
            ("expectancy_usd", "Expectancy ($)", "+.3f"),
        ],
    )
    md += [
        "",
        f"**Best:** {best_regime['regime']} (PF {_fmt_pf(best_regime['profit_factor'])}). "
        f"**Worst:** {worst_regime['regime']} (PF {_fmt_pf(worst_regime['profit_factor'])}).",
        "",
        "## 6. Performance by Hour (UTC)",
        "",
    ]
    md += md_table(
        hourly,
        [
            ("hour", "Hour", "02d"),
            ("trades", "Trades", "d"),
            ("win_rate", "Win Rate", ".1%"),
            ("profit_factor", "PF", "pf"),
            ("net_pnl", "Net PnL ($)", "+.2f"),
            ("return_vs_80", "Return vs $80", "+.1%"),
        ],
    )
    md += [
        "",
        f"**Best PnL hour (≥20):** {int(best_hour['hour']):02d}:00 UTC. "
        f"**Best PF hour (≥20):** {int(best_hour_pf['hour']):02d}:00 UTC. "
        f"**Weakest liquid hour:** {int(worst_hour['hour']):02d}:00 UTC (PF {_fmt_pf(worst_hour['profit_factor'])}).",
        "",
        "## 7. Exit Analysis",
        "",
    ]
    md += md_table(
        exits,
        [
            ("exit_reason", "Exit", "s"),
            ("trades", "Trades", "d"),
            ("win_rate", "Win Rate", ".1%"),
            ("avg_profit_usd", "Avg Profit ($)", "+.3f"),
            ("avg_loss_usd", "Avg Loss ($)", "+.3f"),
            ("avg_mfe_r", "Avg MFE (R)", ".2f"),
            ("avg_mae_r", "Avg MAE (R)", ".2f"),
        ],
    )
    if trail_row is not None and sl_row is not None:
        md += [
            "",
            f"**Read:** TRAIL={int(trail_row['trades'])} (all winners, avg {trail_row['avg_profit_usd']:+.3f}); "
            f"SL={int(sl_row['trades'])} (all losers, avg {sl_row['avg_loss_usd']:+.3f}). "
            f"Break-even WR {breakeven_wr:.1%} — payoff is hit-rate dependent.",
        ]
    md += ["", "## 8. MFE / MAE Analysis", "", "R = 1.5×ATR at entry.", ""]
    md += md_table(
        mfe_mae,
        [
            ("outcome", "Set", "s"),
            ("trades", "Trades", "d"),
            ("mean_mfe_r", "Mean MFE", ".2f"),
            ("median_mfe_r", "Median MFE", ".2f"),
            ("p95_mfe_r", "P95 MFE", ".2f"),
            ("mean_mae_r", "Mean MAE", ".2f"),
            ("median_mae_r", "Median MAE", ".2f"),
            ("p95_mae_r", "P95 MAE", ".2f"),
        ],
    )
    md += [
        "",
        f"**Read:** winners {win_mfe['mean_mfe_r']:.2f}R MFE / {win_mfe['mean_mae_r']:.2f}R MAE; "
        f"losers {loss_mfe['mean_mfe_r']:.2f}R MFE / {loss_mfe['mean_mae_r']:.2f}R MAE. "
        + (
            "Losers barely move in favor → entry quality fails, not the exit."
            if loss_mfe["mean_mfe_r"] <= 0.25
            else "Losers show favorable excursion before reversing."
        ),
        "",
        "## 9. Distributions",
        "",
        "### Holding bars",
        "",
    ]
    md += md_table(hold_dist, [("bucket", "Bars", "s"), ("count", "Trades", "d"), ("share", "Share", ".1%")])
    md += ["", "### Trade return (R)", ""]
    md += md_table(r_dist, [("bucket", "R bucket", "s"), ("count", "Trades", "d"), ("share", "Share", ".1%")])
    md += ["", "### MFE (R)", ""]
    md += md_table(mfe_dist, [("bucket", "MFE", "s"), ("count", "Trades", "d"), ("share", "Share", ".1%")])
    md += ["", "### MAE (R)", ""]
    md += md_table(mae_dist, [("bucket", "MAE", "s"), ("count", "Trades", "d"), ("share", "Share", ".1%")])
    md += [
        "",
        "## 10. Final Conclusion",
        "",
        f"**Best conditions.** {best_regime['regime']} (PF {_fmt_pf(best_regime['profit_factor'])}); "
        f"liquid hours around {int(best_hour['hour']):02d}:00 UTC; month {best_month['month_name']}.",
        "",
        f"**Worst conditions.** {worst_regime['regime']} (PF {_fmt_pf(worst_regime['profit_factor'])}); "
        f"thin Asian hours; month {worst_month['month_name']}.",
        "",
        f"**Edge location.** Both sides — long PF {_fmt_pf(long_row['profit_factor'])}, short PF "
        f"{_fmt_pf(short_row['profit_factor'])} ({long_row['net_pnl'] / max(pnl.sum(), 1e-9):.0%} of net PnL from long).",
        "",
        f"**Year consistency.** {pos_years}/{len(yearly)} positive; PF "
        f"{yearly['profit_factor'].min():.2f}–{yearly['profit_factor'].max():.2f}.",
        "",
        "**Three biggest weaknesses.**",
        "",
        f"1. Ruin risk: worst MaxDD {yearly['max_drawdown'].max():.1%} on $80 — sizing vs 1.5×ATR stop is binding.",
        f"2. Payoff asymmetry: avg loss ${avg_loss:.2f} > avg win ${avg_win:.2f}; only "
        f"{np.mean(pnl > 0) - breakeven_wr:.1%} WR cushion above break-even.",
        f"3. Thin sample: {len(trades)} trades / {len(yearly)} years "
        f"({trades.groupby('year').size().min()}–{trades.groupby('year').size().max()} per year).",
        "",
        "**Three biggest strengths.**",
        "",
        f"1. Clean separation: winners {win_mfe['mean_mfe_r']:.2f}R MFE vs losers {loss_mfe['mean_mfe_r']:.2f}R.",
        f"2. Both directions profitable ({int(long_row['trades'])} long / {int(short_row['trades'])} short).",
        f"3. Fast turnover (mean {trades['holding_bars'].mean():.2f} bars) + {pos_years}/{len(yearly)} positive years.",
        "",
        "### Files",
        "",
        "- `yearly_summary.csv`, `monthly_summary.csv`, `regime_summary.csv`, `hour_summary.csv`",
        "- `exit_summary.csv`, `long_short_summary.csv`, `mfe_mae_summary.csv`",
        "- `trades_enriched.csv`",
        "",
    ]
    (OUT / "performance_attribution.md").write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {OUT}")
    for f in sorted(OUT.glob("*")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
