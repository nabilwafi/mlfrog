"""
Paper trading monitor — summary vs backtest expectations.

Usage:
    python -m src.paper_trading_monitor
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.engine import STARTING_EQUITY
from src.backtest_report_generator import generate_standard_report, normalize_trades

PAPER_DIR = Path("data/paper_trading")
# Sealed-2026 high-tier calibrated WR ~0.46; full v3 WR ~0.49
EXPECTED_WR_LO = 0.40
EXPECTED_WR_HI = 0.52


def _equity_from_trades(trades: pd.DataFrame, starting: float = STARTING_EQUITY) -> pd.DataFrame:
    if trades.empty:
        now = pd.Timestamp.now(tz="UTC")
        return pd.DataFrame({"timestamp": [now], "equity": [starting], "drawdown_pct": [0.0]})
    t = trades.sort_values("exit_time").copy()
    t["exit_time"] = pd.to_datetime(t["exit_time"], utc=True)
    eq = starting
    peak = starting
    rows = [{"timestamp": t["exit_time"].iloc[0] - pd.Timedelta(hours=1), "equity": starting, "drawdown_pct": 0.0}]
    for _, r in t.iterrows():
        eq = eq + float(r["pnl_usd"])
        peak = max(peak, eq)
        dd = (peak - eq) / peak * 100.0 if peak > 0 else 0.0
        rows.append({"timestamp": r["exit_time"], "equity": eq, "drawdown_pct": dd})
    return pd.DataFrame(rows)


def _wr(trades: pd.DataFrame) -> tuple[float, int, int, int]:
    if trades.empty:
        return float("nan"), 0, 0, 0
    o = trades["outcome"].astype(str).str.lower()
    n_w = int((o == "tp").sum())
    n_l = int((o == "sl").sum())
    n_to = int((o == "timeout").sum())
    wr = n_w / (n_w + n_l) if (n_w + n_l) else float("nan")
    return wr, n_w, n_l, n_to


def build_monitor_report(paper_dir: Path = PAPER_DIR) -> str:
    decisions_p = paper_dir / "decisions_log.parquet"
    trades_p = paper_dir / "paper_trades.parquet"
    blocked_p = paper_dir / "blocked_hypothetical.parquet"
    out_md = paper_dir / "paper_monitor_report.md"
    out_std = paper_dir / "standard_report_paper.md"

    lines = [
        "# Paper Trading Monitor",
        "",
        f"_Generated from `{paper_dir.as_posix()}`._",
        "",
        "## Status",
        "",
    ]

    if not decisions_p.exists():
        lines += ["No `decisions_log.parquet` yet — run `python -m src.paper_trading_engine --once`.", ""]
        out_md.write_text("\n".join(lines), encoding="utf-8")
        return "\n".join(lines)

    dec = pd.read_parquet(decisions_p)
    lines += [
        f"- decisions logged: **{len(dec)}**",
        f"- date range: {dec['timestamp'].min()} → {dec['timestamp'].max()}"
        if "timestamp" in dec.columns and len(dec)
        else "- date range: n/a",
        "",
        "### Decision mix",
        "",
    ]
    if "decision" in dec.columns:
        vc = dec["decision"].value_counts()
        lines += ["| decision | n |", "|---|---:|"]
        for k, v in vc.items():
            lines.append(f"| {k} | {int(v)} |")
        lines.append("")

    trades = pd.read_parquet(trades_p) if trades_p.exists() else pd.DataFrame()
    wr, n_w, n_l, n_to = _wr(trades) if len(trades) else (float("nan"), 0, 0, 0)
    lines += [
        "## Paper trades",
        "",
        f"- n_resolved = **{len(trades)}** (TP/SL/TO = {n_w}/{n_l}/{n_to})",
        f"- winrate TP/SL = **{wr:.3f}**" if wr == wr else "- winrate: n/a",
        f"- expected band from backtest ≈ [{EXPECTED_WR_LO:.2f}, {EXPECTED_WR_HI:.2f}]",
        "",
    ]
    if wr == wr and len(trades) >= 30:
        if wr < EXPECTED_WR_LO:
            lines.append(
                f"**ALERT**: live WR {wr:.3f} below backtest band lower bound "
                f"{EXPECTED_WR_LO:.2f} — investigate feature/live drift."
            )
        elif wr > EXPECTED_WR_HI:
            lines.append(
                f"Note: live WR {wr:.3f} above backtest band (small-sample or regime luck)."
            )
        else:
            lines.append("Live WR within expected backtest band.")
        lines.append("")
    elif len(trades) < 30:
        lines += [
            f"_n={len(trades)} < 30 — WR not statistically reliable yet (same rule as standard report)._",
            "",
        ]

    if blocked_p.exists():
        blk = pd.read_parquet(blocked_p)
        lines += [
            "## Blocked hypotheticals (live Gate 4.5 extension)",
            "",
            f"- n_resolved_blocked = **{len(blk)}**",
        ]
        if len(blk) and "hypothetical_outcome" in blk.columns:
            vc = blk["hypothetical_outcome"].value_counts()
            pnl = float(blk["hypothetical_pnl_usd"].sum()) if "hypothetical_pnl_usd" in blk.columns else float("nan")
            lines.append(f"- outcomes: {vc.to_dict()}")
            lines.append(f"- net hypothetical PnL if executed: **{pnl:+.2f} USD**")
            n_sl = int((blk["hypothetical_outcome"] == "sl").sum())
            lines.append(
                f"- would-be loss rate: {100 * n_sl / len(blk):.1f}% "
                f"(guard protective if high + net PnL negative)"
            )
        lines.append("")
    else:
        lines += ["## Blocked hypotheticals", "", "_(none yet)_", ""]

    # Standard report if enough trades
    if len(trades) >= 1:
        t = trades.copy()
        if "outcome" in t.columns and "reason" not in t.columns:
            t["reason"] = t["outcome"]
        t["entry_time"] = pd.to_datetime(t.get("entry_time", t.get("exit_time")), utc=True)
        t["exit_time"] = pd.to_datetime(t["exit_time"], utc=True)
        if "bars_held" not in t.columns:
            t["bars_held"] = np.nan
        if "direction" not in t.columns:
            t["direction"] = "long"
        eq = _equity_from_trades(t)
        generate_standard_report(
            t,
            eq,
            out_std,
            title="Paper trading LONG v3 + guard (live-forward)",
            include_diagnostics=True,
            plot_path=paper_dir / "standard_report_paper.png",
        )
        lines += [
            "## Standard report",
            "",
            f"See `{out_std.as_posix()}` (same schema as backtest baselines).",
            "",
        ]

    state_p = paper_dir / "state.json"
    if state_p.exists():
        import json

        st = json.loads(state_p.read_text(encoding="utf-8"))
        lines += [
            "## Engine state",
            "",
            f"- equity: {st.get('equity')}",
            f"- last_processed_h1_open: {st.get('last_processed_h1_open')}",
            f"- open_position: {'YES' if st.get('open_position') else 'no'}",
            f"- open_hypotheticals: {len(st.get('open_hypotheticals') or [])}",
            "",
        ]

    lines += [
        "## No-order guarantee",
        "",
        "- Paper engine installs an `mt5.order_send` tripwire that raises if called.",
        "- Feed handler only uses `copy_rates_*` via `MT5Connector`.",
        "",
    ]
    text = "\n".join(lines)
    out_md.write_text(text, encoding="utf-8")
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-dir", type=Path, default=PAPER_DIR)
    args = parser.parse_args()
    print(build_monitor_report(args.paper_dir))


if __name__ == "__main__":
    main()
