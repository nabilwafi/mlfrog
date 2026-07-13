"""
Standard backtest report generator — same section layout for every experiment.

Usage:
    from src.backtest_report_generator import generate_standard_report, normalize_trades

    generate_standard_report(trades_df, equity_df, "data/reports/standard_report_....md",
                             title="LONG v3 @0.51 full")

CLI (re-run baselines):
    python -m src.backtest_report_generator
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MIN_N_BREAKDOWN = 30
CONTRACT_SIZE = 100.0
STARTING_EQUITY_DEFAULT = 10_000.0

# UTC session buckets (FX gold-friendly coarse split)
_SESSION_BINS = [
    (0, 7, "Asia"),
    (7, 12, "London"),
    (12, 16, "Overlap"),
    (16, 21, "NY"),
    (21, 24, "Off"),
]


def _session_of(ts: pd.Timestamp) -> str:
    h = int(pd.Timestamp(ts).hour)
    for lo, hi, name in _SESSION_BINS:
        if lo <= h < hi:
            return name
    return "Off"


def normalize_trades(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Adapt engine trade logs to the canonical schema used by the report."""
    df = trades_df.copy()
    rename = {}
    if "outcome" not in df.columns and "reason" in df.columns:
        rename["reason"] = "outcome"
    if rename:
        df = df.rename(columns=rename)

    required = {"entry_time", "exit_time", "outcome", "pnl_usd", "bars_held"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"trades_df missing columns: {sorted(missing)}")

    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)

    if "direction" not in df.columns:
        df["direction"] = "long"

    if "pnl_R" not in df.columns:
        # R = pnl / dollars risked to SL (lots * |entry_ref - sl| * contract)
        if {"entry_ref_close", "sl_level", "lots"}.issubset(df.columns):
            risk = (df["entry_ref_close"] - df["sl_level"]).abs() * CONTRACT_SIZE * df["lots"]
            df["pnl_R"] = np.where(risk > 0, df["pnl_usd"] / risk, np.nan)
        else:
            df["pnl_R"] = np.nan

    if "session" not in df.columns:
        df["session"] = df["entry_time"].map(_session_of)

    return df


def _fmt(x: float, nd: int = 2) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "n/a"
    return f"{x:.{nd}f}"


def _is_win(o: str) -> bool:
    return str(o).lower() in {"tp", "win"}


def _is_loss(o: str) -> bool:
    return str(o).lower() in {"sl", "loss"}


def _resolved_mask(df: pd.DataFrame) -> pd.Series:
    return df["outcome"].map(lambda o: _is_win(o) or _is_loss(o))


def _winrate(df: pd.DataFrame) -> tuple[float, int, int, int]:
    """WR = wins/(wins+losses), timeouts excluded. Returns wr, n_win, n_loss, n_to."""
    n_w = int(df["outcome"].map(_is_win).sum())
    n_l = int(df["outcome"].map(_is_loss).sum())
    n_to = len(df) - n_w - n_l
    wr = n_w / (n_w + n_l) if (n_w + n_l) else float("nan")
    return wr, n_w, n_l, n_to


def _max_consecutive(flags: list[bool]) -> int:
    best = cur = 0
    for f in flags:
        if f:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def _drawdown_episodes(eq: pd.DataFrame) -> list[dict[str, Any]]:
    """Episodes where drawdown_pct > 0; each has max_dd_pct and duration_bars/days."""
    if eq.empty or "drawdown_pct" not in eq.columns:
        return []
    dd = eq["drawdown_pct"].to_numpy(float)
    ts = pd.to_datetime(eq["timestamp"], utc=True)
    episodes: list[dict[str, Any]] = []
    i = 0
    n = len(dd)
    while i < n:
        if dd[i] <= 0:
            i += 1
            continue
        j = i
        peak = dd[i]
        while j < n and dd[j] > 0:
            peak = max(peak, dd[j])
            j += 1
        dur_bars = j - i
        dur_days = (ts.iloc[j - 1] - ts.iloc[i]).total_seconds() / 86400.0 if j > i else 0.0
        episodes.append(
            {
                "max_dd_pct": float(peak),
                "duration_bars": int(dur_bars),
                "duration_days": float(dur_days),
                "start": ts.iloc[i],
                "end": ts.iloc[j - 1],
            }
        )
        i = j
    return episodes


def compute_overall(
    trades: pd.DataFrame,
    equity: pd.DataFrame,
    *,
    starting_equity: float = STARTING_EQUITY_DEFAULT,
) -> dict[str, Any]:
    eq = equity.copy()
    eq["timestamp"] = pd.to_datetime(eq["timestamp"], utc=True)
    end_eq = float(eq["equity"].iloc[-1]) if len(eq) else starting_equity
    total_ret = end_eq / starting_equity - 1.0

    if len(eq) >= 2:
        t0, t1 = eq["timestamp"].iloc[0], eq["timestamp"].iloc[-1]
        years = max((t1 - t0).total_seconds() / (365.25 * 86400), 1e-9)
        cagr = (end_eq / starting_equity) ** (1.0 / years) - 1.0
    else:
        years = 0.0
        cagr = 0.0

    max_dd = float(eq["drawdown_pct"].max()) if len(eq) else 0.0

    sharpe = sortino = float("nan")
    if len(eq) >= 3 and years > 0:
        rets = eq["equity"].pct_change().dropna().to_numpy()
        epy = len(rets) / years
        if rets.std(ddof=1) > 0:
            sharpe = float(rets.mean() / rets.std(ddof=1) * np.sqrt(epy))
        down = rets[rets < 0]
        if len(down) >= 2 and down.std(ddof=1) > 0:
            sortino = float(rets.mean() / down.std(ddof=1) * np.sqrt(epy))
        elif len(down) == 0:
            sortino = float("inf")

    calmar = (cagr * 100.0) / abs(max_dd) if max_dd > 0 else float("nan")

    wr, n_w, n_l, n_to = _winrate(trades)
    wins = trades[trades["outcome"].map(_is_win)]
    losses = trades[trades["outcome"].map(_is_loss)]
    avg_win = float(wins["pnl_usd"].mean()) if len(wins) else float("nan")
    avg_loss = float(losses["pnl_usd"].mean()) if len(losses) else float("nan")  # negative
    avg_win_r = float(wins["pnl_R"].mean()) if len(wins) else float("nan")
    avg_loss_r = float(losses["pnl_R"].mean()) if len(losses) else float("nan")
    loss_rate = n_l / (n_w + n_l) if (n_w + n_l) else float("nan")
    # Expectancy: wr*avg_win - lossrate*|avg_loss|  (== wr*avg_win + lossrate*avg_loss since avg_loss<0)
    if n_w + n_l:
        exp_usd = wr * avg_win + loss_rate * avg_loss
        exp_r = wr * avg_win_r + loss_rate * avg_loss_r
    else:
        exp_usd = exp_r = float("nan")

    pnls = trades["pnl_usd"].to_numpy(float) if len(trades) else np.array([])
    gp = float(pnls[pnls > 0].sum()) if len(pnls) else 0.0
    gl = float(-pnls[pnls < 0].sum()) if len(pnls) else 0.0
    pf = gp / gl if gl > 0 else float("inf")

    # consecutive on resolved sequence only (timeout breaks streak)
    resolved = trades.loc[_resolved_mask(trades)].sort_values("entry_time")
    win_flags = [_is_win(o) for o in resolved["outcome"]]
    loss_flags = [_is_loss(o) for o in resolved["outcome"]]

    return {
        "starting_equity": starting_equity,
        "ending_equity": end_eq,
        "total_return_pct": total_ret * 100.0,
        "cagr_pct": cagr * 100.0,
        "years": years,
        "max_drawdown_pct": max_dd,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "profit_factor": pf,
        "expectancy_usd": exp_usd,
        "expectancy_R": exp_r,
        "winrate": wr,
        "n_trades": len(trades),
        "n_tp": n_w,
        "n_sl": n_l,
        "n_timeout": n_to,
        "avg_win_usd": avg_win,
        "avg_loss_usd": avg_loss,
        "avg_win_R": avg_win_r,
        "avg_loss_R": avg_loss_r,
        "largest_win_usd": float(wins["pnl_usd"].max()) if len(wins) else float("nan"),
        "largest_loss_usd": float(losses["pnl_usd"].min()) if len(losses) else float("nan"),
        "largest_win_R": float(wins["pnl_R"].max()) if len(wins) else float("nan"),
        "largest_loss_R": float(losses["pnl_R"].min()) if len(losses) else float("nan"),
        "avg_bars_held": float(trades["bars_held"].mean()) if len(trades) else float("nan"),
        "max_consec_wins": _max_consecutive(win_flags),
        "max_consec_losses": _max_consecutive(loss_flags),
    }


def compute_drawdown_detail(equity: pd.DataFrame, overall: dict[str, Any]) -> dict[str, Any]:
    eq = equity.copy()
    eq["timestamp"] = pd.to_datetime(eq["timestamp"], utc=True)
    eps = _drawdown_episodes(eq)
    max_dd = overall["max_drawdown_pct"]
    if eps:
        dd_vals = [e["max_dd_pct"] for e in eps]
        avg_dd = float(np.mean(dd_vals))
        med_dd = float(np.median(dd_vals))
        longest = max(eps, key=lambda e: e["duration_bars"])
        longest_bars = longest["duration_bars"]
        longest_days = longest["duration_days"]
    else:
        avg_dd = med_dd = 0.0
        longest_bars = 0
        longest_days = 0.0

    # Recovery Factor = Total Return % / Max DD %
    rf = (
        overall["total_return_pct"] / abs(max_dd)
        if max_dd and abs(max_dd) > 0
        else float("nan")
    )
    # Ulcer Index
    if len(eq):
        ulcer = float(np.sqrt(np.mean(np.square(eq["drawdown_pct"].to_numpy(float)))))
    else:
        ulcer = float("nan")

    return {
        "max_dd_pct": max_dd,
        "avg_dd_pct": avg_dd,
        "median_dd_pct": med_dd,
        "n_dd_episodes": len(eps),
        "longest_dd_bars": longest_bars,
        "longest_dd_days": longest_days,
        "recovery_factor": rf,
        "ulcer_index": ulcer,
    }


def _wr_row(label: str, sub: pd.DataFrame) -> str:
    n = len(sub)
    if n == 0:
        return f"| {label} | 0 | — | — | n=0 |"
    wr, n_w, n_l, n_to = _winrate(sub)
    if n < MIN_N_BREAKDOWN:
        flag = f"**n terlalu kecil ({n}<{MIN_N_BREAKDOWN}), tidak reliable**"
        wr_s = f"{wr:.3f}*" if wr == wr else "n/a*"
    else:
        flag = "ok"
        wr_s = f"{wr:.3f}" if wr == wr else "n/a"
    return f"| {label} | {n} | {n_w}/{n_l}/{n_to} | {wr_s} | {flag} |"


def _ascii_hist(values: np.ndarray, bins: int = 21, width: int = 40) -> str:
    vals = values[np.isfinite(values)]
    if len(vals) == 0:
        return "_(no finite pnl_R)_"
    lo, hi = float(np.percentile(vals, 1)), float(np.percentile(vals, 99))
    if lo == hi:
        lo, hi = float(vals.min()) - 0.5, float(vals.max()) + 0.5
    counts, edges = np.histogram(vals, bins=bins, range=(lo, hi))
    peak = int(counts.max()) or 1
    lines = ["```", f"pnl_R histogram (clipped p1–p99: [{lo:.2f}, {hi:.2f}])", ""]
    for c, left, right in zip(counts, edges[:-1], edges[1:]):
        bar = "█" * int(round(width * c / peak))
        lines.append(f"{left:+6.2f}..{right:+6.2f} | {bar} ({c})")
    lines.append("```")
    return "\n".join(lines)


def _skewness(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return float("nan")
    m = x.mean()
    s = x.std(ddof=1)
    if s == 0:
        return 0.0
    return float(np.mean(((x - m) / s) ** 3))


def _plot_diagnostics(
    equity: pd.DataFrame,
    out_png: Path,
    title: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    eq = equity.copy()
    eq["timestamp"] = pd.to_datetime(eq["timestamp"], utc=True)
    eq = eq.sort_values("timestamp")

    # Daily equity for rolling Sharpe
    daily = eq.set_index("timestamp")["equity"].resample("1D").last().dropna()
    daily_ret = daily.pct_change().dropna()
    roll = daily_ret.rolling(90, min_periods=30)
    roll_sharpe = roll.mean() / roll.std() * np.sqrt(252)

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=False)
    axes[0].plot(eq["timestamp"], eq["equity"], color="#1f77b4", lw=1.4)
    axes[0].set_title(f"{title} — equity")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylabel("USD")

    axes[1].fill_between(eq["timestamp"], eq["drawdown_pct"], 0, color="#d62728", alpha=0.35)
    axes[1].plot(eq["timestamp"], eq["drawdown_pct"], color="#d62728", lw=1.0)
    axes[1].set_title("Drawdown %")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(roll_sharpe.index, roll_sharpe.values, color="#2ca02c", lw=1.2)
    axes[2].axhline(0, color="gray", lw=0.8)
    axes[2].set_title("Rolling Sharpe (90D daily, ann. √252)")
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylabel("Sharpe")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def generate_standard_report(
    trades_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    output_path: str | Path,
    *,
    title: str = "Backtest",
    regime_tier_col: str | None = "regime_tier",
    session_col: str | None = "session",
    starting_equity: float = STARTING_EQUITY_DEFAULT,
    plot_path: str | Path | None = None,
    include_diagnostics: bool = True,
) -> str:
    """
    Write a fixed-structure markdown report. Returns the markdown string.

    Canonical trades columns (after normalize): entry_time, exit_time, direction,
    outcome, pnl_usd, pnl_R, bars_held; optional regime_tier, session.
    """
    trades = normalize_trades(trades_df)
    equity = equity_df.copy()
    equity["timestamp"] = pd.to_datetime(equity["timestamp"], utc=True)

    overall = compute_overall(trades, equity, starting_equity=starting_equity)
    dd = compute_drawdown_detail(equity, overall)

    lines: list[str] = [
        f"# Standard Backtest Report — {title}",
        "",
        f"_Generated by `src/backtest_report_generator.py`. Win-rate = TP/(TP+SL); timeouts excluded._",
        "",
        "=====================================",
        "OVERALL PERFORMANCE",
        "=====================================",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| CAGR % | {_fmt(overall['cagr_pct'])} |",
        f"| Total Return % | {_fmt(overall['total_return_pct'])} |",
        f"| Max Drawdown % | {_fmt(overall['max_drawdown_pct'])} |",
        f"| Sharpe (ann., rf=0) | {_fmt(overall['sharpe'], 4)} |",
        f"| Sortino (ann., rf=0) | {_fmt(overall['sortino'], 4)} |",
        f"| Calmar (CAGR/|MaxDD|) | {_fmt(overall['calmar'], 4)} |",
        f"| Profit Factor | {_fmt(overall['profit_factor'], 4)} |",
        f"| Expectancy USD | {_fmt(overall['expectancy_usd'])} |",
        f"| Expectancy R | {_fmt(overall['expectancy_R'], 4)} |",
        f"| Win Rate (TP/SL) | {_fmt(overall['winrate'], 4)} |",
        f"| Avg Win / Avg Loss USD | {_fmt(overall['avg_win_usd'])} / {_fmt(overall['avg_loss_usd'])} |",
        f"| Avg Win / Avg Loss R | {_fmt(overall['avg_win_R'], 3)} / {_fmt(overall['avg_loss_R'], 3)} |",
        f"| Largest Win / Loss USD | {_fmt(overall['largest_win_usd'])} / {_fmt(overall['largest_loss_usd'])} |",
        f"| Largest Win / Loss R | {_fmt(overall['largest_win_R'], 3)} / {_fmt(overall['largest_loss_R'], 3)} |",
        f"| Avg Holding Bars | {_fmt(overall['avg_bars_held'], 3)} |",
        f"| Max Consecutive Wins / Losses | {overall['max_consec_wins']} / {overall['max_consec_losses']} |",
        f"| N Trades (TP/SL/TO) | {overall['n_trades']} ({overall['n_tp']}/{overall['n_sl']}/{overall['n_timeout']}) |",
        f"| Starting / Ending Equity | {_fmt(overall['starting_equity'])} / {_fmt(overall['ending_equity'])} |",
        "",
        "=====================================",
        "DRAWDOWN DETAIL",
        "=====================================",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Max DD % | {_fmt(dd['max_dd_pct'])} |",
        f"| Avg DD % (per episode) | {_fmt(dd['avg_dd_pct'])} |",
        f"| Median DD % (per episode) | {_fmt(dd['median_dd_pct'])} |",
        f"| N DD episodes | {dd['n_dd_episodes']} |",
        f"| Longest DD duration | {dd['longest_dd_bars']} equity marks "
        f"(~{_fmt(dd['longest_dd_days'], 1)} calendar days) |",
        f"| Recovery Factor (TotalRet%/MaxDD%) | {_fmt(dd['recovery_factor'], 4)} |",
        f"| Ulcer Index | {_fmt(dd['ulcer_index'], 4)} |",
        "",
        "=====================================",
        "WIN RATE BREAKDOWN",
        "=====================================",
        "",
        f"_Rule: categories with n < {MIN_N_BREAKDOWN} are flagged unreliable._",
        "",
        "### Overall WR",
        "",
        "| slice | n | TP/SL/TO | winrate | flag |",
        "|---|---:|---|---:|---|",
        _wr_row("all", trades),
        "",
    ]

    # Direction — only if both long & short present with trades
    dirs = trades["direction"].astype(str).str.lower().unique().tolist()
    has_long = any(d.startswith("long") or d == "buy" for d in dirs)
    has_short = any(d.startswith("short") or d == "sell" for d in dirs)
    lines += ["### WR by direction", ""]
    if has_long and has_short:
        lines += [
            "| slice | n | TP/SL/TO | winrate | flag |",
            "|---|---:|---|---:|---|",
        ]
        for d in sorted(dirs):
            lines.append(_wr_row(d, trades[trades["direction"].astype(str).str.lower() == d]))
        lines.append("")
    else:
        lines += [
            "N/A — SHORT parked; book is LONG-only in this run.",
            "",
        ]

    # Session
    scol = session_col if session_col and session_col in trades.columns else None
    lines += ["### WR by session (UTC)", ""]
    if scol:
        lines += [
            "| slice | n | TP/SL/TO | winrate | flag |",
            "|---|---:|---|---:|---|",
        ]
        for s in ["Asia", "London", "Overlap", "NY", "Off"]:
            sub = trades[trades[scol] == s]
            if len(sub) == 0:
                continue
            lines.append(_wr_row(s, sub))
        lines.append("")
    else:
        lines += ["N/A — no session column.", ""]

    # Regime
    rcol = regime_tier_col if regime_tier_col and regime_tier_col in trades.columns else None
    lines += ["### WR by regime tier", ""]
    if rcol:
        lines += [
            "| slice | n | TP/SL/TO | winrate | flag |",
            "|---|---:|---|---:|---|",
        ]
        for tier in ["normal", "elevated", "extreme"]:
            sub = trades[trades[rcol].astype(str) == tier]
            if len(sub) == 0:
                lines.append(f"| {tier} | 0 | — | — | n=0 |")
            else:
                lines.append(_wr_row(tier, sub))
        lines.append("")
    else:
        lines += ["N/A — regime_tier not attached for this run.", ""]

    lines += [
        "### WR by confidence bucket",
        "",
    ]
    if "confidence_tier" in trades.columns and trades["confidence_tier"].notna().any():
        lines += [
            "| slice | n | TP/SL/TO | winrate | flag |",
            "|---|---:|---|---:|---|",
        ]
        for tier in ["low", "medium", "high"]:
            sub = trades[trades["confidence_tier"].astype(str) == tier]
            if len(sub) == 0:
                continue
            lines.append(_wr_row(tier, sub))
        other = trades[~trades["confidence_tier"].astype(str).isin(["low", "medium", "high"])]
        if len(other):
            lines.append(_wr_row("other/n-a", other))
        lines.append("")
    else:
        lines += [
            "TBD — placeholder until Confidence Engine (Layer 4) is built. Not filled.",
            "",
        ]

    lines += [
        "### Return by year",
        "",
        "| year | n_trades | total_pnl_usd | winrate (TP/SL) | flag |",
        "|---:|---:|---:|---:|---|",
    ]
    trades = trades.copy()
    trades["_year"] = trades["entry_time"].dt.year
    small_years = []
    for y, g in trades.groupby("_year"):
        wr, _, _, _ = _winrate(g)
        flag = "ok" if len(g) >= MIN_N_BREAKDOWN else f"n<{MIN_N_BREAKDOWN}, tidak reliable"
        if len(g) < MIN_N_BREAKDOWN:
            small_years.append(int(y))
        lines.append(
            f"| {y} | {len(g)} | {g['pnl_usd'].sum():+.2f} | {_fmt(wr, 3)} | {flag} |"
        )
    lines.append("")
    if small_years:
        lines.append(
            f"_Catatan: tahun {small_years} punya n < {MIN_N_BREAKDOWN} — angka WR/return "
            f"tahun itu tidak reliable secara statistik._"
        )
        lines.append("")

    lines += [
        "### Return by month",
        "",
        "_Skipped — too granular for current n_trades (optional section)._",
        "",
        "=====================================",
        "TRADE DISTRIBUTION",
        "=====================================",
        "",
        _ascii_hist(trades["pnl_R"].to_numpy(float)),
        "",
        f"- Skewness (pnl_R): **{_fmt(_skewness(trades['pnl_R'].to_numpy(float)), 3)}**",
        f"- Mean pnl_R: {_fmt(float(trades['pnl_R'].mean()), 3)}; "
        f"median: {_fmt(float(trades['pnl_R'].median()), 3)}",
        "",
    ]

    # Diagnostics
    lines += [
        "=====================================",
        "DIAGNOSTIC (opsional)",
        "=====================================",
        "",
    ]
    out_path = Path(output_path)
    if include_diagnostics:
        png = Path(plot_path) if plot_path else out_path.with_suffix(".png")
        _plot_diagnostics(equity, png, title)
        lines += [
            f"- Equity / DD / rolling Sharpe plot: `{png.as_posix()}`",
            "- Rolling Sharpe: 90-day window on daily equity returns (√252 ann.) — cheap drift proxy.",
            "",
        ]
    else:
        lines += ["_(diagnostics skipped)_", ""]

    md = "\n".join(lines)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    return md


def _attach_regime_if_possible(trades: pd.DataFrame) -> pd.DataFrame:
    """Best-effort: map entry_time → causal regime_tier from feature ATR history."""
    if "regime_tier" in trades.columns:
        return trades
    feats = Path("data/features/xauusd_h1_h4_d1_features_v3.parquet")
    if not feats.exists():
        return trades
    try:
        from backtest.regime import (
            ROLLING_WINDOW_BARS,
            Z_REDUCE_THRESHOLD,
            attach_regime_to_frame,
        )
    except ImportError:
        return trades

    atr_hist = pd.read_parquet(feats, columns=["Date", "atr_h1"])
    # Minimal frame for attach
    stub = pd.DataFrame(
        {
            "Date": pd.to_datetime(trades["entry_time"], utc=True),
            "atr_h1": trades["atr"].to_numpy(float)
            if "atr" in trades.columns
            else np.nan,
        }
    )
    # Need full bar history for rolling — use atr_hist as the frame itself
    hist = atr_hist.copy()
    hist["Date"] = pd.to_datetime(hist["Date"], utc=True)
    hist = hist.sort_values("Date").reset_index(drop=True)
    tagged = attach_regime_to_frame(
        hist,
        atr_hist,
        window=ROLLING_WINDOW_BARS,
        z_reduce=Z_REDUCE_THRESHOLD,
        z_block=4.0,
    )
    lookup = tagged.set_index("Date")[["regime_tier", "atr_zscore"]]
    out = trades.copy()
    et = pd.to_datetime(out["entry_time"], utc=True)
    joined = lookup.reindex(et)
    out["regime_tier"] = joined["regime_tier"].to_numpy()
    out["atr_zscore"] = joined["atr_zscore"].to_numpy()
    return out


def main() -> None:
    """Re-generate standard reports for v1, v3, v3+guard (full with-cost windows)."""
    base = Path("data")
    jobs = [
        {
            "title": "LONG LGBM v1 @ thr=0.52 (full test, with cost)",
            "trades": base / "backtest/trades_with_cost.parquet",
            "equity": base / "backtest/equity_with_cost.parquet",
            "out": base / "reports/standard_report_long_v1.md",
            "png": base / "reports/standard_report_long_v1.png",
            "attach_regime": True,
        },
        {
            "title": "LONG LGBM v3 @ thr=0.51 (full test, with cost, no guard)",
            "trades": base / "backtest_v3/trades_with_cost_full.parquet",
            "equity": base / "backtest_v3/equity_with_cost_full.parquet",
            "out": base / "reports/standard_report_long_v3.md",
            "png": base / "reports/standard_report_long_v3.png",
            "attach_regime": True,
        },
        {
            "title": "LONG LGBM v3 @ thr=0.51 + regime guard (Z_BLOCK=4.0, full, with cost)",
            "trades": base / "backtest_v3_guarded/trades_with_guard.parquet",
            "equity": base / "backtest_v3_guarded/equity_with_guard.parquet",
            "out": base / "reports/standard_report_long_v3_guarded.md",
            "png": base / "reports/standard_report_long_v3_guarded.png",
            "attach_regime": False,  # already on trades
        },
    ]

    # Ensure src is importable as package
    Path("src/__init__.py").touch()

    for job in jobs:
        print(f"=== {job['title']}")
        trades = pd.read_parquet(job["trades"])
        equity = pd.read_parquet(job["equity"])
        if job["attach_regime"]:
            trades = _attach_regime_if_possible(trades)
        generate_standard_report(
            trades,
            equity,
            job["out"],
            title=job["title"],
            plot_path=job["png"],
        )
        print(f"wrote {job['out']}")


def _self_check() -> None:
    """ponytail: smallest check that expectancy / WR exclude timeout correctly."""
    trades = pd.DataFrame(
        {
            "entry_time": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-03"], utc=True
            ),
            "exit_time": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-03"], utc=True
            ),
            "outcome": ["tp", "sl", "timeout"],
            "pnl_usd": [100.0, -50.0, 10.0],
            "pnl_R": [1.0, -1.0, 0.1],
            "bars_held": [2, 2, 8],
            "direction": ["long"] * 3,
        }
    )
    eq = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-01", "2024-06-01", "2024-12-31"], utc=True
            ),
            "equity": [10000.0, 10050.0, 10060.0],
            "drawdown_pct": [0.0, 1.0, 0.5],
        }
    )
    m = compute_overall(trades, eq)
    assert abs(m["winrate"] - 0.5) < 1e-9, m["winrate"]
    # expectancy on resolved only: 0.5*100 + 0.5*(-50) = 25
    assert abs(m["expectancy_usd"] - 25.0) < 1e-9, m["expectancy_usd"]
    assert m["n_timeout"] == 1


if __name__ == "__main__":
    _self_check()
    main()
