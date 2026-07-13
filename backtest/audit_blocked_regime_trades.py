"""
Audit hypothetically executed regime-guard blocked LONG signals (Gate 4.5).

Pure diagnostic — does not change guard params.

Usage:
    python -m backtest.audit_blocked_regime_trades
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from backtest.engine import (
    ASSUMED_SLIPPAGE_POINTS,
    CONTRACT_SIZE,
    POINT,
    BacktestConfig,
    _cost_price,
    _round_lots,
    _spread_points,
    load_backtest_frame,
    run_backtest,
)
from backtest.regime import (
    ROLLING_WINDOW_BARS,
    Z_REDUCE_THRESHOLD,
    attach_regime_to_frame,
)

THR = 0.51
SEALED_START = pd.Timestamp("2026-01-02 04:00:00+00:00")
SEALED_END = pd.Timestamp("2026-07-08 11:00:00+00:00")
SL_MULT = 1.5
TP_MULT = 2.0
HORIZON = 8


def _resolve_barrier(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    i: int,
    atr: float,
) -> tuple[str, int, float]:
    """Return (outcome, bars_held, exit_raw_price). SL-first."""
    entry = close[i]
    sl = entry - SL_MULT * atr
    tp = entry + TP_MULT * atr
    end = min(len(close) - 1, i + HORIZON)
    for j in range(i + 1, end + 1):
        if low[j] <= sl:
            return "sl", j - i, sl
        if high[j] >= tp:
            return "tp", j - i, tp
    return "timeout", min(HORIZON, end - i), close[end]


def collect_blocked(
    df: pd.DataFrame,
    *,
    z_block: float,
    equity_curve: pd.DataFrame,
) -> pd.DataFrame:
    """Replay flat/open logic with guard; record blocked extreme signals + hypothetical fill."""
    cfg = BacktestConfig(apply_costs=True, selected_thr=THR, enable_regime_guard=True)
    n = len(df)
    high = df["High"].to_numpy(float)
    low = df["Low"].to_numpy(float)
    close = df["Close"].to_numpy(float)
    atr = df["atr_h1"].to_numpy(float)
    prob = df["y_prob"].to_numpy(float)
    dates = df["Date"].to_numpy()
    regime = df["regime_tier"].astype(str).to_numpy()
    z = df["atr_zscore"].to_numpy(float)

    # Equity as-of for sizing: no-guard curve
    eq = equity_curve.sort_values("timestamp").copy()
    eq["timestamp"] = pd.to_datetime(eq["timestamp"], utc=True)

    def equity_at(ts) -> float:
        ts = pd.Timestamp(ts)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        sub = eq[eq["timestamp"] <= ts]
        if sub.empty:
            return float(cfg.starting_equity)
        return float(sub["equity"].iloc[-1])

    rows = []
    pos_open = False
    entry_i = -1
    sl_level = tp_level = 0.0
    i = 0
    while i < n:
        if pos_open:
            hit_sl = low[i] <= sl_level
            hit_tp = high[i] >= tp_level
            bars_held = i - entry_i
            done = hit_sl or hit_tp or bars_held >= HORIZON
            if done:
                pos_open = False
                # allow same-bar re-entry check below
            else:
                i += 1
                continue

        if (not pos_open) and (prob[i] >= THR):
            if i + 1 >= n:
                i += 1
                continue
            tier = str(regime[i])
            if tier == "extreme":
                # Would be blocked — simulate hypothetical
                a = atr[i]
                if np.isfinite(a) and a > 0 and np.isfinite(z[i]):
                    outcome, bars, exit_raw = _resolve_barrier(high, low, close, i, a)
                    eq_now = equity_at(dates[i])
                    sp_e, _ = _spread_points(df.iloc[i], cfg)
                    sp_x = sp_e  # approx; use exit bar spread if available
                    j_exit = i + bars
                    if j_exit < n:
                        sp_x, _ = _spread_points(df.iloc[j_exit], cfg)
                    entry_cost = _cost_price(sp_e, cfg)
                    exit_cost = _cost_price(sp_x, cfg)
                    entry_ref = close[i]
                    entry_fill = entry_ref + entry_cost
                    exit_fill = exit_raw - exit_cost
                    sl_dist = SL_MULT * a
                    risk_amt = eq_now * cfg.risk_per_trade_pct
                    lots = _round_lots(risk_amt / (sl_dist * CONTRACT_SIZE))
                    if lots <= 0:
                        # still report outcome; pnl with fractional conceptual R
                        lots_concept = risk_amt / (sl_dist * CONTRACT_SIZE)
                        pnl = (exit_fill - entry_fill) * CONTRACT_SIZE * lots_concept
                        lots_used = lots_concept
                    else:
                        pnl = (exit_fill - entry_fill) * CONTRACT_SIZE * lots
                        lots_used = lots
                    pnl_r = pnl / risk_amt if risk_amt > 0 else float("nan")
                    rows.append(
                        {
                            "timestamp": pd.Timestamp(dates[i]),
                            "atr_zscore": float(z[i]),
                            "regime_tier": tier,
                            "z_block_setting": z_block,
                            "hypothetical_outcome": outcome,
                            "hypothetical_pnl_usd": float(pnl),
                            "hypothetical_pnl_R": float(pnl_r),
                            "bars_to_resolution": int(bars),
                            "equity_at_signal": eq_now,
                            "lots": float(lots_used),
                            "atr": float(a),
                            "entry_ref": float(entry_ref),
                            "sl": float(entry_ref - SL_MULT * a),
                            "tp": float(entry_ref + TP_MULT * a),
                            "blocked_at_3_5": float(z[i]) >= 3.5,
                            "blocked_at_4_0": float(z[i]) >= 4.0,
                        }
                    )
                i += 1
                continue

            # Normal/elevated: open like engine (for stacking state)
            a = atr[i]
            if not np.isfinite(a) or a <= 0:
                i += 1
                continue
            size_mult = 0.5 if tier == "elevated" else 1.0
            eq_now = equity_at(dates[i])
            sl_dist = SL_MULT * a
            risk_amt = eq_now * cfg.risk_per_trade_pct * size_mult
            lots = _round_lots(risk_amt / (sl_dist * CONTRACT_SIZE))
            if lots <= 0:
                i += 1
                continue
            entry_ref = close[i]
            sl_level = entry_ref - SL_MULT * a
            tp_level = entry_ref + TP_MULT * a
            pos_open = True
            entry_i = i
            i += 1
            continue

        i += 1

    return pd.DataFrame(rows)


def candle_window(df: pd.DataFrame, i: int, before: int = 4, after: int = 8) -> str:
    lo = max(0, i - before)
    hi = min(len(df), i + after + 1)
    lines = [
        "| idx | Date | O | H | L | C | notes |",
        "|---:|---|---:|---:|---:|---:|---|",
    ]
    for j in range(lo, hi):
        notes = []
        if j == i:
            notes.append("<< SIGNAL")
        lines.append(
            f"| {j} | {df['Date'].iloc[j]} | "
            f"{df['Open'].iloc[j]:.2f} | {df['High'].iloc[j]:.2f} | "
            f"{df['Low'].iloc[j]:.2f} | {df['Close'].iloc[j]:.2f} | "
            f"{' '.join(notes)} |"
        )
    return "\n".join(lines)


def summarize(blocked: pd.DataFrame, label: str) -> list[str]:
    if blocked.empty:
        return [f"### {label}", "", "(no blocked signals)", ""]
    n = len(blocked)
    n_tp = int((blocked["hypothetical_outcome"] == "tp").sum())
    n_sl = int((blocked["hypothetical_outcome"] == "sl").sum())
    n_to = int((blocked["hypothetical_outcome"] == "timeout").sum())
    total_pnl = float(blocked["hypothetical_pnl_usd"].sum())
    total_r = float(blocked["hypothetical_pnl_R"].sum())
    losses = blocked[blocked["hypothetical_outcome"] == "sl"]
    wins = blocked[blocked["hypothetical_outcome"] == "tp"]
    # concentration: share of absolute loss from top 2 losses
    loss_pnls = losses["hypothetical_pnl_usd"].to_numpy() if len(losses) else np.array([])
    if len(loss_pnls):
        abs_losses = np.sort(np.abs(loss_pnls))[::-1]
        top2 = float(abs_losses[: min(2, len(abs_losses))].sum())
        all_loss = float(np.abs(loss_pnls).sum())
        top2_share = top2 / all_loss if all_loss > 0 else 0.0
    else:
        top2_share = float("nan")

    lines = [
        f"### {label}",
        "",
        f"- n_blocked = **{n}**",
        f"- outcomes: TP(win)={n_tp} ({100*n_tp/n:.1f}%), SL(loss)={n_sl} ({100*n_sl/n:.1f}%), "
        f"timeout={n_to} ({100*n_to/n:.1f}%)",
        f"- would-be winrate TP/SL = {n_tp/(n_tp+n_sl):.3f}" if (n_tp + n_sl) else "- would-be winrate: n/a",
        f"- total hypothetical PnL = **{total_pnl:+.2f} USD** ({total_r:+.2f} R)",
        f"- mean PnL/trade = {total_pnl/n:+.2f} USD",
        (
            f"- top-2 loss share of all SL abs PnL = {top2_share:.1%}"
            if len(loss_pnls)
            else "- top-2 loss share: n/a"
        ),
        "",
    ]
    return lines


def main() -> None:
    base = Path("data")
    feats = base / "features/xauusd_h1_h4_d1_features_v3.parquet"
    h1 = base / "raw/XAUUSD_H1.csv"
    sealed_preds = base / "models_v3/long/lightgbm_long_v3_test_preds.parquet"
    atr_hist = pd.read_parquet(feats, columns=["Date", "atr_h1"])

    print("Note: blocked-signal log was NOT saved in prior guard run — regenerating from replay.")

    df0 = load_backtest_frame(preds_path=sealed_preds, features_path=feats, h1_raw_path=h1)
    df0 = df0[(df0["Date"] >= SEALED_START) & (df0["Date"] <= SEALED_END)].reset_index(drop=True)

    # No-guard equity path for sizing
    res_flat = run_backtest(df0, BacktestConfig(apply_costs=True, selected_thr=THR, enable_regime_guard=False))
    eq_curve = res_flat.equity_curve

    tables = {}
    for zb in (3.5, 4.0):
        df = attach_regime_to_frame(
            df0, atr_hist, window=ROLLING_WINDOW_BARS, z_reduce=Z_REDUCE_THRESHOLD, z_block=zb
        )
        blocked = collect_blocked(df, z_block=zb, equity_curve=eq_curve)
        tables[zb] = blocked
        print(f"z_block={zb}: n_blocked={len(blocked)} outcomes={blocked['hypothetical_outcome'].value_counts().to_dict() if len(blocked) else {}}")

    b35 = tables[3.5]
    b40 = tables[4.0]

    # Save raw audit tables
    out_dir = base / "backtest_v3_guarded"
    if len(b35):
        b35.to_parquet(out_dir / "blocked_signals_audit_z35_sealed.parquet", index=False)
    if len(b40):
        b40.to_parquet(out_dir / "blocked_signals_audit_z40_sealed.parquet", index=False)

    def fmt_table(blocked: pd.DataFrame) -> str:
        if blocked.empty:
            return "_(empty)_"
        lines = [
            "| timestamp | atr_zscore | regime_tier | outcome | pnl_usd | pnl_R | bars | mark |",
            "|---|---:|---|---|---:|---:|---:|---|",
        ]
        for _, r in blocked.sort_values("timestamp").iterrows():
            o = r["hypothetical_outcome"]
            mark = {"tp": "WIN", "sl": "LOSS", "timeout": "TIMEOUT"}[o]
            lines.append(
                f"| {r['timestamp']} | {r['atr_zscore']:.3f} | {r['regime_tier']} | "
                f"{o} | {r['hypothetical_pnl_usd']:+.2f} | {r['hypothetical_pnl_R']:+.3f} | "
                f"{int(r['bars_to_resolution'])} | **{mark}** |"
            )
        return "\n".join(lines)

    # Extreme cases for candle dumps — from z=3.5 set
    df35 = attach_regime_to_frame(
        df0, atr_hist, window=ROLLING_WINDOW_BARS, z_reduce=Z_REDUCE_THRESHOLD, z_block=3.5
    )
    case_sections = []
    if len(b35):
        top = b35.nlargest(min(5, len(b35)), "atr_zscore")
        for k, (_, r) in enumerate(top.iterrows(), 1):
            ts = r["timestamp"]
            idxs = df35.index[df35["Date"] == ts]
            if len(idxs) == 0:
                continue
            i = int(idxs[0])
            case_sections.append(
                f"### Extreme case {k}: z={r['atr_zscore']:.3f} @ {ts} → {r['hypothetical_outcome']} "
                f"(PnL {r['hypothetical_pnl_usd']:+.2f} USD / {r['hypothetical_pnl_R']:+.2f} R)\n\n"
                + candle_window(df35, i)
            )

    # Verdicts
    def verdict_bits(blocked: pd.DataFrame) -> dict:
        if blocked.empty:
            return {}
        n = len(blocked)
        n_sl = int((blocked["hypothetical_outcome"] == "sl").sum())
        n_tp = int((blocked["hypothetical_outcome"] == "tp").sum())
        n_to = int((blocked["hypothetical_outcome"] == "timeout").sum())
        total = float(blocked["hypothetical_pnl_usd"].sum())
        losses = blocked[blocked["hypothetical_outcome"] == "sl"]["hypothetical_pnl_usd"]
        if len(losses):
            abs_l = np.sort(np.abs(losses.to_numpy()))[::-1]
            top2_share = float(abs_l[: min(2, len(abs_l))].sum() / abs_l.sum())
        else:
            top2_share = float("nan")
        return {
            "n": n,
            "n_sl": n_sl,
            "n_tp": n_tp,
            "n_to": n_to,
            "pct_loss": 100 * n_sl / n,
            "pct_win": 100 * n_tp / n,
            "total_pnl": total,
            "top2_share": top2_share,
        }

    v35 = verdict_bits(b35)
    v40 = verdict_bits(b40)

    # Wrongly blocked = wins (tp) — check ADX if available
    pattern_note = ""
    if len(b35):
        feat = pd.read_parquet(feats, columns=["Date", "adx_h1", "adx_h4"])
        feat["Date"] = pd.to_datetime(feat["Date"], utc=True)
        m = b35.merge(feat, left_on="timestamp", right_on="Date", how="left")
        wrong = m[m["hypothetical_outcome"] == "tp"]
        right = m[m["hypothetical_outcome"] == "sl"]
        if len(wrong) and len(right):
            pattern_note = (
                f"- Among blocked-that-would-WIN: mean adx_h1={wrong['adx_h1'].mean():.1f}, "
                f"adx_h4={wrong['adx_h4'].mean():.1f}\n"
                f"- Among blocked-that-would-LOSE: mean adx_h1={right['adx_h1'].mean():.1f}, "
                f"adx_h4={right['adx_h4'].mean():.1f}\n"
            )
            if wrong["adx_h1"].mean() > right["adx_h1"].mean() + 2:
                pattern_note += (
                    "- Hint: false blocks (would-be wins) have **higher ADX** on average → "
                    "volatile-but-trending; future guard could allow extreme ATR when ADX high.\n"
                )
            elif right["adx_h1"].mean() > wrong["adx_h1"].mean() + 2:
                pattern_note += (
                    "- Hint: true blocks (would-be losses) have higher ADX — not a clean "
                    "trend-vs-chop filter from this tiny sample.\n"
                )
            else:
                pattern_note += "- ADX gap between wrong/right blocks is small in this sample.\n"

    lines = [
        "# Gate 4.5 — Blocked Trades Audit (sealed 2026)",
        "",
        "## Data source note",
        "",
        "Prior guard run (`trades_with_guard.parquet`) only stores **executed** trades, "
        "not blocked signals. This audit **regenerated** blocked timestamps by replaying "
        "the sealed-2026 frame with causal regime tiers (W=4320, Z_REDUCE=2.0) and the same "
        "entry/no-stacking rules as the engine.",
        "",
        f"Window: `{SEALED_START}` → `{SEALED_END}` | model thr={THR} | LONG barriers SL/TP/horizon = {SL_MULT}/{TP_MULT}/{HORIZON}",
        "",
        "Hypothetical sizing uses **no-guard** equity curve as-of signal time (1% risk, Method A costs).",
        "Group z_block=4.0 is the subset of z_block=3.5 with `atr_zscore >= 4.0`.",
        "",
        f"Replay counts: blocked@3.5 = {len(b35)}, blocked@4.0 = {len(b40)} "
        f"(report previously said 13 / 7 — small diffs possible from equity-path / stacking interaction).",
        "",
        "## Summary statistics",
        "",
        *summarize(b35, "Z_BLOCK = 3.5"),
        *summarize(b40, "Z_BLOCK = 4.0"),
        "## Full table — Z_BLOCK 3.5",
        "",
        fmt_table(b35),
        "",
        "## Full table — Z_BLOCK 4.0 (subset)",
        "",
        fmt_table(b40),
        "",
        "## Candle context — highest-z blocked signals (from 3.5 set)",
        "",
        "\n\n".join(case_sections) if case_sections else "_(none)_",
        "",
        "## Pattern check (ADX on wrong vs right blocks)",
        "",
        pattern_note or "_(insufficient samples)_",
        "",
        "## Required conclusions",
        "",
    ]

    if v35:
        lines.append(
            f"1. **Verdict (z=3.5, n={v35['n']}):** "
            f"would-be LOSS {v35['n_sl']} ({v35['pct_loss']:.0f}%), "
            f"WIN {v35['n_tp']} ({v35['pct_win']:.0f}%), timeout {v35['n_to']}. "
            f"Net hypothetical PnL if allowed: **{v35['total_pnl']:+.2f} USD**."
        )
        if v35["n_sl"] > v35["n_tp"] and v35["total_pnl"] < 0:
            lines.append(
                "   → Closer to **\"mostly would have lost\"** — guard directionally helpful on this window."
            )
        elif v35["n_tp"] > v35["n_sl"] and v35["total_pnl"] > 0:
            lines.append(
                "   → Closer to **\"mostly would have won\"** — guard looks lucky/harmful on this tiny set."
            )
        else:
            lines.append(
                "   → **Ambiguous / mixed** — not a clean 10-loss-vs-3-win story; sample is tiny."
            )

    if v40:
        lines.append(
            f"   **(z=4.0, n={v40['n']}):** LOSS {v40['n_sl']}, WIN {v40['n_tp']}, TO {v40['n_to']}; "
            f"net PnL {v40['total_pnl']:+.2f} USD."
        )

    lines.append("")
    if v35 and v35["n_sl"] > 0:
        lines.append(
            f"2. **Tail concentration:** top-2 SL abs share = {v35['top2_share']:.0%} of all blocked SL losses "
            f"(z=3.5). "
            + (
                "Losses look **somewhat concentrated** (tail-ish)."
                if v35["top2_share"] >= 0.45
                else "Losses look **dispersed** rather than one fat-tail save."
            )
        )
    else:
        lines.append("2. **Tail concentration:** n/a (no SL among blocked).")

    lines += [
        "",
        "3. **Wrong-block patterns:** see ADX section above. Any ADX-based refinement is "
        "**hypothesis only** on n≈10 — do not ship without a larger historical blocked set.",
        "",
        "4. **Gate 5 readiness:** "
    ]

    # Final rec
    if v35 and v35["n"] >= 5:
        if v35["total_pnl"] < 0 and v35["n_sl"] >= v35["n_tp"]:
            rec = (
                "Evidence **mildly supports** \"guard helped on sealed-2026\" but **n is too small** "
                "to claim robustness. Before Gate 5, run a **historical blocked-trade audit** on "
                "2024–2025 (and/or full test) with the same recipe — if blocked bags stay net-negative "
                "OOS, confidence rises; if not, treat sealed improvement as fragile."
            )
        elif v35["total_pnl"] > 0 and v35["n_tp"] > v35["n_sl"]:
            rec = (
                "Blocked bag would have been **net profitable** — sealed CAGR lift may be "
                "**luck / path dependence**. Do **not** treat Gate 4 as settled; expand audit to "
                "earlier years before Gate 5."
            )
        else:
            rec = (
                "Mixed blocked outcomes — **insufficient** to crown the guard. Expand audit "
                "historically before Gate 5."
            )
    else:
        rec = "Too few blocked trades to conclude — expand window/years."

    lines.append(rec)
    lines += [
        "",
        "## Artifacts",
        "- `data/backtest_v3_guarded/blocked_signals_audit_z35_sealed.parquet`",
        "- `data/backtest_v3_guarded/blocked_signals_audit_z40_sealed.parquet`",
        "- `data/reports/gate4_5_blocked_trades_audit.md`",
        "",
    ]

    report_path = base / "reports/gate4_5_blocked_trades_audit.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
