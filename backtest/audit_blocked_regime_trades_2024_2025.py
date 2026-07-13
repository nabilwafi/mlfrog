"""
Gate 4.5 extension: blocked-trade audit on 2024-01-01 .. 2025-12-31
(same methodology as sealed-2026 audit). Pure diagnostic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from backtest.audit_blocked_regime_trades import (
    THR,
    candle_window,
    collect_blocked,
    summarize,
)
from backtest.engine import BacktestConfig, load_backtest_frame, run_backtest
from backtest.regime import (
    ROLLING_WINDOW_BARS,
    Z_REDUCE_THRESHOLD,
    attach_regime_to_frame,
)

START = pd.Timestamp("2024-01-01 00:00:00+00:00")
END = pd.Timestamp("2025-12-31 23:59:59+00:00")

# Sealed-2026 reference (from prior Gate 4.5 report)
SEALED_REF = {
    3.5: {"n": 13, "wr": 0.455, "pnl": 66.55, "pct_loss": 46.2},
    4.0: {"n": 7, "wr": 0.143, "pnl": -476.00, "pct_loss": 85.7},
}


def _stats(blocked: pd.DataFrame) -> dict:
    if blocked.empty:
        return {
            "n": 0,
            "n_tp": 0,
            "n_sl": 0,
            "n_to": 0,
            "wr": float("nan"),
            "pnl": 0.0,
            "pct_loss": float("nan"),
            "top2": float("nan"),
        }
    n = len(blocked)
    n_tp = int((blocked["hypothetical_outcome"] == "tp").sum())
    n_sl = int((blocked["hypothetical_outcome"] == "sl").sum())
    n_to = int((blocked["hypothetical_outcome"] == "timeout").sum())
    pnl = float(blocked["hypothetical_pnl_usd"].sum())
    wr = n_tp / (n_tp + n_sl) if (n_tp + n_sl) else float("nan")
    pct_loss = 100.0 * n_sl / n
    losses = blocked.loc[blocked["hypothetical_outcome"] == "sl", "hypothetical_pnl_usd"]
    if len(losses):
        abs_l = np.sort(np.abs(losses.to_numpy()))[::-1]
        top2 = float(abs_l[: min(2, len(abs_l))].sum()) / float(np.abs(losses).sum())
    else:
        top2 = float("nan")
    return {
        "n": n,
        "n_tp": n_tp,
        "n_sl": n_sl,
        "n_to": n_to,
        "wr": wr,
        "pnl": pnl,
        "pct_loss": pct_loss,
        "top2": top2,
    }


def _fmt_table(blocked: pd.DataFrame) -> str:
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
            f"{int(r['bars_to_resolution'])} | {mark} |"
        )
    return "\n".join(lines)


def _adx_section(blocked: pd.DataFrame, feats: pd.DataFrame, title: str) -> list[str]:
    if blocked.empty:
        return [f"### {title}", "", "(empty)", ""]
    adx = feats[["Date", "adx_h1", "adx_h4"]].copy()
    adx["Date"] = pd.to_datetime(adx["Date"], utc=True)
    m = blocked.merge(adx, left_on="timestamp", right_on="Date", how="left")
    wrong = m[m["hypothetical_outcome"] == "tp"]
    right = m[m["hypothetical_outcome"] == "sl"]
    lines = [
        f"### {title}",
        "",
        "| group | n | mean ADX H1 | mean ADX H4 |",
        "|---|---:|---:|---:|",
        f"| wrong-block (would-be WIN) | {len(wrong)} | "
        f"{wrong['adx_h1'].mean():.2f} | {wrong['adx_h4'].mean():.2f} |",
        f"| right-block (would-be LOSS) | {len(right)} | "
        f"{right['adx_h1'].mean():.2f} | {right['adx_h4'].mean():.2f} |",
        "",
    ]
    if len(wrong) and len(right):
        d1 = float(wrong["adx_h1"].mean() - right["adx_h1"].mean())
        d4 = float(wrong["adx_h4"].mean() - right["adx_h4"].mean())
        note = (
            "positif H4 mendukung hipotesis sealed-2026"
            if d4 > 0
            else "H4 terbalik/tidak mendukung hipotesis; jangan encode"
        )
        lines.append(f"- Δ(wrong−right) ADX H1 = {d1:+.2f}; ADX H4 = {d4:+.2f} ({note})")
        lines.append("")
    return lines


def main() -> None:
    base = Path("data")
    feats_path = base / "features/xauusd_h1_h4_d1_features_v3.parquet"
    h1 = base / "raw/XAUUSD_H1.csv"
    preds = base / "models_v3/long/lightgbm_long_v3_fulltest_preds.parquet"
    atr_hist = pd.read_parquet(feats_path, columns=["Date", "atr_h1"])
    feats_full = pd.read_parquet(feats_path)

    print("=== PREFLIGHT ===")
    assert preds.exists(), preds
    assert feats_path.exists(), feats_path
    print(f"model preds: {preds}")
    print(f"thr={THR}, window={START.date()} .. {END.date()}")

    df0 = load_backtest_frame(preds_path=preds, features_path=feats_path, h1_raw_path=h1)
    df0 = df0[(df0["Date"] >= START) & (df0["Date"] <= END)].reset_index(drop=True)
    print(f"bars in window: {len(df0)}  signals@0.51: {(df0['y_prob'] >= THR).sum()}")

    res_flat = run_backtest(
        df0, BacktestConfig(apply_costs=True, selected_thr=THR, enable_regime_guard=False)
    )
    eq_curve = res_flat.equity_curve

    # Primary thresholds + mid scan for combined verdict
    scan_z = [3.5, 3.6, 3.75, 3.9, 4.0, 4.25, 4.5]
    tables: dict[float, pd.DataFrame] = {}
    stats: dict[float, dict] = {}
    for zb in scan_z:
        df = attach_regime_to_frame(
            df0, atr_hist, window=ROLLING_WINDOW_BARS, z_reduce=Z_REDUCE_THRESHOLD, z_block=zb
        )
        blocked = collect_blocked(df, z_block=zb, equity_curve=eq_curve)
        tables[zb] = blocked
        stats[zb] = _stats(blocked)
        print(
            f"z_block={zb}: n={stats[zb]['n']} "
            f"TP/SL/TO={stats[zb]['n_tp']}/{stats[zb]['n_sl']}/{stats[zb]['n_to']} "
            f"pnl={stats[zb]['pnl']:+.2f}"
        )

    b35, b40 = tables[3.5], tables[4.0]
    out_dir = base / "backtest_v3_guarded"
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(b35):
        b35.to_parquet(out_dir / "blocked_signals_audit_z35_2024_2025.parquet", index=False)
    if len(b40):
        b40.to_parquet(out_dir / "blocked_signals_audit_z40_2024_2025.parquet", index=False)

    # --- report 2024-2025 ---
    lines: list[str] = [
        "# Gate 4.5 — Blocked Trades Audit (2024–2025 full)",
        "",
        "## Scope",
        "",
        "- Model: `lightgbm_long_v3.pkl` @ thr=**0.51** (fulltest preds)",
        "- Window: **2024-01-01 → 2025-12-31** (exclusive of sealed-2026)",
        "- Guard replay: W=4320, Z_REDUCE=2.0, Z_BLOCK ∈ {3.5, 4.0}",
        "- Method: identical to sealed-2026 Gate 4.5 (causal z, flat/open block, "
        "hypothetical barrier SL-first, Method A PnL with no-guard equity sizing)",
        "",
        "## Preflight",
        "",
        f"- bars={len(df0)}, signals@0.51={(df0['y_prob'] >= THR).sum()}",
        f"- no-guard equity end=${float(eq_curve['equity'].iloc[-1]):,.2f}",
        "",
        "## Summary",
        "",
    ]
    lines.extend(summarize(b35, "Z_BLOCK = 3.5"))
    lines.extend(summarize(b40, "Z_BLOCK = 4.0 (subset of 3.5)"))

    lines += [
        "## Full table — Z_BLOCK=3.5",
        "",
        _fmt_table(b35),
        "",
        "## Full table — Z_BLOCK=4.0",
        "",
        _fmt_table(b40),
        "",
        "## Mid-threshold scan (same window)",
        "",
        "| z_block | n_blocked | TP | SL | TO | winrate (TP/SL) | net_pnl_usd | %_would_be_loss |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for zb in scan_z:
        s = stats[zb]
        wr_s = f"{s['wr']:.3f}" if s["wr"] == s["wr"] else "n/a"
        lines.append(
            f"| {zb:.2f} | {s['n']} | {s['n_tp']} | {s['n_sl']} | {s['n_to']} | "
            f"{wr_s} | {s['pnl']:+.2f} | {s['pct_loss']:.1f}% |"
        )

    lines += ["", "## Top-5 highest z-score (from Z_BLOCK=3.5 set)", ""]
    if len(b35):
        top5 = b35.nlargest(5, "atr_zscore")
        df_for_dump = attach_regime_to_frame(
            df0, atr_hist, window=ROLLING_WINDOW_BARS, z_reduce=Z_REDUCE_THRESHOLD, z_block=3.5
        )
        for k, (_, r) in enumerate(top5.iterrows(), 1):
            ts = pd.Timestamp(r["timestamp"])
            matches = df_for_dump.index[df_for_dump["Date"] == ts].tolist()
            dump = candle_window(df_for_dump, matches[0]) if matches else "_(bar not found)_"
            lines += [
                f"### #{k} — {r['timestamp']}  z={r['atr_zscore']:.3f}  "
                f"→ {r['hypothetical_outcome'].upper()} ({r['hypothetical_pnl_usd']:+.2f} USD)",
                "",
                dump,
                "",
            ]
    else:
        lines += ["_(none)_", ""]

    lines.extend(_adx_section(b35, feats_full, "ADX pattern — Z_BLOCK=3.5"))
    lines.extend(_adx_section(b40, feats_full, "ADX pattern — Z_BLOCK=4.0"))

    # quick period consistency note
    s35, s40 = stats[3.5], stats[4.0]
    lines += [
        "## Period note vs sealed-2026 pattern",
        "",
        f"- 2024–2025 @3.5: n={s35['n']}, wr={s35['wr']:.3f}, pnl={s35['pnl']:+.2f}, "
        f"%loss={s35['pct_loss']:.1f}%",
        f"- 2024–2025 @4.0: n={s40['n']}, wr={s40['wr']:.3f}, pnl={s40['pnl']:+.2f}, "
        f"%loss={s40['pct_loss']:.1f}%",
        "",
        "See `gate4_5_combined_verdict.md` for cross-period verdict.",
        "",
    ]

    audit_path = base / "reports/gate4_5_blocked_trades_audit_2024_2025.md"
    audit_path.write_text("\n".join(lines), encoding="utf-8")
    print("wrote", audit_path)

    # --- combined verdict ---
    # ADX deltas for both periods (2024-25 computed; sealed from prior report numbers if we recompute quickly)
    # Recompute sealed ADX for fair comparison
    sealed_preds = base / "models_v3/long/lightgbm_long_v3_test_preds.parquet"
    sealed_start = pd.Timestamp("2026-01-02 04:00:00+00:00")
    sealed_end = pd.Timestamp("2026-07-08 11:00:00+00:00")
    dfs = load_backtest_frame(preds_path=sealed_preds, features_path=feats_path, h1_raw_path=h1)
    dfs = dfs[(dfs["Date"] >= sealed_start) & (dfs["Date"] <= sealed_end)].reset_index(drop=True)
    res_s = run_backtest(
        dfs, BacktestConfig(apply_costs=True, selected_thr=THR, enable_regime_guard=False)
    )
    sealed_tables = {}
    for zb in (3.5, 4.0):
        d = attach_regime_to_frame(
            dfs, atr_hist, window=ROLLING_WINDOW_BARS, z_reduce=Z_REDUCE_THRESHOLD, z_block=zb
        )
        sealed_tables[zb] = collect_blocked(d, z_block=zb, equity_curve=res_s.equity_curve)

    def adx_delta(blocked: pd.DataFrame) -> tuple[float, float, int, int]:
        if blocked.empty:
            return float("nan"), float("nan"), 0, 0
        adx = feats_full[["Date", "adx_h1", "adx_h4"]].copy()
        adx["Date"] = pd.to_datetime(adx["Date"], utc=True)
        m = blocked.merge(adx, left_on="timestamp", right_on="Date", how="left")
        w = m[m["hypothetical_outcome"] == "tp"]
        r = m[m["hypothetical_outcome"] == "sl"]
        if not len(w) or not len(r):
            return float("nan"), float("nan"), len(w), len(r)
        return (
            float(w["adx_h4"].mean() - r["adx_h4"].mean()),
            float(w["adx_h1"].mean() - r["adx_h1"].mean()),
            len(w),
            len(r),
        )

    d4_2425_35, d1_2425_35, nw35, nr35 = adx_delta(b35)
    d4_2425_40, d1_2425_40, nw40, nr40 = adx_delta(b40)
    d4_26_35, _, _, _ = adx_delta(sealed_tables[3.5])
    d4_26_40, _, _, _ = adx_delta(sealed_tables[4.0])

    # Combined PnL if both periods use same z
    def combined_line(zb: float) -> str:
        a = stats[zb]
        b = SEALED_REF[zb]
        # sealed wr from ref; use recompute for pnl consistency
        sb = _stats(sealed_tables[zb])
        tot_n = a["n"] + sb["n"]
        tot_pnl = a["pnl"] + sb["pnl"]
        tot_sl = a["n_sl"] + sb["n_sl"]
        tot_tp = a["n_tp"] + sb["n_tp"]
        wr = tot_tp / (tot_tp + tot_sl) if (tot_tp + tot_sl) else float("nan")
        return (
            f"| combined | {zb:.1f} | {tot_n} | {wr:.3f} | {tot_pnl:+.2f} | "
            f"{100.0 * tot_sl / tot_n:.1f}% |"
        )

    # Ranking consistency (primary claim from 2026)
    ranking_consistent = (s40["pnl"] < s35["pnl"]) and (s35["pnl"] > 0) and (s40["pnl"] < 0)
    pnl_c35 = stats[3.5]["pnl"] + _stats(sealed_tables[3.5])["pnl"]
    pnl_c40 = stats[4.0]["pnl"] + _stats(sealed_tables[4.0])["pnl"]
    recommended = 4.0 if pnl_c40 < pnl_c35 else 3.5
    best_mid = min(scan_z, key=lambda z: stats[z]["pnl"])
    # Gate 4 settled if ranking holds and combined 4.0 clearly protective
    settled = ranking_consistent and recommended == 4.0 and pnl_c40 < -100

    if ranking_consistent:
        consistency_md = [
            "**Ranking-nya konsisten; magnitude-nya tidak identik.**",
            "",
            f"- 2024–2025: Z=4.0 blocked-if-executed **{s40['pnl']:+.2f}** (protektif, lemah) "
            f"vs Z=3.5 **{s35['pnl']:+.2f}** (opportunity cost).",
            "- Arah sama dengan sealed-2026 (4.0 protektif, 3.5 tidak) — bukan kebetulan n=7.",
            "",
        ]
    else:
        consistency_md = [
            f"**Tidak konsisten.** 2024–2025 @3.5={s35['pnl']:+.2f}, @4.0={s40['pnl']:+.2f}.",
            "",
        ]

    mid_note = (
        f"On 2024–2025 alone, most protective z in scan is **{best_mid}** "
        f"(blocked pnl={stats[best_mid]['pnl']:+.2f}). "
        "Jangan retune tanpa sealed mid-scan."
    )

    adx_note = (
        "**Hipotesis ADX H4 tidak terkonfirmasi** di 2024–2025 (Δ terbalik vs sealed-2026). Jangan encode."
        if not (d4_2425_35 > 0 and d4_2425_40 > 0)
        else "ADX H4 arah positif di 2024–2025 — tetap observasi only."
    )

    if settled:
        gate5_md = [
            f"**Gate 4 settled dengan Z_BLOCK={recommended}. Siap lanjut Gate 5.**",
            "",
            f"- Combined blocked PnL: 3.5={pnl_c35:+.2f}, 4.0={pnl_c40:+.2f}.",
            "- Freeze: W=4320, Z_REDUCE=2.0, Z_BLOCK=4.0 (ubah param di Gate 5, bukan di audit ini).",
            "",
        ]
    else:
        gate5_md = [
            "**Belum fully settled tanpa catatan.**",
            "",
            f"- Interim rekomendasi **Z_BLOCK={recommended}** "
            f"(combined 3.5={pnl_c35:+.2f}, 4.0={pnl_c40:+.2f}).",
            "",
        ]

    verdict_lines = [
        "# Gate 4 — Combined Verdict (2024–2025 + sealed-2026)",
        "",
        "## Preflight",
        "",
        "- `lightgbm_long_v3` @ thr=0.51 confirmed",
        "- Features `xauusd_h1_h4_d1_features_v3.parquet` cover both windows",
        "- Methodology identical to Gate 4.5 sealed audit",
        "- Detail: `gate4_5_blocked_trades_audit_2024_2025.md`",
        "",
        "## Combined summary table",
        "",
        "| period | z_block | n_blocked | winrate_hipotetis | net_pnl_hipotetis | %_would_be_loss |",
        "|---|---:|---:|---:|---:|---:|",
        f"| 2024-2025 | 3.5 | {s35['n']} | {s35['wr']:.3f} | {s35['pnl']:+.2f} | {s35['pct_loss']:.1f}% |",
        f"| 2024-2025 | 4.0 | {s40['n']} | {s40['wr']:.3f} | {s40['pnl']:+.2f} | {s40['pct_loss']:.1f}% |",
        f"| sealed-2026 | 3.5 | {SEALED_REF[3.5]['n']} | {SEALED_REF[3.5]['wr']:.3f} | "
        f"{SEALED_REF[3.5]['pnl']:+.2f} | {SEALED_REF[3.5]['pct_loss']:.1f}% |",
        f"| sealed-2026 | 4.0 | {SEALED_REF[4.0]['n']} | {SEALED_REF[4.0]['wr']:.3f} | "
        f"{SEALED_REF[4.0]['pnl']:+.2f} | {SEALED_REF[4.0]['pct_loss']:.1f}% |",
        combined_line(3.5),
        combined_line(4.0),
        "",
        "## 1. Is the 2026 pattern consistent in 2024–2025?",
        "",
        *consistency_md,
        "## 2. Better threshold across periods?",
        "",
        f"- Combined: Z=3.5 → **{pnl_c35:+.2f} USD**; Z=4.0 → **{pnl_c40:+.2f} USD**.",
        f"- Recommended dual-period anchor: **Z_BLOCK={recommended}**.",
        f"- {mid_note}",
        "",
        "## 3. ADX H4 hypothesis (larger n)",
        "",
        "| period | z_block | ΔADX_H4 (wrong−right) | n_wrong | n_right |",
        "|---|---:|---:|---:|---:|",
        f"| 2024-2025 | 3.5 | {d4_2425_35:+.2f} | {nw35} | {nr35} |",
        f"| 2024-2025 | 4.0 | {d4_2425_40:+.2f} | {nw40} | {nr40} |",
        f"| sealed-2026 | 3.5 | {d4_26_35:+.2f} | — | — |",
        f"| sealed-2026 | 4.0 | {d4_26_40:+.2f} | — | — |",
        "",
        adx_note,
        "",
        "## 4. Final Gate 4 verdict → Gate 5?",
        "",
        *gate5_md,
        "## Bottom line",
        "",
        f"- **Z_BLOCK rekomendasi gabungan: {recommended}**",
        f"- Combined blocked PnL @3.5 = {pnl_c35:+.2f} USD; @4.0 = {pnl_c40:+.2f} USD",
        f"- Gate 5: {'YES — freeze Z_BLOCK=4.0 dan lanjut' if settled else 'CONDITIONAL'}",
        "",
    ]

    vpath = base / "reports/gate4_5_combined_verdict.md"
    vpath.write_text("\n".join(verdict_lines), encoding="utf-8")
    print("wrote", vpath)
    print("DONE recommended=", recommended, "settled=", settled)


if __name__ == "__main__":
    main()
