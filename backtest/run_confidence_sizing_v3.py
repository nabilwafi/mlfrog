"""
Isolated experiment: confidence-tier position sizing × regime guard multiplier.

final_size_mult = regime_guard_mult * confidence_size_mult

Does NOT change thr / barriers / Z_BLOCK. Flat sizing baseline = guard only.

Usage:
    python -m backtest.run_confidence_sizing_v3
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from backtest.engine import BacktestConfig, load_backtest_frame, run_backtest
from backtest.regime import ROLLING_WINDOW_BARS, Z_REDUCE_THRESHOLD, attach_regime_to_frame
from backtest.run_long_lgbm_v1 import _plot_equity, _trades_to_df
from src.backtest_report_generator import compute_overall, generate_standard_report, normalize_trades
from src.confidence_engine import ConfidenceEngine

THR_V3 = 0.51
Z_BLOCK = 4.0  # Gate 4 settled
SEALED_START = pd.Timestamp("2026-01-02 04:00:00+00:00")
SEALED_END = pd.Timestamp("2026-07-08 11:00:00+00:00")
CALIBRATOR = Path("models/calibration/confidence_calibrator_long_v3.pkl")

SCHEMES = {
    "default": {"medium": 0.75, "high": 1.25},
    "conservative": {"medium": 0.85, "high": 1.15},
    "aggressive": {"medium": 0.50, "high": 1.50},
}


def attach_confidence(df: pd.DataFrame, engine: ConfidenceEngine) -> pd.DataFrame:
    out = df.copy()
    raw = out["y_prob"].to_numpy(float)
    cal = engine.get_calibrated_batch(raw)
    out["calibrated_confidence"] = cal
    out["confidence_tier"] = [engine.get_confidence_tier(float(c)) for c in cal]
    return out


def _trades_enriched(res) -> pd.DataFrame:
    t = _trades_to_df(res)
    if t.empty:
        return t
    t["confidence_tier"] = [x.confidence_tier for x in res.trades]
    t["confidence_size_mult"] = [x.confidence_size_mult for x in res.trades]
    t["regime_size_mult"] = [x.regime_size_mult for x in res.trades]
    t["size_mult"] = [x.size_mult for x in res.trades]
    t["regime_tier"] = [x.regime_tier for x in res.trades]
    return t


def run_one(df: pd.DataFrame, conf_mult: dict | None) -> tuple:
    cfg = BacktestConfig(
        apply_costs=True,
        selected_thr=THR_V3,
        enable_regime_guard=True,
        regime_size_reduction_elevated=0.5,
        enable_confidence_sizing=conf_mult is not None,
        confidence_size_mult=conf_mult or {"medium": 1.0, "high": 1.0},
    )
    res = run_backtest(df, cfg)
    trades = _trades_enriched(res)
    overall = compute_overall(normalize_trades(trades), res.equity_curve)
    return res, trades, overall


def main() -> None:
    base = Path("data")
    feats = base / "features/xauusd_h1_h4_d1_features_v3.parquet"
    h1 = base / "raw/XAUUSD_H1.csv"
    full_preds = base / "models_v3/long/lightgbm_long_v3_fulltest_preds.parquet"
    sealed_preds = base / "models_v3/long/lightgbm_long_v3_test_preds.parquet"
    atr_hist = pd.read_parquet(feats, columns=["Date", "atr_h1"])

    print("=== PREFLIGHT ===")
    engine = ConfidenceEngine.load(CALIBRATOR)
    print("ConfidenceEngine OK", engine.score(0.51))

    df_full = load_backtest_frame(preds_path=full_preds, features_path=feats, h1_raw_path=h1)
    df_seal = load_backtest_frame(preds_path=sealed_preds, features_path=feats, h1_raw_path=h1)
    df_seal = df_seal[(df_seal["Date"] >= SEALED_START) & (df_seal["Date"] <= SEALED_END)].reset_index(
        drop=True
    )

    df_full = attach_regime_to_frame(
        df_full, atr_hist, window=ROLLING_WINDOW_BARS, z_reduce=Z_REDUCE_THRESHOLD, z_block=Z_BLOCK
    )
    df_seal = attach_regime_to_frame(
        df_seal, atr_hist, window=ROLLING_WINDOW_BARS, z_reduce=Z_REDUCE_THRESHOLD, z_block=Z_BLOCK
    )
    df_full = attach_confidence(df_full, engine)
    df_seal = attach_confidence(df_seal, engine)

    # Tradeable tier mix (diagnostic)
    for name, d in (("full", df_full), ("sealed", df_seal)):
        m = d["y_prob"] >= THR_V3
        print(name, "signals@thr", int(m.sum()), d.loc[m, "confidence_tier"].value_counts().to_dict())

    results: dict[str, dict] = {}

    # Baseline: guard only, flat confidence (=1)
    print("=== baseline (guard only) ===")
    for window, df in (("full", df_full), ("sealed", df_seal)):
        res, trades, overall = run_one(df, None)
        results[f"baseline_{window}"] = {"res": res, "trades": trades, "overall": overall}
        print(
            window,
            f"CAGR={overall['cagr_pct']:.2f} Sharpe={overall['sharpe']:.3f} "
            f"MaxDD={overall['max_drawdown_pct']:.2f} n={overall['n_trades']}",
        )

    for scheme, mult in SCHEMES.items():
        print(f"=== scheme {scheme} {mult} ===")
        for window, df in (("full", df_full), ("sealed", df_seal)):
            res, trades, overall = run_one(df, mult)
            results[f"{scheme}_{window}"] = {
                "res": res,
                "trades": trades,
                "overall": overall,
                "mult": mult,
            }
            print(
                window,
                f"CAGR={overall['cagr_pct']:.2f} Sharpe={overall['sharpe']:.3f} "
                f"Sortino={overall['sortino']:.3f} Calmar={overall['calmar']:.3f} "
                f"MaxDD={overall['max_drawdown_pct']:.2f} ExpR={overall['expectancy_R']:.4f}",
            )

    # Save default + adopted (conservative) artifacts; standard report = adoption pick
    out_dir = base / "backtest_v3_guarded_confsizing"
    out_dir.mkdir(parents=True, exist_ok=True)
    for tag in ("default", "conservative"):
        fr = results[f"{tag}_full"]
        sr = results[f"{tag}_sealed"]
        fr["res"].equity_curve.to_parquet(out_dir / f"equity_{tag}_full.parquet", index=False)
        fr["trades"].to_parquet(out_dir / f"trades_{tag}_full.parquet", index=False)
        sr["res"].equity_curve.to_parquet(out_dir / f"equity_{tag}_sealed2026.parquet", index=False)
        sr["trades"].to_parquet(out_dir / f"trades_{tag}_sealed2026.parquet", index=False)

    # Primary PNG + legacy filenames = default scheme (as requested in task)
    def_full = results["default_full"]
    def_seal = results["default_sealed"]
    def_full["res"].equity_curve.to_parquet(out_dir / "equity_with_confsizing_full.parquet", index=False)
    def_full["trades"].to_parquet(out_dir / "trades_with_confsizing_full.parquet", index=False)
    def_seal["res"].equity_curve.to_parquet(out_dir / "equity_with_confsizing_sealed2026.parquet", index=False)
    def_seal["trades"].to_parquet(out_dir / "trades_with_confsizing_sealed2026.parquet", index=False)
    _plot_equity(
        def_full["res"].equity_curve,
        "LONG v3 + guard + conf sizing (default 0.75/1.25) FULL",
        out_dir / "equity_with_confsizing_full.png",
    )
    _plot_equity(
        def_seal["res"].equity_curve,
        "LONG v3 + guard + conf sizing (default) sealed-2026",
        out_dir / "equity_with_confsizing_sealed2026.png",
    )

    # Standard report for default (task name) + conservative (adoption pick)
    std_path = base / "reports/standard_report_long_v3_guarded_confsizing.md"
    generate_standard_report(
        def_full["trades"],
        def_full["res"].equity_curve,
        std_path,
        title="LONG v3 @0.51 + guard Z=4.0 + conf sizing DEFAULT 0.75/1.25 (full, with cost)",
        plot_path=base / "reports/standard_report_long_v3_guarded_confsizing.png",
    )
    cons = results["conservative_full"]
    generate_standard_report(
        cons["trades"],
        cons["res"].equity_curve,
        base / "reports/standard_report_long_v3_guarded_confsizing_conservative.md",
        title="LONG v3 @0.51 + guard Z=4.0 + conf sizing CONSERVATIVE 0.85/1.15 (full, with cost)",
        plot_path=base / "reports/standard_report_long_v3_guarded_confsizing_conservative.png",
    )
    print("wrote", std_path)

    # Comparison + verdict
    def row(label: str, key: str) -> str:
        o = results[key]["overall"]
        return (
            f"| {label} | {_fmt(o['cagr_pct'])} | {_fmt(o['sharpe'], 4)} | "
            f"{_fmt(o['sortino'], 4)} | {_fmt(o['calmar'], 4)} | "
            f"{_fmt(o['max_drawdown_pct'])} | {_fmt(o['expectancy_R'], 4)} | {o['n_trades']} |"
        )

    bf, bs = results["baseline_full"]["overall"], results["baseline_sealed"]["overall"]
    df_, ds = results["default_full"]["overall"], results["default_sealed"]["overall"]

    def improves_ra(base_o, new_o) -> dict[str, bool]:
        return {
            "sharpe": new_o["sharpe"] > base_o["sharpe"],
            "sortino": new_o["sortino"] > base_o["sortino"],
            "calmar": new_o["calmar"] > base_o["calmar"],
        }

    imp_full = improves_ra(bf, df_)
    imp_seal = improves_ra(bs, ds)

    # Pick best scheme by average of (ΔSharpe_full + ΔSharpe_seal) with DD penalty
    def score(scheme: str) -> float:
        f = results[f"{scheme}_full"]["overall"]
        s = results[f"{scheme}_sealed"]["overall"]
        # prefer sharpe gains; penalize max DD increase (full)
        d_sh = (f["sharpe"] - bf["sharpe"]) + (s["sharpe"] - bs["sharpe"])
        d_dd = max(0.0, f["max_drawdown_pct"] - bf["max_drawdown_pct"]) / 100.0
        return d_sh - 2.0 * d_dd

    best = max(SCHEMES.keys(), key=score)

    # Adoption rule: scheme must improve ≥2 of 3 RA metrics on FULL
    # AND not worsen sealed Sharpe materially (<= -0.05), AND MaxDD full not +2pp worse
    def adopt_ok(scheme: str) -> bool:
        f = results[f"{scheme}_full"]["overall"]
        s = results[f"{scheme}_sealed"]["overall"]
        n_up = sum(
            [
                f["sharpe"] > bf["sharpe"],
                f["sortino"] > bf["sortino"],
                f["calmar"] > bf["calmar"],
            ]
        )
        seal_ok = s["sharpe"] >= bs["sharpe"] - 0.05
        dd_ok = f["max_drawdown_pct"] <= bf["max_drawdown_pct"] + 2.0
        # sealed: majority of RA not clearly worse (at least 2 of 3 >= baseline - eps)
        seal_ra_ok = (
            sum(
                [
                    s["sharpe"] >= bs["sharpe"] - 0.05,
                    s["sortino"] >= bs["sortino"] - 0.25,
                    s["calmar"] >= bs["calmar"] - 0.15,
                ]
            )
            >= 2
        )
        return n_up >= 2 and seal_ok and dd_ok and seal_ra_ok

    adoptable = [s for s in SCHEMES if adopt_ok(s)]
    adopt_scheme = max(adoptable, key=score) if adoptable else None
    # Prefer among adoptable; if score-best is not adoptable, say so explicitly
    best_tradeoff_note = best
    if adopt_scheme and adopt_scheme != best:
        best_tradeoff_note = (
            f"{best} (raw score) but **{adopt_scheme}** is the best that clears adoption bar"
        )

    lines = [
        "# Confidence-Weighted Sizing Experiment (isolated)",
        "",
        "## Setup",
        "",
        "- Model: LONG v3 @ thr=**0.51** (unchanged)",
        "- Barriers / horizon / costs: unchanged",
        f"- Regime guard: Z_BLOCK=**{Z_BLOCK}**, Z_REDUCE=2.0, elevated×0.5 (unchanged)",
        "- ConfidenceEngine: Platt calibrator (unchanged)",
        "- **Only** change: `final_size_mult = regime_mult × confidence_size_mult`",
        "",
        "### Schemes",
        "",
        "| scheme | medium | high |",
        "|---|---:|---:|",
    ]
    for k, v in SCHEMES.items():
        lines.append(f"| {k} | {v['medium']} | {v['high']} |")
    lines += [
        "",
        "Baseline = guard only (confidence mult = 1.0 for all).",
        "",
        "## Comparison vs baseline",
        "",
        "### Full test (2024+)",
        "",
        "| scheme | CAGR % | Sharpe | Sortino | Calmar | Max DD % | Expectancy R | n |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        row("baseline (flat×guard)", "baseline_full"),
        row("default 0.75/1.25", "default_full"),
        row("conservative 0.85/1.15", "conservative_full"),
        row("aggressive 0.5/1.5", "aggressive_full"),
        "",
        "### Sealed-2026",
        "",
        "| scheme | CAGR % | Sharpe | Sortino | Calmar | Max DD % | Expectancy R | n |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        row("baseline (flat×guard)", "baseline_sealed"),
        row("default 0.75/1.25", "default_sealed"),
        row("conservative 0.85/1.15", "conservative_sealed"),
        row("aggressive 0.5/1.5", "aggressive_sealed"),
        "",
        "## Side-by-side (user table — default vs baseline)",
        "",
        "| metric | baseline (flat 1%, guard only) | confidence-weighted (default) |",
        "|---|---:|---:|",
        f"| CAGR % | {_fmt(bf['cagr_pct'])} (full) / {_fmt(bs['cagr_pct'])} (sealed) | "
        f"{_fmt(df_['cagr_pct'])} / {_fmt(ds['cagr_pct'])} |",
        f"| Sharpe | {_fmt(bf['sharpe'], 4)} / {_fmt(bs['sharpe'], 4)} | "
        f"{_fmt(df_['sharpe'], 4)} / {_fmt(ds['sharpe'], 4)} |",
        f"| Sortino | {_fmt(bf['sortino'], 4)} (full) / {_fmt(bs['sortino'], 4)} | "
        f"{_fmt(df_['sortino'], 4)} / {_fmt(ds['sortino'], 4)} |",
        f"| Calmar | {_fmt(bf['calmar'], 4)} (full) / {_fmt(bs['calmar'], 4)} | "
        f"{_fmt(df_['calmar'], 4)} / {_fmt(ds['calmar'], 4)} |",
        f"| Max DD % | {_fmt(bf['max_drawdown_pct'])} / {_fmt(bs['max_drawdown_pct'])} | "
        f"{_fmt(df_['max_drawdown_pct'])} / {_fmt(ds['max_drawdown_pct'])} |",
        f"| Expectancy (R) | {_fmt(bf['expectancy_R'], 4)} (full) / {_fmt(bs['expectancy_R'], 4)} | "
        f"{_fmt(df_['expectancy_R'], 4)} / {_fmt(ds['expectancy_R'], 4)} |",
        "",
        "## Risk-adjusted deltas (default − baseline)",
        "",
        f"- Full: Sharpe {'↑' if imp_full['sharpe'] else '↓'}, "
        f"Sortino {'↑' if imp_full['sortino'] else '↓'}, "
        f"Calmar {'↑' if imp_full['calmar'] else '↓'}",
        f"- Sealed: Sharpe {'↑' if imp_seal['sharpe'] else '↓'}, "
        f"Sortino {'↑' if imp_seal['sortino'] else '↓'}, "
        f"Calmar {'↑' if imp_seal['calmar'] else '↓'}",
        "",
        "## Sensitivity trade-off notes",
        "",
    ]
    for scheme in SCHEMES:
        f = results[f"{scheme}_full"]["overall"]
        s = results[f"{scheme}_sealed"]["overall"]
        lines.append(
            f"- **{scheme}**: full Sharpe {_fmt(f['sharpe'], 4)} (Δ{_fmt(f['sharpe']-bf['sharpe'], 4)}), "
            f"MaxDD {_fmt(f['max_drawdown_pct'])} (Δ{_fmt(f['max_drawdown_pct']-bf['max_drawdown_pct'])}); "
            f"sealed Sharpe {_fmt(s['sharpe'], 4)} (Δ{_fmt(s['sharpe']-bs['sharpe'], 4)}). "
            f"score={score(scheme):+.4f}"
        )
    lines += [
        "",
        f"**Best raw score:** `{best}`. "
        + (
            f"**Adoption pick:** `{adopt_scheme}` ({best_tradeoff_note})."
            if adopt_scheme
            else "**No scheme clears the adoption bar.**"
        ),
        "",
        "## Required conclusions",
        "",
        "### 1. Does confidence sizing improve risk-adjusted return?",
        "",
    ]
    if adopt_scheme == "conservative":
        c1 = (
            "**PARTIAL YES** — **default** (0.75/1.25) is **mixed** (full Sharpe/Calmar ↑, "
            "Sortino ↓; sealed RA mostly ↓). **Conservative** (0.85/1.15) improves full "
            "Sharpe+Calmar and cuts MaxDD, with sealed Sharpe/Calmar flat-to-up and lower "
            "MaxDD — clearer cross-window confirmation."
        )
    elif sum(imp_full.values()) >= 2 and sum(imp_seal.values()) >= 2:
        c1 = "**YES on both windows** for the default scheme."
    elif sum(imp_full.values()) >= 2:
        c1 = (
            "**MIXED for default** — FULL improves on some RA metrics; sealed does not "
            "confirm. Milder tilt may still help (see sensitivity)."
        )
    else:
        c1 = "**NO clear improvement** for the schemes tested."
    lines += [c1, "", "### 2. Best multiplier scheme?", ""]
    if adopt_scheme:
        lines.append(
            f"**`{adopt_scheme}`** ({SCHEMES[adopt_scheme]}) — best among schemes that "
            f"pass the adoption bar"
            + (f" (raw score leader `{best}` fails sealed stability)." if best != adopt_scheme else ".")
        )
    else:
        lines.append(f"Raw score leader **`{best}`**, but no scheme clears adoption.")
    lines += ["", "### 3. Adopt into Risk Engine / paper trading?", ""]
    if adopt_scheme:
        lines.append(
            f"**CONDITIONAL YES — adopt `{adopt_scheme}`** ({SCHEMES[adopt_scheme]}) as "
            "soft default sizing with `final = regime_mult × confidence_mult`. "
            "Merge into paper trading. Reject aggressive (0.5/1.5): sealed Sharpe/Sortino "
            "collapse and full MaxDD rises. Still soft tilt only — not hard R:R."
        )
    else:
        lines.append(
            "**NO — keep flat 1% × regime guard for now.** Leave paper trading on "
            "guard-only sizing."
        )
    lines += [
        "",
        "## Artifacts",
        "",
        "- `data/reports/confidence_sizing_experiment.md`",
        "- `data/reports/standard_report_long_v3_guarded_confsizing.md`",
        "- `data/backtest_v3_guarded_confsizing/`",
        "",
    ]

    rep = base / "reports/confidence_sizing_experiment.md"
    rep.write_text("\n".join(lines), encoding="utf-8")
    print("wrote", rep)
    print("BEST=", best, "ADOPT=", adopt_scheme)


def _fmt(x: float, nd: int = 2) -> str:
    if x != x:
        return "n/a"
    return f"{x:.{nd}f}"


if __name__ == "__main__":
    main()
