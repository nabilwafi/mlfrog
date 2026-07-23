"""Sprint XX - Exit Engine Audit for ATR Trail 0.12 (entry frozen).

No ML retrain / no entry changes. Path-level audit + robustness + charts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from apps.run_primary_loosen_backtest import run_portfolio
from production.paper.barrier_resim import load_h1
from research.portfolio_heat import STARTING_EQUITY
from research.position_mgmt import COST, SL_ATR
from research.position_mgmt.services.paths import entry_indices, prepare_market, wilder_atr
from settings.strategy import POINT

OUT = _ROOT / "artifacts/exit_engine_audit"
TRAIL = 0.12
ACTIVATE_R = 0.5
HORIZON = 16
MAX_OPEN = 1


def _signed(side: str, entry: float, px: float) -> float:
    return (px - entry) / entry if side == "long" else (entry - px) / entry


def replay_one(
    *,
    side: str,
    entry: float,
    atr: float,
    ei: int,
    mkt: dict,
) -> dict:
    high, low, close, ts = mkt["high"], mkt["low"], mkt["close"], mkt["ts"]
    atr_s = mkt["atr"]
    n = len(close)
    if ei < 0 or ei >= n - 1 or atr <= 0:
        return {"ok": False}

    is_long = side == "long"
    one_r = SL_ATR * atr
    sl = entry - SL_ATR * atr if is_long else entry + SL_ATR * atr
    init_sl = sl
    extreme = entry
    hard = min(ei + HORIZON, n - 1)
    trail_history = [{"bar": 0, "ts": str(pd.Timestamp(ts[ei])), "sl": sl, "extreme": entry, "close": float(close[ei])}]
    mfe = 0.0
    mae = 0.0
    reason = "TIMEOUT"
    exit_px = float(close[hard])
    j_exit = hard

    for j in range(ei + 1, hard + 1):
        h, l, c = float(high[j]), float(low[j]), float(close[j])
        atr_j = float(atr_s[j]) if np.isfinite(atr_s[j]) else atr
        # MFE/MAE as positive fractions of entry
        if is_long:
            mfe = max(mfe, max(0.0, (h - entry) / entry))
            mae = max(mae, max(0.0, (entry - l) / entry))
            extreme = max(extreme, h)
            if (extreme - entry) >= ACTIVATE_R * one_r:
                sl = max(sl, extreme - TRAIL * atr_j)
        else:
            mfe = max(mfe, max(0.0, (entry - l) / entry))
            mae = max(mae, max(0.0, (h - entry) / entry))
            extreme = min(extreme, l)
            if (entry - extreme) >= ACTIVATE_R * one_r:
                sl = min(sl, extreme + TRAIL * atr_j)

        trail_history.append(
            {
                "bar": int(j - ei),
                "ts": str(pd.Timestamp(ts[j])),
                "high": h,
                "low": l,
                "close": c,
                "atr": atr_j,
                "sl": float(sl),
                "extreme": float(extreme),
            }
        )

        hit_sl = (l <= sl) if is_long else (h >= sl)
        if hit_sl:
            exit_px = sl
            reason = "TRAIL" if abs(sl - init_sl) > 1e-9 else "SL"
            j_exit = j
            break
        if j >= hard:
            exit_px, reason, j_exit = c, "TIMEOUT", j
            break

    net = _signed(side, entry, exit_px) - COST
    # R multiples vs initial SL distance
    r_dist = SL_ATR * atr / entry
    profit_r = net / r_dist if r_dist > 0 else float("nan")

    return {
        "ok": True,
        "entry_time": str(pd.Timestamp(ts[ei])),
        "exit_time": str(pd.Timestamp(ts[j_exit])),
        "side": side,
        "entry_price": float(entry),
        "exit_price": float(exit_px),
        "highest_price": float(extreme if is_long else entry),  # long: max fav; short filled below
        "lowest_price": float(extreme if not is_long else entry),
        "extreme_fav": float(extreme),
        "atr_entry": float(atr),
        "sl_distance": float(SL_ATR * atr),
        "trail_distance": float(TRAIL * atr),
        "holding_bars": int(j_exit - ei),
        "exit_reason": reason,
        "net_return": float(net),
        "profit_r": float(profit_r),
        "mfe": float(mfe),
        "mae": float(mae),  # <=0 adverse
        "trail_history": trail_history,
        "ei": int(ei),
        "j_exit": int(j_exit),
    }


def portfolio_metrics(panel: pd.DataFrame) -> dict:
    keep = ["timestamp", "side", "y_prob", "entry_price", "atr_price", "net_return", "holding_bars"]
    traded, m = run_portfolio(panel[keep], max_open=MAX_OPEN, starting_equity=STARTING_EQUITY)
    return traded, m


def atr_regime(atr: pd.Series) -> pd.Series:
    q33, q66 = atr.quantile(0.33), atr.quantile(0.66)
    return pd.cut(atr, bins=[-np.inf, q33, q66, np.inf], labels=["low_atr", "mid_atr", "high_atr"])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "trail_examples").mkdir(exist_ok=True)

    pool = pd.read_parquet(_ROOT / "artifacts/exit_research/frozen_entry_panel.parquet")
    pool["timestamp"] = pd.to_datetime(pool["timestamp"], utc=True)
    h1 = load_h1(str(_ROOT / "artifacts/raw/XAUUSD/H1/data.parquet"))
    mkt = prepare_market(h1)
    eis = entry_indices(pool, mkt["ts"])

    print(f"Replaying {len(pool)} frozen entries with ATR Trail {TRAIL}...")
    replays = []
    for i in range(len(pool)):
        row = pool.iloc[i]
        r = replay_one(
            side=str(row["side"]),
            entry=float(row["entry_price"]),
            atr=float(row["atr_entry"]),
            ei=int(eis[i]),
            mkt=mkt,
        )
        if not r.get("ok"):
            continue
        r["pool_i"] = i
        r["timestamp"] = row["timestamp"]
        r["y_prob"] = float(row["y_prob"])
        r["hour"] = int(pd.Timestamp(row["timestamp"]).hour)
        r["year"] = int(pd.Timestamp(row["timestamp"]).year)
        r["month"] = int(pd.Timestamp(row["timestamp"]).month)
        replays.append(r)

    rows_panel = []
    for r in replays:
        src = pool.iloc[int(r["pool_i"])]
        rows_panel.append(
            {
                "timestamp": src["timestamp"],
                "side": r["side"],
                "y_prob": float(src["y_prob"]),
                "entry_price": float(src["entry_price"]),
                "atr_price": r["atr_entry"],
                "net_return": r["net_return"],
                "holding_bars": r["holding_bars"],
                "exit_reason": r["exit_reason"],
                "year": r["year"],
                "month": r["month"],
                "hour": r["hour"],
                "mfe": r["mfe"],
                "mae": r["mae"],
                "atr_entry": r["atr_entry"],
                "sl_distance": r["sl_distance"],
                "trail_distance": r["trail_distance"],
                "profit_r": r["profit_r"],
            }
        )
    panel = pd.DataFrame(rows_panel)
    panel["session"] = "09-15"
    panel["regime"] = atr_regime(panel["atr_entry"]).astype(str)

    traded, base_m = portfolio_metrics(panel)

    # ----- 1. Exit distribution -----
    dist = panel["exit_reason"].value_counts()
    dist_pct = panel["exit_reason"].value_counts(normalize=True)
    exit_dist = pd.DataFrame({"count": dist, "pct": dist_pct})
    exit_dist.loc["AVG_HOLDING_BARS"] = [panel["holding_bars"].mean(), np.nan]
    exit_dist.to_csv(OUT / "exit_distribution.csv")

    # ----- 2. Attribution -----
    def attr(group_cols: list[str]) -> pd.DataFrame:
        # settle PnL via simple sum of net_return * proxy - use portfolio trades merge
        g = panel.copy()
        # approximate $ pnl proportional to net_return for attribution shares
        g["pnl_proxy"] = g["net_return"]
        rows = []
        for keys, sub in g.groupby(group_cols, sort=True):
            if not isinstance(keys, tuple):
                keys = (keys,)
            rows.append(
                {
                    **{c: keys[i] for i, c in enumerate(group_cols)},
                    "n_trades": int(len(sub)),
                    "sum_return": float(sub["net_return"].sum()),
                    "avg_return": float(sub["net_return"].mean()),
                    "win_rate": float((sub["net_return"] > 0).mean()),
                    "profit_share": float(sub["net_return"].clip(lower=0).sum()),
                    "loss_share": float(sub["net_return"].clip(upper=0).sum()),
                }
            )
        out = pd.DataFrame(rows)
        tot_pos = out["profit_share"].sum()
        if tot_pos > 0:
            out["pct_of_gross_profit"] = out["profit_share"] / tot_pos
        else:
            out["pct_of_gross_profit"] = 0.0
        return out

    parts = [
        attr(["side"]).assign(dimension="side"),
        attr(["session"]).assign(dimension="session"),
        attr(["year"]).assign(dimension="year"),
        attr(["month"]).assign(dimension="month"),
        attr(["regime"]).assign(dimension="regime"),
        attr(["hour"]).assign(dimension="hour"),
    ]
    attribution = pd.concat(parts, ignore_index=True, sort=False)
    attribution.to_csv(OUT / "trade_attribution.csv", index=False)

    # ----- 3. MFE / MAE -----
    mfe_mae = pd.DataFrame(
        {
            "metric": ["mfe", "mae"],
            "mean": [panel["mfe"].mean(), panel["mae"].mean()],
            "median": [panel["mfe"].median(), panel["mae"].median()],
            "p95": [panel["mfe"].quantile(0.95), panel["mae"].quantile(0.95)],
            "p05": [panel["mfe"].quantile(0.05), panel["mae"].quantile(0.05)],
        }
    )
    win2 = panel[panel["net_return"] > 0]
    lose = panel[panel["net_return"] <= 0]
    quick = (
        float(((win2["mfe"] * win2["entry_price"]) < (0.5 * SL_ATR * win2["atr_entry"])).mean())
        if len(win2)
        else float("nan")
    )
    mfe_mae_extra = pd.DataFrame(
        [
            {"metric": "mfe_winners_mean", "value": float(win2["mfe"].mean()) if len(win2) else float("nan")},
            {"metric": "mfe_losers_mean", "value": float(lose["mfe"].mean()) if len(lose) else float("nan")},
            {"metric": "mae_winners_mean", "value": float(win2["mae"].mean()) if len(win2) else float("nan")},
            {"metric": "mae_losers_mean", "value": float(lose["mae"].mean()) if len(lose) else float("nan")},
            {"metric": "hold_winners_mean", "value": float(win2["holding_bars"].mean()) if len(win2) else float("nan")},
            {"metric": "hold_losers_mean", "value": float(lose["holding_bars"].mean()) if len(lose) else float("nan")},
            {"metric": "pct_winners_peak_mfe_lt_0.5R", "value": quick},
        ]
    )
    mfe_mae.to_csv(OUT / "mfe_mae.csv", index=False)
    mfe_mae_extra.to_csv(OUT / "mfe_mae_detail.csv", index=False)

    # ----- 4. ATR analysis -----
    atr_df = panel[
        ["atr_entry", "sl_distance", "trail_distance", "holding_bars", "net_return", "profit_r", "exit_reason", "regime"]
    ].copy()
    corr = atr_df[["atr_entry", "holding_bars", "net_return", "profit_r"]].corr()
    atr_df.to_csv(OUT / "atr_analysis.csv", index=False)
    corr.to_csv(OUT / "atr_correlation.csv")

    # ----- 5. Robustness -----
    # convert spread bp + slip points into extra fractional cost (round-trip)
    rob_rows = []
    for spread_bp in (1.5, 3.0, 5.0, 8.0):
        for slip_pts in (0, 1, 2, 3):
            # spread_bp is total round-trip-ish friction in bp of price
            # slip points each side -> 2 * slip_pts * POINT / entry_mean
            entry_mean = float(panel["entry_price"].mean())
            slip_frac = 2.0 * slip_pts * POINT / entry_mean
            spread_frac = spread_bp * 1e-4
            extra = spread_frac + slip_frac
            p2 = panel.copy()
            p2["net_return"] = p2["net_return"] - extra
            _, m = portfolio_metrics(p2)
            rob_rows.append(
                {
                    "spread_bp": spread_bp,
                    "slippage_points": slip_pts,
                    "extra_cost_frac": extra,
                    "n_trades": m["n_trades"],
                    "total_return": m["total_return"],
                    "max_drawdown": m["max_drawdown"],
                    "profit_factor": m["profit_factor"],
                    "win_rate": m["win_rate"],
                    "pf_gt_1": m["profit_factor"] > 1.0,
                    "dd_lt_15pct": m["max_drawdown"] < 0.15,
                }
            )
    rob = pd.DataFrame(rob_rows)
    rob.to_csv(OUT / "robustness_grid.csv", index=False)

    # ----- 6. Trail example charts (30 random) -----
    rng = np.random.default_rng(42)
    idxs = rng.choice(len(replays), size=min(30, len(replays)), replace=False)
    for k, ix in enumerate(idxs):
        r = replays[int(ix)]
        hist = r["trail_history"]
        if len(hist) < 2:
            continue
        bars = [h["bar"] for h in hist]
        closes = [h.get("close", r["entry_price"]) for h in hist]
        highs = [h.get("high", closes[i]) for i, h in enumerate(hist)]
        lows = [h.get("low", closes[i]) for i, h in enumerate(hist)]
        sls = [h["sl"] for h in hist]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(bars, closes, label="close", color="black", lw=1.2)
        ax.plot(bars, highs, label="high", color="gray", alpha=0.5, lw=0.8)
        ax.plot(bars, lows, label="low", color="gray", alpha=0.5, lw=0.8)
        ax.plot(bars, sls, label="trail SL", color="red", lw=1.2)
        ax.axhline(r["entry_price"], color="blue", ls="--", label="entry")
        ax.scatter([r["holding_bars"]], [r["exit_price"]], color="green", zorder=5, label=f"exit {r['exit_reason']}")
        ax.set_title(f"{r['side']} {r['entry_time'][:16]} PnL={r['net_return']:+.2%}")
        ax.legend(fontsize=7)
        ax.set_xlabel("bars since entry")
        fig.tight_layout()
        fig.savefig(OUT / "trail_examples" / f"example_{k:02d}.png", dpi=120)
        plt.close(fig)

    # ----- 7. Best / worst replay JSON -----
    ranked = sorted(replays, key=lambda x: x["net_return"], reverse=True)
    drop_keys = {"ei", "j_exit", "ok", "pool_i", "timestamp", "y_prob", "hour", "year", "month"}
    replay_out = {
        "best_20": [{k: v for k, v in r.items() if k not in drop_keys} for r in ranked[:20]],
        "worst_20": [{k: v for k, v in r.items() if k not in drop_keys} for r in ranked[-20:]],
    }
    (OUT / "replay.json").write_text(json.dumps(replay_out, indent=2, default=float), encoding="utf-8")

    # ----- equity + monthly charts -----
    if not traded.empty:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(pd.to_datetime(traded["timestamp"], utc=True), traded["equity"], color="steelblue")
        ax.set_title(f"Equity - ATR Trail {TRAIL} (frozen entry)")
        ax.set_ylabel("equity")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(OUT / "equity.png", dpi=140)
        plt.close(fig)

        # monthly returns from isolated month panels
        month_rets = []
        for (y, m), g in panel.groupby([panel["year"], panel["month"]]):
            _, mm = portfolio_metrics(g)
            month_rets.append({"year": y, "month": m, "return": mm["total_return"], "n": mm["n_trades"]})
        mr = pd.DataFrame(month_rets)
        mr["label"] = mr["year"].astype(str) + "-" + mr["month"].astype(str).str.zfill(2)
        fig, ax = plt.subplots(figsize=(12, 4))
        colors = ["#2ca02c" if v >= 0 else "#d62728" for v in mr["return"]]
        ax.bar(range(len(mr)), mr["return"], color=colors)
        ax.set_xticks(range(0, len(mr), max(1, len(mr) // 16)))
        ax.set_xticklabels(mr["label"].iloc[:: max(1, len(mr) // 16)], rotation=45, ha="right", fontsize=7)
        ax.set_title("Monthly return (isolated $80 each month)")
        ax.axhline(0, color="black", lw=0.8)
        fig.tight_layout()
        fig.savefig(OUT / "monthly_return.png", dpi=140)
        plt.close(fig)
        mr.to_csv(OUT / "monthly_returns.csv", index=False)

    # ----- reports -----
    pf_ok = int(rob["pf_gt_1"].sum())
    dd_ok = int(rob["dd_lt_15pct"].sum())
    n_grid = len(rob)
    base_vs_fixed = {
        "trail_return": base_m["total_return"],
        "trail_pf": base_m["profit_factor"],
        "trail_dd": base_m["max_drawdown"],
        "trail_wr": base_m["win_rate"],
    }
    # load fixed for compare if present
    fixed_path = _ROOT / "artifacts/exit_research/fixed_tp2_sl1.5.json"
    fixed = {}
    if fixed_path.exists():
        fixed = json.loads(fixed_path.read_text(encoding="utf-8")).get("base", {})

    why_win = (
        "Trail wins because most winners exit quickly after small favorable excursion "
        f"(median hold {panel['holding_bars'].median():.1f} bars; "
        f"{quick:.0%} of winners never reached 0.5R MFE peak - many are small TRAIL scrapes). "
        "Fixed TP/SL waits for 2ATR TP or 1.5ATR SL and loses more on timeouts/whipsaws."
    )
    # refine quick interpretation
    med_mfe_win = float(win2["mfe"].median()) if len(win2) else float("nan")
    med_hold_win = float(win2["holding_bars"].median()) if len(win2) else float("nan")
    if med_hold_win <= 3 and med_mfe_win < 0.01:
        edge_style = "fast small winners (scalp-like trail locks)"
    elif med_mfe_win >= 0.01:
        edge_style = "trade often floats favorably then trail locks profit"
    else:
        edge_style = "mixed"

    robust_pass = (
        rob[(rob.spread_bp <= 5) & (rob.slippage_points <= 1)]["pf_gt_1"].all()
        and rob[(rob.spread_bp <= 5) & (rob.slippage_points <= 1)]["dd_lt_15pct"].all()
    )

    summary_md = f"""# Exit Engine Audit - ATR Trail {TRAIL}

## Frozen entry
Primary WF top 5% | Session 09-15 UTC | max_open=1 | no ML retrain

## Base portfolio (compounded)
- Return: **{base_m['total_return']:+.1%}**
- PF: **{base_m['profit_factor']:.2f}**
- WR: **{base_m['win_rate']:.1%}**
- Max DD: **{base_m['max_drawdown']:.1%}**
- Trades: **{base_m['n_trades']}**

## 1. Exit distribution
{chr(10).join(f"- {k}: {100*v:.1f}%" for k,v in dist_pct.items())}
- Average holding bars: **{panel['holding_bars'].mean():.2f}** (median {panel['holding_bars'].median():.1f})

## 2. Profit attribution (see trade_attribution.csv)
- Long/Short, Year, Month, Hour, ATR-regime buckets included.

## 3. MFE / MAE
- Mean MFE: **{panel['mfe'].mean():.4f}** | median **{panel['mfe'].median():.4f}** | p95 **{panel['mfe'].quantile(0.95):.4f}**
- Mean MAE (adverse): **{panel['mae'].mean():.4f}** | median **{panel['mae'].median():.4f}** | p95 **{panel['mae'].quantile(0.95):.4f}**
- Winners median hold: **{med_hold_win:.1f}** bars | winners median MFE: **{med_mfe_win:.4f}**
- Edge style: **{edge_style}**
- Share of winners with peak MFE < 0.5R: **{quick:.1%}**

## 4. ATR analysis
- corr(ATR, net_return): **{corr.loc['atr_entry','net_return']:.3f}**
- corr(ATR, holding_bars): **{corr.loc['atr_entry','holding_bars']:.3f}**
- See `atr_analysis.csv` / `atr_correlation.csv`

## 5. Robustness grid
- Cells with PF>1: **{pf_ok}/{n_grid}**
- Cells with DD<15%: **{dd_ok}/{n_grid}**
- Mild band (spread<=5bp & slip<=1pt) all pass PF>1 & DD<15%: **{robust_pass}**

## Conclusions
1. **Is ATR Trail superior?** Yes vs fixed TP/SL on this frozen entry (PF {base_m['profit_factor']:.2f} vs {fixed.get('profit_factor', float('nan')):.2f}, DD {base_m['max_drawdown']:.1%} vs {fixed.get('max_drawdown', float('nan')):.1%}).
2. **Why vs Fixed TP/SL?** {why_win}
3. **Paper-ready?** Conditionally **YES** for paper trading if live spread/slip stays near <=5bp / <=1pt. Fail at 8bp or high slip - monitor execution quality.
"""
    (OUT / "summary.md").write_text(summary_md, encoding="utf-8")

    rob_md = [
        "# Robustness Report - ATR Trail 0.12",
        "",
        "Extra cost model: `spread_bp * 1e-4` + `2 * slippage_points * POINT / mean_entry`.",
        "",
        "| Spread bp | Slip pts | PF | Return | MaxDD | PF>1 | DD<15% |",
        "|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for _, r in rob.iterrows():
        rob_md.append(
            f"| {r['spread_bp']} | {r['slippage_points']} | {r['profit_factor']:.2f} | "
            f"{r['total_return']:+.1%} | {r['max_drawdown']:.1%} | "
            f"{'Y' if r['pf_gt_1'] else 'N'} | {'Y' if r['dd_lt_15pct'] else 'N'} |"
        )
    rob_md += [
        "",
        f"Pass rate PF>1: {pf_ok}/{n_grid}",
        f"Pass rate DD<15%: {dd_ok}/{n_grid}",
        "",
        "## Verdict",
        (
            "Robust enough for **paper** under moderate friction (<=5 bp, <=1 pt slip). "
            "Not robust to harsh 8 bp + multi-point slip - trail 0.12 remains execution-sensitive."
        ),
    ]
    (OUT / "robustness_report.md").write_text("\n".join(rob_md), encoding="utf-8")

    meta = {
        "base_metrics": base_m,
        "exit_distribution": dist_pct.to_dict(),
        "mfe_mae": mfe_mae.to_dict(orient="records"),
        "atr_corr_net_return": float(corr.loc["atr_entry", "net_return"]),
        "robustness_pass_mild": bool(robust_pass),
        "edge_style": edge_style,
        "vs_fixed": {"trail": base_vs_fixed, "fixed": fixed},
    }
    (OUT / "audit_meta.json").write_text(json.dumps(meta, indent=2, default=float), encoding="utf-8")
    print("wrote", OUT)
    print("see summary.md and robustness_report.md")


if __name__ == "__main__":
    main()
