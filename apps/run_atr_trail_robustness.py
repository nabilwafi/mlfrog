"""Sprint 28 — ATR Trail Robustness Research (entry frozen).

Only Exit Engine ATR multiplier varies. No ML retrain / no entry changes.
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
from production.paper.barrier_resim import load_h1
from research.portfolio_heat import STARTING_EQUITY
from research.position_mgmt import COST, SL_ATR
from research.position_mgmt.services.paths import entry_indices, prepare_market

OUT = _ROOT / "artifacts/exit_research"
TRAILS = (0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.25)
ACTIVATE_R = 0.5
HORIZON = 16
MAX_OPEN = 1
COST_STRESS = {
    "base": 0.0,
    "plus_3bp": 3.0e-4,
    "plus_5bp": 5.0e-4,
    "plus_8bp": 8.0e-4,
}
KEEP = ["timestamp", "side", "y_prob", "entry_price", "atr_price", "net_return", "holding_bars"]


def _signed(side: str, entry: float, px: float) -> float:
    return (px - entry) / entry if side == "long" else (entry - px) / entry


def replay_trail(
    *,
    side: str,
    entry: float,
    atr: float,
    ei: int,
    mkt: dict,
    trail: float,
) -> dict | None:
    high, low, close = mkt["high"], mkt["low"], mkt["close"]
    atr_s = mkt["atr"]
    n = len(close)
    if ei < 0 or ei >= n - 1 or atr <= 0 or entry <= 0:
        return None

    is_long = side == "long"
    one_r = SL_ATR * atr
    sl = entry - SL_ATR * atr if is_long else entry + SL_ATR * atr
    init_sl = sl
    extreme = entry
    hard = min(ei + HORIZON, n - 1)
    mfe = mae = 0.0
    reason = "TIMEOUT"
    exit_px = float(close[hard])
    j_exit = hard

    for j in range(ei + 1, hard + 1):
        h, l, c = float(high[j]), float(low[j]), float(close[j])
        atr_j = float(atr_s[j]) if np.isfinite(atr_s[j]) else atr
        if is_long:
            mfe = max(mfe, max(0.0, (h - entry) / entry))
            mae = max(mae, max(0.0, (entry - l) / entry))
            extreme = max(extreme, h)
            if (extreme - entry) >= ACTIVATE_R * one_r:
                sl = max(sl, extreme - trail * atr_j)
        else:
            mfe = max(mfe, max(0.0, (entry - l) / entry))
            mae = max(mae, max(0.0, (h - entry) / entry))
            extreme = min(extreme, l)
            if (entry - extreme) >= ACTIVATE_R * one_r:
                sl = min(sl, extreme + trail * atr_j)

        hit_sl = (l <= sl) if is_long else (h >= sl)
        if hit_sl:
            exit_px = sl
            reason = "TRAIL" if abs(sl - init_sl) > 1e-9 else "SL"
            j_exit = j
            break
        if j >= hard:
            exit_px, reason, j_exit = c, "TIMEOUT", j
            break

    return {
        "net_return": float(_signed(side, entry, exit_px) - COST),
        "holding_bars": int(j_exit - ei),
        "exit_reason": reason,
        "mfe": float(mfe),
        "mae": float(mae),
    }


def metrics_bundle(panel: pd.DataFrame, *, extra_cost: float = 0.0) -> dict:
    p = panel.copy()
    if extra_cost:
        p["net_return"] = p["net_return"] - float(extra_cost)
    traded, m = run_portfolio(p[KEEP], max_open=MAX_OPEN, starting_equity=STARTING_EQUITY)
    if traded.empty or m.get("n_trades", 0) == 0:
        return {"n_trades": 0}

    rets = p["net_return"].to_numpy(dtype=float)
    wins = rets[rets > 0]
    losses = rets[rets <= 0]
    ts = pd.to_datetime(p["timestamp"], utc=True)
    years = (ts.max() - ts.min()).days / 365.25
    cagr = float((1.0 + m["total_return"]) ** (1.0 / years) - 1.0) if years > 0 else float("nan")

    by = []
    for y, g in p.groupby(ts.dt.year, sort=True):
        _, ym = run_portfolio(g[KEEP], max_open=MAX_OPEN, starting_equity=STARTING_EQUITY)
        if ym.get("n_trades", 0) == 0:
            continue
        by.append(
            {
                "year": int(y),
                "n_trades": int(ym["n_trades"]),
                "return": float(ym["total_return"]),
                "pf": float(ym["profit_factor"]),
                "win_rate": float(ym["win_rate"]),
                "max_dd": float(ym["max_drawdown"]),
            }
        )

    mix = p["exit_reason"].value_counts(normalize=True).to_dict()
    w = p[p["net_return"] > 0]
    l = p[p["net_return"] <= 0]
    return {
        "n_trades": int(m["n_trades"]),
        "total_return": float(m["total_return"]),
        "max_drawdown": float(m["max_drawdown"]),
        "profit_factor": float(m["profit_factor"]),
        "win_rate": float(m["win_rate"]),
        "cagr": cagr,
        "avg_holding_bars": float(p["holding_bars"].mean()),
        "median_holding_bars": float(p["holding_bars"].median()),
        "avg_winner": float(np.mean(wins)) if len(wins) else 0.0,
        "avg_loser": float(np.mean(losses)) if len(losses) else 0.0,
        "expectancy": float(np.mean(rets)) if len(rets) else float("nan"),
        "exit_trail_pct": float(mix.get("TRAIL", 0.0)),
        "exit_sl_pct": float(mix.get("SL", 0.0)),
        "exit_timeout_pct": float(mix.get("TIMEOUT", 0.0)),
        "mean_mfe": float(p["mfe"].mean()),
        "median_mfe": float(p["mfe"].median()),
        "mean_mae": float(p["mae"].mean()),
        "median_mae": float(p["mae"].median()),
        "winners_hold_mean": float(w["holding_bars"].mean()) if len(w) else float("nan"),
        "losers_hold_mean": float(l["holding_bars"].mean()) if len(l) else float("nan"),
        "by_year": by,
    }


def rank_score(row: dict) -> float:
    """Production-ready ranking: cost robustness first, return last."""
    pf3 = float(row["pf_3bp"])
    pf5 = float(row["pf_5bp"])
    pf8 = float(row["pf_8bp"])
    dd = float(row["max_drawdown"])
    hold = float(row["avg_holding_bars"])
    yrs_pos = float(row["pct_years_positive"])
    yrs_pf = float(row["pct_years_pf_gt_1"])
    pf_std = float(row["yearly_pf_std"])
    ret = float(row["total_return"])

    cost = (
        0.35 * min(max(pf3, 0.0), 2.0)
        + 0.40 * min(max(pf5, 0.0), 2.0)
        + 0.25 * min(max(pf8, 0.0), 2.0)
        + (0.15 if pf5 >= 1.05 else 0.0)
        + (0.20 if pf8 >= 1.0 else 0.0)
        + (0.10 if pf8 >= 1.05 else 0.0)
    )
    # yearly PF stability: high share of PF>1 years, low PF std
    stability = 0.7 * yrs_pf + 0.3 * (1.0 - min(pf_std, 1.0))
    dd_score = 1.0 - min(dd, 0.25) / 0.25
    # prefer ~2-5 bar holds (scalp OK but not pathologically 1-bar)
    if 2.0 <= hold <= 5.0:
        hold_score = 1.0
    elif hold < 2.0:
        hold_score = max(0.0, hold / 2.0)
    else:
        hold_score = max(0.0, 1.0 - (hold - 5.0) / 10.0)
    ret_score = min(max(ret, -1.0), 3.0) / 3.0

    return float(
        0.35 * cost
        + 0.25 * stability
        + 0.15 * dd_score
        + 0.10 * hold_score
        + 0.10 * yrs_pos
        + 0.05 * ret_score
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pool = pd.read_parquet(OUT / "frozen_entry_panel.parquet")
    pool["timestamp"] = pd.to_datetime(pool["timestamp"], utc=True)
    h1 = load_h1(str(_ROOT / "artifacts/raw/XAUUSD/H1/data.parquet"))
    mkt = prepare_market(h1)
    eis = entry_indices(pool, mkt["ts"])
    print(f"Frozen entries: {len(pool)}")

    comparison_rows = []
    yearly_rows = []
    cost_rows = []
    behavior_rows = []
    ranking_rows = []
    details = {}

    for trail in TRAILS:
        print(f"Replaying ATR Trail {trail:.2f}...")
        rows = []
        for i in range(len(pool)):
            src = pool.iloc[i]
            r = replay_trail(
                side=str(src["side"]),
                entry=float(src["entry_price"]),
                atr=float(src["atr_entry"]),
                ei=int(eis[i]),
                mkt=mkt,
                trail=trail,
            )
            if r is None:
                continue
            rows.append(
                {
                    "timestamp": src["timestamp"],
                    "side": src["side"],
                    "y_prob": float(src["y_prob"]),
                    "entry_price": float(src["entry_price"]),
                    "atr_price": float(src["atr_entry"]),
                    **r,
                }
            )
        panel = pd.DataFrame(rows)
        base = metrics_bundle(panel, extra_cost=0.0)
        stress = {k: metrics_bundle(panel, extra_cost=v) for k, v in COST_STRESS.items()}

        by = base.get("by_year") or []
        best = max(by, key=lambda y: y["return"]) if by else None
        worst = min(by, key=lambda y: y["return"]) if by else None
        pf_yrs = [y["pf"] for y in by]
        pos_yrs = [y for y in by if y["return"] > 0]
        pf_gt1 = [y for y in by if y["pf"] > 1.0]

        row = {
            "atr_trail": trail,
            "n_trades": base["n_trades"],
            "total_return": base["total_return"],
            "profit_factor": base["profit_factor"],
            "win_rate": base["win_rate"],
            "max_drawdown": base["max_drawdown"],
            "cagr": base["cagr"],
            "avg_holding_bars": base["avg_holding_bars"],
            "median_holding_bars": base["median_holding_bars"],
            "avg_winner": base["avg_winner"],
            "avg_loser": base["avg_loser"],
            "expectancy": base["expectancy"],
            "pf_3bp": stress["plus_3bp"]["profit_factor"],
            "ret_3bp": stress["plus_3bp"]["total_return"],
            "dd_3bp": stress["plus_3bp"]["max_drawdown"],
            "wr_3bp": stress["plus_3bp"]["win_rate"],
            "pf_5bp": stress["plus_5bp"]["profit_factor"],
            "ret_5bp": stress["plus_5bp"]["total_return"],
            "dd_5bp": stress["plus_5bp"]["max_drawdown"],
            "wr_5bp": stress["plus_5bp"]["win_rate"],
            "pf_8bp": stress["plus_8bp"]["profit_factor"],
            "ret_8bp": stress["plus_8bp"]["total_return"],
            "dd_8bp": stress["plus_8bp"]["max_drawdown"],
            "wr_8bp": stress["plus_8bp"]["win_rate"],
            "pct_years_positive": len(pos_yrs) / max(len(by), 1),
            "pct_years_pf_gt_1": len(pf_gt1) / max(len(by), 1),
            "yearly_pf_std": float(np.std(pf_yrs)) if pf_yrs else float("nan"),
            "best_year": best["year"] if best else None,
            "best_year_return": best["return"] if best else None,
            "worst_year": worst["year"] if worst else None,
            "worst_year_return": worst["return"] if worst else None,
        }
        row["robustness_score"] = rank_score(row)
        comparison_rows.append(row)
        ranking_rows.append(row)

        for y in by:
            yearly_rows.append({"atr_trail": trail, **y})

        for sname, sm in stress.items():
            cost_rows.append(
                {
                    "atr_trail": trail,
                    "stress": sname,
                    "extra_bp": COST_STRESS[sname] * 1e4,
                    "profit_factor": sm["profit_factor"],
                    "total_return": sm["total_return"],
                    "max_drawdown": sm["max_drawdown"],
                    "win_rate": sm["win_rate"],
                    "n_trades": sm["n_trades"],
                }
            )

        behavior_rows.append(
            {
                "atr_trail": trail,
                "exit_trail_pct": base["exit_trail_pct"],
                "exit_sl_pct": base["exit_sl_pct"],
                "exit_timeout_pct": base["exit_timeout_pct"],
                "mean_mfe": base["mean_mfe"],
                "median_mfe": base["median_mfe"],
                "mean_mae": base["mean_mae"],
                "median_mae": base["median_mae"],
                "winners_hold_mean": base["winners_hold_mean"],
                "losers_hold_mean": base["losers_hold_mean"],
                "avg_holding_bars": base["avg_holding_bars"],
                "median_holding_bars": base["median_holding_bars"],
            }
        )
        details[str(trail)] = {"base": base, "stress": stress, "score": row["robustness_score"]}

    comparison = pd.DataFrame(comparison_rows).sort_values("atr_trail")
    ranking = pd.DataFrame(ranking_rows).sort_values("robustness_score", ascending=False)
    yearly = pd.DataFrame(yearly_rows)
    cost = pd.DataFrame(cost_rows)
    behavior = pd.DataFrame(behavior_rows).sort_values("atr_trail")

    comparison.to_csv(OUT / "comparison.csv", index=False)
    ranking.to_csv(OUT / "ranking.csv", index=False)
    yearly.to_csv(OUT / "yearly_report.csv", index=False)
    cost.to_csv(OUT / "cost_stress.csv", index=False)
    behavior.to_csv(OUT / "trade_behavior.csv", index=False)
    (OUT / "atr_trail_robustness_detail.json").write_text(
        json.dumps(details, indent=2, default=float), encoding="utf-8"
    )

    winner = ranking.iloc[0]
    w_trail = float(winner["atr_trail"])

    # losers brief notes
    loser_notes = []
    for _, r in ranking.iloc[1:].iterrows():
        reasons = []
        if r["pf_8bp"] < 1.0:
            reasons.append(f"dies at +8bp (PF={r['pf_8bp']:.2f})")
        if r["pf_5bp"] < winner["pf_5bp"] - 0.05:
            reasons.append(f"weaker at +5bp (PF={r['pf_5bp']:.2f})")
        if r["max_drawdown"] > winner["max_drawdown"] + 0.01:
            reasons.append(f"higher DD ({r['max_drawdown']:.1%})")
        if r["pct_years_pf_gt_1"] < winner["pct_years_pf_gt_1"] - 0.05:
            reasons.append(f"fewer PF>1 years ({r['pct_years_pf_gt_1']:.0%})")
        if not reasons:
            reasons.append("lower composite robustness score")
        loser_notes.append(f"- **{r['atr_trail']:.2f}**: " + "; ".join(reasons))

    sens_lines = [
        "| ATR | Return | PF | WR | DD | Hold | PF@3bp | PF@5bp | PF@8bp |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in comparison.iterrows():
        sens_lines.append(
            f"| {r['atr_trail']:.2f} | {r['total_return']:+.1%} | {r['profit_factor']:.2f} | "
            f"{r['win_rate']:.1%} | {r['max_drawdown']:.1%} | {r['avg_holding_bars']:.2f} | "
            f"{r['pf_3bp']:.2f} | {r['pf_5bp']:.2f} | {r['pf_8bp']:.2f} |"
        )

    paper_ok = bool(winner["pf_5bp"] >= 1.05 and winner["max_drawdown"] < 0.10 and winner["pf_8bp"] >= 0.95)
    paper_line = (
        "Conditionally **YES** for paper trading (monitor live spread; fail closed if friction approaches 8bp)."
        if paper_ok or winner["pf_5bp"] >= 1.0
        else "Not yet - cost robustness too weak."
    )

    summary = f"""# Sprint 28 - ATR Trail Robustness Research

## Frozen entry
Primary WF top 5% | Session 09-15 UTC | max_open=1 | production sizing
Only ATR Trail multiplier varies. No ML retrain.

## Sensitivity

{chr(10).join(sens_lines)}

## Ranking (robustness first, return last)

| Rank | ATR | Score | PF | PF@5bp | PF@8bp | DD | Years+ | Return |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(
    f"| {i} | {r['atr_trail']:.2f} | {r['robustness_score']:.3f} | {r['profit_factor']:.2f} | "
    f"{r['pf_5bp']:.2f} | {r['pf_8bp']:.2f} | {r['max_drawdown']:.1%} | "
    f"{r['pct_years_positive']:.0%} | {r['total_return']:+.1%} |"
    for i, (_, r) in enumerate(ranking.iterrows(), 1)
)}

## Recommendation: ATR Trail **{w_trail:.2f}**

### Why it wins
- Best composite robustness score ({winner['robustness_score']:.3f}) under the production selection rule
- Cost: PF@3bp={winner['pf_3bp']:.2f}, PF@5bp={winner['pf_5bp']:.2f}, PF@8bp={winner['pf_8bp']:.2f}
- Drawdown: {winner['max_drawdown']:.1%}
- Yearly: {winner['pct_years_positive']:.0%} positive years, {winner['pct_years_pf_gt_1']:.0%} years with PF>1
- Hold: avg {winner['avg_holding_bars']:.2f} bars (median {winner['median_holding_bars']:.1f})
- Best year {int(winner['best_year'])} ({winner['best_year_return']:+.1%}); worst {int(winner['worst_year'])} ({winner['worst_year_return']:+.1%})

### Why others lose
{chr(10).join(loser_notes)}

### Paper trading ready?
{paper_line}

## Files
- `comparison.csv`, `ranking.csv`, `yearly_report.csv`, `cost_stress.csv`, `trade_behavior.csv`
"""
    (OUT / "summary.md").write_text(summary, encoding="utf-8")
    print("winner ATR Trail", w_trail)
    print("wrote", OUT / "summary.md")


if __name__ == "__main__":
    main()
