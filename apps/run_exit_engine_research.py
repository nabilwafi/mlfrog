"""Sprint XX — Exit Engine Research (entry stack frozen).

Frozen: data, features, primary WF entries (top 5%), session 09-15 UTC,
max_open=1, primary-edge sizing. Only Exit Engine varies. No ML retrain.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from apps.run_exit_engine_hunt import build_entries
from apps.run_primary_loosen_backtest import run_portfolio
from apps.run_yearly_walkforward_backtest import _load_side
from production.paper.barrier_resim import load_h1
from research.portfolio_backtest.services.metrics import max_dd_from_equity, profit_factor_pnl
from research.portfolio_heat import STARTING_EQUITY
from research.position_mgmt import COST, SL_ATR, TP_ATR
from research.position_mgmt.services.paths import entry_indices, prepare_market
from research.position_mgmt.services.simulator import ManageConfig, simulate_panel

OUT = _ROOT / "artifacts/exit_research"
SESSION = (9, 15)
TOP_PCT = 0.05
MAX_OPEN = 1
YEARS = list(range(2015, 2027))
COST_STRESS = {
    "base": 0.0,
    "plus_3bp": 3.0e-4,
    "plus_5bp": 5.0e-4,
    "plus_8bp": 8.0e-4,
}


@dataclass
class PathResult:
    net_return: float
    holding_bars: int
    exit_reason: str


def _signed(side: str, entry: float, exit_px: float) -> float:
    if side == "long":
        return (exit_px - entry) / entry
    return (entry - exit_px) / entry


def walk_exit(
    *,
    side: str,
    entry: float,
    atr: float,
    ei: int,
    mkt: dict,
    mode: str,
    trail: float | None = None,
    chandelier_k: float = 3.0,
    time_bars: int | None = None,
    hybrid_trail: float | None = None,
    horizon: int = 16,
) -> PathResult:
    """Single-trade exit path. mid OHLC; COST applied once at end."""
    high, low, close = mkt["high"], mkt["low"], mkt["close"]
    atr_s = mkt["atr"]
    n = len(close)
    if ei < 0 or ei >= n - 1 or atr <= 0 or entry <= 0:
        return PathResult(-COST, 0, "bad_entry")

    is_long = side == "long"
    one_r = SL_ATR * atr
    sl = entry - SL_ATR * atr if is_long else entry + SL_ATR * atr
    tp = entry + TP_ATR * atr if is_long else entry - TP_ATR * atr
    max_fav, min_fav = entry, entry
    hard = min(ei + (time_bars or horizon), n - 1)
    # chandelier: no fixed TP; initial SL at chandelier level from entry bar
    if mode == "chandelier":
        tp = float("inf") if is_long else float("-inf")
        # start chandelier from entry extreme = entry
        sl = entry - chandelier_k * atr if is_long else entry + chandelier_k * atr

    j_exit, reason, exit_px = hard, "TIMEOUT", float(close[hard])
    for j in range(ei + 1, hard + 1):
        h, l, c = float(high[j]), float(low[j]), float(close[j])
        atr_j = float(atr_s[j]) if np.isfinite(atr_s[j]) else atr
        if is_long:
            max_fav = max(max_fav, h)
        else:
            min_fav = min(min_fav, l)

        if mode in ("trail", "hybrid") and trail is not None:
            if is_long and (max_fav - entry) >= 0.5 * one_r:
                sl = max(sl, max_fav - trail * atr_j)
            elif (not is_long) and (entry - min_fav) >= 0.5 * one_r:
                sl = min(sl, min_fav + trail * atr_j)

        if mode == "hybrid" and hybrid_trail is not None:
            # same as trail param; hybrid keeps TP
            pass

        if mode == "chandelier":
            if is_long:
                sl = max(sl, max_fav - chandelier_k * atr_j)
            else:
                sl = min(sl, min_fav + chandelier_k * atr_j)

        hit_sl = (l <= sl) if is_long else (h >= sl)
        hit_tp = False
        if mode in ("baseline", "hybrid"):
            hit_tp = (h >= tp) if is_long else (l <= tp)
        # pure trail / chandelier: no fixed TP
        if mode == "trail":
            hit_tp = False

        if mode == "time":
            hit_tp = False

        if hit_sl and hit_tp:
            hit_tp = False
        if hit_sl:
            if mode in ("trail", "hybrid"):
                init_sl = entry - SL_ATR * atr if is_long else entry + SL_ATR * atr
                reason = "TRAIL" if abs(sl - init_sl) > 1e-9 else "SL"
            elif mode == "chandelier":
                reason = "CHAND"
            else:
                reason = "SL"
            return PathResult(_signed(side, entry, sl) - COST, j - ei, reason)
        if hit_tp:
            return PathResult(_signed(side, entry, tp) - COST, j - ei, "TP")
        if mode == "time" and j >= ei + int(time_bars or 8):
            return PathResult(_signed(side, entry, c) - COST, j - ei, "TIME")
        if j >= hard:
            j_exit, reason, exit_px = j, "TIMEOUT", c
            break

    # label trail SL better
    return PathResult(_signed(side, entry, exit_px) - COST, max(j_exit - ei, 0), reason)


def simulate_custom(pool: pd.DataFrame, mkt: dict, eis: np.ndarray, **kw) -> pd.DataFrame:
    rows = []
    for i in range(len(pool)):
        row = pool.iloc[i]
        atr = float(row["atr_entry"])
        r = walk_exit(
            side=str(row["side"]),
            entry=float(row["entry_price"]),
            atr=atr,
            ei=int(eis[i]),
            mkt=mkt,
            **kw,
        )
        rows.append(
            {
                "timestamp": row["timestamp"],
                "side": row["side"],
                "y_prob": float(row["y_prob"]),
                "entry_price": float(row["entry_price"]),
                "atr_price": atr,
                "net_return": r.net_return,
                "holding_bars": r.holding_bars,
                "exit_reason": r.exit_reason,
            }
        )
    return pd.DataFrame(rows)


def from_manage(pool: pd.DataFrame, h1: pd.DataFrame, mkt: dict, eis: np.ndarray, cfg: ManageConfig) -> pd.DataFrame:
    sim = simulate_panel(pool, h1, cfg, mkt=mkt, eis=eis)
    out = sim.copy()
    out["atr_price"] = out["atr_entry"]
    out["holding_bars"] = out["holding_bars_sim"]
    out["exit_reason"] = out["exit_reason_sim"]
    # map SL after trail activation → TRAIL for reporting
    return out[
        ["timestamp", "side", "y_prob", "entry_price", "atr_price", "net_return", "holding_bars", "exit_reason"]
    ]


def enrich_exit_labels(panel: pd.DataFrame, *, trailish: bool) -> pd.DataFrame:
    p = panel.copy()
    if trailish:
        p["exit_reason"] = p["exit_reason"].replace({"SL": "TRAIL"})
    return p


def metrics_bundle(panel: pd.DataFrame, *, extra_cost: float = 0.0) -> dict:
    p = panel.copy()
    p["net_return"] = p["net_return"].astype(float) - float(extra_cost)
    keep = ["timestamp", "side", "y_prob", "entry_price", "atr_price", "net_return", "holding_bars"]
    traded, m = run_portfolio(p[keep], max_open=MAX_OPEN, starting_equity=STARTING_EQUITY)
    if traded.empty:
        return {"n_trades": 0}

    pnl = traded["pnl"].to_numpy(dtype=float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    rets = p["net_return"].to_numpy(dtype=float)
    w_rets = rets[rets > 0]
    l_rets = rets[rets <= 0]

    # yearly isolated
    by = []
    for y, g in p.groupby(pd.to_datetime(p["timestamp"], utc=True).dt.year, sort=True):
        _, ym = run_portfolio(g[keep], max_open=MAX_OPEN, starting_equity=STARTING_EQUITY)
        if ym["n_trades"] == 0:
            continue
        by.append(
            {
                "year": int(y),
                "n_trades": ym["n_trades"],
                "return": ym["total_return"],
                "max_dd": ym["max_drawdown"],
                "pf": ym["profit_factor"],
                "win_rate": ym["win_rate"],
            }
        )

    mix = p["exit_reason"].value_counts(normalize=True).to_dict() if "exit_reason" in p.columns else {}
    expectancy = float(np.mean(rets)) if len(rets) else float("nan")

    return {
        "n_trades": int(m["n_trades"]),
        "total_return": float(m["total_return"]),
        "final_equity": float(m["final_equity"]),
        "max_drawdown": float(m["max_drawdown"]),
        "profit_factor": float(m["profit_factor"]),
        "win_rate": float(m["win_rate"]),
        "expectancy": expectancy,
        "avg_win": float(np.mean(w_rets)) if len(w_rets) else 0.0,
        "avg_loss": float(np.mean(l_rets)) if len(l_rets) else 0.0,
        "avg_holding_bars": float(p["holding_bars"].mean()),
        "exit_reason_distribution": {str(k): float(v) for k, v in mix.items()},
        "by_year": by,
        "extra_cost": float(extra_cost),
    }


def robustness_score(base: dict, stress: dict[str, dict]) -> float:
    """Higher = better. Penalize DD; require stress PF."""
    pf = float(base.get("profit_factor") or 0)
    ret = float(base.get("total_return") or 0)
    dd = float(base.get("max_drawdown") or 1)
    wr = float(base.get("win_rate") or 0)
    # stress: PF at +5bp and +8bp
    pf5 = float((stress.get("plus_5bp") or {}).get("profit_factor") or 0)
    pf8 = float((stress.get("plus_8bp") or {}).get("profit_factor") or 0)
    # years positive
    yrs = base.get("by_year") or []
    pos = sum(1 for y in yrs if y.get("return", 0) > 0)
    n_y = max(len(yrs), 1)
    score = (
        0.25 * min(pf, 2.5)
        + 0.20 * min(max(ret, -1), 3.0)
        + 0.20 * (1.0 - min(dd, 0.5) / 0.5)
        + 0.10 * wr
        + 0.15 * min(max(pf5, 0), 2.0)
        + 0.10 * (1.0 if pf5 >= 1.05 else 0.0)
        + 0.05 * (1.0 if pf8 >= 1.0 else 0.0)
        + 0.10 * (pos / n_y)
    )
    return float(score)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("Building frozen entry panel (top5% + sess 09-15)…")
    long_df = _load_side("long")
    short_df = _load_side("short")
    h1 = load_h1(str(_ROOT / "artifacts/raw/XAUUSD/H1/data.parquet"))
    mkt = prepare_market(h1)
    entries = build_entries(long_df, short_df, h1, top_pct=TOP_PCT, years=YEARS)
    e = entries.copy()
    e["hour"] = pd.to_datetime(e["timestamp"], utc=True).dt.hour
    pool = e[(e["hour"] >= SESSION[0]) & (e["hour"] <= SESSION[1])].drop(columns=["hour"]).reset_index(drop=True)
    eis = entry_indices(pool, mkt["ts"])
    pool.to_parquet(OUT / "frozen_entry_panel.parquet", index=False)
    print(f"frozen entries={len(pool)}")

    strategies: list[tuple[str, pd.DataFrame]] = []

    # 1 baseline fixed TP/SL via ManageConfig
    strategies.append(
        (
            "fixed_tp2_sl1.5",
            from_manage(pool, h1, mkt, eis, ManageConfig(name="baseline", family="baseline", horizon=16, max_bars=16)),
        )
    )
    # 2-4 pure ATR trails (no fixed TP) — custom walker
    for t in (0.12, 0.18, 0.25):
        strategies.append(
            (
                f"atr_trail_{t}",
                simulate_custom(pool, mkt, eis, mode="trail", trail=t, horizon=16),
            )
        )
    # 5 chandelier k=3
    strategies.append(
        (
            "chandelier_k3",
            simulate_custom(pool, mkt, eis, mode="chandelier", chandelier_k=3.0, horizon=24),
        )
    )
    # 6 time exit 8 (SL still on)
    strategies.append(
        (
            "time_exit_8",
            from_manage(
                pool,
                h1,
                mkt,
                eis,
                ManageConfig(name="time_8", family="time", time_exit_bars=8, horizon=8, max_bars=8),
            ),
        )
    )
    # 7 hybrid: keep TP 2ATR + ATR trail 0.18
    strategies.append(
        (
            "hybrid_tp_trail_0.18",
            simulate_custom(pool, mkt, eis, mode="hybrid", trail=0.18, horizon=16),
        )
    )

    summary_rows = []
    all_results = {}

    for name, panel in strategies:
        print(f"\n=== {name} ===")
        base = metrics_bundle(panel, extra_cost=0.0)
        stress = {}
        for sname, extra in COST_STRESS.items():
            stress[sname] = metrics_bundle(panel, extra_cost=extra)
        score = robustness_score(base, stress)
        payload = {"strategy": name, "base": base, "stress": stress, "robustness_score": score}
        all_results[name] = payload
        (OUT / f"{name}.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
        panel.to_parquet(OUT / f"{name}_trades.parquet", index=False)

        # yearly csv
        pd.DataFrame(base.get("by_year") or []).to_csv(OUT / f"{name}_by_year.csv", index=False)

        summary_rows.append(
            {
                "strategy": name,
                "total_return": base.get("total_return"),
                "final_equity": base.get("final_equity"),
                "max_drawdown": base.get("max_drawdown"),
                "profit_factor": base.get("profit_factor"),
                "win_rate": base.get("win_rate"),
                "expectancy": base.get("expectancy"),
                "avg_win": base.get("avg_win"),
                "avg_loss": base.get("avg_loss"),
                "avg_holding_bars": base.get("avg_holding_bars"),
                "n_trades": base.get("n_trades"),
                "pf_plus_3bp": stress["plus_3bp"].get("profit_factor"),
                "pf_plus_5bp": stress["plus_5bp"].get("profit_factor"),
                "pf_plus_8bp": stress["plus_8bp"].get("profit_factor"),
                "ret_plus_5bp": stress["plus_5bp"].get("total_return"),
                "dd_plus_5bp": stress["plus_5bp"].get("max_drawdown"),
                "robustness_score": score,
            }
        )
        print(
            f"  ret={base.get('total_return', float('nan')):+.1%} "
            f"PF={base.get('profit_factor', float('nan')):.2f} "
            f"WR={base.get('win_rate', float('nan')):.1%} "
            f"DD={base.get('max_drawdown', float('nan')):.1%} "
            f"stress5bp PF={stress['plus_5bp'].get('profit_factor', float('nan')):.2f} "
            f"score={score:.3f}"
        )

    summary = pd.DataFrame(summary_rows).sort_values("robustness_score", ascending=False)
    summary.to_csv(OUT / "comparison_summary.csv", index=False)

    best = summary.iloc[0].to_dict()
    # Prefer among profitable at +5bp
    ok = summary[summary["pf_plus_5bp"] >= 1.05]
    if not ok.empty:
        best = ok.sort_values("robustness_score", ascending=False).iloc[0].to_dict()

    report = {
        "objective": "Exit Engine only; entry frozen top5%+sess09-15+max_open1",
        "frozen": [
            "market_data",
            "feature_factory",
            "primary_wf_entries",
            "session_09_15",
            "threshold_top5pct",
            "position_sizing_primary_edge",
            "max_open_1",
        ],
        "comparison": summary_rows,
        "recommendation": {
            "strategy": best["strategy"],
            "reason": (
                "Selected by robustness_score (PF, DD, yearly consistency, "
                "survival under +5bp/+8bp cost stress), not max lab return."
            ),
            "metrics": best,
        },
        "results": all_results,
    }
    (OUT / "exit_engine_research_report.json").write_text(
        json.dumps(report, indent=2, default=float), encoding="utf-8"
    )

    # markdown summary
    lines = [
        "# Exit Engine Research",
        "",
        "Entry frozen: Primary WF top 5% + session 09–15 UTC + max_open 1.",
        "Only Exit Engine varies. No ML retrain.",
        "",
        "## Comparison (base cost)",
        "",
        "| Strategy | Return | PF | WR | MaxDD | Hold | n | PF@+5bp | Score |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['strategy']} | {r['total_return']:+.1%} | {r['profit_factor']:.2f} | "
            f"{r['win_rate']:.1%} | {r['max_drawdown']:.1%} | {r['avg_holding_bars']:.1f} | "
            f"{int(r['n_trades'])} | {r['pf_plus_5bp']:.2f} | {r['robustness_score']:.3f} |"
        )
    lines += [
        "",
        f"## Recommendation: `{best['strategy']}`",
        "",
        report["recommendation"]["reason"],
        "",
    ]
    (OUT / "EXIT_ENGINE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\nRECOMMEND", best["strategy"], "score", best["robustness_score"])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
