"""Walk-forward year-by-year Primary backtest (2015–2026).

For each test year Y: train long+short LGBM on [Y-lookback, Y), score Y,
keep top_pct unique bars (best side), run max_open portfolio.

Isolated = reset equity each year. Continuous = compound across years.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import lightgbm as lgb
import numpy as np
import pandas as pd

from apps.run_primary_loosen_backtest import run_portfolio, wilder_atr
from production.paper.barrier_resim import load_h1
from research.portfolio_heat import STARTING_EQUITY

COST = 1.5e-4
META_COLS = {
    "timestamp",
    "symbol",
    "timeframe",
    "feature_version",
    "label_version",
    "split",
    "side",
    "strategy",
    "label",
    "realized_return",
    "holding_bars",
    "entry_price",
}
SPLITS = ("train", "validation", "test", "sealed")


def _load_side(side: str) -> pd.DataFrame:
    base = _ROOT / f"artifacts/datasets/XAUUSD/H1/{side}/v2"
    parts = [pd.read_parquet(base / f"{s}.parquet") for s in SPLITS]
    d = pd.concat(parts, ignore_index=True)
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
    d["side"] = side
    lab = pd.read_parquet(
        _ROOT / f"artifacts/labels/XAUUSD/H1/{side}/triple_barrier_v1.parquet",
        columns=["timestamp", "realized_return", "holding_bars", "entry_price"],
    )
    lab["timestamp"] = pd.to_datetime(lab["timestamp"], utc=True)
    d = d.merge(lab, on="timestamp", how="left")
    return d.dropna(subset=["label", "realized_return"]).sort_values("timestamp")


def _feat_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS]


def _train_predict(train: pd.DataFrame, test: pd.DataFrame, feat: list[str]) -> np.ndarray:
    """ponytail: last 20% of train as early-stop val; fine for yearly WF."""
    n = len(train)
    cut = max(int(n * 0.8), 1)
    tr, va = train.iloc[:cut], train.iloc[cut:]
    if len(va) < 50:
        tr, va = train, train.iloc[-max(50, n // 10) :]
    y_tr = tr["label"].astype(int).to_numpy()
    y_va = va["label"].astype(int).to_numpy()
    spw = max((y_tr == 0).sum(), 1) / max((y_tr == 1).sum(), 1)
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 40,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": float(spw),
        "verbosity": -1,
        "seed": 42,
    }
    dtr = lgb.Dataset(tr[feat], label=y_tr, feature_name=list(feat), free_raw_data=False)
    dva = lgb.Dataset(va[feat], label=y_va, reference=dtr, free_raw_data=False)
    booster = lgb.train(
        params,
        dtr,
        num_boost_round=500,
        valid_sets=[dva],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    return booster.predict(test[feat])


def build_year_panel(
    long_df: pd.DataFrame,
    short_df: pd.DataFrame,
    h1: pd.DataFrame,
    *,
    year: int,
    lookback: int,
    top_pct: float,
) -> pd.DataFrame:
    lo = year - lookback
    rows = []
    for side, df in (("long", long_df), ("short", short_df)):
        years = df["timestamp"].dt.year
        train = df[(years >= lo) & (years < year)]
        test = df[years == year]
        if len(train) < 500 or test.empty:
            continue
        feat = _feat_cols(train)
        prob = _train_predict(train, test, feat)
        g = test[["timestamp", "realized_return", "holding_bars", "entry_price"]].copy()
        g["side"] = side
        g["y_prob"] = prob
        rows.append(g)
    if not rows:
        return pd.DataFrame()
    p = pd.concat(rows, ignore_index=True)
    p = p.loc[p.groupby("timestamp")["y_prob"].idxmax()].copy()
    k = max(1, int(round(len(p) * float(top_pct))))
    u = p.nlargest(k, "y_prob").sort_values("timestamp")

    h = h1.copy()
    h["timestamp"] = pd.to_datetime(h["timestamp"], utc=True)
    h["atr"] = wilder_atr(h)
    m = u.merge(h[["timestamp", "close", "atr"]], on="timestamp", how="left")
    m["entry_price"] = m["entry_price"].fillna(m["close"]).astype(float)
    m["atr_price"] = m["atr"].astype(float)
    m["net_return"] = m["realized_return"].astype(float) - COST
    m["holding_bars"] = m["holding_bars"].astype(float).clip(lower=1.0)
    return m.dropna(subset=["entry_price", "atr_price", "net_return"])


def main() -> None:
    years = list(range(2015, 2027))
    lookback = 7
    top_pct = 0.15
    max_open = 5

    print("Loading long/short datasets + H1…")
    long_df = _load_side("long")
    short_df = _load_side("short")
    h1 = load_h1(str(_ROOT / "artifacts/raw/XAUUSD/H1/data.parquet"))

    isolated_rows = []
    continuous_parts = []
    print(
        f"\nWalk-forward | lookback={lookback}y | top={top_pct:.0%} unique bars | "
        f"max_open={max_open} | ATR TP/SL 2.0/1.5 | primary-edge sizing\n"
    )
    print(f"{'year':>6} {'cands':>6} {'trades':>6} {'isol_ret':>10} {'isol_dd':>8} {'win':>6}")
    for y in years:
        panel = build_year_panel(
            long_df, short_df, h1, year=y, lookback=lookback, top_pct=top_pct
        )
        if panel.empty:
            print(f"{y:6d} {'—':>6} {'—':>6} {'n/a':>10} {'n/a':>8} {'n/a':>6}")
            isolated_rows.append({"year": y, "n_candidates": 0, "n_trades": 0})
            continue
        traded, m = run_portfolio(panel, max_open=max_open, starting_equity=STARTING_EQUITY)
        by = (m.get("by_year") or [{}])[0]
        isol_ret = float(by.get("year_return", m["total_return"]))
        isol_dd = float(by.get("max_drawdown", m["max_drawdown"]))
        row = {
            "year": y,
            "n_candidates": int(len(panel)),
            "n_trades": int(m["n_trades"]),
            "isolated_return": isol_ret,
            "isolated_max_dd": isol_dd,
            "win_rate": float(m.get("win_rate", float("nan"))),
            "profit_factor": float(m.get("profit_factor", float("nan"))),
            "final_equity": float(m["final_equity"]),
        }
        isolated_rows.append(row)
        continuous_parts.append(panel)
        print(
            f"{y:6d} {len(panel):6d} {m['n_trades']:6d} "
            f"{isol_ret:+10.1%} {isol_dd:8.1%} {m.get('win_rate', float('nan')):6.1%}"
        )

    # continuous equity across full span
    cont_metrics = {}
    if continuous_parts:
        all_panel = pd.concat(continuous_parts, ignore_index=True).sort_values("timestamp")
        _, cont_metrics = run_portfolio(
            all_panel, max_open=max_open, starting_equity=STARTING_EQUITY
        )
        print("\nContinuous (compounded equity across years):")
        print(
            f"  trades={cont_metrics['n_trades']} ret={cont_metrics['total_return']:+.1%} "
            f"maxDD={cont_metrics['max_drawdown']:.1%} "
            f"pf={cont_metrics.get('profit_factor', float('nan')):.2f}"
        )
        for r in cont_metrics.get("by_year", []):
            print(
                f"  {r['year']}: trades={r['n_trades']} "
                f"ret={r['year_return']:+.1%} dd={r['max_drawdown']:.1%}"
            )

    iso = pd.DataFrame(isolated_rows)
    pos = int((iso.get("isolated_return", pd.Series(dtype=float)).fillna(0) > 0).sum())
    neg = int((iso.get("isolated_return", pd.Series(dtype=float)).fillna(0) < 0).sum())
    print(f"\nIsolated years positive/negative: {pos}/{neg} (of {len(iso.dropna(subset=['isolated_return']))})")

    out = _ROOT / "artifacts/pipeline_backtest"
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "policy": {
            "lookback_years": lookback,
            "top_pct": top_pct,
            "max_open": max_open,
            "years": years,
            "note": "Walk-forward: train only on prior lookback years. Not frozen production model.",
        },
        "isolated_by_year": isolated_rows,
        "continuous": cont_metrics,
    }
    (out / "yearly_walkforward_2015_2026.json").write_text(
        json.dumps(payload, indent=2, default=float), encoding="utf-8"
    )
    iso.to_csv(out / "yearly_walkforward_2015_2026.csv", index=False)
    print("\nwrote", out / "yearly_walkforward_2015_2026.json")


if __name__ == "__main__":
    main()
