"""Try trend-align + simple meta gate for WR/PF on WF 2015–2026."""

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

from apps.run_primary_loosen_backtest import run_portfolio, wilder_atr, COST
from apps.run_yearly_walkforward_backtest import _feat_cols, _load_side, _train_predict
from production.paper.barrier_resim import load_h1
from research.portfolio_heat import STARTING_EQUITY


def score_side(df: pd.DataFrame, year: int, lookback: int = 7) -> pd.DataFrame:
    years = df["timestamp"].dt.year
    train = df[(years >= year - lookback) & (years < year)]
    test = df[years == year]
    if len(train) < 500 or test.empty:
        return pd.DataFrame()
    feat = _feat_cols(train)
    prob = _train_predict(train, test, feat)
    out = test[
        ["timestamp", "realized_return", "holding_bars", "entry_price", "side"]
        + [c for c in feat if c in test.columns]
    ].copy()
    out["y_prob"] = prob
    return out


def attach_atr(panel: pd.DataFrame, h1: pd.DataFrame) -> pd.DataFrame:
    h = h1.copy()
    h["timestamp"] = pd.to_datetime(h["timestamp"], utc=True)
    h["atr"] = wilder_atr(h)
    m = panel.merge(h[["timestamp", "close", "atr"]], on="timestamp", how="left")
    m["entry_price"] = m["entry_price"].fillna(m["close"]).astype(float)
    m["atr_price"] = m["atr"].astype(float)
    m["net_return"] = m["realized_return"].astype(float) - COST
    m["holding_bars"] = m["holding_bars"].astype(float).clip(lower=1.0)
    return m.dropna(subset=["entry_price", "atr_price", "net_return"])


def trend_align(p: pd.DataFrame) -> pd.DataFrame:
    """Keep long only if ema_alignment>0, short if <0 (if col exists)."""
    if "ema_alignment_score" not in p.columns:
        return p
    long_ok = (p["side"] == "long") & (p["ema_alignment_score"] > 0)
    short_ok = (p["side"] == "short") & (p["ema_alignment_score"] < 0)
    return p.loc[long_ok | short_ok].copy()


_META_BAN = {
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
    "close",
    "atr",
    "atr_price",
    "net_return",
    "meta",
    "hour",
}


def meta_filter(
    train_cands: pd.DataFrame, test_cands: pd.DataFrame, *, keep_frac: float
) -> pd.DataFrame:
    """ponytail: meta = LGBM on primary feat + y_prob predicting net_return>0."""
    if train_cands.empty or test_cands.empty:
        return test_cands
    feat = [c for c in train_cands.columns if c not in _META_BAN and c in test_cands.columns]
    if "y_prob" not in feat:
        feat.append("y_prob")
    y = (train_cands["realized_return"].astype(float) > 0).astype(int).to_numpy()
    if y.sum() < 20 or (1 - y).sum() < 20:
        return test_cands
    dtr = lgb.Dataset(train_cands[feat], label=y, free_raw_data=False)
    bst = lgb.train(
        {
            "objective": "binary",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 20,
            "verbosity": -1,
            "seed": 42,
        },
        dtr,
        num_boost_round=120,
    )
    score = bst.predict(test_cands[feat])
    t = test_cands.copy()
    t["meta"] = score
    k = max(1, int(round(len(t) * keep_frac)))
    return t.nlargest(k, "meta")


def eval_parts(parts: list[pd.DataFrame], mo: int = 1) -> dict:
    all_p = pd.concat(parts, ignore_index=True).sort_values("timestamp")
    # drop meta/helper cols portfolio doesn't need
    keep = [
        c
        for c in all_p.columns
        if c
        in {
            "timestamp",
            "side",
            "y_prob",
            "entry_price",
            "atr_price",
            "net_return",
            "holding_bars",
        }
    ]
    _, cont = run_portfolio(all_p[keep], max_open=mo, starting_equity=STARTING_EQUITY)
    # isolated
    iso_wr, iso_pf = [], []
    for y, g in all_p.groupby(pd.to_datetime(all_p["timestamp"], utc=True).dt.year):
        _, m = run_portfolio(g[keep], max_open=mo, starting_equity=STARTING_EQUITY)
        if m["n_trades"] >= 3:
            iso_wr.append(m["win_rate"])
            iso_pf.append(m["profit_factor"])
    return {
        "n_trades": cont["n_trades"],
        "cont_wr": cont["win_rate"],
        "cont_pf": cont["profit_factor"],
        "cont_ret": cont["total_return"],
        "cont_dd": cont["max_drawdown"],
        "avg_iso_wr": float(np.mean(iso_wr)) if iso_wr else float("nan"),
        "avg_iso_pf": float(np.mean(iso_pf)) if iso_pf else float("nan"),
        "years": int(len(iso_wr)),
    }


def main() -> None:
    long_df = _load_side("long")
    short_df = _load_side("short")
    h1 = load_h1(str(_ROOT / "artifacts/raw/XAUUSD/H1/data.parquet"))
    years = list(range(2015, 2027))

    # per-year full scored frames (both sides)
    print("scoring all years…")
    scored: dict[int, pd.DataFrame] = {}
    for y in years:
        parts = []
        for df in (long_df, short_df):
            s = score_side(df, y)
            if not s.empty:
                parts.append(s)
        if not parts:
            continue
        p = pd.concat(parts, ignore_index=True)
        # best side per bar
        p = p.loc[p.groupby("timestamp")["y_prob"].idxmax()].copy()
        scored[y] = attach_atr(p, h1)

    results = []

    def add(name: str, parts: list[pd.DataFrame], mo: int = 1) -> None:
        if not parts:
            print(f"{name}: empty")
            return
        m = eval_parts(parts, mo=mo)
        m["name"] = name
        results.append(m)
        print(
            f"{name:40} n={m['n_trades']:4d} avgWR={m['avg_iso_wr']:.1%} "
            f"contWR={m['cont_wr']:.1%} contPF={m['cont_pf']:.2f} "
            f"ret={m['cont_ret']:+.1%} dd={m['cont_dd']:.1%} yrs={m['years']}"
        )

    # A: top pct only
    for pct in (0.01, 0.02, 0.03):
        parts = []
        for y, p in scored.items():
            k = max(1, int(round(len(p) * pct)))
            parts.append(p.nlargest(k, "y_prob"))
        add(f"top{pct:.0%}_mo1", parts, 1)

    # B: trend align then top pct
    for pct in (0.02, 0.05, 0.10):
        parts = []
        for y, p in scored.items():
            t = trend_align(p)
            if t.empty:
                continue
            k = max(1, int(round(len(t) * pct)))
            parts.append(t.nlargest(k, "y_prob"))
        add(f"trend+top{pct:.0%}_mo1", parts, 1)

    # C: session + top
    for pct in (0.02, 0.05):
        parts = []
        for y, p in scored.items():
            t = p.copy()
            t["hour"] = pd.to_datetime(t["timestamp"], utc=True).dt.hour
            t = t[(t["hour"] >= 8) & (t["hour"] <= 16)]
            k = max(1, int(round(len(t) * pct)))
            parts.append(t.nlargest(k, "y_prob"))
        add(f"sess+top{pct:.0%}_mo1", parts, 1)

    # D: walk-forward meta on top15 candidates, keep top frac
    for keep in (0.05, 0.10, 0.20, 0.30):
        parts = []
        # build candidate pool history
        hist = []
        for y in years:
            if y not in scored:
                continue
            p = scored[y]
            k = max(1, int(round(len(p) * 0.15)))
            cands = p.nlargest(k, "y_prob").copy()
            train = pd.concat(hist, ignore_index=True) if hist else pd.DataFrame()
            # need prior years only
            if not train.empty:
                kept = meta_filter(train, cands, keep_frac=keep)
            else:
                # first year: just top keep of primary
                kk = max(1, int(round(len(cands) * keep)))
                kept = cands.nlargest(kk, "y_prob")
            if not kept.empty:
                parts.append(kept)
            hist.append(cands)
        add(f"meta_keep{keep:.0%}_mo1", parts, 1)

    # E: trend + session + top2%
    parts = []
    for y, p in scored.items():
        t = trend_align(p)
        t["hour"] = pd.to_datetime(t["timestamp"], utc=True).dt.hour
        t = t[(t["hour"] >= 8) & (t["hour"] <= 16)]
        if t.empty:
            continue
        k = max(1, int(round(len(t) * 0.02)))
        parts.append(t.nlargest(k, "y_prob"))
    add("trend+sess+top2%_mo1", parts, 1)

    # F: oracle ceiling — winners only from top15 (CHEAT upper bound)
    parts = []
    for y, p in scored.items():
        k = max(1, int(round(len(p) * 0.15)))
        c = p.nlargest(k, "y_prob")
        w = c[c["net_return"] > 0]
        if not w.empty:
            parts.append(w)
    add("ORACLE_winners_from_top15", parts, 1)

    out = _ROOT / "artifacts/pipeline_backtest/wr60_pf17_extra.json"
    out.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    print("\nwrote", out)


if __name__ == "__main__":
    main()
