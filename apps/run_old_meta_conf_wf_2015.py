"""Walk-forward old pipeline (Primary -> Meta -> Confidence) 2015-2026 yearly.

Approximate research freeze on available H1 v2 features (some FINAL_FEATURES
proxied). Top-3% primary candidates/year/side, meta LOO, conf LOO, then
meta>=0.45 + conf>=40 + single + meta-edge + heat.
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
from sklearn.linear_model import LogisticRegression

from apps.run_old_meta_pipeline_yearly import isolated_years, run_old
from apps.run_primary_loosen_backtest import wilder_atr
from apps.run_yearly_walkforward_backtest import _feat_cols, _load_side, _train_predict
from production.paper.barrier_resim import load_h1

COST = 1.5e-4
TOP_PCT = 0.03
META_GATE = 0.45
CONF_GATE = 40.0
LOOKBACK = 7

# ponytail: FINAL_FEATURES subset + proxies available on v2 datasets
META_FEATS = [
    "raw_probability",
    "probability_rank",
    "probability_percentile",
    "probability_margin",
    "ctx_h4_swing_quality",
    "ctx_h4_trend_strength",  # proxy for missing h4 structure cols
    "rolling_quantile",
    "ema_trend_duration",
    "volatility_rank",
    "atr_percent",
    "hour_of_day",
    "day_of_week",
    "month",
    "rolling_volatility",
]


def _score_year(df: pd.DataFrame, year: int) -> pd.DataFrame:
    years = df["timestamp"].dt.year
    train = df[(years >= year - LOOKBACK) & (years < year)]
    test = df[years == year]
    if len(train) < 500 or test.empty:
        return pd.DataFrame()
    feat = _feat_cols(train)
    prob = _train_predict(train, test, feat)
    out = test.copy()
    out["raw_probability"] = prob
    out["y_prob"] = prob
    return out


def _candidates(scored: pd.DataFrame) -> pd.DataFrame:
    """Top TOP_PCT by y_prob within (year, side)."""
    if scored.empty:
        return scored
    parts = []
    for (_, _), g in scored.groupby([scored["timestamp"].dt.year, "side"]):
        k = max(1, int(round(len(g) * TOP_PCT)))
        parts.append(g.nlargest(k, "y_prob"))
    c = pd.concat(parts, ignore_index=True)
    # rank features within year+side
    c["probability_rank"] = c.groupby([c["timestamp"].dt.year, "side"])["y_prob"].rank(
        ascending=False, method="average"
    )
    c["probability_percentile"] = c.groupby([c["timestamp"].dt.year, "side"])["y_prob"].rank(
        pct=True
    )
    c["probability_margin"] = c.groupby([c["timestamp"].dt.year, "side"])["y_prob"].transform(
        lambda s: s - s.min()
    )
    ts = pd.to_datetime(c["timestamp"], utc=True)
    c["hour_of_day"] = ts.dt.hour.astype(float)
    c["day_of_week"] = ts.dt.dayofweek.astype(float)
    c["month"] = ts.dt.month.astype(float)
    c["valid_year"] = ts.dt.year.astype(int)
    c["meta_label"] = (c["realized_return"].astype(float) > 0).astype(int)
    c["net_return"] = c["realized_return"].astype(float) - COST
    # fill missing meta feats
    for f in META_FEATS:
        if f not in c.columns:
            c[f] = 0.0
        else:
            c[f] = c[f].astype(float)
    return c


def _train_meta(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    feats = [f for f in META_FEATS if f in train.columns]
    x_tr = train[feats].astype(float).fillna(train[feats].median(numeric_only=True))
    y_tr = train["meta_label"].astype(int).to_numpy()
    if y_tr.sum() < 10 or (1 - y_tr).sum() < 10:
        return np.full(len(test), 0.5)
    med = x_tr.median()
    x_te = test[feats].astype(float).fillna(med)
    # hold out last 20% of train chronologically for early stop
    tr = train.sort_values("timestamp")
    cut = max(int(len(tr) * 0.8), 1)
    tr_a, tr_b = tr.iloc[:cut], tr.iloc[cut:]
    if len(tr_b) < 30:
        tr_a, tr_b = tr, tr.iloc[-max(30, len(tr) // 10) :]
    xa = tr_a[feats].astype(float).fillna(med)
    xb = tr_b[feats].astype(float).fillna(med)
    spw = max((tr_a["meta_label"] == 0).sum(), 1) / max((tr_a["meta_label"] == 1).sum(), 1)
    dtr = lgb.Dataset(xa, label=tr_a["meta_label"].astype(int), free_raw_data=False)
    dva = lgb.Dataset(xb, label=tr_b["meta_label"].astype(int), reference=dtr, free_raw_data=False)
    bst = lgb.train(
        {
            "objective": "binary",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": float(spw),
            "verbosity": -1,
            "seed": 42,
        },
        dtr,
        num_boost_round=200,
        valid_sets=[dva],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
    )
    return bst.predict(x_te)


def _h4_context(frame: pd.DataFrame) -> np.ndarray:
    cols = [c for c in ("ctx_h4_swing_quality", "ctx_h4_trend_strength") if c in frame.columns]
    if not cols:
        return np.full(len(frame), 0.5)
    return frame[cols].astype(float).rank(pct=True).mean(axis=1).to_numpy(dtype=float)


def _fit_conf_loo(panel: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-year-out logistic confidence 0-100 (simplified components)."""
    p = panel.copy()
    p["h4_context"] = _h4_context(p)
    p["d1_trend"] = 0.5  # ponytail: no d1 in v2 panel; neutral
    p["m5_entry_quality_01"] = 0.5
    feats = ["meta_proba", "raw_probability", "h4_context", "d1_trend", "m5_entry_quality_01"]
    conf = np.full(len(p), np.nan)
    years = sorted(p["valid_year"].unique())
    for y in years:
        tr = p[p["valid_year"] != y]
        te_idx = p["valid_year"] == y
        if tr["meta_label"].nunique() < 2 or te_idx.sum() == 0:
            continue
        x_tr = tr[feats].astype(float).fillna(0.5)
        y_tr = tr["meta_label"].astype(int)
        x_te = p.loc[te_idx, feats].astype(float).fillna(0.5)
        clf = LogisticRegression(max_iter=500, random_state=42)
        clf.fit(x_tr, y_tr)
        proba = clf.predict_proba(x_te)[:, 1]
        conf[np.where(te_idx)[0]] = np.clip(proba * 100.0, 0.0, 100.0)
    p["confidence"] = conf
    # fill any gaps with meta-scaled fallback
    miss = p["confidence"].isna()
    p.loc[miss, "confidence"] = (p.loc[miss, "meta_proba"] * 100.0).clip(0, 100)
    return p


def attach_atr(panel: pd.DataFrame, h1: pd.DataFrame) -> pd.DataFrame:
    h = h1.copy()
    h["timestamp"] = pd.to_datetime(h["timestamp"], utc=True)
    h["atr"] = wilder_atr(h)
    m = panel.merge(h[["timestamp", "close", "atr"]], on="timestamp", how="left")
    m["entry_price"] = m["entry_price"].fillna(m["close"]).astype(float)
    m["atr_entry"] = m["atr"].astype(float)
    m["atr_price"] = m["atr_entry"]
    return m.dropna(subset=["entry_price", "atr_entry", "net_return", "holding_bars"])


def main() -> None:
    years = list(range(2015, 2027))
    print("Loading datasets…")
    long_df = _load_side("long")
    short_df = _load_side("short")
    h1 = load_h1(str(_ROOT / "artifacts/raw/XAUUSD/H1/data.parquet"))

    # score primary for all years needed (2010+ for meta train hist)
    score_years = list(range(2010, 2027))
    scored_parts = []
    print("Primary walk-forward scoring…")
    for y in score_years:
        for df in (long_df, short_df):
            s = _score_year(df, y)
            if not s.empty:
                scored_parts.append(s)
        print(f"  scored {y}")
    scored = pd.concat(scored_parts, ignore_index=True)
    scored["timestamp"] = pd.to_datetime(scored["timestamp"], utc=True)

    print("Building top-3% candidates…")
    cands = _candidates(scored)
    print(f"candidates={len(cands)} years={sorted(cands['valid_year'].unique())}")

    # Meta: expanding window (train years < Y) for causal WF
    print("Meta expanding-window predict…")
    meta_parts = []
    for y in years:
        te = cands[cands["valid_year"] == y].copy()
        tr = cands[cands["valid_year"] < y]
        if te.empty:
            continue
        if len(tr) < 100:
            # bootstrap: use other years LOO if not enough history
            tr = cands[cands["valid_year"] != y]
        if len(tr) < 100:
            te["meta_proba"] = te["raw_probability"]
        else:
            te["meta_proba"] = _train_meta(tr, te)
        meta_parts.append(te)
    meta_panel = pd.concat(meta_parts, ignore_index=True)
    print(f"meta_panel={len(meta_panel)}")

    print("Confidence LOO…")
    conf_panel = _fit_conf_loo(meta_panel)
    conf_panel = attach_atr(conf_panel, h1)

    # save panel
    out = _ROOT / "artifacts/pipeline_backtest"
    out.mkdir(parents=True, exist_ok=True)
    conf_panel.to_parquet(out / "wf_meta_conf_panel_2015_2026.parquet", index=False)

    # funnel
    print("\nFunnel by year:")
    for y, g in conf_panel.groupby("valid_year", sort=True):
        n = len(g)
        n_m = int((g["meta_proba"] >= META_GATE).sum())
        n_c = int(((g["meta_proba"] >= META_GATE) & (g["confidence"] >= CONF_GATE)).sum())
        print(f"  {y}: cands={n}  meta>={META_GATE}={n_m}  +conf>={CONF_GATE:.0f}={n_c}")

    results = {}
    for mo in (1, 3):
        iso = isolated_years(
            conf_panel, meta_gate=True, conf_gate=True, max_open=mo
        )
        _, ov = run_old(conf_panel, meta_gate=True, conf_gate=True, max_open=mo)
        print(f"\n=== WF meta+conf | max_open={mo} | isolated $80/yr ===")
        print(f"{'year':>6} {'trades':>7} {'ret':>8} {'dd':>7} {'wr':>7} {'pf':>6}")
        for r in iso:
            if r.get("n_trades", 0) == 0:
                print(f"{r['year']:6d} {'—':>7}")
                continue
            print(
                f"{r['year']:6d} {r['n_trades']:7d} {r['ret']:+8.1%} {r['dd']:7.1%} "
                f"{r['wr']:7.1%} {r['pf']:6.2f}"
            )
        print(
            f"{'ALL':>6} {ov['n_trades']:7d} {ov['total_return']:+8.1%} "
            f"{ov['max_drawdown']:7.1%} {ov['win_rate']:7.1%} {ov['profit_factor']:6.2f}"
            "  (compounded)"
        )
        results[f"max_open_{mo}"] = {"isolated": iso, "continuous": ov}

    payload = {
        "note": (
            "Walk-forward approx of old pipeline 2015-2026. "
            "Meta features proxied from H1 v2 (not full FINAL_FEATURES). "
            "Confidence without real d1/m5 (neutral 0.5)."
        ),
        "top_pct": TOP_PCT,
        "meta_gate": META_GATE,
        "conf_gate": CONF_GATE,
        "results": results,
    }
    (out / "old_meta_conf_wf_2015_2026.json").write_text(
        json.dumps(payload, indent=2, default=float), encoding="utf-8"
    )
    print("\nwrote", out / "old_meta_conf_wf_2015_2026.json")


if __name__ == "__main__":
    main()
