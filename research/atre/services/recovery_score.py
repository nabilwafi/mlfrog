"""LOO Recovery Score (0-100, high = failure likely) — not a new classifier gate."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

SCORE_FEATS = (
    "adverse_atr",
    "underwater_bars",
    "atr_ratio",
    "adx_ratio",
    "vol_ratio",
    "dist_sl_atr",
    "dist_tp_atr",
    "struct_fail",
    "mom_collapse",
    "m5_q",
    "h4_swing",
    "conf01",
    "vol_rank",
    "d1_bull",
    "d1_bear",
)


def _event_rows(lib: list[dict[str, Any]], min_mae: float = 0.4) -> pd.DataFrame:
    """One row per trade at first bar where adverse >= min_mae (causal snapshot)."""
    rows = []
    for p in lib:
        if not p.get("ok") or not p.get("bars"):
            continue
        snap = next((b for b in p["bars"] if b["adverse_atr"] >= min_mae), None)
        if snap is None:
            continue
        d1 = str(p.get("d1_regime", ""))
        rows.append(
            {
                "trade_i": p["trade_i"],
                "valid_year": p["valid_year"],
                "y_fail": int(p["hit_sl"]),  # predict eventual SL
                "adverse_atr": snap["adverse_atr"],
                "underwater_bars": snap["underwater_bars"],
                "atr_ratio": snap["atr_ratio"],
                "adx_ratio": snap["adx_ratio"],
                "vol_ratio": snap["vol_ratio"],
                "dist_sl_atr": snap["dist_sl_atr"],
                "dist_tp_atr": snap["dist_tp_atr"],
                "struct_fail": float(snap["struct_fail"]),
                "mom_collapse": float(snap["mom_collapse"]),
                "m5_q": float(p.get("m5_q", 0.5)),
                "h4_swing": float(p.get("h4_swing", 0.5)),
                "conf01": float(p.get("confidence", 50)) / 100.0,
                "vol_rank": float(p.get("vol_rank", 0.5)),
                "d1_bull": float("Bull" in d1),
                "d1_bear": float("Bear" in d1),
            }
        )
    return pd.DataFrame(rows)


def fit_loo_recovery_scores(lib: list[dict[str, Any]], min_mae: float = 0.4) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    LOO logistic: score = 100 * P(eventual SL | snapshot).
    Returns (events_with_score, importance_proxy).
    """
    ev = _event_rows(lib, min_mae=min_mae)
    if ev.empty or ev["y_fail"].nunique() < 2:
        return ev, pd.DataFrame()

    years = sorted(ev["valid_year"].unique())
    score = np.full(len(ev), np.nan)
    coef_rows = []
    for year in years:
        tr = ev.loc[ev["valid_year"] != year]
        va_idx = ev.index[ev["valid_year"] == year]
        if tr.empty or len(va_idx) == 0 or tr["y_fail"].nunique() < 2:
            continue
        x_tr = tr[list(SCORE_FEATS)].astype(float).fillna(0.5).to_numpy()
        y_tr = tr["y_fail"].to_numpy(dtype=int)
        sc = StandardScaler()
        xs = sc.fit_transform(x_tr)
        clf = LogisticRegression(max_iter=400, random_state=42)
        clf.fit(xs, y_tr)
        x_va = sc.transform(ev.loc[va_idx, list(SCORE_FEATS)].astype(float).fillna(0.5).to_numpy())
        p = clf.predict_proba(x_va)[:, 1]
        score[ev.index.get_indexer(va_idx)] = p * 100.0
        coef_rows.append(dict(zip(SCORE_FEATS, np.abs(clf.coef_[0]))))
    ev = ev.copy()
    ev["recovery_score"] = score
    imp = pd.DataFrame(coef_rows).mean().reset_index()
    imp.columns = ["feature", "mean_abs_coef"]
    imp = imp.sort_values("mean_abs_coef", ascending=False)
    return ev, imp


def score_map(events: pd.DataFrame) -> dict[int, float]:
    if events.empty or "recovery_score" not in events.columns:
        return {}
    return {int(r.trade_i): float(r.recovery_score) for r in events.itertuples() if r.recovery_score == r.recovery_score}
