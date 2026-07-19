"""Rolling / aging / retrain / recency experiment runners."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from research.temporal_stability import (
    HISTORY_WINDOWS,
    META_GATE,
    RETRAIN_HORIZONS_MONTHS,
    STARTING_EQUITY,
    THRESHOLDS,
)
from research.temporal_stability.services import class_metrics, optimal_threshold, trade_metrics_from_returns
from research.temporal_stability.services.data import fit_predict, ml_feature_names, prepare_xy, recency_weights

logger = logging.getLogger(__name__)


def _proxy_returns(y_true: np.ndarray, y_prob: np.ndarray, *, thr: float) -> np.ndarray:
    """
    Research proxy PnL: take trade when p>=thr; +cost-free signed outcome.
    ponytail: uses label as R-proxy (+1/-1), not live ATR path.
    """
    take = y_prob >= thr
    r = np.where(y_true == 1, 0.01, -0.0075)  # rough TP/SL expectancy proxy
    return np.where(take, r, 0.0)[take]


def rolling_performance(panel: pd.DataFrame, *, min_train_years: int = 5) -> pd.DataFrame:
    years = sorted(int(y) for y in panel["year"].unique())
    features = ml_feature_names(panel)
    rows = []
    for i, test_year in enumerate(years):
        train_years = [y for y in years if y < test_year]
        if len(train_years) < min_train_years:
            continue
        train = panel.loc[panel["year"].isin(train_years)]
        test = panel.loc[panel["year"] == test_year]
        try:
            y, p, _ = fit_predict(train, test, features)
        except ValueError as exc:
            logger.warning("rolling skip year=%s err=%s", test_year, exc)
            continue
        cm = class_metrics(y, p, thr=0.5)
        rets = _proxy_returns(y, p, thr=0.5)
        tm = trade_metrics_from_returns(rets, starting=STARTING_EQUITY)
        rows.append(
            {
                "train_start": min(train_years),
                "train_end": max(train_years),
                "test_year": test_year,
                **cm,
                **tm,
            }
        )
        logger.info("rolling test=%s roc=%.3f pf=%.3f", test_year, cm["roc_auc"], tm["profit_factor"])
    return pd.DataFrame(rows)


def year_by_year_ml(panel: pd.DataFrame, *, leave_one_out: bool = True) -> pd.DataFrame:
    """Per-year metrics via LOO train (all other years) or expanding."""
    years = sorted(int(y) for y in panel["year"].unique())
    features = ml_feature_names(panel)
    rows = []
    for year in years:
        if leave_one_out:
            train = panel.loc[panel["year"] != year]
        else:
            train = panel.loc[panel["year"] < year]
        test = panel.loc[panel["year"] == year]
        if train.empty or test.empty:
            continue
        try:
            y, p, _ = fit_predict(train, test, features)
        except ValueError:
            continue
        cm = class_metrics(y, p)
        rets = _proxy_returns(y, p, thr=0.5)
        tm = trade_metrics_from_returns(rets, starting=STARTING_EQUITY)
        rows.append({"year": year, **cm, **tm})
    return pd.DataFrame(rows)


def year_by_year_frozen(trades: pd.DataFrame) -> pd.DataFrame:
    """Evaluate frozen production trade log by calendar year."""
    rows = []
    for year, g in trades.groupby("year"):
        if "net_return" in g.columns:
            rets = g["net_return"].astype(float).to_numpy()
        elif "pnl" in g.columns and "equity_before" in g.columns:
            rets = (g["pnl"].astype(float) / g["equity_before"].astype(float).replace(0, np.nan)).to_numpy()
        else:
            rets = np.array([])
        tm = trade_metrics_from_returns(rets, starting=STARTING_EQUITY)
        # classification proxies from meta/confidence if present
        y = None
        p = None
        if "meta_label" in g.columns and "meta_proba" in g.columns:
            y = g["meta_label"].astype(int).to_numpy()
            p = g["meta_proba"].astype(float).to_numpy()
        elif "meta_label" in g.columns and "raw_probability" in g.columns:
            y = g["meta_label"].astype(int).to_numpy()
            p = g["raw_probability"].astype(float).to_numpy()
        cm = class_metrics(y, p) if y is not None and p is not None else {}
        rows.append({"year": int(year), "source": "frozen_production", **cm, **tm})
    return pd.DataFrame(rows)


def dataset_aging(panel: pd.DataFrame) -> pd.DataFrame:
    years = sorted(int(y) for y in panel["year"].unique())
    features = ml_feature_names(panel)
    rows = []
    for test_year in years:
        for win in HISTORY_WINDOWS:
            if win == 0:
                train_years = [y for y in years if y < test_year]
                label = "full"
            else:
                train_years = [y for y in years if test_year - win <= y < test_year]
                label = f"{win}y"
            if len(train_years) < 1:
                continue
            if win != 0 and len(train_years) < min(win, 2):
                continue
            train = panel.loc[panel["year"].isin(train_years)]
            test = panel.loc[panel["year"] == test_year]
            try:
                y, p, _ = fit_predict(train, test, features)
            except ValueError:
                continue
            cm = class_metrics(y, p)
            rets = _proxy_returns(y, p, thr=0.5)
            tm = trade_metrics_from_returns(rets, starting=STARTING_EQUITY)
            rows.append(
                {
                    "test_year": test_year,
                    "history_window": label,
                    "train_years": len(train_years),
                    "train_start": min(train_years),
                    "train_end": max(train_years),
                    **cm,
                    **tm,
                }
            )
    return pd.DataFrame(rows)


def recency_weight_experiment(panel: pd.DataFrame) -> pd.DataFrame:
    years = sorted(int(y) for y in panel["year"].unique())
    features = ml_feature_names(panel)
    modes = ("uniform", "linear", "exponential", "half_life")
    rows = []
    for test_year in years:
        train = panel.loc[panel["year"] < test_year].copy()
        test = panel.loc[panel["year"] == test_year]
        if train.empty or test.empty:
            continue
        for mode in modes:
            w = recency_weights(train["year"].to_numpy(), mode=mode)
            try:
                y, p, _ = fit_predict(train, test, features, sample_weight=w)
            except ValueError:
                continue
            cm = class_metrics(y, p)
            rets = _proxy_returns(y, p, thr=0.5)
            tm = trade_metrics_from_returns(rets, starting=STARTING_EQUITY)
            rows.append({"test_year": test_year, "weight_mode": mode, **cm, **tm})
    return pd.DataFrame(rows)


def retrain_frequency(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Freeze model at train cutoff, score subsequent months without retrain.
    Measures degradation vs months since train end.
    """
    years = sorted(int(y) for y in panel["year"].unique())
    features = ml_feature_names(panel)
    rows = []
    # use mid-history cutoffs
    cutoffs = [y for y in years if y >= years[0] + 5 and y <= years[-1] - 1]
    for cutoff in cutoffs:
        train = panel.loc[panel["year"] <= cutoff]
        future = panel.loc[panel["year"] > cutoff].copy()
        if train.empty or future.empty:
            continue
        future = future.sort_values("timestamp")
        try:
            # score entire future once with frozen model
            y_all, p_all, _ = fit_predict(train, future, features)
        except ValueError:
            continue
        # align back to future rows with exclude_timeout mask
        raw = future["label"].astype(int)
        mask = (raw != 0).to_numpy()
        fut = future.loc[mask].copy()
        fut["y"] = y_all
        fut["p"] = p_all
        fut["months_since"] = (
            (fut["timestamp"].dt.year - cutoff) * 12 + fut["timestamp"].dt.month - 12
        ).clip(lower=1)
        for months in RETRAIN_HORIZONS_MONTHS:
            chunk = fut.loc[fut["months_since"] <= months]
            if chunk.empty:
                continue
            cm = class_metrics(chunk["y"].to_numpy(), chunk["p"].to_numpy())
            rets = _proxy_returns(chunk["y"].to_numpy(), chunk["p"].to_numpy(), thr=0.5)
            tm = trade_metrics_from_returns(rets, starting=STARTING_EQUITY)
            rows.append(
                {
                    "train_cutoff_year": cutoff,
                    "freeze_months": months,
                    **cm,
                    **tm,
                }
            )
    return pd.DataFrame(rows)


def threshold_by_year(panel: pd.DataFrame) -> pd.DataFrame:
    years = sorted(int(y) for y in panel["year"].unique())
    features = ml_feature_names(panel)
    rows = []
    for year in years:
        train = panel.loc[panel["year"] < year]
        test = panel.loc[panel["year"] == year]
        if train.empty or test.empty:
            continue
        try:
            y, p, _ = fit_predict(train, test, features)
        except ValueError:
            continue
        opt = optimal_threshold(y, p, THRESHOLDS)
        cm45 = class_metrics(y, p, thr=META_GATE)
        rows.append(
            {
                "year": year,
                "best_threshold": opt["best_threshold"],
                "best_f1": opt["best_f1"],
                "fixed_0_45_roc": cm45["roc_auc"],
                "fixed_0_45_accuracy": cm45["accuracy"],
                "n": cm45["n"],
            }
        )
    return pd.DataFrame(rows)


def probability_stability(panel: pd.DataFrame) -> pd.DataFrame:
    years = sorted(int(y) for y in panel["year"].unique())
    features = ml_feature_names(panel)
    rows = []
    for year in years:
        train = panel.loc[panel["year"] < year]
        test = panel.loc[panel["year"] == year]
        if train.empty or test.empty:
            continue
        try:
            y, p, _ = fit_predict(train, test, features)
        except ValueError:
            continue
        for cls, name in ((1, "positive"), (0, "negative")):
            sub = p[y == cls]
            if len(sub) == 0:
                continue
            rows.append(
                {
                    "year": year,
                    "class": name,
                    "n": len(sub),
                    "mean_prob": float(np.mean(sub)),
                    "std_prob": float(np.std(sub)),
                    "p10": float(np.percentile(sub, 10)),
                    "p50": float(np.percentile(sub, 50)),
                    "p90": float(np.percentile(sub, 90)),
                }
            )
        cm = class_metrics(y, p)
        rows.append(
            {
                "year": year,
                "class": "calibration",
                "n": cm["n"],
                "mean_prob": float(np.mean(p)),
                "std_prob": float(np.std(p)),
                "p10": cm["ece"],
                "p50": cm["brier"],
                "p90": cm["cal_slope"],
                "ece": cm["ece"],
                "brier": cm["brier"],
                "cal_slope": cm["cal_slope"],
                "cal_intercept": cm["cal_intercept"],
            }
        )
    return pd.DataFrame(rows)
