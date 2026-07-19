"""Fixed-param Meta LightGBM ablation runner (leave-one-year-out)."""

from __future__ import annotations

import logging
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from research.feature_ablation.services.walk_forward_runner import expected_calibration_error
from research.probability_calibration.services.metrics import (
    apply_cost,
    max_drawdown,
    profit_factor,
)

logger = logging.getLogger(__name__)

DEFAULT_LGBM_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.0,
    "reg_lambda": 0.0,
    "verbosity": -1,
}


def _sharpe(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2 or np.std(r, ddof=1) == 0:
        return float("nan")
    return float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(len(r)))


def _sortino(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return float("nan")
    downside = r[r < 0]
    if len(downside) == 0 or np.std(downside, ddof=1) == 0:
        return float("nan")
    return float(np.mean(r) / np.std(downside, ddof=1) * np.sqrt(len(r)))


def trading_metrics(
    frame: pd.DataFrame,
    *,
    taken_mask: np.ndarray,
    cost: float,
) -> dict[str, float]:
    sub = frame.loc[taken_mask]
    if sub.empty:
        # Zero takes = zero PnL for stage means (don't drop WF years as NaN)
        return {
            "n_trades": 0,
            "win_rate": float("nan"),
            "avg_return": 0.0,
            "expectancy": 0.0,
            "profit_factor": float("nan"),
            "max_drawdown": 0.0,
            "sharpe": float("nan"),
            "sortino": float("nan"),
            "avg_holding_bars": float("nan"),
            "annual_return": 0.0,
        }
    # Prefer precomputed net_return else haircut realized
    if "net_return" in sub.columns:
        net = sub["net_return"].to_numpy(dtype=float)
    else:
        net = apply_cost(sub["realized_return"].to_numpy(dtype=float), cost)
    wins = net > 0
    # Annual return: sum of net returns within the evaluation slice
    # (ponytail: additive trade PnL, not geometric)
    annual = float(np.nansum(net))
    hold = (
        float(np.nanmean(sub["holding_bars"].to_numpy(dtype=float)))
        if "holding_bars" in sub.columns
        else float("nan")
    )
    return {
        "n_trades": int(len(sub)),
        "win_rate": float(np.mean(wins)),
        "avg_return": float(np.mean(net)),
        "expectancy": float(np.mean(net)),
        "profit_factor": profit_factor(net),
        "max_drawdown": max_drawdown(net),
        "sharpe": _sharpe(net),
        "sortino": _sortino(net),
        "avg_holding_bars": hold,
        "annual_return": annual,
    }


def classify_metrics(y_true: np.ndarray, proba: np.ndarray, *, threshold: float = 0.5) -> dict:
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(proba, dtype=float), 1e-6, 1 - 1e-6)
    pred = (p >= threshold).astype(int)
    if len(np.unique(y)) < 2:
        nan = float("nan")
        return {
            "roc_auc": nan,
            "pr_auc": nan,
            "precision": nan,
            "recall": nan,
            "f1": nan,
            "brier": nan,
            "log_loss": nan,
            "ece": nan,
        }
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "ece": float(expected_calibration_error(y, p)),
    }


class MetaAblationRunner:
    def __init__(
        self,
        *,
        lgbm_params: dict[str, Any] | None = None,
        num_boost_round: int = 200,
        early_stopping_rounds: int = 30,
        threshold: float = 0.5,
        random_seed: int = 42,
        cost: float = 0.00015,
        compute_shap: bool = True,
    ) -> None:
        self.params = {**DEFAULT_LGBM_PARAMS, **(lgbm_params or {})}
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.threshold = threshold
        self.random_seed = random_seed
        self.cost = cost
        self.compute_shap = compute_shap

    def run_stage(
        self,
        panel: pd.DataFrame,
        features: list[str],
        *,
        stage_id: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Leave-one-year-out. Returns:
          window_metrics, oof_predictions, importance_long
        """
        years = sorted(int(y) for y in panel["valid_year"].dropna().unique())
        pred_rows: list[pd.DataFrame] = []
        win_rows: list[dict] = []
        imp_rows: list[dict] = []

        for year in years:
            val = panel.loc[panel["valid_year"] == year].copy()
            train = panel.loc[panel["valid_year"] != year].copy()
            if train.empty or val.empty:
                continue
            # Need both classes
            if train["meta_label"].nunique() < 2 or val["meta_label"].nunique() < 2:
                logger.warning("Skip year=%s degenerate labels", year)
                continue

            x_tr = train[features].astype(float)
            y_tr = train["meta_label"].astype(int).to_numpy()
            x_va = val[features].astype(float)
            y_va = val["meta_label"].astype(int).to_numpy()

            # Fill NaN with train median
            med = x_tr.median(numeric_only=True)
            x_tr = x_tr.fillna(med)
            x_va = x_va.fillna(med)

            n_pos = max(int((y_tr == 1).sum()), 1)
            n_neg = max(int((y_tr == 0).sum()), 1)
            params = {
                **self.params,
                "seed": self.random_seed,
                "feature_fraction_seed": self.random_seed,
                "bagging_seed": self.random_seed,
                "scale_pos_weight": n_neg / n_pos,
            }
            dtrain = lgb.Dataset(x_tr, label=y_tr, feature_name=features, free_raw_data=False)
            dval = lgb.Dataset(
                x_va, label=y_va, reference=dtrain, feature_name=features, free_raw_data=False
            )
            callbacks = [lgb.log_evaluation(0)]
            if self.early_stopping_rounds > 0:
                callbacks.append(lgb.early_stopping(self.early_stopping_rounds, verbose=False))

            booster = lgb.train(
                params,
                dtrain,
                num_boost_round=self.num_boost_round,
                valid_sets=[dval],
                valid_names=["val"],
                callbacks=callbacks,
            )
            proba = booster.predict(x_va, num_iteration=booster.best_iteration)
            clf = classify_metrics(y_va, proba, threshold=self.threshold)
            taken = proba >= self.threshold
            trd = trading_metrics(val, taken_mask=taken, cost=self.cost)

            logger.info(
                "Ablation %s | year=%s roc=%.4f n_take=%s E=%.5f",
                stage_id,
                year,
                clf["roc_auc"],
                trd["n_trades"],
                trd["expectancy"] if trd["expectancy"] == trd["expectancy"] else -1,
            )
            win_rows.append(
                {
                    "stage_id": stage_id,
                    "valid_year": year,
                    "n_features": len(features),
                    **clf,
                    **trd,
                }
            )

            part = val[
                [
                    c
                    for c in (
                        "side",
                        "window",
                        "valid_year",
                        "timestamp",
                        "meta_label",
                        "net_return",
                        "holding_bars",
                    )
                    if c in val.columns
                ]
            ].copy()
            part["stage_id"] = stage_id
            part["meta_proba"] = proba
            part["meta_taken"] = taken.astype(int)
            pred_rows.append(part)

            # Importances
            gain = booster.feature_importance(importance_type="gain")
            split = booster.feature_importance(importance_type="split")
            shap_mean = np.full(len(features), np.nan)
            if self.compute_shap:
                try:
                    contrib = booster.predict(x_va, pred_contrib=True)
                    arr = np.asarray(contrib, dtype=float)
                    if arr.ndim == 2 and arr.shape[1] == len(features) + 1:
                        shap_mean = np.abs(arr[:, :-1]).mean(axis=0)
                except Exception:
                    shap_mean = np.full(len(features), np.nan)

            # Permutation importance (ROC drop) — small, research only
            base_roc = clf["roc_auc"]
            rng = np.random.default_rng(self.random_seed + int(year))
            for i, feat in enumerate(features):
                x_perm = x_va.copy()
                x_perm[feat] = rng.permutation(x_perm[feat].to_numpy())
                p_perm = booster.predict(x_perm, num_iteration=booster.best_iteration)
                try:
                    roc_p = float(roc_auc_score(y_va, p_perm))
                    perm = base_roc - roc_p
                except Exception:
                    perm = float("nan")
                imp_rows.append(
                    {
                        "stage_id": stage_id,
                        "valid_year": year,
                        "feature": feat,
                        "gain": float(gain[i]),
                        "split": float(split[i]),
                        "shap": float(shap_mean[i]) if shap_mean[i] == shap_mean[i] else float("nan"),
                        "permutation": float(perm) if perm == perm else float("nan"),
                    }
                )

        window_metrics = pd.DataFrame(win_rows)
        oof = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
        importance = pd.DataFrame(imp_rows)
        return window_metrics, oof, importance
