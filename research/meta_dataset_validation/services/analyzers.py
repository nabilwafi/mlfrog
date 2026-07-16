"""Candidate trade metrics for meta-dataset validation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.probability_calibration.services.metrics import (
    apply_cost,
    max_drawdown,
    profit_factor,
    select_top_percentile,
)

CANDIDATE_PERCENTILES: tuple[float, ...] = (0.01, 0.03, 0.05, 0.10, 0.15, 0.20)

# Ponytail floor for "enough to train" heuristics (not a hard rule).
MIN_TOTAL_SAMPLES = 200
MIN_PER_WINDOW = 25


def compute_path_excursions(
    candles: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    side: str,
) -> pd.DataFrame:
    """
    MAE / MFE from OHLC path between entry and exit (diagnostic only).

    Uses metadata entry_index / exit_index when present; otherwise holding_bars.
    """
    c = candles.copy()
    c["timestamp"] = pd.to_datetime(c["timestamp"], utc=True)
    c = c.sort_values("timestamp").reset_index(drop=True)
    high = c["high"].to_numpy(dtype=float)
    low = c["low"].to_numpy(dtype=float)
    n = len(c)

    lab = labels.copy()
    lab["timestamp"] = pd.to_datetime(lab["timestamp"], utc=True)

    entry_idx = np.full(len(lab), -1, dtype=int)
    exit_idx = np.full(len(lab), -1, dtype=int)

    if "metadata_json" in lab.columns:
        import json

        for i, raw in enumerate(lab["metadata_json"].tolist()):
            try:
                meta = json.loads(raw) if isinstance(raw, str) else dict(raw or {})
            except Exception:
                meta = {}
            if "entry_index" in meta and "exit_index" in meta:
                entry_idx[i] = int(meta["entry_index"])
                exit_idx[i] = int(meta["exit_index"])

    # Fallback: map timestamp → row, exit = entry + holding_bars
    ts_to_i = {t: i for i, t in enumerate(c["timestamp"].tolist())}
    for i, row in enumerate(lab.itertuples()):
        if entry_idx[i] < 0:
            ei = ts_to_i.get(row.timestamp)
            if ei is None:
                continue
            entry_idx[i] = int(ei)
            hb = int(getattr(row, "holding_bars", 0) or 0)
            exit_idx[i] = min(int(ei + max(hb, 1)), n - 1)

    mae = np.full(len(lab), np.nan)
    mfe = np.full(len(lab), np.nan)
    entry_prices = lab["entry_price"].to_numpy(dtype=float) if "entry_price" in lab.columns else None

    for i in range(len(lab)):
        ei, xj = int(entry_idx[i]), int(exit_idx[i])
        if ei < 0 or xj < 0 or ei >= n or xj >= n or xj < ei:
            continue
        entry = float(entry_prices[i]) if entry_prices is not None else float(c["close"].iloc[ei])
        if entry == 0 or not np.isfinite(entry):
            continue
        path_h = high[ei : xj + 1]
        path_l = low[ei : xj + 1]
        if side == "long":
            mfe[i] = float((np.max(path_h) - entry) / entry)
            mae[i] = float((entry - np.min(path_l)) / entry)
        else:
            mfe[i] = float((entry - np.min(path_l)) / entry)
            mae[i] = float((np.max(path_h) - entry) / entry)

    out = lab[["timestamp"]].copy()
    out["mae"] = mae
    out["mfe"] = mfe
    if "holding_bars" in lab.columns:
        out["holding_bars"] = lab["holding_bars"].to_numpy()
    else:
        out["holding_bars"] = np.maximum(exit_idx - entry_idx, 0)
    return out


def enrich_predictions(
    predictions: pd.DataFrame,
    labels_by_side: dict[str, pd.DataFrame],
    candles: pd.DataFrame | None,
) -> pd.DataFrame:
    """Attach holding / MAE / MFE / net-ready fields onto OOF predictions."""
    parts: list[pd.DataFrame] = []
    for side, g in predictions.groupby("side"):
        lab = labels_by_side.get(str(side))
        if lab is None or lab.empty:
            gg = g.copy()
            gg["holding_bars"] = np.nan
            gg["mae"] = np.nan
            gg["mfe"] = np.nan
            parts.append(gg)
            continue
        lab = lab.copy()
        lab["timestamp"] = pd.to_datetime(lab["timestamp"], utc=True)
        # Restrict path stats to prediction timestamps only (ponytail: avoid full-history scan)
        pred_ts = set(pd.to_datetime(g["timestamp"], utc=True).tolist())
        lab_sub = lab.loc[lab["timestamp"].isin(pred_ts)].copy()
        keep = ["timestamp", "holding_bars", "realized_return", "entry_price", "exit_reason"]
        if "metadata_json" in lab_sub.columns:
            keep.append("metadata_json")
        keep = [c for c in keep if c in lab_sub.columns]
        merged = g.copy()
        merged["timestamp"] = pd.to_datetime(merged["timestamp"], utc=True)
        extra = lab_sub[keep].drop_duplicates("timestamp")
        # Prefer prediction realized_return if already present
        if "realized_return" in merged.columns and "realized_return" in extra.columns:
            extra = extra.drop(columns=["realized_return"])
        merged = merged.merge(extra, on="timestamp", how="left")

        if candles is not None and not candles.empty and not lab_sub.empty:
            path = compute_path_excursions(candles, lab_sub, side=str(side))
            path = path.drop(columns=["holding_bars"], errors="ignore")
            merged = merged.merge(path, on="timestamp", how="left")
        else:
            merged["mae"] = np.nan
            merged["mfe"] = np.nan
        parts.append(merged)
    return pd.concat(parts, ignore_index=True) if parts else predictions


def trade_quality_stats(frame: pd.DataFrame, *, cost: float) -> dict[str, float]:
    if frame.empty:
        nan = float("nan")
        return {
            "n_trades": 0,
            "win_rate": nan,
            "avg_return": nan,
            "median_return": nan,
            "expectancy": nan,
            "profit_factor": nan,
            "max_drawdown": nan,
            "avg_holding_bars": nan,
            "avg_mae": nan,
            "avg_mfe": nan,
        }
    rets = frame["realized_return"].to_numpy(dtype=float)
    net = apply_cost(rets, cost)
    meta_win = (net > 0).astype(int)
    return {
        "n_trades": int(len(frame)),
        "win_rate": float(np.mean(meta_win)),
        "avg_return": float(np.mean(net)),
        "median_return": float(np.median(net)),
        "expectancy": float(np.mean(net)),
        "profit_factor": profit_factor(net),
        "max_drawdown": max_drawdown(net),
        "avg_holding_bars": float(np.nanmean(frame["holding_bars"].to_numpy(dtype=float)))
        if "holding_bars" in frame.columns
        else float("nan"),
        "avg_mae": float(np.nanmean(frame["mae"].to_numpy(dtype=float)))
        if "mae" in frame.columns
        else float("nan"),
        "avg_mfe": float(np.nanmean(frame["mfe"].to_numpy(dtype=float)))
        if "mfe" in frame.columns
        else float("nan"),
    }


def build_candidate_trades(
    predictions: pd.DataFrame,
    *,
    prob_col: str = "y_prob_raw",
    percentiles: tuple[float, ...] = CANDIDATE_PERCENTILES,
) -> pd.DataFrame:
    """Per-window top-percentile candidates with meta_label."""
    if prob_col not in predictions.columns and "y_prob" in predictions.columns:
        prob_col = "y_prob"
    rows: list[pd.DataFrame] = []
    for (side, window), g in predictions.groupby(["side", "window"], sort=True):
        for pct in percentiles:
            picked = select_top_percentile(g, prob_col, pct)
            if picked.empty:
                continue
            part = picked.copy()
            part["percentile"] = pct
            part["prob_col"] = prob_col
            rows.append(part)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def attach_meta_label(candidates: pd.DataFrame, *, cost: float) -> pd.DataFrame:
    out = candidates.copy()
    net = apply_cost(out["realized_return"].to_numpy(dtype=float), cost)
    out["net_return"] = net
    out["meta_label"] = (net > 0).astype(int)
    return out


def build_percentile_summary(candidates: pd.DataFrame, *, cost: float) -> pd.DataFrame:
    rows = []
    for (side, pct), g in candidates.groupby(["side", "percentile"], sort=True):
        st = trade_quality_stats(g, cost=cost)
        rows.append({"side": side, "percentile": float(pct), **st})
    return pd.DataFrame(rows)


def build_wf_summary(candidates: pd.DataFrame, *, cost: float) -> pd.DataFrame:
    rows = []
    for (side, pct, window, year), g in candidates.groupby(
        ["side", "percentile", "window", "valid_year"], sort=True
    ):
        st = trade_quality_stats(g, cost=cost)
        rows.append(
            {
                "side": side,
                "percentile": float(pct),
                "window": window,
                "valid_year": int(year),
                **st,
                "positive_expectancy": bool(
                    st["n_trades"] > 0
                    and st["expectancy"] == st["expectancy"]
                    and st["expectancy"] > 0
                ),
            }
        )
    return pd.DataFrame(rows)


def build_stability(candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (side, pct), g in candidates.groupby(["side", "percentile"], sort=True):
        per_win = g.groupby("window").size()
        g2 = g.copy()
        g2["year"] = pd.to_datetime(g2["timestamp"], utc=True).dt.year
        per_year = g2.groupby("year").size()
        pos = int((g["meta_label"] == 1).sum())
        neg = int((g["meta_label"] == 0).sum())
        n = int(len(g))
        rows.append(
            {
                "side": side,
                "percentile": float(pct),
                "n_total": n,
                "n_windows": int(g["window"].nunique()),
                "samples_per_wf_mean": float(per_win.mean()) if len(per_win) else float("nan"),
                "samples_per_wf_min": int(per_win.min()) if len(per_win) else 0,
                "samples_per_year_mean": float(per_year.mean()) if len(per_year) else float("nan"),
                "positive_labels": pos,
                "negative_labels": neg,
                "positive_pct": float(pos / n) if n else float("nan"),
                "negative_pct": float(neg / n) if n else float("nan"),
                "imbalance_ratio_neg_over_pos": float(neg / pos) if pos else float("nan"),
                "enough_total": bool(n >= MIN_TOTAL_SAMPLES),
                "enough_per_window": bool(len(per_win) > 0 and int(per_win.min()) >= MIN_PER_WINDOW),
                "meta_trainable_heuristic": bool(
                    n >= MIN_TOTAL_SAMPLES
                    and len(per_win) > 0
                    and int(per_win.min()) >= MIN_PER_WINDOW
                    and pos >= 30
                    and neg >= 30
                ),
            }
        )
    return pd.DataFrame(rows)


def answer_research(
    *,
    summary: pd.DataFrame,
    wf: pd.DataFrame,
    stability: pd.DataFrame,
) -> dict:
    """Pick best quality / stability / sample percentiles per side."""

    def _best_quality(side: str) -> dict:
        sub = summary.loc[summary["side"] == side].copy()
        if sub.empty:
            return {"percentile": float("nan")}
        pos = sub.loc[sub["expectancy"] > 0]
        cand = pos if not pos.empty else sub
        cand = cand.sort_values(
            ["expectancy", "profit_factor", "win_rate"], ascending=[False, False, False]
        )
        r = cand.iloc[0]
        return {
            "percentile": float(r["percentile"]),
            "expectancy": float(r["expectancy"]),
            "profit_factor": float(r["profit_factor"]),
            "win_rate": float(r["win_rate"]),
            "n_trades": int(r["n_trades"]),
        }

    def _best_stable(side: str) -> dict:
        sub = wf.loc[wf["side"] == side]
        if sub.empty:
            return {"percentile": float("nan")}
        rows = []
        for pct, g in sub.groupby("percentile"):
            n = len(g)
            pos_w = int(g["positive_expectancy"].sum())
            rows.append(
                {
                    "percentile": float(pct),
                    "wf_positive_frac": float(pos_w / n) if n else 0.0,
                    "wf_expectancy_mean": float(g["expectancy"].mean()),
                    "wf_expectancy_std": float(g["expectancy"].std(ddof=1)) if n > 1 else 0.0,
                    "n_windows": n,
                }
            )
        tab = pd.DataFrame(rows)
        tab = tab.sort_values(
            ["wf_positive_frac", "wf_expectancy_mean", "wf_expectancy_std"],
            ascending=[False, False, True],
        )
        r = tab.iloc[0]
        return r.to_dict()

    def _enough(side: str) -> list[float]:
        sub = stability.loc[
            (stability["side"] == side) & (stability["meta_trainable_heuristic"])
        ]
        return [float(x) for x in sub["percentile"].tolist()]

    out = {}
    for side in ("long", "short"):
        out[side] = {
            "best_quality": _best_quality(side),
            "best_stable": _best_stable(side),
            "enough_percentiles": _enough(side),
        }
    return out
