"""Descriptive ATRE diagnostics (MAE, underwater, P(TP|), sessions)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def mae_sl_rate(lib: list[dict[str, Any]], thresholds: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0, 1.2)) -> pd.DataFrame:
    rows = []
    for thr in thresholds:
        hit = [p for p in lib if p.get("ok") and p.get("max_adverse_atr", 0) >= thr]
        if not hit:
            rows.append({"mae_atr": thr, "n": 0, "sl_rate": np.nan, "tp_rate": np.nan, "mean_net": np.nan})
            continue
        sl = np.mean([p["hit_sl"] for p in hit])
        tp = np.mean([p["hit_tp"] for p in hit])
        rows.append(
            {
                "mae_atr": thr,
                "n": len(hit),
                "sl_rate": float(sl),
                "tp_rate": float(tp),
                "mean_net": float(np.mean([p["net_return"] for p in hit])),
            }
        )
    return pd.DataFrame(rows)


def underwater_analysis(lib: list[dict[str, Any]], bars=(1, 2, 4, 8, 12)) -> pd.DataFrame:
    rows = []
    for b in bars:
        hit = [p for p in lib if p.get("ok") and p.get("underwater_bars", 0) >= b]
        if not hit:
            rows.append({"underwater_bars": b, "n": 0, "tp_rate": np.nan, "mean_net": np.nan})
            continue
        rows.append(
            {
                "underwater_bars": b,
                "n": len(hit),
                "tp_rate": float(np.mean([p["hit_tp"] for p in hit])),
                "sl_rate": float(np.mean([p["hit_sl"] for p in hit])),
                "mean_net": float(np.mean([p["net_return"] for p in hit])),
            }
        )
    return pd.DataFrame(rows)


def recovery_probability(lib: list[dict[str, Any]], thresholds=(0.3, 0.5, 0.6, 0.8, 1.0, 1.2)) -> pd.DataFrame:
    """P(TP | first time floating adverse >= thr ATR) — using path max_adverse gate as proxy."""
    rows = []
    for thr in thresholds:
        # trades that reached thr adverse before finish
        reached = []
        for p in lib:
            if not p.get("ok") or not p.get("bars"):
                continue
            first = next((b for b in p["bars"] if b["adverse_atr"] >= thr), None)
            if first is None:
                continue
            reached.append(p)
        if not reached:
            rows.append({"cond_mae_atr": thr, "n": 0, "p_tp": np.nan, "p_sl": np.nan, "mean_net": np.nan})
            continue
        rows.append(
            {
                "cond_mae_atr": thr,
                "n": len(reached),
                "p_tp": float(np.mean([p["hit_tp"] for p in reached])),
                "p_sl": float(np.mean([p["hit_sl"] for p in reached])),
                "mean_net": float(np.mean([p["net_return"] for p in reached])),
            }
        )
    return pd.DataFrame(rows)


def momentum_collapse_edge(lib: list[dict[str, Any]]) -> dict[str, Any]:
    with_m = [p for p in lib if p.get("ok") and any(b.get("mom_collapse") for b in p.get("bars", []))]
    without = [p for p in lib if p.get("ok") and not any(b.get("mom_collapse") for b in p.get("bars", []))]
    return {
        "n_collapse": len(with_m),
        "n_no": len(without),
        "mean_net_collapse": float(np.mean([p["net_return"] for p in with_m])) if with_m else np.nan,
        "mean_net_no": float(np.mean([p["net_return"] for p in without])) if without else np.nan,
        "tp_rate_collapse": float(np.mean([p["hit_tp"] for p in with_m])) if with_m else np.nan,
        "tp_rate_no": float(np.mean([p["hit_tp"] for p in without])) if without else np.nan,
    }


def structure_fail_edge(lib: list[dict[str, Any]]) -> dict[str, Any]:
    with_s = [p for p in lib if p.get("ok") and any(b.get("struct_fail") for b in p.get("bars", []))]
    without = [p for p in lib if p.get("ok") and not any(b.get("struct_fail") for b in p.get("bars", []))]
    return {
        "n_fail": len(with_s),
        "n_ok": len(without),
        "mean_net_fail": float(np.mean([p["net_return"] for p in with_s])) if with_s else np.nan,
        "mean_net_ok": float(np.mean([p["net_return"] for p in without])) if without else np.nan,
        "tp_rate_fail": float(np.mean([p["hit_tp"] for p in with_s])) if with_s else np.nan,
    }


def session_recovery(lib: list[dict[str, Any]], mae_thr: float = 0.6) -> pd.DataFrame:
    rows = []
    for name, pred in (
        ("asia", lambda p: p.get("session_asia", 0) >= 0.5),
        ("london", lambda p: p.get("session_london", 0) >= 0.5),
        ("overlap", lambda p: p.get("session_overlap", 0) >= 0.5),
        ("newyork", lambda p: p.get("session_newyork", 0) >= 0.5),
    ):
        sub = [p for p in lib if p.get("ok") and pred(p) and p.get("max_adverse_atr", 0) >= mae_thr]
        rows.append(
            {
                "session": name,
                "n": len(sub),
                "p_tp": float(np.mean([p["hit_tp"] for p in sub])) if sub else np.nan,
                "mean_net": float(np.mean([p["net_return"] for p in sub])) if sub else np.nan,
            }
        )
    return pd.DataFrame(rows)


def vol_recovery(lib: list[dict[str, Any]], mae_thr: float = 0.6) -> pd.DataFrame:
    reached = [p for p in lib if p.get("ok") and p.get("max_adverse_atr", 0) >= mae_thr]
    if not reached:
        return pd.DataFrame()
    med = float(np.median([p.get("vol_rank", 0.5) for p in reached]))
    rows = []
    for name, pred in (("high_vol", lambda p: p.get("vol_rank", 0.5) >= med), ("low_vol", lambda p: p.get("vol_rank", 0.5) < med)):
        sub = [p for p in reached if pred(p)]
        rows.append(
            {
                "vol_bucket": name,
                "n": len(sub),
                "p_tp": float(np.mean([p["hit_tp"] for p in sub])) if sub else np.nan,
                "mean_net": float(np.mean([p["net_return"] for p in sub])) if sub else np.nan,
            }
        )
    return pd.DataFrame(rows)


def side_recovery(lib: list[dict[str, Any]], mae_thr: float = 0.6) -> pd.DataFrame:
    rows = []
    for side in ("long", "short"):
        sub = [p for p in lib if p.get("ok") and p.get("side") == side and p.get("max_adverse_atr", 0) >= mae_thr]
        rows.append(
            {
                "side": side,
                "n": len(sub),
                "p_tp": float(np.mean([p["hit_tp"] for p in sub])) if sub else np.nan,
                "mean_net": float(np.mean([p["net_return"] for p in sub])) if sub else np.nan,
            }
        )
    return pd.DataFrame(rows)
