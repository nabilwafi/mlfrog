"""ATRE recovery policies applied on path library (same entries)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from research.atre import COST, SL_ATR
from research.atre.services.path_diag import _pnl_frac


@dataclass(frozen=True)
class AtrePolicy:
    name: str
    family: str  # baseline | early_exit | reduce | compress | score_exit | score_reduce | mom | struct
    mae_atr: float | None = None
    underwater: int | None = None
    score_thr: float | None = None  # exit/reduce if recovery_score >= thr
    reduce_frac: float = 0.0
    compress_sl_atr: float | None = None
    require_mom_collapse: bool = False
    require_struct_fail: bool = False
    side_filter: str | None = None  # long | short | None


def build_atre_policies() -> list[AtrePolicy]:
    pols = [AtrePolicy(name="baseline", family="baseline")]
    for m in (0.4, 0.6, 0.8, 1.0, 1.2):
        pols.append(AtrePolicy(name=f"early_exit_{m}atr", family="early_exit", mae_atr=m))
    for u in (2, 4, 6, 8):
        pols.append(AtrePolicy(name=f"exit_uw_{u}b", family="early_exit", underwater=u))
    for m in (0.6, 0.8, 1.0):
        pols.append(
            AtrePolicy(name=f"reduce50_at_{m}atr", family="reduce", mae_atr=m, reduce_frac=0.5)
        )
    for c in (1.0, 0.8):
        for m in (0.5, 0.7):
            pols.append(
                AtrePolicy(
                    name=f"compress_sl_{c}_trig_{m}",
                    family="compress",
                    mae_atr=m,
                    compress_sl_atr=c,
                )
            )
    for thr in (55.0, 60.0, 65.0, 70.0, 75.0):
        pols.append(AtrePolicy(name=f"score_exit_{int(thr)}", family="score_exit", score_thr=thr, mae_atr=0.4))
        pols.append(
            AtrePolicy(
                name=f"score_reduce50_{int(thr)}",
                family="score_reduce",
                score_thr=thr,
                mae_atr=0.4,
                reduce_frac=0.5,
            )
        )
    pols.append(AtrePolicy(name="mom_collapse_exit", family="mom", require_mom_collapse=True, mae_atr=0.4))
    pols.append(AtrePolicy(name="struct_fail_exit", family="struct", require_struct_fail=True, mae_atr=0.3))
    # side-specific early exit
    for side in ("long", "short"):
        pols.append(
            AtrePolicy(name=f"early_exit_0.8atr_{side}", family="early_exit", mae_atr=0.8, side_filter=side)
        )
    return pols


def apply_policy(
    path: dict[str, Any],
    policy: AtrePolicy,
    *,
    recovery_score: float | None = None,
) -> dict[str, Any]:
    """Return net_return under policy; baseline uses path net."""
    if not path.get("ok"):
        return {"net_return": float(path.get("net_return", -COST)), "intervened": False, "reason": "bad"}

    if policy.family == "baseline":
        return {"net_return": float(path["net_return"]), "intervened": False, "reason": "baseline"}

    if policy.side_filter and path.get("side") != policy.side_filter:
        return {"net_return": float(path["net_return"]), "intervened": False, "reason": "side_skip"}

    side = path["side"]
    entry = path["entry"]
    atr = path["atr_entry"]
    bars = path["bars"]
    if not bars:
        return {"net_return": float(path["net_return"]), "intervened": False, "reason": "nobars"}

    # compress: tighten SL from bar trigger onward (re-simulate remainder)
    if policy.compress_sl_atr is not None and policy.mae_atr is not None:
        return _compress(path, policy)

    size = 1.0
    realized = 0.0
    intervened = False
    reason = "baseline_finish"

    for b in bars:
        fire = True
        if policy.mae_atr is not None and b["adverse_atr"] < policy.mae_atr:
            fire = False
        if policy.underwater is not None and b["underwater_bars"] < policy.underwater:
            fire = False
        if policy.require_mom_collapse and not b.get("mom_collapse"):
            fire = False
        if policy.require_struct_fail and not b.get("struct_fail"):
            fire = False
        if policy.score_thr is not None:
            if recovery_score is None or recovery_score < policy.score_thr:
                fire = False
            # only after mild adverse so we don't exit winners early for no reason
            if policy.mae_atr is not None and b["adverse_atr"] < policy.mae_atr:
                fire = False

        if fire and size > 0:
            intervened = True
            px = float(b["close"])
            if policy.reduce_frac > 0 and policy.reduce_frac < 1:
                take = policy.reduce_frac * size
                realized += take * _pnl_frac(side, entry, px)
                size -= take
                reason = "reduce"
                # continue with remainder to original outcome approximation:
                # use final path exit for remainder
                final_px = path["bars"][-1]["close"]
                if path["hit_sl"]:
                    final_px = path["sl"]
                elif path["hit_tp"]:
                    final_px = path["tp"]
                realized += size * _pnl_frac(side, entry, final_px)
                size = 0.0
                reason = "reduce_then_finish"
                break
            else:
                realized += size * _pnl_frac(side, entry, px)
                size = 0.0
                reason = "early_exit"
                break

    if size > 0:
        # finish original
        final_px = path["bars"][-1]["close"]
        if path["hit_sl"]:
            final_px = path["sl"]
        elif path["hit_tp"]:
            final_px = path["tp"]
        realized += size * _pnl_frac(side, entry, final_px)

    return {"net_return": float(realized - COST), "intervened": intervened, "reason": reason}


def _compress(path: dict[str, Any], policy: AtrePolicy) -> dict[str, Any]:
    """From first bar with adverse>=trig, SL becomes compress_sl_atr * atr."""
    side = path["side"]
    entry = path["entry"]
    atr = path["atr_entry"]
    is_long = side == "long"
    sl = path["sl"]
    tp = path["tp"]
    trig = float(policy.mae_atr or 0.5)
    new_dist = float(policy.compress_sl_atr or 1.0) * atr
    activated = False
    for b in path["bars"]:
        if not activated and b["adverse_atr"] >= trig:
            activated = True
            sl = entry - new_dist if is_long else entry + new_dist
        # approximate: if adverse already beyond new SL, exit at compressed SL price
        if activated:
            adv_px = entry - b["adverse_atr"] * atr if is_long else entry + b["adverse_atr"] * atr
            hit_new = (adv_px <= sl) if is_long else (adv_px >= sl)
            if hit_new:
                return {
                    "net_return": float(_pnl_frac(side, entry, sl) - COST),
                    "intervened": True,
                    "reason": "compress_hit",
                }
            # if favor reached TP after compress
            if b["favor_atr"] * atr >= abs(tp - entry):
                return {
                    "net_return": float(_pnl_frac(side, entry, tp) - COST),
                    "intervened": True,
                    "reason": "compress_tp",
                }
    # no hit of compressed stop — use original end
    return {
        "net_return": float(path["net_return"]),
        "intervened": activated,
        "reason": "compress_held" if activated else "no_trig",
    }


def simulate_policy_panel(
    trades: pd.DataFrame,
    lib: list[dict[str, Any]],
    policy: AtrePolicy,
    scores: dict[int, float],
) -> pd.DataFrame:
    out = trades.reset_index(drop=True).copy()
    nets, intervening, reasons = [], [], []
    for i, p in enumerate(lib):
        r = apply_policy(p, policy, recovery_score=scores.get(i))
        nets.append(r["net_return"])
        intervening.append(r["intervened"])
        reasons.append(r["reason"])
    out["net_return_base"] = out["net_return"].astype(float)
    out["net_return"] = nets
    out["atre_intervened"] = intervening
    out["atre_reason"] = reasons
    out["policy"] = policy.name
    out["family"] = policy.family
    return out
