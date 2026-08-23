"""Deterministic M5 execution engine — no ML."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from research.mtf_h4_h1_m5.contracts import M5Action, M5ExecutionSignal

StrategyName = Literal[
    "immediate",
    "delay_1",
    "delay_2",
    "delay_3",
    "delay_4",
    "momentum",
    "pullback_recovery",
    "structure",
]

MAX_WAIT_M5 = 48  # = 4 H1 bars
MAX_WAIT_M15 = 16  # = 4 H1 bars
PULLBACK_ATR = 0.15
RECOVERY_FRAC = 0.5
ADV_INVALIDATE_R = 1.0  # skip if M5 moves 1R against before execute


def _ts(v) -> pd.Timestamp:
    t = pd.Timestamp(v)
    return t.tz_convert("UTC") if t.tzinfo else t.tz_localize("UTC")


def _wilder_atr(h: np.ndarray, l: np.ndarray, c: np.ndarray, n: int = 14) -> np.ndarray:
    prev = np.roll(c, 1)
    prev[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev), np.abs(l - prev)))
    atr = np.full_like(c, np.nan)
    if len(c) < n:
        return atr
    atr[n - 1] = tr[:n].mean()
    alpha = 1.0 / n
    for i in range(n, len(c)):
        atr[i] = atr[i - 1] * (1 - alpha) + tr[i] * alpha
    return atr


def build_m5_features(m5: pd.DataFrame) -> pd.DataFrame:
    """Causal M5 native + structure features (compact set)."""
    x = m5.sort_values("timestamp").reset_index(drop=True).copy()
    x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True)
    o = x["open"].astype(float).to_numpy()
    h = x["high"].astype(float).to_numpy()
    l = x["low"].astype(float).to_numpy()
    c = x["close"].astype(float).to_numpy()
    atr = _wilder_atr(h, l, c)
    rng = np.maximum(h - l, 1e-12)
    n = len(c)

    def ret(k: int) -> np.ndarray:
        out = np.full(n, np.nan)
        out[k:] = (c[k:] - c[:-k]) / np.maximum(atr[k:], 1e-12)
        return out

    r1, r3, r6, r12 = ret(1), ret(3), ret(6), ret(12)
    mom_slope = pd.Series(r3).diff(3).to_numpy() / 3.0
    hi20 = pd.Series(h).rolling(20, min_periods=20).max().to_numpy()
    lo20 = pd.Series(l).rolling(20, min_periods=20).min().to_numpy()
    close_pos = (c - lo20) / np.maximum(hi20 - lo20, 1e-12)
    dist_hi = (hi20 - c) / np.maximum(atr, 1e-12)
    dist_lo = (c - lo20) / np.maximum(atr, 1e-12)
    body = np.abs(c - o) / rng
    up_wick = (h - np.maximum(o, c)) / rng
    lo_wick = (np.minimum(o, c) - l) / rng
    range_atr = rng / np.maximum(atr, 1e-12)
    sh = pd.Series(h).rolling(5, min_periods=5).max().to_numpy()
    sl = pd.Series(l).rolling(5, min_periods=5).min().to_numpy()
    swing_dir = np.sign(sh - np.roll(sh, 5)) - np.sign(sl - np.roll(sl, 5))

    out = pd.DataFrame(
        {
            "timestamp": x["timestamp"],
            "open": o,
            "high": h,
            "low": l,
            "close": c,
            "atr": atr,
            "m5_ret_1": r1,
            "m5_ret_3": r3,
            "m5_ret_6": r6,
            "m5_ret_12": r12,
            "m5_mom_slope": mom_slope,
            "m5_close_pos": close_pos,
            "m5_dist_high": dist_hi,
            "m5_dist_low": dist_lo,
            "m5_body_ratio": body,
            "m5_upper_wick": up_wick,
            "m5_lower_wick": lo_wick,
            "m5_range_atr": range_atr,
            "m5_swing_dir": swing_dir,
        }
    )
    return out


@dataclass(frozen=True)
class M5ExecResult:
    action: M5Action
    reason: str
    execute_idx: int | None
    delay_m5_bars: int
    fill_price: float | None
    available_timestamp: pd.Timestamp | None


def _aligned_momentum(side: str, ret3: float) -> bool:
    if not np.isfinite(ret3):
        return False
    return ret3 > 0 if side == "long" else ret3 < 0


def _pullback_recovery(
    side: str,
    ref: float,
    atr: float,
    low: float,
    high: float,
    close: float,
    pullback_atr: float = PULLBACK_ATR,
    recovery_frac: float = RECOVERY_FRAC,
) -> bool:
    if atr <= 0:
        return False
    if side == "long":
        pb = (ref - low) / atr
        if pb < pullback_atr:
            return False
        return close > ref - (ref - low) * (1 - recovery_frac)
    pb = (high - ref) / atr
    if pb < pullback_atr:
        return False
    return close < ref + (high - ref) * (1 - recovery_frac)


def build_m15_bars(m15: pd.DataFrame) -> pd.DataFrame:
    """M15 OHLC + ATR for pullback entry rules."""
    x = m15.sort_values("timestamp").reset_index(drop=True).copy()
    x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True)
    h = x["high"].astype(float).to_numpy()
    l = x["low"].astype(float).to_numpy()
    c = x["close"].astype(float).to_numpy()
    atr = _wilder_atr(h, l, c)
    return pd.DataFrame({
        "timestamp": x["timestamp"],
        "open": x["open"].astype(float),
        "high": h,
        "low": l,
        "close": c,
        "atr": atr,
    })


def _scan_pullback_execution(
    *,
    side: str,
    h1_signal_ts: pd.Timestamp,
    h1_ref_price: float,
    one_r: float,
    bars: pd.DataFrame,
    start: int,
    max_wait: int,
    strategy: StrategyName,
    pullback_atr: float,
    recovery_frac: float,
) -> M5ExecResult:
    delay_map = {"immediate": 0, "delay_1": 1, "delay_2": 2, "delay_3": 3, "delay_4": 4}
    if strategy in delay_map:
        d = delay_map[strategy]
        idx = start + d
        if idx >= len(bars):
            return M5ExecResult("INVALIDATE", "delay_past_horizon", None, d, None, None)
        row = bars.iloc[idx]
        avail = _ts(row["timestamp"])
        fill = float(row["open"])
        return M5ExecResult("EXECUTE", f"fixed_delay_{d}", idx, d, fill, avail)

    end = min(len(bars), start + max_wait)
    for j in range(start, end):
        row = bars.iloc[j]
        atr = float(row["atr"])
        if atr <= 0 or not np.isfinite(atr):
            continue
        h, l, c = float(row["high"]), float(row["low"]), float(row["close"])
        adv = ((h1_ref_price - l) if side == "long" else (h - h1_ref_price)) / max(one_r, 1e-12)
        if adv >= ADV_INVALIDATE_R:
            return M5ExecResult("INVALIDATE", "adverse_move_limit", None, j - start, None, None)

        if strategy == "pullback_recovery":
            if _pullback_recovery(side, h1_ref_price, atr, l, h, c, pullback_atr, recovery_frac):
                nxt = j + 1
                if nxt >= len(bars):
                    return M5ExecResult("EXECUTE", "pullback_recovery", j, j - start, float(row["close"]), _ts(row["timestamp"]))
                return M5ExecResult(
                    "EXECUTE",
                    "pullback_recovery",
                    nxt,
                    j - start + 1,
                    float(bars.iloc[nxt]["open"]),
                    _ts(bars.iloc[nxt]["timestamp"]),
                )

    return M5ExecResult("INVALIDATE", "max_wait_exceeded", None, max_wait, None, None)


def decide_m15_execution(
    *,
    side: str,
    h1_signal_ts: pd.Timestamp,
    h1_ref_price: float,
    one_r: float,
    m15: pd.DataFrame,
    strategy: StrategyName = "pullback_recovery",
    pullback_atr: float = PULLBACK_ATR,
    recovery_frac: float = RECOVERY_FRAC,
) -> M5ExecResult:
    """Scan M15 bars after H1 signal; same pullback rule as M5."""
    h1_signal_ts = pd.to_datetime(h1_signal_ts, utc=True)
    ts = pd.to_datetime(m15["timestamp"], utc=True)
    start = int(ts.searchsorted(h1_signal_ts, side="right"))
    if start >= len(m15):
        return M5ExecResult("INVALIDATE", "no_m15_after_signal", None, 0, None, None)
    return _scan_pullback_execution(
        side=side, h1_signal_ts=h1_signal_ts, h1_ref_price=h1_ref_price, one_r=one_r,
        bars=m15, start=start, max_wait=MAX_WAIT_M15, strategy=strategy,
        pullback_atr=pullback_atr, recovery_frac=recovery_frac,
    )


def decide_m5_execution(
    *,
    side: str,
    h1_signal_ts: pd.Timestamp,
    h1_ref_price: float,
    one_r: float,
    m5: pd.DataFrame,
    strategy: StrategyName,
    pullback_atr: float = PULLBACK_ATR,
    recovery_frac: float = RECOVERY_FRAC,
) -> M5ExecResult:
    """Scan M5 bars after H1 signal; deterministic rule only."""
    h1_signal_ts = pd.to_datetime(h1_signal_ts, utc=True)
    ts = pd.to_datetime(m5["timestamp"], utc=True)
    start = int(ts.searchsorted(h1_signal_ts, side="right"))
    if start >= len(m5):
        return M5ExecResult("INVALIDATE", "no_m5_after_signal", None, 0, None, None)

    delay_map = {"immediate": 0, "delay_1": 1, "delay_2": 2, "delay_3": 3, "delay_4": 4}
    if strategy in delay_map:
        d = delay_map[strategy]
        idx = start + d
        if idx >= len(m5):
            return M5ExecResult("INVALIDATE", "delay_past_horizon", None, d, None, None)
        row = m5.iloc[idx]
        avail = _ts(row["timestamp"])
        fill = float(row["open"])
        return M5ExecResult("EXECUTE", f"fixed_delay_{d}", idx, d, fill, avail)

    end = min(len(m5), start + MAX_WAIT_M5)
    best_pb = 0.0
    for j in range(start, end):
        row = m5.iloc[j]
        atr = float(row["atr"])
        if atr <= 0 or not np.isfinite(atr):
            continue
        h, l, c = float(row["high"]), float(row["low"]), float(row["close"])
        adv = ((h1_ref_price - l) if side == "long" else (h - h1_ref_price)) / max(one_r, 1e-12)
        if adv >= ADV_INVALIDATE_R:
            return M5ExecResult("INVALIDATE", "adverse_move_limit", None, j - start, None, None)

        if strategy == "momentum":
            if _aligned_momentum(side, float(row["m5_ret_3"])):
                fill = float(m5.iloc[j + 1]["open"]) if j + 1 < len(m5) else float(row["close"])
                avail = _ts(m5.iloc[j + 1]["timestamp"]) if j + 1 < len(m5) else _ts(row["timestamp"])
                return M5ExecResult("EXECUTE", "momentum_aligned", j + 1 if j + 1 < len(m5) else j, j - start + 1, fill, avail)

        elif strategy == "pullback_recovery":
            if side == "long":
                best_pb = max(best_pb, (h1_ref_price - l) / atr)
            else:
                best_pb = max(best_pb, (h - h1_ref_price) / atr)
            if _pullback_recovery(side, h1_ref_price, atr, l, h, c, pullback_atr, recovery_frac):
                nxt = j + 1
                if nxt >= len(m5):
                    return M5ExecResult("EXECUTE", "pullback_recovery", j, j - start, float(row["close"]), _ts(row["timestamp"]))
                return M5ExecResult(
                    "EXECUTE",
                    "pullback_recovery",
                    nxt,
                    j - start + 1,
                    float(m5.iloc[nxt]["open"]),
                    _ts(m5.iloc[nxt]["timestamp"]),
                )

        elif strategy == "structure":
            sd = float(row["m5_swing_dir"])
            ok = (sd > 0 and side == "long") or (sd < 0 and side == "short")
            if ok:
                nxt = j + 1
                if nxt >= len(m5):
                    return M5ExecResult("EXECUTE", "structure_confirm", j, j - start, float(row["close"]), _ts(row["timestamp"]))
                return M5ExecResult(
                    "EXECUTE",
                    "structure_confirm",
                    nxt,
                    j - start + 1,
                    float(m5.iloc[nxt]["open"]),
                    _ts(m5.iloc[nxt]["timestamp"]),
                )

    return M5ExecResult("INVALIDATE", "max_wait_exceeded", None, MAX_WAIT_M5, None, None)


def to_execution_signal(res: M5ExecResult, *, m5_ts: pd.Timestamp | None = None) -> M5ExecutionSignal:
    ts = res.available_timestamp or m5_ts or pd.Timestamp.now(tz="UTC")
    return M5ExecutionSignal(
        action=res.action,
        reason=res.reason,
        signal_timestamp=ts.isoformat(),
        available_timestamp=ts.isoformat(),
        delay_m5_bars=int(res.delay_m5_bars),
    )
