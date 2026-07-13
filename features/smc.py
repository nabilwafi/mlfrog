"""
Smart Money Concepts (SMC) features with explicit no-lookahead.

Swing fractal confirmation:
  A swing high at index i (N=2) requires highs[i] > highs[i-N:i] and
  highs[i] > highs[i+1:i+N+1]. It is only CONFIRMED at index i+N.
  For features at time T (index t), we only use swings with confirm_index <= t.

Same rule for BOS/CHoCH/OB/FVG: events become usable only after they are
fully determined from past+present bars, never from future bars.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


SENTINEL_DIST = -1.0


@dataclass(frozen=True)
class SMCConfig:
    fractal_n: int = 2
    bos_lookback_bars: int = 3
    ob_impulse_atr_mult: float = 1.5
    ob_lookback_bars: int = 50
    fvg_lookback_bars: int = 50


def detect_fractals(high: np.ndarray, low: np.ndarray, n: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Return swing_high_price[i], swing_low_price[i] at pivot index i (NaN if not).

    Confirmation happens at i+n; callers must gate by confirm index.
    """
    m = len(high)
    sh = np.full(m, np.nan)
    sl = np.full(m, np.nan)
    for i in range(n, m - n):
        window_h = high[i - n : i + n + 1]
        window_l = low[i - n : i + n + 1]
        if high[i] == np.max(window_h) and high[i] > np.max(high[i - n : i]) and high[i] > np.max(high[i + 1 : i + n + 1]):
            sh[i] = high[i]
        if low[i] == np.min(window_l) and low[i] < np.min(low[i - n : i]) and low[i] < np.min(low[i + 1 : i + n + 1]):
            sl[i] = low[i]
    return sh, sl


def compute_smc_features(
    df: pd.DataFrame,
    *,
    atr_col: str = "atr_h1",
    prefix: str = "",
    cfg: SMCConfig = SMCConfig(),
) -> pd.DataFrame:
    """Compute SMC features aligned to df rows (Date must be close-time).

    Output columns use optional prefix (e.g. '' for H1, or handled separately for H4).
    """
    need = ["Date", "Open", "High", "Low", "Close", atr_col]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"SMC input missing: {missing}")

    out = pd.DataFrame({"Date": df["Date"].to_numpy()})
    high = df["High"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)
    close = df["Close"].to_numpy(dtype=float)
    open_ = df["Open"].to_numpy(dtype=float)
    atr = df[atr_col].to_numpy(dtype=float)
    n_bars = len(df)
    n = cfg.fractal_n

    sh_price, sl_price = detect_fractals(high, low, n=n)

    # Confirmed swings available at confirm_idx = pivot_idx + n
    # Build arrays of last confirmed swing as of each bar t.
    dist_sh = np.full(n_bars, SENTINEL_DIST)
    dist_sl = np.full(n_bars, SENTINEL_DIST)
    bars_sh = np.full(n_bars, SENTINEL_DIST)
    bars_sl = np.full(n_bars, SENTINEL_DIST)

    last_sh_idx = -1
    last_sh_px = np.nan
    last_sl_idx = -1
    last_sl_px = np.nan

    # Also keep history of confirmed swings for structure/trend
    sh_hist: list[tuple[int, float]] = []  # (pivot_idx, price)
    sl_hist: list[tuple[int, float]] = []

    bos_bull = np.zeros(n_bars, dtype=np.int8)
    bos_bear = np.zeros(n_bars, dtype=np.int8)
    choch_bull = np.zeros(n_bars, dtype=np.int8)
    choch_bear = np.zeros(n_bars, dtype=np.int8)

    # Structure state: 1=uptrend, -1=downtrend, 0=unknown
    # Uptrend: last two swing highs HH and/or last two swing lows HL
    # Downtrend: LH and/or LL
    structure = 0

    # Order blocks: list of dicts {side, top, bottom, born_idx, mitigated}
    bull_obs: list[dict] = []
    bear_obs: list[dict] = []
    dist_bull_ob = np.full(n_bars, SENTINEL_DIST)
    dist_bear_ob = np.full(n_bars, SENTINEL_DIST)

    # FVGs: {side, top, bottom, born_idx, filled}
    bull_fvgs: list[dict] = []
    bear_fvgs: list[dict] = []
    dist_fvg_bull = np.full(n_bars, SENTINEL_DIST)
    dist_fvg_bear = np.full(n_bars, SENTINEL_DIST)

    # Recent break events for lookback flags (confirm bar index, type)
    recent_events: list[tuple[int, str]] = []

    for t in range(n_bars):
        # 1) Newly confirmed swings at this bar t: pivot at t-n
        pivot = t - n
        if pivot >= n:
            if np.isfinite(sh_price[pivot]):
                last_sh_idx = pivot
                last_sh_px = float(sh_price[pivot])
                sh_hist.append((pivot, last_sh_px))
                structure = _update_structure(structure, sh_hist, sl_hist)
            if np.isfinite(sl_price[pivot]):
                last_sl_idx = pivot
                last_sl_px = float(sl_price[pivot])
                sl_hist.append((pivot, last_sl_px))
                structure = _update_structure(structure, sh_hist, sl_hist)

        # 2) Distances / bars since (only if confirmed swing exists)
        a = atr[t] if np.isfinite(atr[t]) and atr[t] > 0 else np.nan
        if last_sh_idx >= 0 and np.isfinite(a):
            dist_sh[t] = (last_sh_px - close[t]) / a
            bars_sh[t] = float(t - last_sh_idx)
        if last_sl_idx >= 0 and np.isfinite(a):
            dist_sl[t] = (close[t] - last_sl_px) / a
            bars_sl[t] = float(t - last_sl_idx)

        # 3) BOS / CHoCH on this bar using last confirmed swings BEFORE this break.
        # Break detected when high/low crosses the level on bar t.
        event = None
        if last_sh_idx >= 0 and high[t] > last_sh_px:
            if structure >= 0:
                event = "bos_bull"
            else:
                event = "choch_bull"
        if last_sl_idx >= 0 and low[t] < last_sl_px:
            # If both somehow, prefer the more extreme vs structure; keep both flags possible
            if structure <= 0:
                event2 = "bos_bear"
            else:
                event2 = "choch_bear"
            recent_events.append((t, event2))

        if event is not None:
            recent_events.append((t, event))

        # Flags: any matching event in last bos_lookback_bars including t
        lo = t - cfg.bos_lookback_bars + 1
        for et, typ in recent_events:
            if et < lo:
                continue
            if typ == "bos_bull":
                bos_bull[t] = 1
            elif typ == "bos_bear":
                bos_bear[t] = 1
            elif typ == "choch_bull":
                choch_bull[t] = 1
            elif typ == "choch_bear":
                choch_bear[t] = 1

        # Trim old events
        recent_events = [(et, typ) for et, typ in recent_events if et >= t - 50]

        # 4) Order blocks: impulse candle body >= mult*ATR
        if np.isfinite(a):
            body = abs(close[t] - open_[t])
            if body >= cfg.ob_impulse_atr_mult * a:
                if close[t] > open_[t]:
                    # bullish impulse -> last bearish candle before t is bullish OB
                    for j in range(t - 1, max(-1, t - 6), -1):
                        if close[j] < open_[j]:
                            bull_obs.append(
                                {
                                    "top": float(open_[j]),
                                    "bottom": float(close[j]),
                                    "born": t,
                                    "mitigated": False,
                                }
                            )
                            break
                else:
                    for j in range(t - 1, max(-1, t - 6), -1):
                        if close[j] > open_[j]:
                            bear_obs.append(
                                {
                                    "top": float(close[j]),
                                    "bottom": float(open_[j]),
                                    "born": t,
                                    "mitigated": False,
                                }
                            )
                            break

        # Mitigate OBs if price revisits zone on this bar
        for ob in bull_obs:
            if not ob["mitigated"] and low[t] <= ob["top"] and high[t] >= ob["bottom"]:
                # touch zone after birth
                if t > ob["born"]:
                    ob["mitigated"] = True
        for ob in bear_obs:
            if not ob["mitigated"] and low[t] <= ob["top"] and high[t] >= ob["bottom"]:
                if t > ob["born"]:
                    ob["mitigated"] = True

        # Nearest unmitigated OB within lookback
        if np.isfinite(a):
            best_b = None
            best_d = None
            for ob in bull_obs:
                if ob["mitigated"] or t - ob["born"] > cfg.ob_lookback_bars:
                    continue
                mid = 0.5 * (ob["top"] + ob["bottom"])
                d = abs(close[t] - mid) / a
                if best_d is None or d < best_d:
                    best_d = d
                    best_b = ob
            dist_bull_ob[t] = float(best_d) if best_d is not None else SENTINEL_DIST

            best_d = None
            for ob in bear_obs:
                if ob["mitigated"] or t - ob["born"] > cfg.ob_lookback_bars:
                    continue
                mid = 0.5 * (ob["top"] + ob["bottom"])
                d = abs(close[t] - mid) / a
                if best_d is None or d < best_d:
                    best_d = d
            dist_bear_ob[t] = float(best_d) if best_d is not None else SENTINEL_DIST

        # 5) FVG detection at bar t (uses t, t-1, t-2 only — no future)
        if t >= 2:
            # bullish FVG: low[t] > high[t-2]
            if low[t] > high[t - 2]:
                bull_fvgs.append(
                    {"top": float(low[t]), "bottom": float(high[t - 2]), "born": t, "filled": False}
                )
            # bearish FVG: high[t] < low[t-2]
            if high[t] < low[t - 2]:
                bear_fvgs.append(
                    {"top": float(low[t - 2]), "bottom": float(high[t]), "born": t, "filled": False}
                )

        for g in bull_fvgs:
            if not g["filled"] and t > g["born"] and low[t] <= g["bottom"]:
                g["filled"] = True
        for g in bear_fvgs:
            if not g["filled"] and t > g["born"] and high[t] >= g["top"]:
                g["filled"] = True

        if np.isfinite(a):
            best_d = None
            for g in bull_fvgs:
                if g["filled"] or t - g["born"] > cfg.fvg_lookback_bars:
                    continue
                mid = 0.5 * (g["top"] + g["bottom"])
                d = abs(close[t] - mid) / a
                if best_d is None or d < best_d:
                    best_d = d
            dist_fvg_bull[t] = float(best_d) if best_d is not None else SENTINEL_DIST

            best_d = None
            for g in bear_fvgs:
                if g["filled"] or t - g["born"] > cfg.fvg_lookback_bars:
                    continue
                mid = 0.5 * (g["top"] + g["bottom"])
                d = abs(close[t] - mid) / a
                if best_d is None or d < best_d:
                    best_d = d
            dist_fvg_bear[t] = float(best_d) if best_d is not None else SENTINEL_DIST

    p = prefix
    out[f"{p}dist_to_last_swing_high_h1" if not prefix else f"{p}dist_to_last_swing_high"] = dist_sh
    out[f"{p}dist_to_last_swing_low_h1" if not prefix else f"{p}dist_to_last_swing_low"] = dist_sl
    out[f"{p}bars_since_last_swing_high" if not prefix else f"{p}bars_since_last_swing_high"] = bars_sh
    out[f"{p}bars_since_last_swing_low" if not prefix else f"{p}bars_since_last_swing_low"] = bars_sl
    out[f"{p}bos_bullish_h1" if not prefix else f"{p}bos_bullish"] = bos_bull
    out[f"{p}bos_bearish_h1" if not prefix else f"{p}bos_bearish"] = bos_bear
    out[f"{p}choch_bullish_h1" if not prefix else f"{p}choch_bullish"] = choch_bull
    out[f"{p}choch_bearish_h1" if not prefix else f"{p}choch_bearish"] = choch_bear
    out[f"{p}dist_to_bullish_ob_h1" if not prefix else f"{p}dist_to_bullish_ob"] = dist_bull_ob
    out[f"{p}dist_to_bearish_ob_h1" if not prefix else f"{p}dist_to_bearish_ob"] = dist_bear_ob
    out[f"{p}dist_to_nearest_fvg_bullish"] = dist_fvg_bull
    out[f"{p}dist_to_nearest_fvg_bearish"] = dist_fvg_bear

    # Keep debug helpers for sample validation (not necessarily in final model cols)
    out["_last_sh_idx"] = _carry_last_idx(n_bars, sh_price, n)
    out["_last_sl_idx"] = _carry_last_idx(n_bars, sl_price, n)
    return out


def _update_structure(
    structure: int,
    sh_hist: list[tuple[int, float]],
    sl_hist: list[tuple[int, float]],
) -> int:
    """Update structure from swing sequence.

    Uptrend (1): last two swing highs form HH, or last two swing lows form HL.
    Downtrend (-1): LH or LL.
    """
    if len(sh_hist) >= 2:
        if sh_hist[-1][1] > sh_hist[-2][1]:
            structure = 1
        elif sh_hist[-1][1] < sh_hist[-2][1]:
            structure = -1
    if len(sl_hist) >= 2:
        if sl_hist[-1][1] > sl_hist[-2][1]:
            structure = 1
        elif sl_hist[-1][1] < sl_hist[-2][1]:
            structure = -1
    return structure


def _carry_last_idx(n_bars: int, swing_price: np.ndarray, n: int) -> np.ndarray:
    """For each t, index of last swing pivot confirmed by t (or -1)."""
    out = np.full(n_bars, -1, dtype=int)
    last = -1
    for t in range(n_bars):
        pivot = t - n
        if pivot >= n and np.isfinite(swing_price[pivot]):
            last = pivot
        out[t] = last
    return out


def compute_h4_bos_features(df_h4: pd.DataFrame, atr_col: str = "atr_h4") -> pd.DataFrame:
    """H4 BOS flags only (broadcast later via merge_asof on Date=close time)."""
    tmp = compute_smc_features(df_h4, atr_col=atr_col, prefix="h4tmp_", cfg=SMCConfig())
    out = pd.DataFrame(
        {
            "Date": tmp["Date"],
            "bos_bullish_h4": tmp["h4tmp_bos_bullish"],
            "bos_bearish_h4": tmp["h4tmp_bos_bearish"],
        }
    )
    return out


def write_smc_sample_validation(
    df_h1: pd.DataFrame,
    smc: pd.DataFrame,
    out_path,
    *,
    n_samples: int = 5,
    seed: int = 42,
    window: int = 8,
) -> None:
    """Text sample check around random rows for manual SMC sanity."""
    rng = np.random.default_rng(seed)
    # Prefer rows where some SMC event fired
    event_mask = (
        (smc["bos_bullish_h1"] == 1)
        | (smc["bos_bearish_h1"] == 1)
        | (smc["choch_bullish_h1"] == 1)
        | (smc["choch_bearish_h1"] == 1)
    )
    candidates = np.flatnonzero(event_mask.to_numpy() & (smc["_last_sh_idx"].to_numpy() >= 0))
    if len(candidates) < n_samples:
        candidates = np.arange(window, len(df_h1) - window)
    picks = rng.choice(candidates, size=min(n_samples, len(candidates)), replace=False)

    lines = [
        "# SMC Feature Validation Samples",
        "",
        "No-lookahead reminder: swing at pivot i is only usable from confirm bar i+N onward.",
        "",
    ]
    for k, t in enumerate(picks, 1):
        t = int(t)
        lo = max(0, t - window)
        hi = min(len(df_h1), t + window + 1)
        lines.append(f"## Sample {k} — focus Date={df_h1['Date'].iloc[t]}")
        lines.append(
            f"- flags @T: bos_bull={int(smc['bos_bullish_h1'].iloc[t])} "
            f"bos_bear={int(smc['bos_bearish_h1'].iloc[t])} "
            f"choch_bull={int(smc['choch_bullish_h1'].iloc[t])} "
            f"choch_bear={int(smc['choch_bearish_h1'].iloc[t])}"
        )
        lines.append(
            f"- dist_sh={smc['dist_to_last_swing_high_h1'].iloc[t]:.3f} "
            f"dist_sl={smc['dist_to_last_swing_low_h1'].iloc[t]:.3f} "
            f"ob_bull={smc['dist_to_bullish_ob_h1'].iloc[t]:.3f} "
            f"ob_bear={smc['dist_to_bearish_ob_h1'].iloc[t]:.3f} "
            f"fvg_bull={smc['dist_to_nearest_fvg_bullish'].iloc[t]:.3f} "
            f"fvg_bear={smc['dist_to_nearest_fvg_bearish'].iloc[t]:.3f}"
        )
        lines.append("")
        lines.append("| idx | Date | O | H | L | C | notes |")
        lines.append("|---:|---|---:|---:|---:|---:|---|")
        for i in range(lo, hi):
            notes = []
            if i == t:
                notes.append("<< T")
            if smc["_last_sh_idx"].iloc[t] == i:
                notes.append("last_SH_pivot_used")
            if smc["_last_sl_idx"].iloc[t] == i:
                notes.append("last_SL_pivot_used")
            if int(smc["bos_bullish_h1"].iloc[i]):
                notes.append("bos_bull")
            if int(smc["bos_bearish_h1"].iloc[i]):
                notes.append("bos_bear")
            if int(smc["choch_bullish_h1"].iloc[i]):
                notes.append("choch_bull")
            if int(smc["choch_bearish_h1"].iloc[i]):
                notes.append("choch_bear")
            lines.append(
                f"| {i} | {df_h1['Date'].iloc[i]} | "
                f"{df_h1['Open'].iloc[i]:.2f} | {df_h1['High'].iloc[i]:.2f} | "
                f"{df_h1['Low'].iloc[i]:.2f} | {df_h1['Close'].iloc[i]:.2f} | "
                f"{' '.join(notes)} |"
            )
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _self_check_no_lookahead() -> None:
    """ponytail: fails if confirmed swing is usable before confirm bar i+N."""
    n = 2
    high = np.array([1.0, 2.0, 5.0, 3.0, 2.5, 2.0, 1.5], dtype=float)
    low = np.array([0.5, 1.0, 2.0, 1.5, 1.0, 0.8, 0.5], dtype=float)
    sh, _ = detect_fractals(high, low, n=n)
    # pivot at i=2 (high=5) needs confirm at i=4
    assert np.isfinite(sh[2])
    carry = _carry_last_idx(len(high), sh, n)
    assert carry[3] == -1  # not yet confirmed
    assert carry[4] == 2  # confirmed at i+n


if __name__ == "__main__":
    _self_check_no_lookahead()
    print("smc self-check OK")
