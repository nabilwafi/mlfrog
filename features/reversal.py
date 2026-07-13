"""
Reversal / mean-reversion / exhaustion features for H1 (v3).

No-lookahead rules:
- RSI divergence: only uses swing pivots that are already confirmed (confirm=pivot+N).
  Divergence becomes available at the confirm bar of the *second* swing.
- Liquidity sweep: wick pierce of a previously confirmed swing + close reject on the
  same bar. Pattern is complete at that bar's close (no future bars required).
  Flag = any such event in the last lookback bars including t.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from features.smc import detect_fractals


@dataclass(frozen=True)
class ReversalConfig:
    fractal_n: int = 2
    sweep_lookback: int = 10
    rsi_ob: float = 70.0
    rsi_os: float = 30.0
    zscore_window: int = 100
    atr_pct_window: int = 100
    atr_contraction_pct: float = 20.0


REVERSAL_FEATURE_COLS = [
    "bearish_divergence_h1",
    "bullish_divergence_h1",
    "rsi_extreme_duration_h1",
    "rsi_extreme_duration_oversold",
    "consecutive_higher_closes_h1",
    "consecutive_lower_closes_h1",
    "liquidity_sweep_high_h1",
    "liquidity_sweep_low_h1",
    "dist_from_ema_h4_zscore",
    "dist_from_ema_d1_zscore",
    "atr_contraction_flag_h1",
]


def compute_reversal_features(
    df_h1: pd.DataFrame,
    *,
    ema_h4: pd.Series | None = None,
    ema_d1: pd.Series | None = None,
    cfg: ReversalConfig = ReversalConfig(),
) -> pd.DataFrame:
    """df_h1 must have Date, OHLC, atr_h1, rsi_h1; optional ema columns for z-scores.

    If ema_h4 / ema_d1 not provided, expect columns ema_h4_ref / ema_d1_ref on df
    (already asof-merged to H1 close time).
    """
    need = ["Date", "Open", "High", "Low", "Close", "atr_h1", "rsi_h1"]
    missing = [c for c in need if c not in df_h1.columns]
    if missing:
        raise ValueError(f"reversal input missing: {missing}")

    n = len(df_h1)
    high = df_h1["High"].to_numpy(float)
    low = df_h1["Low"].to_numpy(float)
    close = df_h1["Close"].to_numpy(float)
    open_ = df_h1["Open"].to_numpy(float)
    rsi = df_h1["rsi_h1"].to_numpy(float)
    atr = df_h1["atr_h1"].to_numpy(float)
    fn = cfg.fractal_n

    sh_price, sl_price = detect_fractals(high, low, n=fn)
    # RSI at pivot (same index as swing); usable only after confirm
    sh_rsi = np.where(np.isfinite(sh_price), rsi, np.nan)
    sl_rsi = np.where(np.isfinite(sl_price), rsi, np.nan)

    bear_div = np.zeros(n, dtype=np.int8)
    bull_div = np.zeros(n, dtype=np.int8)
    sweep_hi = np.zeros(n, dtype=np.int8)
    sweep_lo = np.zeros(n, dtype=np.int8)

    # Confirmed swing history: (pivot_idx, price, rsi)
    sh_hist: list[tuple[int, float, float]] = []
    sl_hist: list[tuple[int, float, float]] = []
    last_sh_px = np.nan
    last_sl_px = np.nan
    recent_sweep_hi: list[int] = []
    recent_sweep_lo: list[int] = []

    rsi_ob_dur = np.zeros(n, dtype=float)
    rsi_os_dur = np.zeros(n, dtype=float)
    streak_up = np.zeros(n, dtype=float)
    streak_dn = np.zeros(n, dtype=float)
    ob_run = 0
    os_run = 0
    up_run = 0
    dn_run = 0

    for t in range(n):
        # --- RSI extreme duration / close streaks ---
        if np.isfinite(rsi[t]) and rsi[t] > cfg.rsi_ob:
            ob_run += 1
        else:
            ob_run = 0
        if np.isfinite(rsi[t]) and rsi[t] < cfg.rsi_os:
            os_run += 1
        else:
            os_run = 0
        rsi_ob_dur[t] = float(ob_run)
        rsi_os_dur[t] = float(os_run)

        if t > 0 and close[t] > close[t - 1]:
            up_run += 1
            dn_run = 0
        elif t > 0 and close[t] < close[t - 1]:
            dn_run += 1
            up_run = 0
        else:
            up_run = 0
            dn_run = 0
        streak_up[t] = float(up_run)
        streak_dn[t] = float(dn_run)

        # --- Newly confirmed swings at t ---
        pivot = t - fn
        if pivot >= fn:
            if np.isfinite(sh_price[pivot]) and np.isfinite(sh_rsi[pivot]):
                px, rv = float(sh_price[pivot]), float(sh_rsi[pivot])
                if len(sh_hist) >= 1:
                    prev_i, prev_px, prev_r = sh_hist[-1]
                    # bearish div: HH price + LH RSI (confirmed now)
                    if px > prev_px and rv < prev_r:
                        bear_div[t] = 1
                sh_hist.append((pivot, px, rv))
                last_sh_px = px
            if np.isfinite(sl_price[pivot]) and np.isfinite(sl_rsi[pivot]):
                px, rv = float(sl_price[pivot]), float(sl_rsi[pivot])
                if len(sl_hist) >= 1:
                    prev_i, prev_px, prev_r = sl_hist[-1]
                    if px < prev_px and rv > prev_r:
                        bull_div[t] = 1
                sl_hist.append((pivot, px, rv))
                last_sl_px = px

        # --- Liquidity sweep (complete on this bar vs last confirmed swing) ---
        if np.isfinite(last_sh_px):
            # wick above swing high, close back at/below
            if high[t] > last_sh_px and close[t] <= last_sh_px:
                recent_sweep_hi.append(t)
        if np.isfinite(last_sl_px):
            if low[t] < last_sl_px and close[t] >= last_sl_px:
                recent_sweep_lo.append(t)

        lo = t - cfg.sweep_lookback + 1
        recent_sweep_hi = [i for i in recent_sweep_hi if i >= lo]
        recent_sweep_lo = [i for i in recent_sweep_lo if i >= lo]
        if recent_sweep_hi:
            sweep_hi[t] = 1
        if recent_sweep_lo:
            sweep_lo[t] = 1

    # --- HTF overextension z-scores ---
    if ema_h4 is None:
        ema_h4 = df_h1["ema_h4_ref"] if "ema_h4_ref" in df_h1.columns else pd.Series(np.nan, index=df_h1.index)
    if ema_d1 is None:
        ema_d1 = df_h1["ema_d1_ref"] if "ema_d1_ref" in df_h1.columns else pd.Series(np.nan, index=df_h1.index)

    dist_h4 = (pd.Series(close) - ema_h4.reset_index(drop=True)) 
    dist_d1 = (pd.Series(close) - ema_d1.reset_index(drop=True))
    z_h4 = _rolling_zscore(dist_h4, cfg.zscore_window)
    z_d1 = _rolling_zscore(dist_d1, cfg.zscore_window)

    # ATR contraction: ATR below 20th pct of rolling window
    atr_s = pd.Series(atr)
    q = atr_s.rolling(cfg.atr_pct_window, min_periods=cfg.atr_pct_window).quantile(
        cfg.atr_contraction_pct / 100.0
    )
    atr_flag = ((atr_s <= q) & atr_s.notna() & q.notna()).astype(np.int8).to_numpy()

    out = pd.DataFrame(
        {
            "Date": df_h1["Date"].to_numpy(),
            "bearish_divergence_h1": bear_div,
            "bullish_divergence_h1": bull_div,
            "rsi_extreme_duration_h1": rsi_ob_dur,
            "rsi_extreme_duration_oversold": rsi_os_dur,
            "consecutive_higher_closes_h1": streak_up,
            "consecutive_lower_closes_h1": streak_dn,
            "liquidity_sweep_high_h1": sweep_hi,
            "liquidity_sweep_low_h1": sweep_lo,
            "dist_from_ema_h4_zscore": z_h4.to_numpy(),
            "dist_from_ema_d1_zscore": z_d1.to_numpy(),
            "atr_contraction_flag_h1": atr_flag,
        }
    )
    return out


def _rolling_zscore(s: pd.Series, window: int) -> pd.Series:
    mu = s.rolling(window, min_periods=window).mean()
    sd = s.rolling(window, min_periods=window).std(ddof=0)
    z = (s - mu) / sd.replace(0, np.nan)
    return z


def write_reversal_samples(
    df_h1: pd.DataFrame,
    rev: pd.DataFrame,
    out_path: Path,
    *,
    n_samples: int = 3,
    seed: int = 7,
    window: int = 6,
) -> None:
    rng = np.random.default_rng(seed)
    lines = [
        "# Reversal Feature Validation Samples",
        "",
        "No-lookahead: divergence uses confirmed swings only (pivot+N). "
        "Liquidity sweep completes on pierce+reject close vs last confirmed swing.",
        "",
    ]
    checks = [
        ("bearish_divergence_h1", "Bearish RSI divergence"),
        ("bullish_divergence_h1", "Bullish RSI divergence"),
        ("liquidity_sweep_high_h1", "Liquidity sweep high"),
        ("liquidity_sweep_low_h1", "Liquidity sweep low"),
        ("atr_contraction_flag_h1", "ATR contraction"),
    ]
    for col, title in checks:
        cand = np.flatnonzero(rev[col].to_numpy() == 1)
        if len(cand) == 0:
            lines.append(f"## {title}\n\n(no positive samples)\n")
            continue
        picks = rng.choice(cand, size=min(n_samples, len(cand)), replace=False)
        lines.append(f"## {title}")
        for k, t in enumerate(picks, 1):
            t = int(t)
            lo = max(0, t - window)
            hi = min(len(df_h1), t + 1)  # no future in sample table past T
            lines.append(f"### Sample {k} @ {df_h1['Date'].iloc[t]}")
            lines.append(
                f"- rsi={df_h1['rsi_h1'].iloc[t]:.2f} "
                f"ob_dur={rev['rsi_extreme_duration_h1'].iloc[t]:.0f} "
                f"streak_up={rev['consecutive_higher_closes_h1'].iloc[t]:.0f} "
                f"z_h4={rev['dist_from_ema_h4_zscore'].iloc[t]:.3f}"
            )
            lines.append("| idx | Date | O | H | L | C | notes |")
            lines.append("|---:|---|---:|---:|---:|---:|---|")
            for i in range(lo, hi):
                notes = []
                if i == t:
                    notes.append("<< T")
                if int(rev[col].iloc[i]):
                    notes.append(col)
                lines.append(
                    f"| {i} | {df_h1['Date'].iloc[i]} | "
                    f"{df_h1['Open'].iloc[i]:.2f} | {df_h1['High'].iloc[i]:.2f} | "
                    f"{df_h1['Low'].iloc[i]:.2f} | {df_h1['Close'].iloc[i]:.2f} | "
                    f"{' '.join(notes)} |"
                )
            lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
