from __future__ import annotations

"""
Triple-barrier labeling for XAUUSD using H1 candles only.

For each H1 row (entry at H1 close):
LONG:
  TP = close + tp_multiplier * atr_h1
  SL = close - sl_multiplier * atr_h1
SHORT:
  TP = close - tp_multiplier * atr_h1
  SL = close + sl_multiplier * atr_h1

We scan forward up to `horizon_bars` using High/Low touch logic.
Ambiguous touch within the same candle is resolved conservatively as SL first.

Outputs:
- label_long, label_short in {1, 0, -1} where:
  1 = win (TP first)
  0 = loss (SL first)
  -1 = timeout (no touch within horizon)
- bars_to_resolution_long/short: number of bars until resolution

Also prints and saves distribution reports plus implied breakeven checks.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LabelConfig:
    sl_atr_multiplier: float = 1.5
    tp_atr_multiplier: float = 2.0
    horizon_bars: int = 8


def load_h1_for_labeling(h1_path: Path) -> pd.DataFrame:
    """Load H1 OHLC and shift Date from OPEN to CLOSE time (Date + 1 hour)."""
    df = pd.read_csv(h1_path, parse_dates=["Date"])
    if getattr(df["Date"].dtype, "tz", None) is None:
        df["Date"] = pd.to_datetime(df["Date"], utc=True)
    df["Date"] = df["Date"] + pd.Timedelta(hours=1)
    df = df.sort_values("Date").reset_index(drop=True)

    needed = ["Date", "High", "Low", "Close"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"H1 CSV missing columns: {missing}")
    return df[needed]


def label_triple_barrier(
    df_feat_with_atr_and_session: pd.DataFrame,
    df_h1_ohlc: pd.DataFrame,
    *,
    sl_atr_multiplier: float,
    tp_atr_multiplier: float,
    horizon_bars: int,
) -> pd.DataFrame:
    """Compute triple-barrier labels for LONG and SHORT directions.

    Input:
    - df_feat_with_atr_and_session must contain:
      Date, atr_h1, session
    - df_h1_ohlc must contain:
      Date, High, Low, Close
    """
    base_cols = ["Date", "atr_h1", "session"]
    missing_base = [c for c in base_cols if c not in df_feat_with_atr_and_session.columns]
    if missing_base:
        raise ValueError(f"Feature table missing columns: {missing_base}")

    merged = df_feat_with_atr_and_session.merge(df_h1_ohlc, on="Date", how="inner")
    merged = merged.sort_values("Date").reset_index(drop=True)

    # Extract numpy arrays for speed.
    high = merged["High"].to_numpy(dtype=float)
    low = merged["Low"].to_numpy(dtype=float)
    close = merged["Close"].to_numpy(dtype=float)
    atr = merged["atr_h1"].to_numpy(dtype=float)

    n = len(merged)
    labels_long = np.full(n, -1, dtype=int)
    labels_short = np.full(n, -1, dtype=int)
    bars_long = np.full(n, horizon_bars, dtype=int)
    bars_short = np.full(n, horizon_bars, dtype=int)

    max_j_exclusive = n  # j is inclusive in loops; we cap internally
    for i in range(n):
        if not np.isfinite(atr[i]):
            continue

        entry = close[i]

        # LONG levels
        tp_long = entry + tp_atr_multiplier * atr[i]
        sl_long = entry - sl_atr_multiplier * atr[i]

        # SHORT levels
        tp_short = entry - tp_atr_multiplier * atr[i]
        sl_short = entry + sl_atr_multiplier * atr[i]

        end_j = min(n - 1, i + horizon_bars)

        # LONG scan
        resolved = False
        for j in range(i + 1, end_j + 1):
            # Conservative ambiguity resolution: SL first within the same candle.
            if low[j] <= sl_long:
                labels_long[i] = 0
                bars_long[i] = j - i
                resolved = True
                break
            if high[j] >= tp_long:
                labels_long[i] = 1
                bars_long[i] = j - i
                resolved = True
                break

        if not resolved:
            labels_long[i] = -1
            bars_long[i] = min(horizon_bars, n - 1 - i)

        # SHORT scan
        resolved = False
        for j in range(i + 1, end_j + 1):
            if high[j] >= sl_short:
                labels_short[i] = 0
                bars_short[i] = j - i
                resolved = True
                break
            if low[j] <= tp_short:
                labels_short[i] = 1
                bars_short[i] = j - i
                resolved = True
                break

        if not resolved:
            labels_short[i] = -1
            bars_short[i] = min(horizon_bars, n - 1 - i)

    out = merged.copy()
    out["label_long"] = labels_long
    out["bars_to_resolution_long"] = bars_long
    out["label_short"] = labels_short
    out["bars_to_resolution_short"] = bars_short
    return out


def _distribution_stats(labels: pd.Series, bars_to_resolution: pd.Series) -> Dict[str, float]:
    """Compute label distribution and basic stats.

    labels are expected to be in {1, 0, -1}.
    """
    import math
    n = len(labels)
    wins = int((labels == 1).sum())
    losses = int((labels == 0).sum())
    timeouts = int((labels == -1).sum())

    def pct(x: int) -> float:
        return float(x) * 100.0 / float(n) if n else 0.0

    win_rate_train = wins / (wins + losses) if (wins + losses) else 0.0

    win_bars = bars_to_resolution[labels == 1].astype(float)
    loss_bars = bars_to_resolution[labels == 0].astype(float)

    return {
        "count_win": wins,
        "count_loss": losses,
        "count_timeout": timeouts,
        "pct_win": pct(wins),
        "pct_loss": pct(losses),
        "pct_timeout": pct(timeouts),
        "win_rate_train": float(win_rate_train),
        "avg_bars_win": float(win_bars.mean()) if len(win_bars) else math.nan,
        "median_bars_win": float(win_bars.median()) if len(win_bars) else math.nan,
        "avg_bars_loss": float(loss_bars.mean()) if len(loss_bars) else math.nan,
        "median_bars_loss": float(loss_bars.median()) if len(loss_bars) else math.nan,
    }


def implied_breakeven_winrate(sl_mult: float, tp_mult: float) -> float:
    """Compute min win-rate for breakeven:
    min_winrate = SL / (SL + TP)
    With SL = sl_mult * ATR and TP = tp_mult * ATR, ATR cancels out.
    """
    return float(sl_mult) / float(sl_mult + tp_mult)


def make_label_report_markdown(
    df_labeled: pd.DataFrame,
    *,
    sl_mult: float,
    tp_mult: float,
    horizon: int,
) -> str:
    """Create a markdown report string for label distribution and breakeven checks."""
    import math

    long_stats = _distribution_stats(df_labeled["label_long"], df_labeled["bars_to_resolution_long"])
    short_stats = _distribution_stats(df_labeled["label_short"], df_labeled["bars_to_resolution_short"])

    be = implied_breakeven_winrate(sl_mult, tp_mult)
    long_edge = long_stats["win_rate_train"] - be
    short_edge = short_stats["win_rate_train"] - be

    def flag(edge: float) -> str:
        if edge >= 0:
            return "ABOVE (edge ok)"
        return "BELOW (needs better params)"

    # Session breakdown
    session_rows: List[str] = []
    for session, g in df_labeled.groupby("session"):
        lg = _distribution_stats(g["label_long"], g["bars_to_resolution_long"])
        sg = _distribution_stats(g["label_short"], g["bars_to_resolution_short"])
        session_rows.append(
            f"| {session} | {lg['pct_win']:.2f}% | {lg['pct_loss']:.2f}% | {lg['pct_timeout']:.2f}% | {sg['pct_win']:.2f}% | {sg['pct_loss']:.2f}% | {sg['pct_timeout']:.2f}% |"
        )

    header = [
        f"# Triple-Barrier Report",
        f"- sl_mult (ATR): {sl_mult}",
        f"- tp_mult (ATR): {tp_mult}",
        f"- horizon_bars: {horizon}",
        "",
        "## LONG label distribution",
        f"- win (1): {long_stats['count_win']} ({long_stats['pct_win']:.2f}%)",
        f"- loss (0): {long_stats['count_loss']} ({long_stats['pct_loss']:.2f}%)",
        f"- timeout (-1): {long_stats['count_timeout']} ({long_stats['pct_timeout']:.2f}%)",
        f"- win_rate_train (win/(win+loss)): {long_stats['win_rate_train']:.4f}",
        f"- avg/median bars_to_resolution (win): {long_stats['avg_bars_win']:.3f} / {long_stats['median_bars_win']:.3f}",
        f"- avg/median bars_to_resolution (loss): {long_stats['avg_bars_loss']:.3f} / {long_stats['median_bars_loss']:.3f}",
        "",
        "## SHORT label distribution",
        f"- win (1): {short_stats['count_win']} ({short_stats['pct_win']:.2f}%)",
        f"- loss (0): {short_stats['count_loss']} ({short_stats['pct_loss']:.2f}%)",
        f"- timeout (-1): {short_stats['count_timeout']} ({short_stats['pct_timeout']:.2f}%)",
        f"- win_rate_train (win/(win+loss)): {short_stats['win_rate_train']:.4f}",
        f"- avg/median bars_to_resolution (win): {short_stats['avg_bars_win']:.3f} / {short_stats['median_bars_win']:.3f}",
        f"- avg/median bars_to_resolution (loss): {short_stats['avg_bars_loss']:.3f} / {short_stats['median_bars_loss']:.3f}",
        "",
        "## Breakeven check (implied win-rate)",
        f"- min_winrate = SL / (SL + TP) = {be:.4f}",
        f"- LONG win_rate_train={long_stats['win_rate_train']:.4f} -> {flag(long_edge)}",
        f"- SHORT win_rate_train={short_stats['win_rate_train']:.4f} -> {flag(short_edge)}",
        "",
        "## Session breakdown (pct of labels)",
        "| session | LONG win% | LONG loss% | LONG timeout% | SHORT win% | SHORT loss% | SHORT timeout% |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    body = session_rows
    return "\n".join(header + body)


def run_labeling_once(
    *,
    features_path: Path,
    h1_path: Path,
    out_parquet_path: Path,
    report_md_path: Path,
    cfg: LabelConfig = LabelConfig(),
) -> pd.DataFrame:
    """Run triple-barrier labeling for one parameter set and save results + report."""
    df_feat = pd.read_parquet(features_path)
    df_h1 = load_h1_for_labeling(h1_path)

    df_labeled = label_triple_barrier(
        df_feat,
        df_h1,
        sl_atr_multiplier=cfg.sl_atr_multiplier,
        tp_atr_multiplier=cfg.tp_atr_multiplier,
        horizon_bars=cfg.horizon_bars,
    )

    report = make_label_report_markdown(
        df_labeled,
        sl_mult=cfg.sl_atr_multiplier,
        tp_mult=cfg.tp_atr_multiplier,
        horizon=cfg.horizon_bars,
    )
    print(report)

    report_md_path.parent.mkdir(parents=True, exist_ok=True)
    report_md_path.write_text(report, encoding="utf-8")

    out_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df_labeled.to_parquet(out_parquet_path, index=False)

    return df_labeled


def run_labeling_grid_search(
    *,
    features_path: Path,
    h1_path: Path,
    sl_multipliers: List[float],
    tp_multipliers: List[float],
    horizons: List[int],
) -> pd.DataFrame:
    """Run labeling for multiple parameter combinations and summarize label health.

    NOTE: This is optional and may be slow for large datasets.
    """
    df_feat = pd.read_parquet(features_path)
    df_h1 = load_h1_for_labeling(h1_path)

    results: List[Dict[str, float]] = []
    for sl in sl_multipliers:
        for tp in tp_multipliers:
            be = implied_breakeven_winrate(sl, tp)
            for horizon in horizons:
                cfg = LabelConfig(sl_atr_multiplier=sl, tp_atr_multiplier=tp, horizon_bars=horizon)
                df_labeled = label_triple_barrier(
                    df_feat,
                    df_h1,
                    sl_atr_multiplier=cfg.sl_atr_multiplier,
                    tp_atr_multiplier=cfg.tp_atr_multiplier,
                    horizon_bars=cfg.horizon_bars,
                )

                long_stats = _distribution_stats(df_labeled["label_long"], df_labeled["bars_to_resolution_long"])
                short_stats = _distribution_stats(df_labeled["label_short"], df_labeled["bars_to_resolution_short"])

                results.append(
                    {
                        "sl_mult": float(sl),
                        "tp_mult": float(tp),
                        "horizon": int(horizon),
                        "winrate_long": float(long_stats["win_rate_train"]),
                        "timeout_pct_long": float(long_stats["pct_timeout"]),
                        "winrate_short": float(short_stats["win_rate_train"]),
                        "timeout_pct_short": float(short_stats["pct_timeout"]),
                        "breakeven_winrate": float(be),
                        "edge_vs_breakeven": float(long_stats["win_rate_train"] - be),
                    }
                )

    return pd.DataFrame(results).sort_values(["horizon", "sl_mult", "tp_mult"]).reset_index(drop=True)


def main() -> None:
    base = Path("data")
    features_path = base / "features/xauusd_h1_h4_d1_features.parquet"
    h1_path = base / "raw/XAUUSD_H1.csv"

    # --- grid search ---
    # once (single parameter labeling) sengaja di-comment sesuai permintaan kamu.
    cfg = LabelConfig(sl_atr_multiplier=1.5, tp_atr_multiplier=2.0, horizon_bars=8)
    out_parquet = base / "labels/xauusd_triple_barrier_labels.parquet"
    report_md = base / "reports/xauusd_triple_barrier_report.md"
    run_labeling_once(
        features_path=features_path,
        h1_path=h1_path,
        out_parquet_path=out_parquet,
        report_md_path=report_md,
        cfg=cfg,
    )

    # sl_multipliers = [1.0, 1.5, 2.0]
    # tp_multipliers = [2.0, 3.0, 4.0]
    # horizons = [4, 6, 8]

    # df = run_labeling_grid_search(
    #     features_path=features_path,
    #     h1_path=h1_path,
    #     sl_multipliers=sl_multipliers,
    #     tp_multipliers=tp_multipliers,
    #     horizons=horizons,
    # )
    # print(df)

    # out_grid_parquet = base / "reports/xauusd_triple_barrier_grid_search.parquet"
    # out_grid_parquet.parent.mkdir(parents=True, exist_ok=True)
    # df.to_parquet(out_grid_parquet, index=False)
    # print("saved", out_grid_parquet)


if __name__ == "__main__":
    main()

