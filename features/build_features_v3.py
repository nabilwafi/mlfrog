"""
Build feature table v3 = v2 (classic+SMC) + reversal/exhaustion features.

Does NOT overwrite v1/v2.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from ta.trend import EMAIndicator

from features.feature_engineering import load_ohlc_csv
from features.reversal import (
    REVERSAL_FEATURE_COLS,
    compute_reversal_features,
    write_reversal_samples,
)


def build_features_v3(
    *,
    v2_path: Path,
    h1_raw: Path,
    h4_raw: Path,
    d1_raw: Path,
    out_path: Path,
    sample_report: Path,
) -> pd.DataFrame:
    v2 = pd.read_parquet(v2_path)
    v2["Date"] = pd.to_datetime(v2["Date"], utc=True)

    h1 = load_ohlc_csv(h1_raw)
    h1["Date"] = h1["Date"] + pd.Timedelta(hours=1)
    h4 = load_ohlc_csv(h4_raw)
    h4["Date"] = h4["Date"] + pd.Timedelta(hours=4)
    d1 = load_ohlc_csv(d1_raw)
    d1["Date"] = d1["Date"] + pd.Timedelta(days=1)

    h4["ema_h4_ref"] = EMAIndicator(close=h4["Close"], window=20, fillna=False).ema_indicator()
    d1["ema_d1_ref"] = EMAIndicator(close=d1["Close"], window=20, fillna=False).ema_indicator()

    # Align OHLC + HTF EMA refs onto v2 close-time rows
    base = v2.merge(
        h1[["Date", "Open", "High", "Low", "Close"]],
        on="Date",
        how="left",
        suffixes=("", "_h1"),
    )
    # if Open already in v2 from somewhere, prefer merge columns
    for c in ("Open", "High", "Low", "Close"):
        if f"{c}_h1" in base.columns:
            base[c] = base[f"{c}_h1"]
            base.drop(columns=[f"{c}_h1"], inplace=True)

    base = pd.merge_asof(
        base.sort_values("Date"),
        h4[["Date", "ema_h4_ref"]].sort_values("Date"),
        on="Date",
        direction="backward",
    )
    base = pd.merge_asof(
        base.sort_values("Date"),
        d1[["Date", "ema_d1_ref"]].sort_values("Date"),
        on="Date",
        direction="backward",
    )

    if "rsi_h1" not in base.columns:
        raise ValueError("v2 missing rsi_h1")

    print("Computing reversal features...")
    rev = compute_reversal_features(base)
    write_reversal_samples(base, rev, sample_report)
    print(f"Wrote samples -> {sample_report}")

    merged = v2.merge(rev, on="Date", how="left")
    before = len(merged)
    # drop rows missing z-scores (warmup)
    merged = merged.dropna(subset=["dist_from_ema_h4_zscore", "dist_from_ema_d1_zscore"]).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path, index=False)
    print(f"Saved v3 -> {out_path} rows={len(merged)} (from {before})")
    print(f"Date range: {merged['Date'].min()} -> {merged['Date'].max()}")
    print(f"New cols: {REVERSAL_FEATURE_COLS}")
    return merged


def main() -> None:
    base = Path("data")
    build_features_v3(
        v2_path=base / "features/xauusd_h1_h4_d1_features_v2.parquet",
        h1_raw=base / "raw/XAUUSD_H1.csv",
        h4_raw=base / "raw/XAUUSD_H4.csv",
        d1_raw=base / "raw/XAUUSD_D1.csv",
        out_path=base / "features/xauusd_h1_h4_d1_features_v3.parquet",
        sample_report=base / "reports/reversal_feature_validation_samples.md",
    )


if __name__ == "__main__":
    main()
